"""Batch profiler contracts, configuration, shared sample, and orchestration."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations, islice
from collections import Counter
from typing import Any
import numpy as np
import pandas as pd
from pandas.api import types as pt


@dataclass
class SectionResult:
    name: str
    data: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class ProfileConfig:
    batch_size: int = 50_000
    sample_size: int = 10_000
    seed: int = 42
    max_unique: int = 10_000
    max_pairs: int = 100
    max_categories: int = 50
    top_k: int = 10
    histogram_bins: int = 30
    roles: dict = field(default_factory=dict)
    pairs: Any = None
    correlations: bool = True
    interactions: bool = True
    duplicate_limit: int = 0
    categorical_threshold: int = 10

    def __post_init__(self):
        for name in ('batch_size', 'sample_size', 'max_unique', 'max_categories', 'top_k', 'histogram_bins'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('max_pairs', 'duplicate_limit', 'categorical_threshold'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if any(x not in {'numeric', 'category', 'text', 'datetime', 'timedelta'} for x in self.roles.values()):
            raise ValueError('Unsupported role override')


def infer_role(series):
    dtype = series.dtype
    if pt.is_complex_dtype(dtype):
        raise TypeError('Complex numbers need separate analysis; project to real features first')
    if pt.is_datetime64_any_dtype(dtype):
        return 'datetime'
    if pt.is_timedelta64_dtype(dtype):
        return 'timedelta'
    if pt.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return 'category'
    if pt.is_numeric_dtype(dtype):
        return 'numeric'
    return 'text'


def finite_values(series):
    values = pd.to_numeric(series, errors='raise').to_numpy(dtype=float, na_value=np.nan)
    return values[np.isfinite(values)]


class Context:
    def __init__(self, config):
        self.config = config
        self.rows = 0
        self.columns = None
        self.roles = {}
        self.category_counts = {}
        self.dtypes = {}
        self.pairs = []
        self.omitted_pairs = 0
        self.sample = pd.DataFrame()
        self._keys = np.empty(0)
        self.rng = np.random.default_rng(config.seed)

    def accept(self, frame):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError('Each batch must be a pandas DataFrame')
        if not frame.columns.is_unique or any(not isinstance(x, str) for x in frame.columns):
            raise ValueError('Column names must be unique strings')
        if self.columns is None:
            self.columns = list(frame.columns)
            unknown = set(self.config.roles) - set(self.columns)
            if unknown:
                raise ValueError(f'Unknown role columns: {unknown}')
            self.sample = frame.iloc[:0].copy()
            for col in self.columns:
                self.roles[col] = self.config.roles[col] if col in self.config.roles else infer_role(frame[col])
                self.dtypes[col] = set()
                if col not in self.config.roles and self.roles[col] in {'numeric', 'text'} and self.config.categorical_threshold:
                    self.category_counts[col] = Counter()
            self.select_pairs(allow_text=True)
        if set(frame.columns) != set(self.columns):
            raise ValueError('Batch column names changed')
        for col in self.columns:
            self.dtypes[col].add(str(frame[col].dtype))
            if frame[col].notna().any() and col not in self.config.roles:
                actual = infer_role(frame[col])
                if actual != self.roles[col]:
                    raise TypeError(f'Role changed for {col!r}; specify loader dtype and/or roles')
        for col, counts in list(self.category_counts.items()):
            if counts is None:
                continue
            for value in frame[col].dropna():
                counts[value] += 1
                if len(counts) > self.config.categorical_threshold:
                    self.category_counts[col] = None
                    break
        return frame.loc[:, self.columns]

    def select_pairs(self, allow_text=False):
        eligible = [c for c in self.columns if self.roles[c] in {'numeric', 'category'}]
        if self.config.pairs is None:
            total = len(eligible) * (len(eligible)-1)//2
            self.pairs = list(islice(combinations(eligible, 2), self.config.max_pairs))
            self.omitted_pairs = max(0, total-len(self.pairs))
        else:
            seen = set()
            self.pairs = []
            for a, b in self.config.pairs:
                allowed = eligible + ([c for c in self.columns if self.roles[c] == 'text'] if allow_text else [])
                if a == b or a not in allowed or b not in allowed:
                    raise ValueError('Pairs must reference two distinct numeric/category columns')
                key = frozenset((a, b))
                if key not in seen:
                    seen.add(key)
                    self.pairs.append((a, b))
            if len(self.pairs) > self.config.max_pairs:
                raise ValueError('Explicit pairs exceed max_pairs')

    def finalize_roles(self):
        # Full-stream cardinality, never inferred from the first batch or sample.
        for col, counts in self.category_counts.items():
            if counts is not None and 1 <= len(counts) <= self.config.categorical_threshold:
                self.roles[col] = 'category'
        self.select_pairs()

    def retain_sample(self, frame):
        priorities = self.rng.random(len(frame))
        keys = np.concatenate([self._keys, priorities])
        candidates = pd.concat([self.sample, frame], ignore_index=True)
        idx = np.argsort(keys, kind='stable')[:self.config.sample_size]
        self._keys = keys[idx]
        self.sample = candidates.iloc[idx].copy().reset_index(drop=True)
        self.rows += len(frame)

    def sample_metadata(self):
        return {'method': 'exact' if self.rows <= self.config.sample_size else 'sampled',
                'rows_examined': len(self.sample), 'population_rows': self.rows,
                'sampling': 'uniform random priorities without replacement', 'seed': self.config.seed}


class ProfilerComponent(ABC):
    name: str
    def __init__(self, context):
        self.context = context
    @abstractmethod
    def update(self, batch):
        """Consume a batch without retaining it."""
    @abstractmethod
    def finalize(self, results):
        """Return a SectionResult; dependencies finalize earlier."""


class BaseProfiler(ABC):
    def __init__(self, source, config=None):
        self.source = source
        self.config = config or ProfileConfig()

    @abstractmethod
    def component_types(self):
        """Return classes in dependency order; instances are per run."""

    def run(self):
        from ..loader import open_dataset
        context = Context(self.config)
        components = [cls(context) for cls in self.component_types()]
        source = self.source
        if hasattr(source, 'iter_batches'):
            iterator = iter(source.iter_batches(batch_size=self.config.batch_size))
        elif isinstance(source, (pd.DataFrame, str)) or hasattr(source, '__fspath__'):
            iterator = iter(open_dataset(source).iter_batches(batch_size=self.config.batch_size))
        else:
            iterator = iter(source)
        try:
            for batch in iterator:
                batch = context.accept(batch)
                for component in components:
                    component.update(batch)
                context.retain_sample(batch)
        finally:
            close = getattr(iterator, 'close', None)
            if close:
                close()
        if context.columns is None:
            empty = source.iloc[:0] if isinstance(source, pd.DataFrame) else pd.DataFrame(
                columns=list(source.schema()) if hasattr(source, 'schema') else [])
            context.accept(empty)
            for component in components:
                component.update(empty)
        context.finalize_roles()
        results = {}
        for component in components:
            results[component.name] = component.finalize(results)
        return results
