from collections import Counter
import math
import numpy as np
import pandas as pd
from .base import Transform, batches


def scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or not isinstance(value, (str, bool, int, float)):
        raise ValueError('Fill values must be non-null JSON scalars')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Fill values must be finite')
    return value


class FillMissing(Transform):
    def __init__(self, columns, strategy='constant', value=None, fallback=None, max_unique=100_000):
        super().__init__(columns)
        if strategy not in {'constant', 'mean', 'median', 'mode'}:
            raise ValueError('Invalid imputation strategy')
        if type(max_unique) is not int or max_unique < 1:
            raise ValueError('max_unique must be a positive integer')
        self.strategy, self.max_unique = strategy, max_unique
        self.fallback = None if fallback is None else scalar(fallback)
        self.requires_fit = strategy != 'constant'
        self.fills = {c: scalar(value) for c in self.columns} if not self.requires_fit else {}
        self.counts = {}

    def fit(self, factory):
        if not self.requires_fit:
            return self
        counters = {c: Counter() for c in self.columns}
        means = {c: 0.0 for c in self.columns}
        counts = {c: 0 for c in self.columns}
        with batches(factory) as iterator:
            for frame in iterator:
                self.output_columns(frame.columns)
                for c in self.columns:
                    present = frame[c].dropna()
                    if self.strategy in {'mean', 'median'}:
                        present = pd.to_numeric(present, errors='raise')
                        if not np.isfinite(present.to_numpy(dtype=float)).all():
                            raise ValueError(f'{c}: resolve nonfinite values before fitting')
                    if self.strategy == 'mean':
                        n = len(present)
                        if n:
                            total = counts[c]+n
                            means[c] += (float(present.mean())-means[c])*(n/total)
                            counts[c] = total
                    else:
                        for item in present:
                            item = scalar(item)
                            counters[c][item] += 1
                            if len(counters[c]) > self.max_unique:
                                raise ValueError(f'{c}: exact {self.strategy} exceeds max_unique={self.max_unique}; '
                                                 'increase the cap or choose a different strategy')
                        counts[c] += len(present)
        fills = {}
        for c in self.columns:
            if not counts[c]:
                if self.fallback is None:
                    raise ValueError(f'{c}: no nonmissing fitting values; supply an explicit fallback')
                fills[c] = self.fallback
            elif self.strategy == 'mean':
                fills[c] = scalar(means[c])
            elif self.strategy == 'mode':
                maximum = max(counters[c].values())
                tied = [v for v, n in counters[c].items() if n == maximum]
                fills[c] = sorted(tied, key=lambda v: (type(v).__name__, repr(v)))[0]
            else:
                left, right = (counts[c]-1)//2, counts[c]//2
                found, cumulative = [], 0
                for value, n in sorted(counters[c].items()):
                    if cumulative <= left < cumulative+n:
                        found.append(float(value))
                    if cumulative <= right < cumulative+n:
                        found.append(float(value))
                    cumulative += n
                    if cumulative > right:
                        break
                fills[c] = scalar(found[0]/2+found[1]/2)
        self.fills, self.counts = fills, counts
        return self

    def state(self):
        return {'fills': dict(self.fills), 'nonmissing_fit_rows': dict(self.counts),
                'strategy': self.strategy, 'computation': 'exact',
                'mode_ties': 'type name then repr'}

    def restore(self, state):
        if state.get('strategy') != self.strategy or set(state.get('fills', {})) != set(self.columns):
            raise ValueError('Imputer state does not match its configuration')
        self.fills = {c: scalar(v) for c, v in state['fills'].items()}
        self.counts = dict(state.get('nonmissing_fit_rows', {}))
        return self

    def apply(self, frame):
        self.output_columns(frame.columns)
        if set(self.fills) != set(self.columns):
            raise ValueError('Imputer must be fitted before applying')
        out = frame.copy()
        for c in self.columns:
            if self.strategy in {'mean', 'median'}:
                out[c] = pd.to_numeric(out[c], errors='raise').astype('Float64').fillna(self.fills[c])
            else:
                series = out[c]
                if isinstance(series.dtype, pd.CategoricalDtype) and self.fills[c] not in series.cat.categories:
                    series = series.cat.add_categories([self.fills[c]])
                try:
                    out[c] = series.fillna(self.fills[c])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{c}: fill value incompatible with dtype; add convert_type first') from exc
        return out


class DropMissing(Transform):
    def __init__(self, columns, how='any'):
        super().__init__(columns)
        if how not in {'any', 'all'}:
            raise ValueError('how must be any or all')
        self.how = how

    def apply(self, frame):
        self.output_columns(frame.columns)
        return frame.dropna(subset=list(self.columns), how=self.how).copy()
