import math
import pandas as pd
from .base import Transform


class NormalizeCategories(Transform):
    def __init__(self, columns, strip=True, case='preserve', collapse_whitespace=False, empty_to_missing=False):
        super().__init__(columns)
        if case not in {'preserve', 'lower', 'upper', 'casefold'}:
            raise ValueError('Invalid case normalization')
        if any(type(x) is not bool for x in (strip, collapse_whitespace, empty_to_missing)):
            raise ValueError('Normalization flags must be booleans')
        self.strip, self.case = strip, case
        self.collapse_whitespace, self.empty_to_missing = collapse_whitespace, empty_to_missing

    def apply(self, frame):
        self.output_columns(frame.columns)
        out = frame.copy()
        for c in self.columns:
            if not out[c].dropna().map(lambda v: isinstance(v, str)).all():
                raise ValueError(f'{c}: category normalization requires strings')
            s = out[c].astype('string')
            if self.strip:
                s = s.str.strip()
            if self.collapse_whitespace:
                s = s.str.replace(r'\s+', ' ', regex=True)
            if self.case != 'preserve':
                s = getattr(s.str, self.case)()
            if self.empty_to_missing:
                s = s.mask(s == '')
            out[c] = s
        return out


class MapCategories(Transform):
    def __init__(self, columns, mapping, unknown='keep'):
        super().__init__(columns)
        if unknown not in {'keep', 'missing', 'error'} or not isinstance(mapping, list):
            raise ValueError('mapping must be a list of {from, to} entries; invalid unknown policy')
        self.mapping = {}
        for entry in mapping:
            if not isinstance(entry, dict) or set(entry) != {'from', 'to'}:
                raise ValueError('Mapping entries need exactly from and to')
            old, new = entry['from'], entry['to']
            for value in (old, new):
                if value is not None and (not isinstance(value, (str, int, float, bool)) or
                                          isinstance(value, float) and not math.isfinite(value)):
                    raise ValueError('Category mappings require finite JSON scalar values')
            if old is None or old in self.mapping:
                raise ValueError('Missing/duplicate mapping keys; use fill_missing for nulls')
            self.mapping[old] = new
        self.unknown = unknown

    def apply(self, frame):
        self.output_columns(frame.columns)
        out = frame.copy()
        for c in self.columns:
            original = out[c]
            known = original.isin(list(self.mapping))
            if self.unknown == 'error' and (original.notna() & ~known).any():
                raise ValueError(f'{c}: encountered an unmapped category')
            mapped = original.astype(object).map(self.mapping)
            if self.unknown == 'keep':
                mapped = mapped.where(known, original.astype(object))
            out[c] = mapped.where(original.notna(), pd.NA)
        return out
