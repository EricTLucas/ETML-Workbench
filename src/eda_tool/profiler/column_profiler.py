"""Full-scan column aggregates plus explicitly labeled sample diagnostics."""
from collections import Counter
import numpy as np
import pandas as pd
from .base import ProfilerComponent, SectionResult, finite_values
from .aggregations import Moments


class ColumnState:
    def __init__(self):
        self.missing = self.infinity = self.zeros = self.negative = 0
        self.memory = 0
        self.moments = Moments()
        self.lengths = Moments()
        self.counts = Counter()
        self.capped = False
        self.first = self.last = None
        self.increasing = self.decreasing = True
        self.minimum = self.maximum = None
        self.examples = []


class ColumnProfiler(ProfilerComponent):
    name = 'columns'

    def __init__(self, context):
        super().__init__(context)
        self.states = {}

    def update(self, batch):
        for col in batch.columns:
            state = self.states.setdefault(col, ColumnState())
            series = batch[col]
            role = self.context.roles[col]
            state.missing += int(series.isna().sum())
            state.memory += int(series.memory_usage(index=False, deep=True))
            present = series.dropna()
            for value in present:
                try:
                    hash(value)
                except TypeError as exc:
                    raise TypeError(f'{col}: nested values must be normalized before profiling') from exc
                if len(state.examples) < 5 and value not in state.examples:
                    state.examples.append(value)
                if not state.capped:
                    if value not in state.counts and len(state.counts) >= self.context.config.max_unique:
                        state.capped = True
                        state.counts.clear()
                    else:
                        state.counts[value] += 1
            if role == 'numeric':
                values = pd.to_numeric(series, errors='raise').to_numpy(dtype=float, na_value=np.nan)
                state.infinity += int(np.isinf(values).sum())
                finite = values[np.isfinite(values)]
                state.zeros += int((finite == 0).sum())
                state.negative += int((finite < 0).sum())
                state.moments.update(finite)
                if len(finite):
                    state.increasing &= bool(np.all(finite[1:] >= finite[:-1]))
                    state.decreasing &= bool(np.all(finite[1:] <= finite[:-1]))
                    if state.last is not None:
                        state.increasing &= bool(finite[0] >= state.last)
                        state.decreasing &= bool(finite[0] <= state.last)
                    state.last = float(finite[-1])
            elif role in {'text', 'category'}:
                state.lengths.update(np.array([len(str(x)) for x in present], dtype=float))
            elif role in {'datetime', 'timedelta'}:
                values = pd.to_datetime(present, errors='raise') if role == 'datetime' else pd.to_timedelta(present, errors='raise')
                if len(values):
                    lo, hi = values.min(), values.max()
                    state.minimum = lo if state.minimum is None else min(state.minimum, lo)
                    state.maximum = hi if state.maximum is None else max(state.maximum, hi)

    def finalize(self, results):
        output = {}
        rows = self.context.rows
        config = self.context.config
        sample_meta = self.context.sample_metadata()
        for col, state in self.states.items():
            role = self.context.roles[col]
            sample = self.context.sample[col]
            present = sample.dropna()
            inferred_counts = self.context.category_counts.get(col)
            capped = state.capped and not (role == 'category' and inferred_counts is not None)
            counts = inferred_counts if role == 'category' and inferred_counts is not None else (Counter(present) if capped else state.counts)
            denominator = len(sample) if capped else rows
            common = [[value, count, count/denominator if denominator else 0.0]
                      for value, count in counts.most_common(config.top_k)]
            unique = None if capped else len(counts)
            p = {'dtype': ', '.join(sorted(self.context.dtypes[col])), 'type': role,
                 'num_missing': state.missing, 'pct_missing': state.missing/rows if rows else 0.0,
                 'num_unique': unique, 'pct_unique': unique/rows if rows and unique is not None else None,
                 'unique_lower_bound': config.max_unique+1 if capped else unique,
                 'sample_values': state.examples, 'memory_size': state.memory,
                 'is_constant': unique == 1, 'all_missing': rows > 0 and state.missing == rows,
                 'possible_id': rows > 0 and unique == rows,
                 'cardinality': 'high' if capped else ('low' if unique < 20 else 'medium' if unique < 1000 else 'high'),
                 'common_values': common,
                 'methods': {'missing': 'exact', 'unique': 'unavailable_above_cap' if capped else 'exact',
                             'frequencies': sample_meta if capped else {'method': 'exact', 'rows_examined': rows},
                             'memory_size': 'sum of batch value memory; not process peak memory'}}
            if role == 'numeric':
                stats = state.moments.result()
                p.update(stats)
                p.update(num_infinity=state.infinity, num_zeros=state.zeros, num_neg=state.negative)
                for name, count in [('infinity', state.infinity), ('zeros', state.zeros), ('neg', state.negative)]:
                    p['pct_'+name] = count/rows if rows else 0.0
                p['range'] = stats['max']-stats['min'] if stats['count'] else None
                p['coefficient_of_variation'] = stats['std']/stats['mean'] if stats['std'] is not None and stats['mean'] else None
                p['monotonicity'] = ('undefined' if not stats['count'] else 'constant' if state.increasing and state.decreasing
                                    else 'increasing' if state.increasing else 'decreasing' if state.decreasing else 'none')
                values = finite_values(sample)
                q = np.quantile(values, [.05, .25, .5, .75, .95]) if len(values) else [None]*5
                p.update(zip(['5thp', 'Q1', 'median', 'Q3', '95thp'], q))
                p['IQR'] = q[3]-q[1] if len(values) else None
                p['MAD'] = float(np.median(np.abs(values-q[2]))) if len(values) else None
                if len(values):
                    counts_h, edges = np.histogram(values, bins=config.histogram_bins)
                    outliers = (values < q[1]-1.5*p['IQR']) | (values > q[3]+1.5*p['IQR'])
                    p['outliers'] = {'count': int(outliers.sum()), 'fraction': float(outliers.mean()),
                                     'rule': '1.5 IQR on analyzed finite rows', **sample_meta}
                    p['histogram'] = {'counts': counts_h.tolist(), 'edges': edges.tolist(), **sample_meta}
                else:
                    p['outliers'] = {'count': 0, 'fraction': None, **sample_meta}
                    p['histogram'] = {'counts': [], 'edges': [], **sample_meta}
                p['methods'].update(moments='exact finite values, float64; variance ddof=1; adjusted skew; excess kurtosis',
                                    quantiles={**sample_meta, 'finite_rows': len(values)},
                                    monotonicity='exact over finite values in input order')
            elif role in {'text', 'category'}:
                if role == 'category' and inferred_counts is not None:
                    lengths = Moments()
                    for value, count in counts.items():
                        block = Moments()
                        block.update([len(str(value))])
                        block.n = count
                        lengths.merge(block)
                    p['length_stats'] = lengths.result()
                else:
                    p['length_stats'] = state.lengths.result()
                p['length_stats']['definition'] = 'characters per nonmissing row after str conversion'
                p['counts'] = dict(counts.most_common(config.top_k))
                p['percentages'] = {v: c/denominator for v, c in p['counts'].items()} if denominator else {}
                p['top'] = common[0][0] if common else None
                if role == 'text':
                    words = Counter()
                    for value in present:
                        words.update(str(value).lower().split())
                    total_words = sum(words.values())
                    p['top words'] = dict(words.most_common(config.top_k))
                    p['top words percentages'] = {w: c/total_words for w, c in p['top words'].items()} if total_words else {}
                    p['methods']['words'] = sample_meta
            else:
                p['min'], p['max'] = state.minimum, state.maximum
            output[col] = p
        return SectionResult(self.name, output, {'rows_examined': rows, 'nonfinite_policy':
            'missing and +/-infinity excluded from numeric moments and associations; counted separately'})
