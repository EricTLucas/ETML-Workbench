"""Exact streamed Pearson; labeled sample-based categorical associations.

No p-values are claimed. Association magnitudes of different kinds are not
interchangeable. Categories include low-cardinality inferred roles and explicit overrides.
"""
import numpy as np
import pandas as pd
from .base import ProfilerComponent, SectionResult
from .aggregations import Covariance


def cramers_v(x, y):
    """Uncorrected Cramer's V on pairwise-complete data; None if undefined."""
    frame = pd.DataFrame({'x': list(x), 'y': list(y)}).dropna()
    table = pd.crosstab(frame.x, frame.y).to_numpy(dtype=float)
    if not table.size or min(table.shape) < 2:
        return None
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0))/n
    chi2 = np.sum((table-expected)**2/expected)
    return float(np.clip(np.sqrt(chi2/(n*(min(table.shape)-1))), 0, 1))


def correlation_ratio(categories, values):
    """Eta: square root of between-group / total sum of squares."""
    frame = pd.DataFrame({'category': list(categories), 'value': list(values)}).dropna()
    frame = frame[np.isfinite(frame.value.to_numpy(dtype=float))]
    if len(frame) < 2 or frame.category.nunique() < 2:
        return None
    overall = frame.value.mean()
    denominator = float(((frame.value-overall)**2).sum())
    if denominator <= 0:
        return None
    grouped = frame.groupby('category', observed=True).value.agg(['mean', 'count'])
    numerator = float((grouped['count']*(grouped['mean']-overall)**2).sum())
    return float(np.clip(np.sqrt(numerator/denominator), 0, 1))


class CorrelationsComponent(ProfilerComponent):
    name = 'correlations'
    def __init__(self, context):
        super().__init__(context)
        self.states = {}

    def update(self, batch):
        if not self.context.config.correlations:
            return
        for a, b in self.context.pairs:
            if self.context.roles[a] == self.context.roles[b] == 'numeric':
                state = self.states.setdefault((a, b), Covariance())
                x = pd.to_numeric(batch[a], errors='raise').to_numpy(dtype=float, na_value=np.nan)
                y = pd.to_numeric(batch[b], errors='raise').to_numpy(dtype=float, na_value=np.nan)
                state.update(x, y)

    def finalize(self, results):
        c = self.context
        # Matrix is bounded by selected pairs, not all input columns.
        names = list(dict.fromkeys(x for pair in c.pairs for x in pair)) if c.config.correlations else []
        matrix = pd.DataFrame(np.nan, index=names, columns=names)
        records = []
        for a, b in c.pairs if c.config.correlations else []:
            record = {'columns': (a, b), 'value': None, 'status': 'undefined'}
            if c.roles[a] == c.roles[b] == 'numeric':
                state = self.states[a, b]
                record.update(method='pearson', computation='exact', rows_examined=state.n,
                              value=state.correlation())
            else:
                roles = [c.roles[a], c.roles[b]]
                categories = [x for x in (a, b) if c.roles[x] == 'category']
                too_many = any(results['columns'].data[x]['num_unique'] is None or
                               results['columns'].data[x]['num_unique'] > c.config.max_categories for x in categories)
                record.update(method='cramers_v' if roles == ['category', 'category'] else 'correlation_ratio_eta',
                              computation=c.sample_metadata()['method'], rows_examined=0)
                if too_many:
                    record['status'] = 'skipped_category_limit'
                else:
                    frame = c.sample[[a, b]].dropna()
                    for col in (a, b):
                        if c.roles[col] == 'numeric':
                            frame = frame[np.isfinite(pd.to_numeric(frame[col]).to_numpy(dtype=float))]
                    record['rows_examined'] = len(frame)
                    if roles == ['category', 'category']:
                        record['value'] = cramers_v(frame[a], frame[b])
                    else:
                        cat = categories[0]
                        num = b if cat == a else a
                        record['value'] = correlation_ratio(frame[cat], pd.to_numeric(frame[num]))
            if record['value'] is not None:
                record['status'] = 'ok'
                matrix.loc[a, b] = matrix.loc[b, a] = record['value']
            records.append(record)
        return SectionResult(self.name, matrix, {'pairs': records, 'omitted_pairs': c.omitted_pairs,
            'enabled': c.config.correlations, 'sample': c.sample_metadata(),
            'diagonal': 'not computed', 'p_values': 'not computed',
            'interpretation': 'Pearson is signed; V and eta are unsigned; do not use one universal threshold'})
