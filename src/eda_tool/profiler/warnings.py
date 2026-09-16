"""Evidence-based data quality findings; no distribution claims from heuristics."""
from .base import ProfilerComponent, SectionResult


class WarningsComponent(ProfilerComponent):
    name = 'warnings'
    def update(self, batch):
        pass

    def finalize(self, results):
        warnings = {}
        for col, profile in results['columns'].data.items():
            messages = {}
            if profile['num_missing']:
                messages['missing'] = f"{profile['num_missing']} missing values ({profile['pct_missing']:.2%})"
            if profile['all_missing']:
                messages['all_missing'] = 'All values are missing'
            if profile['is_constant']:
                messages['constant'] = 'One distinct nonmissing value'
            if profile['possible_id']:
                messages['unique'] = 'Every row has a distinct value; inspect whether this is an identifier'
            if profile.get('num_infinity', 0):
                messages['infinity'] = 'Infinite values excluded from numeric statistics'
            if profile['num_unique'] is None:
                messages['cardinality_capped'] = 'Exact distinct tracking exceeded its limit; frequencies use the sample'
            if len(self.context.dtypes[col]) > 1:
                messages['dtype_drift'] = 'Storage dtype varied between batches; inspect loader dtype overrides'
            warnings[col] = messages
        return SectionResult(self.name, warnings, {'zeros_and_negatives': 'reported in column statistics; not automatically errors'})
