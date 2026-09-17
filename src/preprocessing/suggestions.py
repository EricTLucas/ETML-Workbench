from dataclasses import dataclass
from .recipe import Recipe, Step


@dataclass(frozen=True)
class SuggestionConfig:
    numeric_missing: str = 'mean'
    category_missing: str = 'mode'
    suggest_constant_drop: bool = True
    suggest_trim: bool = True


def suggest_recipe(profile, *, config=None, protected_columns=()) -> Recipe:
    """Return proposals only. Target/ID/time columns can be protected by the caller."""
    config = config or SuggestionConfig()
    if config.numeric_missing not in {'mean', 'median'} or config.category_missing != 'mode':
        raise ValueError('Unsupported suggestion strategies')
    results = profile.run() if hasattr(profile, 'run') else profile
    section = results.get('columns', {})
    columns = section.data if hasattr(section, 'data') else section
    protected = set(protected_columns)
    if protected-set(columns):
        raise ValueError('Unknown protected columns')
    steps = []
    for name, stats in columns.items():
        if name in protected:
            continue
        role = stats.get('type')
        evidence = {'role': role, 'missing': stats.get('num_missing'),
                    'distinct': stats.get('num_unique')}
        if stats.get('all_missing'):
            continue  # No basis for inferring a fill value or dropping this feature.
        if config.suggest_constant_drop and stats.get('is_constant') and not stats.get('num_missing'):
            if len(columns) > 1:
                steps.append(Step('drop_columns', (name,), reason='Constant nonmissing feature; review whether it is needed.',
                                  evidence=evidence))
            continue
        examples = [v for v in stats.get('sample_values', []) if isinstance(v, str)]
        if config.suggest_trim and role in {'text', 'category'} and any(v != v.strip() for v in examples):
            steps.append(Step('normalize_categories', (name,), {'strip': True},
                              reason='Observed leading/trailing whitespace; trimming may merge labels. Review first.',
                              evidence={**evidence, 'basis': 'profile examples; not a full scan'}))
        if stats.get('num_missing', 0) and role in {'numeric', 'category'}:
            strategy = config.numeric_missing if role == 'numeric' else config.category_missing
            steps.append(Step('fill_missing', (name,), {'strategy': strategy},
                              reason=f'Propose {strategy} imputation; fit on an explicitly selected fitting dataset.',
                              evidence=evidence))
    # Keep one feature if every column was proposed for removal.
    drops = [s for s in steps if s.operation == 'drop_columns']
    if len(drops) == len(columns) and drops:
        steps.remove(drops[-1])
    return Recipe(tuple(steps), name='Proposed preprocessing')
