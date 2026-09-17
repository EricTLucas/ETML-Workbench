from dataclasses import dataclass
import pandas as pd
from .executor import FittedRecipe, fit_recipe, _source
from .recipe import Recipe
from .transforms.base import batches


@dataclass(frozen=True)
class PreviewResult:
    before: pd.DataFrame
    after: pd.DataFrame
    changes: dict
    metadata: dict


def preview_recipe(source, recipe: Recipe, *, fitted: FittedRecipe | None = None,
                   max_rows=200, loader_options=None) -> PreviewResult:
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError('max_rows must be a positive integer')
    dataset, factory = _source(source, min(max_rows+1, 50_000), loader_options)
    frames, n = [], 0
    with batches(factory) as iterator:
        for batch in iterator:
            part = batch.head(max_rows+1-n)
            frames.append(part)
            n += len(part)
            if n > max_rows:
                break
    sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(dataset.schema()))
    before = sample.head(max_rows).copy()
    if fitted is None:
        fitted = fit_recipe(before, recipe, batch_size=max_rows, require_approved=False,
                            fit_label='preview rows only; do not reuse for execution')
        fitting = 'preview rows only; execution fits may differ'
    else:
        if recipe.fingerprint != fitted.recipe_fingerprint:
            raise ValueError('Fitted state does not match preview recipe')
        fitting = 'provided fitted state; no refitting'
    after = fitted.apply(before)
    changes = {'rows_before': len(before), 'rows_after': len(after),
               'rows_removed': len(before)-len(after),
               'columns_added': [c for c in after if c not in before],
               'columns_removed': [c for c in before if c not in after],
               'missing_before': {c: int(n) for c, n in before.isna().sum().items()},
               'missing_after': {c: int(n) for c, n in after.isna().sum().items()},
               'dtypes_before': {c: str(t) for c, t in before.dtypes.items()},
               'dtypes_after': {c: str(t) for c, t in after.dtypes.items()}}
    return PreviewResult(before, after, changes,
                         {'selection': 'first rows in source order, not a random sample',
                          'truncated': n > max_rows, 'max_rows': max_rows,
                          'fitting': fitting, 'scope': 'preview only; not population estimates'})
