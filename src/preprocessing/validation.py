from dataclasses import dataclass
from .recipe import Recipe
from .transforms import build_transform


@dataclass(frozen=True)
class ValidationResult:
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    requires_fit: bool


def validate_recipe(recipe: Recipe, columns, *, require_approved=True) -> ValidationResult:
    if not isinstance(recipe, Recipe):
        raise TypeError('Expected a Recipe')
    names = list(columns)
    original = tuple(names)
    if not names or len(set(names)) != len(names) or not all(isinstance(c, str) for c in names):
        raise ValueError('Input must have nonempty, unique string column names')
    requires_fit = False
    for step in recipe.active_steps:
        if require_approved and not step.approved:
            raise ValueError(f'Step {step.step_id} needs approval (or changed since approval)')
        transform = build_transform(step)
        names = transform.output_columns(names)
        requires_fit |= transform.requires_fit
    return ValidationResult(original, tuple(names), requires_fit)
