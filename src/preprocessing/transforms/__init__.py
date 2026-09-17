from .base import Transform
from .missing import FillMissing, DropMissing
from .types import ConvertType
from .categories import NormalizeCategories, MapCategories
from .columns import SelectColumns, DropColumns, RenameColumns

TRANSFORMS = {
    'fill_missing': FillMissing,
    'drop_missing': DropMissing,
    'convert_type': ConvertType,
    'normalize_categories': NormalizeCategories,
    'map_categories': MapCategories,
    'select_columns': SelectColumns,
    'drop_columns': DropColumns,
    'rename_columns': RenameColumns,
}


def build_transform(step):
    if step.operation not in TRANSFORMS:
        raise ValueError(f'Unknown operation: {step.operation}')
    try:
        return TRANSFORMS[step.operation](step.columns, **step.params)
    except TypeError as exc:
        raise ValueError(f'{step.operation}: invalid parameters: {exc}') from exc


def available_transforms():
    import inspect
    return {name: [p for p in inspect.signature(cls.__init__).parameters if p not in {'self', 'columns'}]
            for name, cls in TRANSFORMS.items()}


__all__ = ['Transform', 'FillMissing', 'DropMissing', 'ConvertType', 'NormalizeCategories',
           'MapCategories', 'SelectColumns', 'DropColumns', 'RenameColumns',
           'build_transform', 'available_transforms']
