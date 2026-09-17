import numpy as np
import pandas as pd
from preprocessing.validation import validate_recipe

ROW_ID = '__etml_row_id__'


def validate_task(task, columns, split_config, recipe):
    names = list(columns)
    if not names or len(set(names)) != len(names) or not all(isinstance(c, str) for c in names):
        raise ValueError('Input columns must be unique strings')
    if any(c.startswith('__etml_') for c in names):
        raise ValueError('Input column names beginning __etml_ are reserved')
    required = {task.target, *task.excluded_columns}
    controls = [c for c in (split_config.group_column, split_config.time_column) if c is not None]
    required.update(controls)
    if required-set(names):
        raise ValueError(f'Unknown task/split columns: {sorted(required-set(names))}')
    if task.target in controls:
        raise ValueError('The target cannot be a group/time split column')
    if split_config.strategy == 'stratified' and task.task_type != 'classification':
        raise ValueError('Stratification requires a classification task')
    features = tuple(c for c in names if c not in {task.target, *task.excluded_columns, *controls})
    if not features:
        raise ValueError('Task has no feature columns')
    validated = validate_recipe(recipe, features)
    if task.target in validated.output_columns or any(c.startswith('__etml_') for c in validated.output_columns):
        raise ValueError('Recipe output collides with the target or reserved row identifier')
    return features


def eligible_rows(frame, task):
    missing = frame[task.target].isna()
    if missing.any() and task.missing_target == 'error':
        raise ValueError('Missing target values found; explicitly choose missing_target="drop" to exclude them')
    if task.task_type == 'regression':
        values = frame.loc[~missing, task.target]
        if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
            raise ValueError('Regression targets must be numeric; convert the source explicitly if needed')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError('Regression targets must be finite')
    return ~missing
