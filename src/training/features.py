import numpy as np
import pandas as pd
from numbers import Real


def _category(value):
    if pd.isna(value):
        return '__missing__'
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        return 'number:' + (str(int(value)) if float(value).is_integer() else repr(float(value)))
    return 'value:' + str(value)


def infer_schema(frame, categorical_columns=()):
    unknown = set(categorical_columns)-set(frame.columns)
    if unknown:
        raise ValueError(f'Unknown categorical columns: {sorted(unknown)}')
    schema = {}
    for c in frame:
        dtype = frame[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            raise ValueError(f'{c}: derive numeric/date features before model training')
        schema[c] = 'numeric' if (c not in categorical_columns and pd.api.types.is_numeric_dtype(dtype)
                                  and not pd.api.types.is_bool_dtype(dtype)) else 'categorical'
    return schema


def normalize_features(frame, schema):
    output = pd.DataFrame(index=frame.index)
    for c, kind in schema.items():
        if c not in frame:
            raise ValueError(f'Missing model feature: {c}')
        if kind == 'numeric':
            values = pd.to_numeric(frame[c], errors='raise').to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f'{c}: infinite values must be resolved before modeling')
            output[c] = values
        elif kind == 'categorical':
            output[c] = frame[c].map(_category).astype(object)
        else:
            raise ValueError('Unknown feature schema kind')
    return output


def fit_encoder(frame, schema, *, scale=False, max_features=50_000):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = [c for c, kind in schema.items() if kind == 'numeric']
    categorical = [c for c, kind in schema.items() if kind == 'categorical']
    normalized = normalize_features(frame, schema)
    if len(numeric)+sum(normalized[c].nunique() for c in categorical) > max_features:
        raise ValueError('One-hot feature count exceeds max_features; drop IDs or reduce cardinality')
    transforms = []
    if numeric:
        steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
        if scale:
            steps.append(('scaler', StandardScaler(with_mean=False)))
        transforms.append(('numeric', Pipeline(steps), numeric))
    if categorical:
        transforms.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True,
                                                         dtype=np.float32), categorical))
    encoder = ColumnTransformer(transforms, sparse_threshold=1.0)
    encoder.fit(normalized)
    return encoder


def encode(encoder, frame, schema, max_bytes=512*1024**2):
    from scipy import sparse
    result = encoder.transform(normalize_features(frame, schema)).astype(np.float32)
    size = result.data.nbytes+result.indices.nbytes+result.indptr.nbytes if sparse.issparse(result) else result.nbytes
    if size > max_bytes:
        raise ValueError('Encoded features exceed the memory budget')
    return result
