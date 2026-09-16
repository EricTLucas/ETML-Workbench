"""Versioned JSON-safe values. No pickle, executable imports or eval."""
from datetime import date, datetime
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from ..profiler.base import SectionResult


def encode(value):
    if value is pd.NA:
        return {'type': 'pd_na'}
    if value is pd.NaT:
        return {'type': 'pd_nat'}
    if isinstance(value, pd.Timestamp):
        return {'type': 'timestamp', 'value': value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {'type': 'timedelta', 'value': value.value}
    if isinstance(value, np.generic):
        return encode(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'type': 'float', 'value': 'nan' if math.isnan(value) else 'inf' if value > 0 else '-inf'}
    if isinstance(value, datetime):
        return {'type': 'timestamp', 'value': value.isoformat()}
    if isinstance(value, date):
        return {'type': 'date', 'value': value.isoformat()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SectionResult):
        return {'type': 'section', 'name': value.name, 'data': encode(value.data), 'metadata': encode(value.metadata)}
    if isinstance(value, pd.DataFrame):
        # Profile frames are small association tables. Raw rows are removed by
        # persistence policy before encoding; samples have their own Parquet file.
        return {'type': 'frame', 'columns': encode(list(value.columns)), 'index': encode(list(value.index)),
                'values': encode(value.to_numpy().tolist())}
    if isinstance(value, np.ndarray):
        return {'type': 'array', 'values': encode(value.tolist())}
    if isinstance(value, dict):
        # Tag *every* mapping, so user keys cannot be mistaken for codec tags.
        return {'type': 'mapping', 'items': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {'type': 'tuple', 'items': [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    raise TypeError(f'Cannot serialize {type(value).__name__}; provide JSON-compatible configuration values')


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get('type')
    if kind == 'pd_na': return pd.NA
    if kind == 'pd_nat': return pd.NaT
    if kind == 'timestamp': return pd.Timestamp(value['value'])
    if kind == 'date': return date.fromisoformat(value['value'])
    if kind == 'timedelta': return pd.Timedelta(value['value'], unit='ns')
    if kind == 'float':
        if value['value'] not in {'nan', 'inf', '-inf'}: raise ValueError('Invalid float tag')
        return float(value['value'])
    if kind == 'mapping': return {decode(k): decode(v) for k, v in value['items']}
    if kind == 'tuple': return tuple(decode(v) for v in value['items'])
    if kind == 'array': return np.asarray(decode(value['values']))
    if kind == 'frame': return pd.DataFrame(decode(value['values']), columns=decode(value['columns']), index=decode(value['index']))
    if kind == 'section': return SectionResult(value['name'], decode(value['data']), decode(value['metadata']))
    raise ValueError(f'Unknown JSON value tag: {kind!r}')


def dump(path, value):
    Path(path).write_text(json.dumps(encode(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def load(path):
    return decode(json.loads(Path(path).read_text(encoding='utf-8')))
