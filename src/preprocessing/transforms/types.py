import numpy as np
import pandas as pd
from .base import Transform


class ConvertType(Transform):
    def __init__(self, columns, dtype, errors='raise', format=None, utc=True):
        super().__init__(columns)
        if dtype not in {'Int64', 'Float64', 'string', 'boolean', 'datetime'}:
            raise ValueError('dtype must be Int64, Float64, string, boolean, or datetime')
        if errors not in {'raise', 'coerce'} or type(utc) is not bool:
            raise ValueError('Invalid conversion options')
        if dtype == 'datetime' and not isinstance(format, str):
            raise ValueError('Datetime conversion requires an explicit format')
        self.dtype, self.errors, self.format, self.utc = dtype, errors, format, utc

    def apply(self, frame):
        self.output_columns(frame.columns)
        out = frame.copy()
        for c in self.columns:
            s = out[c]
            if self.dtype == 'datetime':
                out[c] = pd.to_datetime(s, format=self.format, utc=self.utc, errors=self.errors)
            elif self.dtype in {'Int64', 'Float64'}:
                values = pd.to_numeric(s, errors=self.errors)
                invalid = values.notna() & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
                if self.dtype == 'Int64':
                    invalid |= values.notna() & (values % 1 != 0)
                if invalid.any() and self.errors == 'raise':
                    raise ValueError(f'{c}: nonfinite or fractional values cannot be converted')
                out[c] = values.mask(invalid).astype(self.dtype)
            elif self.dtype == 'boolean':
                invalid = s.notna() & ~s.isin([True, False, 0, 1])
                if invalid.any() and self.errors == 'raise':
                    raise ValueError(f'{c}: map text labels explicitly before boolean conversion')
                out[c] = s.mask(invalid).astype('boolean')
            else:
                out[c] = s.astype('string')
        return out
