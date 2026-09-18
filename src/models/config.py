from dataclasses import dataclass, field, asdict
import json


@dataclass(frozen=True)
class ModelConfig:
    backend: str = 'sklearn'
    algorithm: str = 'linear'
    params: dict = field(default_factory=dict)
    seed: int = 42
    categorical_columns: tuple[str, ...] = ()
    scale_numeric: bool | None = None

    def __post_init__(self):
        if type(self.seed) is not int or not isinstance(self.params, dict):
            raise ValueError('seed must be an integer and params a dictionary')
        if isinstance(self.categorical_columns, str):
            raise ValueError('categorical_columns must be a sequence')
        object.__setattr__(self, 'categorical_columns', tuple(self.categorical_columns))
        if self.scale_numeric is not None and type(self.scale_numeric) is not bool:
            raise ValueError('scale_numeric must be a boolean or None')
        object.__setattr__(self, 'params', json.loads(json.dumps(self.params, allow_nan=False)))

    def to_dict(self):
        value = asdict(self)
        value['categorical_columns'] = list(self.categorical_columns)
        return value
