from dataclasses import dataclass, asdict
from data.manifest import validate_name


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    target: str
    task_type: str
    excluded_columns: tuple[str, ...] = ()
    missing_target: str = 'error'
    format_version: int = 1

    def __post_init__(self):
        validate_name(self.task_id)
        if not isinstance(self.target, str) or not self.target:
            raise ValueError('target must be a nonempty column name')
        if self.task_type not in {'classification', 'regression'}:
            raise ValueError('task_type must be classification or regression')
        if self.missing_target not in {'error', 'drop'}:
            raise ValueError('missing_target must be error or drop')
        if isinstance(self.excluded_columns, str):
            raise ValueError('excluded_columns must be a sequence')
        object.__setattr__(self, 'excluded_columns', tuple(self.excluded_columns))
        if not all(isinstance(c, str) and c for c in self.excluded_columns):
            raise ValueError('Excluded column names must be nonempty strings')
        if len(set(self.excluded_columns)) != len(self.excluded_columns) or self.target in self.excluded_columns:
            raise ValueError('Duplicate excluded columns or target also excluded')
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError('Unsupported task format')

    def to_dict(self):
        value = asdict(self)
        value['excluded_columns'] = list(self.excluded_columns)
        return value

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
