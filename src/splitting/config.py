from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = 'random'
    train: float = 0.7
    validation: float = 0.15
    test: float = 0.15
    seed: int = 42
    group_column: str | None = None
    time_column: str | None = None
    time_format: str | None = None
    max_classes: int = 10_000

    def __post_init__(self):
        if self.strategy not in {'random', 'stratified', 'group', 'chronological'}:
            raise ValueError('Unknown split strategy')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
               for x in self.ratios) or self.train <= 0 or not math.isclose(sum(self.ratios), 1.0, abs_tol=1e-9):
            raise ValueError('Split fractions must be nonnegative, sum to 1, and include positive training data')
        if type(self.seed) is not int or type(self.max_classes) is not int or self.max_classes < 2:
            raise ValueError('seed must be an integer and max_classes at least 2')
        if (self.group_column is not None) != (self.strategy == 'group'):
            raise ValueError('group_column is required only for group splitting')
        if (self.time_column is not None) != (self.strategy == 'chronological'):
            raise ValueError('time_column is required only for chronological splitting')
        if self.strategy == 'chronological' and not self.time_format:
            raise ValueError('Chronological splitting needs an explicit time_format, e.g. ISO8601')
        if self.strategy != 'chronological' and self.time_format is not None:
            raise ValueError('time_format applies only to chronological splitting')

    @property
    def ratios(self):
        return (self.train, self.validation, self.test)

    def to_dict(self):
        return asdict(self)
