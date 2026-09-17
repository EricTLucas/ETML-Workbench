from abc import ABC, abstractmethod
from contextlib import contextmanager
import pandas as pd


def check_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Expected pandas DataFrame batches')
    if not frame.columns.is_unique or not all(isinstance(c, str) for c in frame.columns):
        raise ValueError('Column names must be unique strings')


@contextmanager
def batches(factory):
    iterator = iter(factory())
    try:
        yield iterator
    finally:
        close = getattr(iterator, 'close', None)
        if close:
            close()


class Transform(ABC):
    requires_fit = False

    def __init__(self, columns):
        self.columns = tuple(columns)

    def output_columns(self, names):
        missing = set(self.columns)-set(names)
        if missing:
            raise ValueError(f'Unknown columns: {sorted(missing)}')
        return list(names)

    def fit(self, factory):
        return self

    def state(self):
        return {}

    def restore(self, state):
        if state:
            raise ValueError('Unexpected state for stateless transformation')
        return self

    @abstractmethod
    def apply(self, frame):
        pass
