from .base import Transform


class SelectColumns(Transform):
    def output_columns(self, names):
        super().output_columns(names)
        return list(self.columns)

    def apply(self, frame):
        self.output_columns(frame.columns)
        return frame.loc[:, self.columns].copy()


class DropColumns(Transform):
    def output_columns(self, names):
        super().output_columns(names)
        result = [c for c in names if c not in self.columns]
        if not result:
            raise ValueError('Cannot drop every column')
        return result

    def apply(self, frame):
        return frame.loc[:, self.output_columns(frame.columns)].copy()


class RenameColumns(Transform):
    def __init__(self, columns, names):
        super().__init__(columns)
        if not isinstance(names, list) or len(names) != len(columns) or not all(isinstance(n, str) and n for n in names):
            raise ValueError('names must contain one nonempty new name per selected column')
        self.mapping = dict(zip(columns, names))

    def output_columns(self, names):
        super().output_columns(names)
        result = [self.mapping.get(c, c) for c in names]
        if len(set(result)) != len(result):
            raise ValueError('Renaming would create duplicate column names')
        return result

    def apply(self, frame):
        self.output_columns(frame.columns)
        return frame.rename(columns=self.mapping).copy()
