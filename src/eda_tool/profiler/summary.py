"""Dataset counts, previews, and optionally capped exact duplicate detection."""
import pandas as pd
from .base import ProfilerComponent, SectionResult


class SummaryComponent(ProfilerComponent):
    name = 'summary'
    def __init__(self, context):
        super().__init__(context)
        self.missing = self.memory = self.duplicates = 0
        self.first = self.last = pd.DataFrame()
        self.seen = set()
        self.exact_duplicates = context.config.duplicate_limit > 0

    def update(self, batch):
        self.missing += int(batch.isna().sum().sum())
        self.memory += int(batch.memory_usage(index=False, deep=True).sum())
        self.first = pd.concat([self.first, batch.head(max(0, 10-len(self.first)))], ignore_index=True)
        self.last = pd.concat([self.last, batch.tail(10)], ignore_index=True).tail(10)
        if self.exact_duplicates:
            for row in batch.itertuples(index=False, name=None):
                # Tagged nulls cannot collide with actual user values.
                key = tuple((0,) if pd.isna(value) else (1, value) for value in row)
                if key in self.seen:
                    self.duplicates += 1
                elif len(self.seen) < self.context.config.duplicate_limit:
                    self.seen.add(key)
                else:
                    self.exact_duplicates = False
                    self.seen.clear()
                    break

    def finalize(self, results):
        rows, cols = self.context.rows, len(self.context.columns)
        sample = self.context.sample
        sample_duplicates = int(sample.duplicated().sum()) if cols else max(0, len(sample)-1)
        exact = self.exact_duplicates or rows <= self.context.config.sample_size
        count = self.duplicates if self.exact_duplicates else sample_duplicates if exact else None
        return SectionResult(self.name, {
            'rows': rows, 'cols': cols, 'num_missing_cells': self.missing,
            'percent_missing_cells': self.missing/(rows*cols) if rows*cols else 0.0,
            'memory_usage': self.memory, 'total_memory_usage': self.memory,
            'sample_values_first': self.first, 'sample_values_last': self.last,
            'num_duplicates': count, 'percent_duplicates': count/rows if count is not None and rows else (0.0 if not rows else None),
            'sample_duplicates': sample_duplicates, 'sample_rows': len(sample),
            'duplicate_method': 'exact' if exact else 'unavailable; sample duplicate count is NOT a population estimate',
        }, {'counts': 'exact', 'memory': 'sum of batch value memory; not process peak',
            'sample': self.context.sample_metadata(), 'duplicate_limit': self.context.config.duplicate_limit})
