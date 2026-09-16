from .base import BaseProfiler, ProfilerComponent, ProfileConfig, SectionResult
from .summary import SummaryComponent
from .column_profiler import ColumnProfiler
from .correlations import CorrelationsComponent
from .interactions import InteractionsComponent
from .warnings import WarningsComponent


class Profiler(BaseProfiler):
    def component_types(self):
        return (SummaryComponent, ColumnProfiler, CorrelationsComponent,
                InteractionsComponent, WarningsComponent)


__all__ = ['Profiler', 'ProfileConfig', 'SectionResult', 'ProfilerComponent', 'BaseProfiler']
