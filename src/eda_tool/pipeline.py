"""Compatibility entry point. The implementation lives in workflows.eda."""
import warnings
from .workflows import run_eda, EDARun, ProgressEvent


def analyze(path_or_object, output_dir='reports', open_html=False, **kwargs):
    """Deprecated old wrapper; now returns EDARun, not (df, profiler)."""
    warnings.warn('analyze() now returns EDARun; prefer run_eda(). No browser is opened.',
                  DeprecationWarning, stacklevel=2)
    kwargs.setdefault('generate_summary',True)
    kwargs.setdefault('export_html',True)
    return run_eda(path_or_object,output_dir=output_dir,**kwargs)


__all__ = ['run_eda','EDARun','ProgressEvent','analyze']
