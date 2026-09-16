"""ML workbench components; public orchestration entry point."""
__version__ = '0.2.0'


def run_eda(*args, **kwargs):
    from .workflows import run_eda as run
    return run(*args, **kwargs)
