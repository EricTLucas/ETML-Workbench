"""Compatibility entry point: python -m etml.app."""
from etml.ui import launch

if __name__ == '__main__':
    raise SystemExit(launch())
