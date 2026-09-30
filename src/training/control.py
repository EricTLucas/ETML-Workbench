"""Thread-local cooperative controls for UI training jobs."""
from contextvars import ContextVar

CONTROL=ContextVar('etml_training_control',default=None)

class TrainingCancelled(Exception):
    pass

def check_cancelled():
    control=CONTROL.get()
    if control and control['cancel'].is_set():
        raise TrainingCancelled('Training cancelled at a safe boundary. Saved checkpoints and completed trials are preserved.')

def stop_requested():
    control=CONTROL.get()
    return bool(control and control['stop'].is_set())
