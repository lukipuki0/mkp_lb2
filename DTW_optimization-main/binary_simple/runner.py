"""
Runner para Fire D2 (estrategia A3).
Wrapper sobre el runner genérico con fire_fn = fire_d2.
"""

from mkp_common.runner import run_experiment as _run_experiment
from mkp_common.runner import run_epochs as _run_epochs

from .config import fire_d2


def run_experiment(**kwargs):
    """run_experiment con estrategia A3 (D2 puro)."""
    kwargs.setdefault("fire_fn", fire_d2)
    return _run_experiment(**kwargs)


def run_epochs(**kwargs):
    """run_epochs con estrategia A3 (D2 puro)."""
    kwargs.setdefault("fire_fn", fire_d2)
    return _run_epochs(**kwargs)
