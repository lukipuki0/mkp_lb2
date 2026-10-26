"""Study defaults for adaptive selection.

CLI and campaign read these values. An explicit flag overrides only its own
field. Native solver limits stay in hybrid_mkp and are not copied here.
"""

from dataclasses import dataclass, field, replace
import math

from dtw_stagnation import StagnationConfig
from hybrid_mkp.orchestrator import POOL_POBLACIONAL, POOL_TRAYECTORIA, COLORES_MH

POPULATION_POOL = tuple(POOL_POBLACIONAL)
TRAJECTORY_POOL = tuple(POOL_TRAYECTORIA)
COLORS = COLORES_MH


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def default_monitor():
    # Do not import the benchmark runner: input validation must not load plots.
    return StagnationConfig(window=70, band=7, min_slope=2, plateau_max=25,
                            patience=15, use_ddtw=True, adapt_thresholds=True,
                            p_low=30., p_high=70., improvement_tol=0.)


def monitor_with_cli_overrides(window, min_slope, plateau_max, patience):
    """Apply the existing monitor flags without dropping configured fields."""
    return replace(default_monitor(), window=window, min_slope=min_slope,
                   plateau_max=plateau_max, patience=patience)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    selector: str = 'adaptive'
    max_iters: int = 1000
    max_epoch_iters: int | None = None
    alpha: float = .3
    epsilon: float = .2
    use_ddtw: bool = True
    pop_injection_mode: str = 'mixed'
    monitor: StagnationConfig = field(default_factory=default_monitor)
    population_pool: tuple[str, ...] | None = None
    trajectory_pool: tuple[str, ...] | None = None

    def __post_init__(self):
        for name, native_pool in (('population_pool', POPULATION_POOL),
                                  ('trajectory_pool', TRAJECTORY_POOL)):
            names = getattr(self, name)
            if names is None:
                names = native_pool
            if (not isinstance(names, (tuple, list)) or not names
                    or any(not isinstance(mh, str) or mh not in native_pool for mh in names)
                    or len(set(names)) != len(names)):
                raise ValueError(f'{name} must be a nonempty unique subset of ' + ', '.join(native_pool))
            object.__setattr__(self, name, tuple(names))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an unsigned 32-bit integer')
        if self.selector not in ('adaptive', 'uniform', 'adaptive_exploratory'):
            raise ValueError('selector must be adaptive, uniform or adaptive_exploratory')
        positive_integer(self.max_iters, 'max_iters')
        if self.max_epoch_iters is not None:
            positive_integer(self.max_epoch_iters, 'max_epoch_iters')
        for name in ('alpha', 'epsilon'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite and in (0, 1]')
        if not isinstance(self.use_ddtw, bool):
            raise ValueError('use_ddtw must be a boolean')
        if self.pop_injection_mode not in ('mixed', 'random', 'mutated'):
            raise ValueError('pop_injection_mode must be mixed, random or mutated')
        if not isinstance(self.monitor, StagnationConfig):
            raise ValueError('monitor must be a StagnationConfig')
        for name in ('window', 'patience'):
            positive_integer(getattr(self.monitor, name), f'monitor.{name}')
        if self.monitor.window < 2:
            raise ValueError('monitor.window must be at least 2')
        for name in ('band', 'plateau_max'):
            value = getattr(self.monitor, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'monitor.{name} must be a nonnegative integer')
        for name in ('min_slope', 'improvement_tol'):
            value = getattr(self.monitor, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'monitor.{name} must be finite and nonnegative')
        if not 0 <= self.monitor.p_low <= self.monitor.p_high <= 100:
            raise ValueError('monitor percentiles must satisfy 0 <= p_low <= p_high <= 100')
        # Store actual settings, including detector, without mutating caller config.
        object.__setattr__(self, 'monitor', replace(self.monitor, use_ddtw=self.use_ddtw))
