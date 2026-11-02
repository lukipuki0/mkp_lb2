"""Variant mode of WOA-ABC: the a/rho recovery clock and variants v0-v4.

Restored from commit 8cc99f5 (``woa_abc/solver.py`` and README). This mode is
separate from the GWO-DE policies of ``profiles``; the two never combine.

Clock (original horizon ``T``):
    before the first fire   s = cycle index, a = max(0, 2 - 2s/(T-1)), rho = 1
    after an applied fire   the next executed cycle starts at s = 0 with
                            a = max(0, 2 - 2s/(T-1)), rho = max(1, 2 - s/(T-1))
A fire resets the monitor (``reset()``); population, failure counters and the
archive are kept. A fire is marked ``recovery_applied`` only when a following
cycle exists and can run one LB2 pair (two evaluations remain). ``a2`` follows
the ordinary cycle index and is never restarted. ``G3`` is fixed at 0.5.

Variants change one control each:
    v0  baseline
    v1  an omitted scout limit resolves to 100
    v2  after an applied recovery a restarts at 1 and rho at 1.5 (same T)
    v3  after an applied recovery a/rho decay over the R cycles still left
        (R = 1: a = 0, rho = 1)
    v4  employed bees and onlookers perturb one random coordinate
"""

from __future__ import annotations

from dataclasses import dataclass

from dtw_stagnation import StagnationMonitor


VARIANTS = ("v0", "v1", "v2", "v3", "v4")
VARIANT_ERROR = "variant must be one of v0, v1, v2, v3, v4"
VARIANT_SCOUT_LIMIT = 100  # v1, only when the scout limit is omitted
VARIANT_G3 = 0.5

# Segment of a cycle, for records and plots: "base" before the first applied
# recovery, "recovery" from each applied recovery on.
SEGMENT_BASE = "base"
SEGMENT_RECOVERY = "recovery"


def check_variant(value: object) -> str:
    if not isinstance(value, str) or value not in VARIANTS:
        raise ValueError(VARIANT_ERROR)
    return value


def control_parameters(
    schedule_index: int,
    iterations: int,
    recovery_active: bool,
    variant: str = "v0",
    recovery_span: int | None = None,
) -> tuple[float, float]:
    """Return ``(a, rho)`` for the current clock.

    Before any recovery, rho stays 1. After a hook reset, the next cycle is
    called with schedule index 0 and ``recovery_active`` so rho restarts at 2
    on v0. v2 changes only the post-fire amplitudes. v3 changes only the
    post-fire horizon. v1 and v4 keep this baseline clock.
    """

    if recovery_active and variant == "v2":
        span = iterations - 1
        return (
            max(0.0, 1.0 - 2.0 * schedule_index / span),
            max(1.0, 1.5 - schedule_index / span),
        )
    if recovery_active and variant == "v3":
        if recovery_span is None:
            raise RuntimeError("v3 recovery is missing its remaining-cycle span")
        if recovery_span == 0:
            return 0.0, 1.0
        return (
            max(0.0, 2.0 - 2.0 * schedule_index / recovery_span),
            max(1.0, 2.0 - schedule_index / recovery_span),
        )
    span = iterations - 1
    a = max(0.0, 2.0 - 2.0 * schedule_index / span)
    if recovery_active:
        rho = max(1.0, 2.0 - schedule_index / span)
    else:
        rho = 1.0
    return a, rho


@dataclass
class ClockCycleRecord:
    """One variant-mode cycle, including a cycle stopped at a phase boundary.

    ``schedule_index`` is the clock used for ``a`` and ``rho``; it matches
    ``cycle`` until an applied recovery resets it. ``segment`` is "base"
    before the first applied recovery and "recovery" after it. ``G3`` is the
    fixed LB2 value. ``evaluations`` is the cumulative objective count.
    """

    cycle: int
    schedule_index: int
    segment: str
    a: float
    rho: float
    G3: float
    best: float
    evaluations: int
    completeness: bool


class RecoveryClock:
    """The a/rho clock of one run and the monitor it restarts from."""

    def __init__(self, monitor: StagnationMonitor | None, variant: str, iterations: int) -> None:
        self.monitor = monitor
        self.variant = check_variant(variant)
        self.iterations = iterations
        self.schedule_index = 0
        self.recovery_active = False
        self.recovery_span: int | None = None

    @property
    def segment(self) -> str:
        return SEGMENT_RECOVERY if self.recovery_active else SEGMENT_BASE

    def parameters(self) -> tuple[float, float]:
        return control_parameters(
            self.schedule_index,
            self.iterations,
            self.recovery_active,
            self.variant,
            self.recovery_span,
        )

    def observe(self, best: float) -> dict | None:
        """Feed one archived best. On fire the monitor is reset after the update.

        Returns the monitor status, or None when detection is off.
        """

        if self.monitor is None:
            return None
        status = self.monitor.update(float(best))
        if status.get("fire"):
            self.monitor.reset()
        return status

    def advance(self, fired: bool, cycle_index: int, can_run_pair: bool) -> bool:
        """Move the clock after complete cycle ``cycle_index``; return recovery_applied.

        A claimed restart requires a later cycle that can execute one LB2 pair.
        """

        applied = bool(fired and cycle_index + 1 < self.iterations and can_run_pair)
        if applied:
            self.schedule_index = 0
            self.recovery_active = True
            if self.variant == "v3":
                remaining_cycles = self.iterations - (cycle_index + 1)
                self.recovery_span = remaining_cycles - 1
        else:
            self.schedule_index += 1
        return applied
