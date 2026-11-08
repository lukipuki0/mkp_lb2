"""Approved MKP OAT catalog. Nineteen monitor settings; band stays 4 when W changes."""

from dataclasses import dataclass

from dtw_stagnation import StagnationConfig

INSTANCE_FILE = "instancias/mknapcb1.txt"
INDICES = (0, 15, 29)
SEEDS = tuple(range(1, 32))
POLICIES = ("binary_simple", "binary_complex")
POP_SIZE = 30
ITERATIONS = 1000
G1 = 0.5
G2 = 0.5
G3_EXPLOIT = 0.1
G3_EXPLORE = 0.1
BINARIZATION = "lb2"
HYBRID_MODE = "competitive"
SIGNAL = "ddtw"
MIN_SLOPE = 2.0
IMPROVEMENT_TOL = 0.0
FIELDS = ("window", "band", "p_low", "p_high", "plateau_max", "patience")

BASELINE_VALUES = {
    "window": 40,
    "band": 4,
    "p_low": 40.0,
    "p_high": 60.0,
    "plateau_max": 15,
    "patience": 8,
}
ALTERNATIVES = (
    ("window", (20, 80, 160)),
    ("band", (1, 2, 8)),
    ("p_low", (20.0, 30.0, 50.0)),
    ("p_high", (50.0, 70.0, 80.0)),
    ("plateau_max", (5, 10, 30)),
    ("patience", (2, 4, 16)),
)


def _shown(value):
    number = float(value)
    return int(number) if number == int(number) else number


@dataclass(frozen=True)
class MonitorSpec:
    name: str
    parameter: str
    value: object
    window: int
    band: int
    p_low: float
    p_high: float
    plateau_max: int
    patience: int

    def stagnation_config(self) -> StagnationConfig:
        """Fresh monitor config. Absolute band is never derived from W."""
        return StagnationConfig(
            window=self.window, band=self.band, min_slope=MIN_SLOPE,
            plateau_max=self.plateau_max, patience=self.patience,
            use_ddtw=True, adapt_thresholds=True,
            p_low=float(self.p_low), p_high=float(self.p_high),
            improvement_tol=IMPROVEMENT_TOL,
        )


def _spec(parameter: str, value) -> MonitorSpec:
    values = dict(BASELINE_VALUES)
    if parameter != "baseline":
        values[parameter] = value
    name = "baseline" if parameter == "baseline" else f"{parameter}_{_shown(values[parameter])}"
    return MonitorSpec(name, parameter, None if parameter == "baseline" else values[parameter],
                       **values)


def oat_problems(spec, baseline) -> list:
    """Empty when `spec` changes at most the one named monitor field."""
    diffs = [field for field in FIELDS if getattr(spec, field) != getattr(baseline, field)]
    problems = []
    if spec.name == "baseline" or spec.parameter == "baseline":
        if diffs or spec.band != 4:
            problems.append("OAT invariant: baseline must keep W40 and absolute band 4")
    elif diffs != [spec.parameter] or (spec.parameter == "window" and spec.band != 4):
        problems.append(
            f"OAT invariant: {spec.name} must change only {spec.parameter} "
            f"(band stays 4 when W changes); changed {diffs or [spec.parameter]}"
        )
    if spec.p_low >= spec.p_high:
        problems.append("OAT invariant: p_low must be below p_high")
    for field in ("window", "band", "plateau_max", "patience"):
        if getattr(spec, field) <= 0:
            problems.append(f"OAT invariant: {field} must be positive")
    return problems


CONFIGURATIONS = (_spec("baseline", None),) + tuple(
    _spec(parameter, value) for parameter, values in ALTERNATIVES for value in values
)
FULL_RUNS = len(CONFIGURATIONS) * len(POLICIES) * len(INDICES) * len(SEEDS)
