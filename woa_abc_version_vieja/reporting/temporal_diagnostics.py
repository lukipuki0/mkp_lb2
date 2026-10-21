"""Figura de diagnóstico temporal para una corrida MKP ya guardada.

Provenance: layout and drawing copied from ``a/temporal_diagnostics.py``
(``_window_rise`` ~260, ``_longest_true_run``/``_window_end`` through ~328,
``_shade_state``/``_mark_windows``/``plot_figure`` ~359–524). That script is
not imported: importing it inserts a global ROOT and can launch a BDE run.

Adaptations, only where the source would misrepresent this monitor:

- M0 returns None. No monitor series is invented.
- Window, band and slope come from ``configuration['algorithm']['dtw']``.
- Ready rows use the fixed threshold as well as later percentiles.
- A window never crosses a confirmed fire; the fire snapshot is kept.
- ``base`` is its own state. Four-state aliases keep their names.
- Panel (d) annotates actual ``a_eff``/``phi_eff``, not BDE F/CR.
- Window and delta y-limits follow plotted data. The source limits
  ``(-30, 460)`` and delta pad ``4`` clip or flatten these traces.
- PDF only. No series export and no raster copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from woa_abc_version_vieja.config import DTWConfig
from woa_abc_version_vieja.core.binary_woa_abc import MKPOptimizationResult
from woa_abc_version_vieja.dtw.monitor import ddtw_distance, dtw_distance
from woa_abc_version_vieja.dtw.strategies import M0_NO_DTW
from woa_abc_version_vieja.runners.checkpoints import load_typed_rows


_RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
_RAMP = "#0072B2"
_CONSTANT = "#D55E00"
_DELTA = "#009E73"
_EXPLORE = "#F4A261"
_EXPLOIT = "#8ECAE6"
_BASE = "#B0B0B0"
_DTW_FIELDS = (
    "window",
    "band",
    "min_slope",
    "plateau_max",
    "patience",
    "use_ddtw",
    "adapt_thresholds",
    "p_low",
    "p_high",
)
_STATE_ORDER = (
    "base",
    "exploit_high",
    "exploit",
    "exploit_low",
    "explore_low",
    "explore",
    "explore_high",
)
_UNAVAILABLE = "Representative panel unavailable"
_DISTANCE_TOLERANCE = 1e-6


def _is_m0(result: Any) -> bool:
    return getattr(result, "variant", None) == M0_NO_DTW


def _dtw_config(configuration: dict[str, Any]) -> DTWConfig:
    """Read the run's DTW block. Missing fields are an error, not a default."""

    try:
        raw = configuration["algorithm"]["dtw"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "la configuración real debe incluir algorithm.dtw; "
            "no se asume la ventana del paper ni un default del código"
        ) from exc
    if not isinstance(raw, dict) or any(key not in raw for key in _DTW_FIELDS):
        raise ValueError(
            "algorithm.dtw está incompleto; no se rellenan W, banda ni pendiente"
        )
    return DTWConfig(**{key: raw[key] for key in _DTW_FIELDS})


def _family(mode: str) -> str:
    if mode == "base":
        return "base"
    if str(mode).startswith("explore"):
        return "explore"
    if str(mode).startswith("exploit"):
        return "exploit"
    return "other"


def _state_color(mode: str) -> str:
    family = _family(mode)
    if family == "base":
        return _BASE
    if family == "explore":
        return _EXPLORE
    if family == "exploit":
        return _EXPLOIT
    return "#CCCCCC"


def _references(
    observed: np.ndarray,
    min_slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Same ramp/constant construction as ``StagnationMonitor.update``."""

    window = int(observed.size)
    start = float(observed[0])
    slope = float(min_slope)
    if slope == 0.0:
        observed_range = max(1.0, abs(float(observed[-1] - observed[0])))
        slope = 0.01 * observed_range / window
    ramp = start + slope * np.arange(window, dtype=float)
    constant = np.full(window, start, dtype=float)
    return ramp, constant


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return the inclusive index range of the longest True run."""

    best = (0, -1)
    start = None
    for index, flag in enumerate(mask.tolist() + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - 1 - start > best[1] - best[0]:
                best = (start, index - 1)
            start = None
    return best


def _walk_forward(series: dict[str, Any], progress_index: int) -> int:
    """Keep the rising level visible without crossing a monitor reset.

    The source walks while the end value is held and the state stays in
    exploitation. Here the walk also stops at the next episode, at a
    not-ready row, and when the applied mode changes. A warmup or post-fire
    row is not treated as part of the previous window.
    """

    fitness = series["f_star"]
    rise = series["rise"]
    modes = series["mode"]
    episode = series["episode_start"]
    ready = series["ready"]
    end_value = fitness[progress_index]
    target_rise = rise[progress_index]
    mode = modes[progress_index]
    index = progress_index
    while (
        index + 1 < len(fitness)
        and int(episode[index + 1]) == int(episode[index])
        and bool(ready[index + 1])
        and modes[index + 1] == mode
        and fitness[index + 1] == end_value
        and np.isfinite(rise[index + 1])
        and rise[index + 1] >= 0.9 * target_rise
    ):
        index += 1
    return index


def _select_windows(series: dict[str, Any]) -> dict[str, Any]:
    """Prefer the source rules, then any real ready window, else none.

    Source ``_window_end`` can call ``nanargmin`` on an empty pool and treats
    state as binary exploitation/exploration. This picker never fabricates a
    window or an explore label for ``base``.
    """

    ready = np.asarray(series["ready"], dtype=bool)
    rise = np.asarray(series["rise"], dtype=float)
    delta = np.asarray(series["delta"], dtype=float)
    modes = list(series["mode"])
    families = np.asarray([_family(mode) for mode in modes])
    flat = ready & np.isfinite(rise) & (rise == 0.0)
    rising = ready & np.isfinite(rise) & (rise > 0.0)
    selected = dict(series)
    selected["progress_index"] = None
    selected["plateau_index"] = None
    selected["progress_title"] = "Rising window unavailable"
    selected["plateau_title"] = "Flat window unavailable"

    explore_flat = flat & (families == "explore")
    start, stop = _longest_true_run(explore_flat)
    if stop >= start:
        selected["plateau_index"] = (start + stop) // 2
        selected["plateau_title"] = "Flat window"
    elif np.any(flat):
        start, stop = _longest_true_run(flat)
        selected["plateau_index"] = (start + stop) // 2
        selected["plateau_title"] = f"Flat window ({modes[selected['plateau_index']]})"

    separated = np.ones(len(modes), dtype=bool)
    plateau_index = selected["plateau_index"]
    if plateau_index is not None:
        separated = np.abs(series["t"] - series["t"][plateau_index]) >= series["window"] / 2
    pools = (
        (rising & (families == "exploit") & separated & np.isfinite(delta), False),
        (rising & separated & np.isfinite(delta), True),
        (rising & np.isfinite(delta), True),
    )
    for pool, fallback in pools:
        if not np.any(pool):
            continue
        chosen = int(np.flatnonzero(pool)[np.nanargmin(delta[pool])])
        chosen = _walk_forward(series, chosen)
        selected["progress_index"] = chosen
        if fallback and _family(modes[chosen]) != "exploit":
            selected["progress_title"] = f"Rising window ({modes[chosen]})"
        else:
            selected["progress_title"] = "Rising window"
        break
    return selected


def reconstruct_temporal_series(
    result: MKPOptimizationResult,
    configuration: dict[str, Any],
) -> dict[str, Any] | None:
    """Rebuild ready windows from convergence and the stored fire resets.

    Returns None for M0. Non-ready rows stay NaN even if a fixture left a
    sentinel distance in ``dtw_history``. Ready D1/D2 must match the public
    distance on the actual window; otherwise the figure is refused.
    """

    if _is_m0(result):
        return None
    dtw = _dtw_config(configuration)
    values = [float(value) for value in result.convergence_history]
    rows = list(result.dtw_history)
    modes = [str(mode) for mode in result.mode_history]
    if len(rows) != len(values) or len(modes) != len(values):
        raise ValueError(
            "convergence_history, dtw_history y mode_history deben alinearse"
        )
    count = len(values)
    window = int(dtw.window)
    ramp = np.full((count, window), np.nan)
    constant = np.full((count, window), np.nan)
    observed_windows = np.full((count, window), np.nan)
    distance_r = np.full(count, np.nan)
    distance_c = np.full(count, np.nan)
    delta = np.full(count, np.nan)
    theta_r = np.full(count, np.nan)
    theta_c = np.full(count, np.nan)
    theta_delta = np.full(count, np.nan)
    rise = np.full(count, np.nan)
    ready = np.zeros(count, dtype=bool)
    fire = np.zeros(count, dtype=bool)
    adaptive = np.zeros(count, dtype=bool)
    episode_start = np.zeros(count, dtype=int)
    distance = ddtw_distance if dtw.use_ddtw else dtw_distance
    band = dtw.effective_band
    start = 0
    ready_in_episode = 0
    for index, row in enumerate(rows):
        episode_start[index] = start
        is_ready = bool(row.get("ready", False))
        is_fire = bool(row.get("fire", False))
        ready[index] = is_ready
        fire[index] = is_fire
        episode_length = index - start + 1
        stored_n = row.get("n")
        if stored_n is not None and int(stored_n) != episode_length:
            raise ValueError(
                f"n={stored_n} no coincide con las observaciones desde el rearme "
                f"en el índice {index}"
            )
        if is_ready:
            if episode_length < window or index - window + 1 < start:
                raise ValueError(
                    f"la ventana lista del índice {index} cruzaría el reset o el calentamiento"
                )
            observed = np.asarray(values[index - window + 1 : index + 1], dtype=float)
            ramp_row, constant_row = _references(observed, dtw.min_slope)
            recomputed_r = float(distance(observed, ramp_row, band))
            recomputed_c = float(distance(observed, constant_row, band))
            stored_r = row.get("D1_vs_ramp")
            stored_c = row.get("D2_vs_const")
            if stored_r is None or stored_c is None:
                raise RuntimeError(f"falta D1/D2 almacenado en el índice {index}")
            if (
                abs(recomputed_r - float(stored_r)) > _DISTANCE_TOLERANCE
                or abs(recomputed_c - float(stored_c)) > _DISTANCE_TOLERANCE
            ):
                raise RuntimeError(
                    f"D1/D2 recomputado no coincide en el índice {index}: "
                    f"D1 {recomputed_r} vs {stored_r}, D2 {recomputed_c} vs {stored_c}"
                )
            stored_delta = row.get("delta")
            recomputed_delta = recomputed_r - recomputed_c
            if (
                stored_delta is not None
                and abs(float(stored_delta) - recomputed_delta) > _DISTANCE_TOLERANCE
            ):
                raise RuntimeError(
                    f"delta almacenado no coincide con D1-D2 en el índice {index}"
                )
            ramp[index] = ramp_row
            constant[index] = constant_row
            observed_windows[index] = observed
            distance_r[index] = recomputed_r
            distance_c[index] = recomputed_c
            delta[index] = float(stored_delta) if stored_delta is not None else recomputed_delta
            for source_key, destination in (
                ("theta_r", theta_r),
                ("theta_c", theta_c),
                ("theta_delta", theta_delta),
            ):
                if row.get(source_key) is not None:
                    destination[index] = float(row[source_key])
            rise[index] = float(observed[-1] - observed[0])
            ready_in_episode += 1
            # Early ready rows still have a real window. The percentile flag
            # is recorded only so it is not reused as the eligibility mask.
            adaptive[index] = ready_in_episode >= 10
        if is_fire:
            start = index + 1
            ready_in_episode = 0
    series = {
        "t": np.arange(1, count + 1),
        "f_star": np.asarray(values, dtype=float),
        "R": ramp,
        "C": constant,
        "X": observed_windows,
        "D_R": distance_r,
        "D_C": distance_c,
        "delta": delta,
        "theta_R": theta_r,
        "theta_C": theta_c,
        "theta_delta": theta_delta,
        "rise": rise,
        "ready": ready,
        "fire": fire,
        "adaptive_threshold": adaptive,
        "episode_start": episode_start,
        "mode": modes,
        "window": window,
        "band": int(dtw.band),
        "effective_band": int(band),
        "min_slope": float(dtw.min_slope),
        "use_ddtw": bool(dtw.use_ddtw),
    }
    return _select_windows(series)


def _shade_state(axis: plt.Axes, t: np.ndarray, modes: list[str], label: bool = True) -> None:
    """Source binary shade, extended with base and four-state aliases.

    ``base`` is gray, not exploitation. Aliases share the explore or exploit
    color but keep their own legend text. No state is drawn as 0 or 1.
    """

    if not modes:
        return
    start = 0
    labeled: set[str] = set()
    for index in range(1, len(modes) + 1):
        if index == len(modes) or modes[index] != modes[start]:
            mode = modes[start]
            span_label = None
            if label and mode not in labeled:
                span_label = mode
                labeled.add(mode)
            axis.axvspan(
                t[start] - 0.5,
                t[index - 1] + 0.5,
                color=_state_color(mode),
                alpha=0.35,
                linewidth=0,
                label=span_label,
                zorder=0,
            )
            start = index


def _mark_windows(axis: plt.Axes, marks: list[tuple[int, str]]) -> None:
    for position, color in marks:
        axis.axvline(position, color=color, lw=0.7, ls=":", zorder=2)


def _scale_to_data(axis: plt.Axes, arrays: list[np.ndarray]) -> None:
    """Replace source ``ylim(-30, 460)`` and delta pad 4 for this fitness scale."""

    if not arrays:
        return
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    pad = 0.12 * span if span > 0 else max(0.05 * max(abs(lower), abs(upper), 1e-9), 1e-3)
    axis.set_ylim(lower - pad, upper + pad)


def _parameter_note(parameter_history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for key, label in (("woa_a_effective", "a_eff"), ("abc_phi_effective", "phi_eff")):
        values = [
            float(row[key])
            for row in parameter_history
            if row.get(key) is not None
        ]
        if not values:
            continue
        lower = min(values)
        upper = max(values)
        rendered = f"{lower:.4g}" if abs(upper - lower) <= 1e-9 else f"{lower:.4g}–{upper:.4g}"
        lines.append(f"{label} {rendered}")
    return "\n".join(lines)


def _state_levels(modes: list[str]) -> list[str]:
    ordered = [mode for mode in _STATE_ORDER if mode in modes]
    ordered.extend(mode for mode in modes if mode not in ordered)
    return ordered


def _unavailable(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=8.5)
    axis.text(
        0.5,
        0.5,
        _UNAVAILABLE,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8,
    )
    axis.set_xlabel("Iteration")


def _draw_window(
    axis: plt.Axes,
    series: dict[str, Any],
    index: int,
    title: str,
) -> tuple[Any, Any, Any]:
    window = int(series["window"])
    iterations = series["t"][index - window + 1 : index + 1]
    observed = series["X"][index] - series["X"][index, 0]
    reference_r = series["R"][index] - series["R"][index, 0]
    reference_c = series["C"][index] - series["C"][index, 0]
    ramp_line, = axis.plot(iterations, reference_r, color=_RAMP, lw=1.05, label=r"$R_t$", zorder=2)
    observed_line, = axis.step(
        iterations,
        observed,
        where="pre",
        color="#1a1a1a",
        lw=1.25,
        label=r"$X_t$",
        zorder=3,
    )
    const_line, = axis.plot(
        iterations,
        reference_c,
        color=_CONSTANT,
        lw=1.15,
        ls="--",
        label=r"$C_t$",
        zorder=4,
    )
    axis.set_xlim(iterations[0], iterations[-1])
    axis.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    axis.set_title(title, loc="left", fontsize=8.5)
    axis.set_xlabel("Iteration")
    return observed_line, ramp_line, const_line


def draw_temporal_diagnostics(
    result: MKPOptimizationResult,
    configuration: dict[str, Any],
) -> plt.Figure | None:
    """Draw the four-row diagnostic figure, or None when the run is M0."""

    if _is_m0(result):
        return None
    series = reconstruct_temporal_series(result, configuration)
    if series is None:
        return None
    previous = {key: plt.rcParams[key] for key in _RC}
    plt.rcParams.update(_RC)
    try:
        return _draw(result, series)
    finally:
        plt.rcParams.update(previous)


def _draw(result: MKPOptimizationResult, series: dict[str, Any]) -> plt.Figure:
    t = series["t"]
    modes = list(series["mode"])
    window = int(series["window"])
    ready = np.asarray(series["ready"], dtype=bool)
    progress_index = series["progress_index"]
    plateau_index = series["plateau_index"]
    # Full-width rows. A half-width window panel cannot hold 4-digit iteration
    # labels, and a twin axis collides with the controller-state labels.
    figure = plt.figure(figsize=(7.4, 8.6))
    gridspec = figure.add_gridspec(4, 1, height_ratios=[1.15, 1.05, 1.05, 1.35], hspace=0.55)
    axis_a = figure.add_subplot(gridspec[0, 0])
    gridspec_b = gridspec[1, 0].subgridspec(1, 2, wspace=0.18)
    axis_rise = figure.add_subplot(gridspec_b[0, 0])
    axis_flat = figure.add_subplot(gridspec_b[0, 1], sharey=axis_rise)
    axis_c = figure.add_subplot(gridspec[2, 0], sharex=axis_a)
    gridspec_d = gridspec[3, 0].subgridspec(2, 1, height_ratios=[1.45, 0.85], hspace=0.18)
    axis_delta = figure.add_subplot(gridspec_d[0, 0], sharex=axis_a)
    axis_state = figure.add_subplot(gridspec_d[1, 0], sharex=axis_a)

    _shade_state(axis_a, t, modes)
    if len(t):
        axis_a.step(t, series["f_star"], where="post", color="#1a1a1a", lw=1.05, label=r"$f_t^*$", zorder=3)
        axis_a.set_xlim(t[0], t[-1])
    for index, color in ((progress_index, _RAMP), (plateau_index, _CONSTANT)):
        if index is None:
            continue
        axis_a.axvspan(
            t[index] - window + 1,
            t[index],
            facecolor="none",
            edgecolor=color,
            lw=1.15,
            zorder=4,
        )
    axis_a.set_ylabel(r"Best-so-far fitness $f_t^*$")
    axis_a.legend(loc="lower right", frameon=False)
    axis_a.set_title("(a)  Trajectory and controller state", loc="left")

    legend_lines = None
    if progress_index is None:
        _unavailable(axis_rise, f"(b)  {series['progress_title']}")
    else:
        legend_lines = _draw_window(
            axis_rise,
            series,
            progress_index,
            rf"(b)  {series['progress_title']}, $t={int(t[progress_index])}$",
        )
        axis_rise.set_ylabel("Fitness change in the window")
    if plateau_index is None:
        _unavailable(axis_flat, series["plateau_title"])
    else:
        drawn = _draw_window(
            axis_flat,
            series,
            plateau_index,
            rf"{series['plateau_title']}, $t={int(t[plateau_index])}$",
        )
        legend_lines = legend_lines or drawn
        axis_flat.tick_params(labelleft=False)
    window_values: list[np.ndarray] = []
    for index in (progress_index, plateau_index):
        if index is None:
            continue
        window_values.extend([
            series["X"][index] - series["X"][index, 0],
            series["R"][index] - series["R"][index, 0],
            series["C"][index] - series["C"][index, 0],
        ])
    _scale_to_data(axis_rise, window_values)
    if legend_lines is not None:
        axis_rise.legend(
            list(legend_lines),
            [r"$X_t$", r"$R_t$", r"$C_t$"],
            loc="upper left",
            frameon=False,
            handlelength=1.8,
        )

    distance_label = "DDTW distance" if series["use_ddtw"] else "DTW distance"
    axis_c.plot(t[ready], series["D_R"][ready], color=_RAMP, lw=0.95, label=r"$D_R(t)$")
    axis_c.plot(t[ready], series["theta_R"][ready], color=_RAMP, lw=0.8, ls="--", label=r"$\theta_R(t)$")
    axis_c.plot(t[ready], series["D_C"][ready], color=_CONSTANT, lw=0.95, label=r"$D_C(t)$")
    axis_c.plot(t[ready], series["theta_C"][ready], color=_CONSTANT, lw=0.8, ls="--", label=r"$\theta_C(t)$")
    marks = [
        (int(t[index]), color)
        for index, color in ((progress_index, _RAMP), (plateau_index, _CONSTANT))
        if index is not None
    ]
    _mark_windows(axis_c, marks)
    if len(t):
        axis_c.set_xlim(t[0], t[-1])
    axis_c.set_ylabel(distance_label)
    axis_c.legend(loc="upper right", frameon=False, ncol=2)
    axis_c.set_title(r"(c)  Distances $D_R$, $D_C$ and thresholds", loc="left")
    _scale_to_data(axis_c, [
        series["D_R"][ready],
        series["D_C"][ready],
        series["theta_R"][ready],
        series["theta_C"][ready],
    ])

    axis_delta.plot(t[ready], series["delta"][ready], color=_DELTA, lw=0.95, label=r"$\Delta_t$")
    axis_delta.plot(
        t[ready],
        series["theta_delta"][ready],
        color=_DELTA,
        lw=0.8,
        ls="--",
        label=r"$\theta_\Delta(t)$",
    )
    _mark_windows(axis_delta, marks)
    axis_delta.set_ylabel(r"$\Delta_t = D_R-D_C$")
    axis_delta.legend(loc="lower right", frameon=False)
    axis_delta.set_title(r"(d)  Relative indicator and controller state", loc="left")
    _scale_to_data(axis_delta, [series["delta"][ready], series["theta_delta"][ready]])

    levels = _state_levels(modes)
    level_of = {mode: index for index, mode in enumerate(levels)}
    coded = np.asarray([level_of[mode] for mode in modes], dtype=float) if modes else np.asarray([])
    _shade_state(axis_state, t, modes, label=False)
    if len(t):
        axis_state.step(t, coded, where="post", color="#1a1a1a", lw=1.0, zorder=3)
    entered = {"explore": False, "exploit": False, "base": False}
    markers = {
        "explore": ("v", _CONSTANT, "To exploration"),
        "exploit": ("^", _RAMP, "To exploitation"),
        "base": ("s", _BASE, "To base"),
    }
    for index in range(1, len(modes)):
        if modes[index] == modes[index - 1]:
            continue
        family = _family(modes[index])
        if family not in markers or entered[family]:
            continue
        marker, color, label = markers[family]
        axis_state.plot(
            [t[index]],
            [level_of[modes[index]]],
            ls="none",
            marker=marker,
            ms=4.5,
            color=color,
            zorder=5,
            label=label,
        )
        entered[family] = True
    if levels:
        axis_state.set_yticks(range(len(levels)))
        axis_state.set_yticklabels(levels)
        axis_state.set_ylim(-0.35, max(len(levels) - 0.65, 0.65))
    axis_state.set_xlabel("Iteration $t$")
    axis_state.set_ylabel(r"$s_t$")
    note = _parameter_note(list(getattr(result, "parameter_history", []) or []))
    if note:
        axis_state.text(
            0.01,
            0.98,
            note,
            transform=axis_state.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#333333",
            zorder=6,
        )
    if axis_state.get_legend_handles_labels()[0]:
        axis_state.legend(loc="lower right", frameon=False, ncol=2)
    for axis in (axis_c, axis_delta):
        axis.tick_params(labelbottom=False)
        axis.set_xlabel("")
    axis_a.tick_params(labelbottom=True)
    figure.subplots_adjust(left=0.16, right=0.985, top=0.975, bottom=0.07)
    return figure


def save_temporal_diagnostics(
    run_dir: Path,
    result: MKPOptimizationResult,
    configuration: dict[str, Any],
) -> Path | None:
    """Write ``temporal_diagnostics.pdf`` only. M0 writes nothing."""

    figure = draw_temporal_diagnostics(result, configuration)
    if figure is None:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "temporal_diagnostics.pdf"
    fonttype = plt.rcParams["pdf.fonttype"]
    ps_fonttype = plt.rcParams["ps.fonttype"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    try:
        figure.savefig(path, bbox_inches="tight", pad_inches=0.2)
    finally:
        plt.rcParams["pdf.fonttype"] = fonttype
        plt.rcParams["ps.fonttype"] = ps_fonttype
        plt.close(figure)
    return path


def _load_table(
    path: Path,
    *,
    boolean_fields: set[str],
    integer_fields: set[str],
) -> list[dict[str, Any]]:
    rows = load_typed_rows(
        path,
        integer_fields=integer_fields,
        boolean_fields=boolean_fields,
    )
    for row in rows:
        for key, value in list(row.items()):
            if not isinstance(value, str):
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            row[key] = int(value) if key in integer_fields else numeric
    if rows and "iteration" in rows[0]:
        rows.sort(key=lambda row: int(row["iteration"]))
    return rows


def load_saved_mkp_run(
    run_dir: Path,
) -> tuple[MKPOptimizationResult, dict[str, Any]]:
    """Rebuild one saved MKP run from its CSV/JSON. Does not optimize."""

    payload = json.loads((run_dir / "resultado.json").read_text(encoding="utf-8"))
    summary = dict(payload["summary"])
    configuration = dict(payload["configuration"])
    required = (
        "problem",
        "variant",
        "seed",
        "best_profit",
        "objective_evaluations",
        "iterations_completed",
        "optimum_reached",
        "stop_reason",
        "fire_count",
        "transition_count",
        "parameter_update_count",
        "local_search_improvements",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"resultado.json no trae {', '.join(missing)}")
    convergence = _load_table(
        run_dir / "historial_convergencia.csv",
        boolean_fields=set(),
        integer_fields={"iteration"},
    )
    dtw_rows = _load_table(
        run_dir / "historial_dtw.csv",
        boolean_fields={
            "ready",
            "fire",
            "cond_plateau",
            "cond_constant",
            "cond_ramp",
            "transition",
            "parameter_changed",
            "raw_fire",
        },
        integer_fields={"iteration", "n", "no_improve_len", "trigger_streak"},
    )
    parameters = _load_table(
        run_dir / "historial_parametros.csv",
        boolean_fields=set(),
        integer_fields={"iteration"},
    )
    events_path = run_dir / "eventos_control.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.is_file() else []
    best_known = summary.get("best_known")
    result = MKPOptimizationResult(
        problem=str(summary["problem"]),
        variant=str(summary["variant"]),
        seed=int(summary["seed"]),
        best_profit=float(summary["best_profit"]),
        best_solution=[int(bit) for bit in payload["best_solution"]],
        best_known=None if best_known is None else float(best_known),
        objective_evaluations=int(summary["objective_evaluations"]),
        iterations_completed=int(summary["iterations_completed"]),
        optimum_reached=bool(summary["optimum_reached"]),
        stop_reason=str(summary["stop_reason"]),
        fire_count=int(summary["fire_count"]),
        transition_count=int(summary["transition_count"]),
        parameter_update_count=int(summary["parameter_update_count"]),
        local_search_improvements=int(summary["local_search_improvements"]),
        convergence_history=[float(row["best_profit"]) for row in convergence],
        woa_history=[float(row["best_after_woa"]) for row in convergence],
        abc_history=[float(row["best_after_abc"]) for row in convergence],
        mode_history=[str(row["mode"]) for row in convergence],
        intensity_history=[
            None if row.get("intensity") in (None, "") else float(row["intensity"])
            for row in convergence
        ],
        dtw_history=dtw_rows,
        parameter_history=parameters,
        control_events=list(events),
    )
    return result, configuration


__all__ = [
    "draw_temporal_diagnostics",
    "load_saved_mkp_run",
    "reconstruct_temporal_series",
    "save_temporal_diagnostics",
]
