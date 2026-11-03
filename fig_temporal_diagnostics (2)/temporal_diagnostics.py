"""COPIA para llevar a otro proyecto. La guía está en COMO_HACERLO.md.

Este archivo es el script que generó la figura de diagnóstico temporal.
Sigue funcionando dentro de DTW_optimization: la carpeta `lucas copiar`
está un nivel bajo la raíz, y ROOT apunta a esa raíz.

En otro proyecto hay que cambiar solo el bloque marcado ADAPTAR:
los imports, la corrida de una semilla y los nombres de las series.
El guardado, la figura de cuatro paneles y el texto LaTeX se reutilizan.

Temporal diagnostics for one representative Binary-Complex run.

Produces the series requested for the reviewer response (R3.4, R3.8, R2.3)
and a four-panel figure. One run only: it does not repeat the benchmark.

The run is the first independent replicate of the manuscript protocol:
Binary-Complex, BDE, Chu-Beasley mknapcb1 (index 0), seed q = 1.
Trajectory-analysis parameters are the values published in the manuscript
(W = 200, Sakoe-Chiba half-width 2, s_min = 2, percentiles 40/60,
pi_max = 5, tau_pat = 3, DDTW on). They are set here on purpose, because
mkp_common.config currently holds a different sensitivity setting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ADAPTAR: en este repo el script vive en "<raíz>/lucas copiar/", así que
# la raíz del proyecto es parents[1]. Si movés la carpeta, apuntá ROOT
# a la raíz del otro proyecto.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ADAPTAR: imports del controlador y del monitor del otro proyecto.
from binary_complex.config import HysteresisController
from mkp_common import BinaryDE, StagnationConfig, StagnationMonitor, cargar_instancia
from mkp_common.monitor import constant_baseline, ddtw_distance, ramp_baseline

OUT_DIR = ROOT / "analisis" / "temporal_diagnostics"
LATEX_DIR = ROOT / "contexto" / "latex"

# ADAPTAR: parámetros publicados en el manuscrito, no los del config global.
PAPER_CFG = StagnationConfig(
    window=200,
    band=2,
    min_slope=2.0,
    plateau_max=5,
    patience=3,
    use_ddtw=True,
    adapt_thresholds=True,
    p_low=40.0,
    p_high=60.0,
)

INSTANCE = "instances/mknapcb1.txt"
INSTANCE_INDEX = 0
SEED = 1
N_PARTICLES = 20
N_ITERATIONS = 2000
STRATEGY = "Binary-Complex"
ALGORITHM = "BDE"


def run_representative() -> dict:
    """ADAPTAR: una sola corrida, con el mismo lazo que el runner del paper."""
    inst = cargar_instancia(str(ROOT / INSTANCE), INSTANCE_INDEX)
    rng = np.random.default_rng(SEED)
    mh = BinaryDE(inst, rng, num_particulas=N_PARTICLES)
    monitor = StagnationMonitor(cfg=PAPER_CFG)
    controller = HysteresisController(initial_mode="explore")

    mh.initialize()
    mh.adapt(True)  # initial configuration is exploration, as in binary_complex

    t_count = N_ITERATIONS
    window = PAPER_CFG.window
    f_star = np.empty(t_count, dtype=float)
    mode_used = np.empty(t_count, dtype=np.int8)  # configuration that generated t
    s_t = np.empty(t_count, dtype=np.int8)  # decision after observing t
    ready = np.zeros(t_count, dtype=bool)
    adaptive = np.zeros(t_count, dtype=bool)
    fire = np.zeros(t_count, dtype=bool)
    d_r = np.full(t_count, np.nan)
    d_c = np.full(t_count, np.nan)
    delta = np.full(t_count, np.nan)
    theta_r = np.full(t_count, np.nan)
    theta_c = np.full(t_count, np.nan)
    theta_delta = np.full(t_count, np.nan)
    no_improve = np.zeros(t_count, dtype=int)
    trigger_streak = np.zeros(t_count, dtype=int)
    F_param = np.empty(t_count, dtype=float)
    CR_param = np.empty(t_count, dtype=float)
    # R_t and C_t are length-W references. Undefined until the window is full.
    ramp = np.full((t_count, window), np.nan)
    const = np.full((t_count, window), np.nan)

    t0 = time.perf_counter()
    for it in range(t_count):
        mode_used[it] = 1 if mh.mode == "explore" else 0
        fitness = mh.step()
        out = monitor.update(fitness)
        if out.get("ready"):
            fire_now = controller(out)
            mh.adapt(fire_now)
        else:
            # Same warm-up path as run_experiment(decision_on_early=True):
            # the controller is queried and keeps the initial mode.
            fire_now = controller(out)
            mh.adapt(fire_now)

        s_t[it] = 1 if mh.mode == "explore" else 0
        F_param[it] = mh.F
        CR_param[it] = mh.CR
        f_star[it] = float(monitor.best_so_far[-1])
        no_improve[it] = int(out["no_improve_len"])
        trigger_streak[it] = int(out["trigger_streak"])
        fire[it] = bool(out["fire"])

        if out.get("ready"):
            ready[it] = True
            X = np.asarray(monitor.best_so_far[-window:], dtype=float)
            r = ramp_baseline(float(X[0]), PAPER_CFG.min_slope, window)
            c = constant_baseline(float(X[0]), window)
            ramp[it] = r
            const[it] = c
            d_r[it] = float(out["D1_vs_ramp"])
            d_c[it] = float(out["D2_vs_const"])
            delta[it] = float(out["delta"])
            theta_r[it] = float(out["theta_r"])
            theta_c[it] = float(out["theta_c"])
            theta_delta[it] = float(out["theta_delta"])
            # The monitor uses fixed placeholders until 10 ready samples exist.
            adaptive[it] = len(monitor.d1_hist) >= 10

            recomputed_r = ddtw_distance(X, r, window=PAPER_CFG.band)
            recomputed_c = ddtw_distance(X, c, window=PAPER_CFG.band)
            if abs(recomputed_r - d_r[it]) > 1e-6 or abs(recomputed_c - d_c[it]) > 1e-6:
                raise RuntimeError(
                    f"DDTW recomputation diverged at iteration {it + 1}: "
                    f"D_R {recomputed_r} vs {d_r[it]}, D_C {recomputed_c} vs {d_c[it]}"
                )

    elapsed = time.perf_counter() - t0
    best_fit = float(mh.get_best()[1])
    optimum = float(inst["optimo"])
    entries = int(np.sum((s_t[1:] == 1) & (s_t[:-1] == 0)))

    meta = {
        "strategy": STRATEGY,
        "algorithm": ALGORITHM,
        "instance": INSTANCE,
        "instance_index": INSTANCE_INDEX,
        "n_items": int(inst["n"]),
        "n_constraints": int(inst["m"]),
        "optimum": optimum,
        "seed": SEED,
        "population": N_PARTICLES,
        "iterations": N_ITERATIONS,
        "initial_mode": "explore",
        "best_fitness": best_fit,
        "gap_percent": (optimum - best_fit) / optimum * 100.0,
        "explore_entries": entries,
        "elapsed_seconds": elapsed,
        "dtw_window": PAPER_CFG.window,
        "dtw_band": PAPER_CFG.band,
        "dtw_min_slope": PAPER_CFG.min_slope,
        "dtw_p_low": PAPER_CFG.p_low,
        "dtw_p_high": PAPER_CFG.p_high,
        "dtw_plateau_max": PAPER_CFG.plateau_max,
        "dtw_patience": PAPER_CFG.patience,
        "dtw_use_ddtw": PAPER_CFG.use_ddtw,
        "bde_exploit": {"F": BinaryDE.F_EXPLOIT, "CR": BinaryDE.CR_EXPLOIT},
        "bde_explore": {"F": BinaryDE.F_EXPLORE, "CR": BinaryDE.CR_EXPLORE},
        "series_notes": {
            "t": "Iteration index in the manuscript, t = 1..T.",
            "f_star": "Best-so-far fitness after iteration t.",
            "R": "Ramp reference of the window ending at t. Shape (T, W). NaN before warm-up.",
            "C": "Constant reference of the window ending at t. Shape (T, W). NaN before warm-up.",
            "D_R": "DDTW(X_t, R_t). NaN before warm-up.",
            "D_C": "DDTW(X_t, C_t). NaN before warm-up.",
            "delta": "D_R - D_C.",
            "theta_R": "Threshold on D_R. Fixed placeholder for the first 9 ready iterations, then percentile p_high.",
            "theta_C": "Threshold on D_C. Fixed placeholder for the first 9 ready iterations, then percentile p_low.",
            "theta_delta": "Threshold on delta. Fixed placeholder for the first 9 ready iterations, then percentile p_high.",
            "adaptive_threshold": "True once thresholds come from the historical percentiles.",
            "s": "Controller state after observing iteration t. 0 exploitation, 1 exploration. This is the configuration used at t+1.",
            "mode_used": "Configuration that generated the candidates of iteration t.",
            "F": "BDE scale factor active after the decision at t.",
            "CR": "BDE crossover rate active after the decision at t.",
        },
    }
    return {
        "meta": meta,
        "t": np.arange(1, t_count + 1),
        "f_star": f_star,
        "R": ramp,
        "C": const,
        "D_R": d_r,
        "D_C": d_c,
        "delta": delta,
        "theta_R": theta_r,
        "theta_C": theta_c,
        "theta_delta": theta_delta,
        "s": s_t,
        "mode_used": mode_used,
        "F": F_param,
        "CR": CR_param,
        "ready": ready,
        "adaptive_threshold": adaptive,
        "fire": fire,
        "no_improve_len": no_improve,
        "trigger_streak": trigger_streak,
    }


def save_series(data: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / "series_binary_complex_bde_mknapcb1_seed1.npz"
    arrays = {k: v for k, v in data.items() if k != "meta"}
    np.savez_compressed(npz_path, **arrays)
    meta_path = OUT_DIR / "series_binary_complex_bde_mknapcb1_seed1.json"
    meta_path.write_text(json.dumps(data["meta"], indent=2), encoding="utf-8")

    t = data["t"]
    columns = [
        t,
        data["f_star"],
        data["D_R"],
        data["D_C"],
        data["delta"],
        data["theta_R"],
        data["theta_C"],
        data["theta_delta"],
        data["s"],
        data["mode_used"],
        data["F"],
        data["CR"],
        data["ready"].astype(int),
        data["adaptive_threshold"].astype(int),
        data["fire"].astype(int),
        data["no_improve_len"],
        data["trigger_streak"],
    ]
    header = (
        "t,f_star,D_R,D_C,delta,theta_R,theta_C,theta_delta,"
        "s,mode_used,F,CR,ready,adaptive_threshold,fire,no_improve_len,trigger_streak"
    )
    csv_path = OUT_DIR / "series_binary_complex_bde_mknapcb1_seed1.csv"
    np.savetxt(csv_path, np.column_stack(columns), delimiter=",", header=header, comments="")
    return npz_path


def _window_rise(data: dict) -> np.ndarray:
    """End-to-start change of the best-so-far window. NaN before it exists."""
    f_star = data["f_star"]
    window = int(data["meta"]["dtw_window"])
    rise = np.full(f_star.shape, np.nan)
    ready = np.flatnonzero(data["adaptive_threshold"])
    for i in ready:
        if i + 1 >= window:
            rise[i] = f_star[i] - f_star[i - window + 1]
    return rise


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return the inclusive index range of the longest True run."""
    best = (0, -1)
    start = None
    for i, flag in enumerate(mask.tolist() + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


def _window_end(data: dict) -> tuple[int, int]:
    """Indexes of the progress window and the plateau window.

    Progress window: exploitation iteration whose window rises, chosen by
    the smallest relative indicator Delta. On this run that window gains
    almost the same fitness as the ramp template.

    Plateau window: midpoint of the longest exploration episode whose whole
    window is exactly flat, so the constant reference lies on the data.
    """
    ready = data["adaptive_threshold"]
    s = data["s"]
    delta = data["delta"]
    rise = _window_rise(data)

    flat_explore = ready & (s == 1) & (rise == 0)
    p0, p1 = _longest_true_run(flat_explore)
    if p1 < p0:
        plateau_i = int(np.nanargmin(np.where(ready, data["D_C"], np.nan)))
    else:
        plateau_i = (p0 + p1) // 2

    separated = np.abs(data["t"] - data["t"][plateau_i]) >= data["meta"]["dtw_window"] / 2
    progress_pool = ready & (s == 0) & (rise > 0) & separated & np.isfinite(delta)
    if not np.any(progress_pool):
        progress_pool = ready & (rise > 0) & separated & np.isfinite(delta)
    progress_i = int(np.flatnonzero(progress_pool)[np.nanargmin(delta[progress_pool])])
    # The closest-to-ramp iteration can fall on the improvement itself, so a
    # step plot hides that level. Walk forward while the same best-so-far
    # value is held and the window still contains the rise.
    f_star = data["f_star"]
    end_value = f_star[progress_i]
    target_rise = rise[progress_i]
    j = progress_i
    while (
        j + 1 < len(f_star)
        and s[j + 1] == 0
        and f_star[j + 1] == end_value
        and rise[j + 1] >= 0.9 * target_rise
    ):
        j += 1
    return j, plateau_i


def _shade_state(ax, t: np.ndarray, s: np.ndarray, label: bool = True) -> None:
    explore = "#F4A261"
    exploit = "#8ECAE6"
    start = 0
    labeled = {0: False, 1: False}
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            state = int(s[start])
            span_label = None
            if label and not labeled[state]:
                span_label = "Exploration" if state == 1 else "Exploitation"
                labeled[state] = True
            ax.axvspan(
                t[start] - 0.5,
                t[i - 1] + 0.5,
                color=explore if state == 1 else exploit,
                alpha=0.35,
                linewidth=0,
                label=span_label,
                zorder=0,
            )
            start = i


def _mark_windows(ax, t_progress: int, t_plateau: int) -> None:
    ax.axvline(t_progress, color="#0072B2", lw=0.7, ls=":", zorder=2)
    ax.axvline(t_plateau, color="#D55E00", lw=0.7, ls=":", zorder=2)


def plot_figure(data: dict) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
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
    })

    t = data["t"]
    f_star = data["f_star"]
    s = data["s"]
    d_r = data["D_R"]
    d_c = data["D_C"]
    delta = data["delta"]
    theta_r = data["theta_R"]
    theta_c = data["theta_C"]
    theta_delta = data["theta_delta"]
    adaptive = data["adaptive_threshold"]
    window = int(data["meta"]["dtw_window"])
    progress_i, plateau_i = _window_end(data)
    t_progress = int(t[progress_i])
    t_plateau = int(t[plateau_i])
    mask = adaptive

    # Full-width rows. A half-width window panel cannot hold 4-digit iteration
    # labels, and a twin axis collides with the controller-state labels.
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.05, 1.05, 1.35], hspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gs[1, 0].subgridspec(1, 2, wspace=0.18)
    ax_bp = fig.add_subplot(gs_b[0, 0])
    ax_bc = fig.add_subplot(gs_b[0, 1], sharey=ax_bp)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    gs_d = gs[3, 0].subgridspec(2, 1, height_ratios=[1.45, 0.85], hspace=0.18)
    ax_delta = fig.add_subplot(gs_d[0, 0], sharex=ax_a)
    ax_s = fig.add_subplot(gs_d[1, 0], sharex=ax_a)

    # (a) best-so-far. Background is the controller state.
    _shade_state(ax_a, t, s)
    ax_a.step(t, f_star, where="post", color="#1a1a1a", lw=1.05, label=r"$f_t^*$", zorder=3)
    for idx, color in ((progress_i, "#0072B2"), (plateau_i, "#D55E00")):
        ax_a.axvspan(
            t[idx] - window + 1,
            t[idx],
            facecolor="none",
            edgecolor=color,
            lw=1.15,
            zorder=4,
        )
    ax_a.set_xlim(1, t[-1])
    ax_a.set_ylabel(r"Best-so-far fitness $f_t^*$")
    ax_a.legend(loc="lower right", frameon=False)
    ax_a.set_title("(a)  Trajectory and controller state", loc="left")

    # (b) same two references, both windows measured from their own start.
    def draw_window(ax, idx: int, title: str) -> None:
        i0 = idx - window + 1
        iterations = np.arange(i0 + 1, idx + 2)
        observed = f_star[i0:idx + 1] - f_star[i0]
        reference_r = data["R"][idx] - data["R"][idx, 0]
        reference_c = data["C"][idx] - data["C"][idx, 0]
        ramp_line, = ax.plot(
            iterations, reference_r, color="#0072B2", lw=1.05, label=r"$R_t$", zorder=2
        )
        # steps-pre keeps the value at the right edge of the window visible.
        observed_line, = ax.step(
            iterations, observed, where="pre", color="#1a1a1a", lw=1.25, label=r"$X_t$", zorder=3
        )
        # Drawn last so a flat window still shows the constant reference on top of X_t.
        const_line, = ax.plot(
            iterations, reference_c, color="#D55E00", lw=1.15, ls="--", label=r"$C_t$", zorder=4
        )
        ax.set_xlim(iterations[0], iterations[-1])
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("Iteration")
        return observed_line, ramp_line, const_line

    observed_line, ramp_line, const_line = draw_window(
        ax_bp, progress_i, rf"(b)  Rising window, $t={t_progress}$"
    )
    draw_window(ax_bc, plateau_i, rf"Flat window, $t={t_plateau}$")
    ax_bp.set_ylabel("Fitness change in the window")
    ax_bp.set_ylim(-30, 460)
    ax_bc.tick_params(labelleft=False)
    ax_bp.legend(
        [observed_line, ramp_line, const_line],
        [r"$X_t$", r"$R_t$", r"$C_t$"],
        loc="upper left",
        frameon=False,
        handlelength=1.8,
    )

    # (c) distances to each reference and the percentiles used as thresholds.
    ax_c.plot(t[mask], d_r[mask], color="#0072B2", lw=0.95, label=r"$D_R(t)$")
    ax_c.plot(t[mask], theta_r[mask], color="#0072B2", lw=0.8, ls="--", label=r"$\theta_R(t)$")
    ax_c.plot(t[mask], d_c[mask], color="#D55E00", lw=0.95, label=r"$D_C(t)$")
    ax_c.plot(t[mask], theta_c[mask], color="#D55E00", lw=0.8, ls="--", label=r"$\theta_C(t)$")
    _mark_windows(ax_c, t_progress, t_plateau)
    ax_c.set_xlim(1, t[-1])
    ax_c.set_ylabel("DDTW distance")
    ax_c.legend(loc="upper right", frameon=False, ncol=2)
    ax_c.set_title(r"(c)  Distances $D_R$, $D_C$ and thresholds", loc="left")
    ax_c.tick_params(labelbottom=False)
    ax_c.set_xlabel("")

    # (d) relative indicator on its own scale, controller state underneath.
    ax_delta.plot(t[mask], delta[mask], color="#009E73", lw=0.95, label=r"$\Delta_t$")
    ax_delta.plot(
        t[mask], theta_delta[mask], color="#009E73", lw=0.8, ls="--", label=r"$\theta_\Delta(t)$"
    )
    _mark_windows(ax_delta, t_progress, t_plateau)
    finite_delta = delta[mask]
    lo = float(np.nanmin(finite_delta))
    hi = float(np.nanmax(finite_delta))
    pad = max(4.0, 0.25 * (hi - lo))
    ax_delta.set_ylim(lo - pad, hi + pad)
    ax_delta.set_ylabel(r"$\Delta_t = D_R-D_C$")
    ax_delta.legend(loc="lower right", frameon=False)
    ax_delta.set_title(r"(d)  Relative indicator and controller state", loc="left")
    ax_delta.tick_params(labelbottom=False)

    _shade_state(ax_s, t, s, label=False)
    ax_s.step(t, s, where="post", color="#1a1a1a", lw=1.0, zorder=3)
    to_explore = np.flatnonzero((s[1:] == 1) & (s[:-1] == 0)) + 1
    to_exploit = np.flatnonzero((s[1:] == 0) & (s[:-1] == 1)) + 1
    ax_s.plot(
        t[to_explore], np.ones(to_explore.size),
        ls="none", marker="v", ms=4.5, color="#D55E00", zorder=5,
        label="To exploration",
    )
    ax_s.plot(
        t[to_exploit], np.zeros(to_exploit.size),
        ls="none", marker="^", ms=4.5, color="#0072B2", zorder=5,
        label="To exploitation",
    )
    ax_s.set_ylim(-0.18, 1.28)
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels([
        "0\n" + r"$F{=}0.5,\ CR{=}0.9$",
        "1\n" + r"$F{=}0.9,\ CR{=}0.3$",
    ])
    ax_s.set_xlabel("Iteration $t$")
    ax_s.set_ylabel(r"$s_t$")
    ax_s.legend(loc="lower right", frameon=False, ncol=2)
    for ax in (ax_c, ax_delta):
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    ax_a.tick_params(labelbottom=True)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.975, bottom=0.055)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "fig_temporal_diagnostics.pdf"
    png_path = OUT_DIR / "fig_temporal_diagnostics.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    for src in (pdf_path, png_path):
        target = LATEX_DIR / src.name
        target.write_bytes(src.read_bytes())

    # Drop windows from a previous plot so only the illustrated ones remain.
    for stale in OUT_DIR.glob("window_*.csv"):
        stale.unlink()
    for name, idx in (("progress", progress_i), ("plateau", plateau_i)):
        i0 = idx - window + 1
        rows = np.column_stack([
            np.arange(1, window + 1),
            np.arange(i0 + 1, idx + 2),
            f_star[i0:idx + 1],
            data["R"][idx],
            data["C"][idx],
        ])
        np.savetxt(
            OUT_DIR / f"window_{name}_t{int(t[idx])}.csv",
            rows,
            delimiter=",",
            header="i,iteration,f_star,R,C",
            comments="",
        )
    return pdf_path, png_path


def load_series(path: Path) -> dict:
    blob = np.load(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = {key: blob[key] for key in blob.files}
    data["meta"] = meta
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the figure from the saved series.",
    )
    args = parser.parse_args()
    npz_path = OUT_DIR / "series_binary_complex_bde_mknapcb1_seed1.npz"
    if args.plot_only:
        if not npz_path.exists():
            raise SystemExit(f"Missing series file: {npz_path}")
        data = load_series(npz_path)
    else:
        data = run_representative()
        npz_path = save_series(data)
        meta = data["meta"]
        print(
            f"Run finished in {meta['elapsed_seconds']:.1f}s | "
            f"fitness {meta['best_fitness']:.0f} | "
            f"gap {meta['gap_percent']:.3f}% | "
            f"explore entries {meta['explore_entries']}"
        )
        print(f"Series: {npz_path}")
    pdf_path, png_path = plot_figure(data)
    print(f"Figure: {pdf_path}")
    print(f"Figure: {png_path}")


if __name__ == "__main__":
    main()
