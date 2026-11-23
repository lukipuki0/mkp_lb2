"""Generate the MKP gap figures (one per problem size) and two binary temporal diagnostics.

Gap figures: mean gap to f* per case and policy, 100 * (f* - mean profit) / f*,
recomputed from the raw runs.csv of the adaptive and vanilla campaigns. Before
plotting, the script checks run counts, seeds and status, agreement with
statistics.csv, the f* column of Table tab:mkp_best and the two-decimal values
of Table tab:mkp_gap in results.tex. Any mismatch aborts without writing.

Temporal figures: re-rendered from the saved temporal_series.csv of two saved
best runs (only winners retain traces): Binary-Simple and Binary-Complex on
mknapcb1[0]. The plotted columns are used as saved; nothing is interpolated
across monitor warm-up gaps. For Binary-Complex, the three conditions of its
stagnation rule are evaluated from the saved distances, thresholds and
no-improvement counter, and must reproduce the saved trigger streak, fire flag
and profile decisions before anything is written.

Usage (from the repository root):
    python figuras/resultados-gwo-de/generate_gap_figures.py
    python figuras/resultados-gwo-de/generate_gap_figures.py --check-only
    python figuras/resultados-gwo-de/generate_gap_figures.py --only mkp-temporal-mknapcb1-0-binary-complex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "gwo_de_dtw" / "results" / "resultados gwo de dtw"
DEFAULT_TEX = REPO / "overleaf" / "gwo_de_dtw_paper" / "sections" / "06_results" / "results.tex"
DEFAULT_OUT = REPO / "overleaf" / "gwo_de_dtw_paper" / "figura"

ADAPTIVE = "gwo_de_22349"
VANILLA = "gwo_de_vanilla_22875"

# Same order, colours (Okabe-Ito) and marker shapes as mkp-gap-by-case.
POLICIES = [
    ("binary_simple", "Binary-Simple", "#0072B2", "o", ADAPTIVE),
    ("binary_complex", "Binary-Complex", "#56B4E9", "s", ADAPTIVE),
    ("continuous_simple", "Continuous-Simple", "#009E73", "^", ADAPTIVE),
    ("continuous_complex", "Continuous-Complex", "#E69F00", "D", ADAPTIVE),
    ("vanilla_exploitation", "Vanilla-Exploitation", "#CC79A7", "v", VANILLA),
    ("vanilla_exploration", "Vanilla-Exploration", "#D55E00", "X", VANILLA),
]
# Problem size -> mknapcb files for m = 5, 10, 30.
SIZES = {100: (1, 4, 7), 250: (2, 5, 8), 500: (3, 6, 9)}
M_OF_FILE = {1: 5, 2: 5, 3: 5, 4: 10, 5: 10, 6: 10, 7: 30, 8: 30, 9: 30}
INDICES = (0, 1, 2)
N_SEEDS = 31

# Saved best runs selected for the temporal diagnostics (case, index, policy, label, seed).
TEMPORAL = [
    (1, 0, "binary_simple", "Binary-Simple", 31),
    (1, 0, "binary_complex", "Binary-Complex", 26),
]

EXPLOIT_COLOR = "#D6E6F2"
EXPLORE_COLOR = "#FBE3CF"
WIDTH_IN = 5.40  # MDPI \textwidth is 394 pt (13.9 cm); included at \textwidth.

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "svg.hashsalt": "mkp",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checksums(results: Path) -> dict[str, str]:
    sums = {}
    for line in (results / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        sums[rel.strip()] = digest
    return sums


def verify_source(results: Path, sums: dict[str, str], path: Path) -> None:
    """Match SHA256SUMS on the raw bytes or, for checkouts with core.autocrlf, on LF line ends."""
    rel = path.relative_to(results).as_posix()
    data = path.read_bytes()
    digests = {hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}
    if sums.get(rel) not in digests:
        raise SystemExit(f"checksum mismatch or missing entry for {rel}")


def parse_tex_table(tex: str, label: str) -> dict[str, list[str]]:
    """Return {case: [cells...]} for the rows of the table with this label."""
    start = tex.index(f"\\label{{{label}}}")
    body = tex[start:tex.index("\\bottomrule", start)]
    rows = {}
    for line in body.splitlines():
        m = re.match(r"\s*(mknapcb\d\[\d\])\s*&(.*)\\\\", line)
        if m:
            cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip() for c in m.group(2).split("&")]
            rows[m.group(1)] = cells
    return rows


def compute_gaps(results: Path, tex_path: Path) -> pd.DataFrame:
    sums = load_checksums(results)
    tex = tex_path.read_text(encoding="utf-8")
    gap_table = parse_tex_table(tex, "tab:mkp_gap")
    best_table = parse_tex_table(tex, "tab:mkp_best")
    records = []
    for g in range(1, 10):
        for i in INDICES:
            case = f"mknapcb{g}[{i}]"
            fstar_tex = float(best_table[case][0].replace(",", ""))
            for col, (pol, label, _, _, campaign) in enumerate(POLICIES):
                folder = results / campaign / f"mknapcb{g}" / f"inst_{i:02d}"
                runs_path, stats_path = folder / "runs.csv", folder / "statistics.csv"
                verify_source(results, sums, runs_path)
                verify_source(results, sums, stats_path)
                runs = pd.read_csv(runs_path)
                runs = runs[runs.policy == pol]
                if len(runs) != N_SEEDS or sorted(runs.seed) != list(range(1, N_SEEDS + 1)):
                    raise SystemExit(f"{case} {pol}: expected seeds 1..{N_SEEDS}, got {len(runs)} runs")
                if not (runs.status == "ok").all():
                    raise SystemExit(f"{case} {pol}: run with status other than ok")
                fstar = runs.known_optimum.unique()
                if len(fstar) != 1 or fstar[0] != fstar_tex:
                    raise SystemExit(f"{case} {pol}: f* {fstar} differs from Table tab:mkp_best {fstar_tex}")
                fstar = float(fstar[0])
                mean_profit = float(runs.best_fitness.mean())
                gap = 100.0 * (fstar - mean_profit) / fstar
                stats = pd.read_csv(stats_path).set_index("policy").loc[pol]
                if abs(stats.mean_fitness - mean_profit) > 1e-6 or abs(stats.mean_gap_pct - gap) > 1e-9:
                    raise SystemExit(f"{case} {pol}: raw runs disagree with statistics.csv")
                printed = gap_table[case][col]
                if f"{gap:.2f}" != printed:
                    raise SystemExit(f"{case} {pol}: gap {gap:.6f} rounds to {gap:.2f}, table prints {printed}")
                records.append(dict(file=g, inst=i, case=case, m=M_OF_FILE[g], policy=pol,
                                    label=label, fstar=fstar, mean_profit=mean_profit, gap=gap,
                                    printed=printed))
    df = pd.DataFrame(records)
    if len(df) != 162:
        raise SystemExit(f"expected 162 case-policy means, got {len(df)}")
    return df


def save(fig: plt.Figure, out: Path, stem: str) -> list[Path]:
    paths = [out / f"{stem}.pdf", out / f"{stem}.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None, "ModDate": None, "Creator": None,
                                    "Producer": None})
    fig.savefig(paths[1], dpi=300, metadata={"Software": None})
    plt.close(fig)
    return paths


def plot_gap(df: pd.DataFrame, n: int, out: Path) -> list[Path]:
    files = SIZES[n]
    sub = df[df.file.isin(files)]
    if len(sub) != 54:
        raise SystemExit(f"n={n}: expected 54 points, got {len(sub)}")
    group_gap = 0.7
    xpos = {}
    for k, g in enumerate(files):
        for i in INDICES:
            xpos[(g, i)] = k * (len(INDICES) + group_gap) + i
    offsets = np.linspace(-0.3, 0.3, len(POLICIES))

    # Common vertical span (4 percentage points) so the three figures share a scale.
    span = 4.0
    lo = np.floor((sub.gap.min() - 0.2) * 2) / 2
    hi = lo + span
    if sub.gap.max() > hi - 0.1:
        raise SystemExit(f"n={n}: gaps exceed the common {span} pp span")

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    for (g, i), x in xpos.items():
        ax.axvspan(x - 0.45, x + 0.45, color="#F2F2F2", zorder=0, lw=0)
    for k, (pol, label, color, marker, _) in enumerate(POLICIES):
        rows = sub[sub.policy == pol]
        xs = [xpos[(g, i)] + offsets[k] for g, i in zip(rows.file, rows.inst)]
        ax.scatter(xs, rows.gap, s=24, marker=marker, color=color, edgecolor="black",
                   linewidth=0.5, zorder=3, label=label)
    ax.set_xticks([xpos[key] for key in xpos])
    ax.set_xticklabels([f"[{i}]" for (_, i) in xpos])
    for k, g in enumerate(files):
        centre = xpos[(g, 1)]
        ax.annotate(f"mknapcb{g}  ($m = {M_OF_FILE[g]}$)", xy=(centre, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -20), textcoords="offset points", ha="center", va="top", fontsize=8.5)
    ax.set_xlim(-0.6, xpos[(files[-1], 2)] + 0.6)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 1e-9, 0.5))
    ax.yaxis.grid(True, color="#DDDDDD", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Mean gap to $f^*$ (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0.3)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.17)
    return save(fig, out, f"mkp-gap-n{n}")


def shade_profiles(ax: plt.Axes, t: np.ndarray, mode: np.ndarray) -> None:
    starts = np.r_[0, np.flatnonzero(np.diff(mode)) + 1]
    ends = np.r_[starts[1:], len(t)]
    for a, b in zip(starts, ends):
        color = EXPLORE_COLOR if mode[a] == 1 else EXPLOIT_COLOR
        ax.axvspan(t[a] - 0.5, t[b - 1] + 0.5, color=color, lw=0, zorder=0)


def check_temporal(results: Path, sums: dict[str, str], g: int, i: int, pol: str, seed: int):
    folder = results / ADAPTIVE / f"mknapcb{g}" / f"inst_{i:02d}"
    run_dir = folder / pol / "best_run"
    for p in (run_dir / "temporal_series.csv", run_dir / "temporal_metadata.json",
              run_dir / "result.json", folder / "runs.csv", folder / "statistics.csv"):
        verify_source(results, sums, p)
    series = pd.read_csv(run_dir / "temporal_series.csv")
    meta = json.loads((run_dir / "temporal_metadata.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(folder / "runs.csv")
    row = runs[(runs.policy == pol) & (runs.seed == seed)].iloc[0]
    stats = pd.read_csv(folder / "statistics.csv").set_index("policy").loc[pol]
    mode = series.mode_used.to_numpy().astype(int)
    switches = int(np.count_nonzero(np.diff(mode)))
    checks = {
        "metadata seed": meta["seed"] == seed and meta["policy"] == pol,
        "statistics best_seed": int(stats.best_seed) == seed,
        "1000 generations": len(series) == 1000 and series.t.iloc[-1] == 1000,
        "final best equals runs.csv": series.f_star.iloc[-1] == row.best_fitness == meta["best_fitness"],
        "switches equal runs.csv changes": switches == int(row.changes),
        "monitor uses DDTW": meta["monitor_config"]["use_ddtw"] is True,
        "decision applied at t+1": bool((series.s.shift(1).iloc[1:].to_numpy()
                                         == series.mode_used.iloc[1:].to_numpy()).all()),
    }
    if pol.startswith("continuous"):
        checks["mode is intensity > 0.5"] = bool(((series.intensity > 0.5).astype(int) == series.mode_used).all())
    if pol == "binary_complex":
        checks.update(check_complex_rule(series, meta["monitor_config"]))
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"mknapcb{g}[{i}] {pol} seed {seed}: failed checks {failed}")
    return series, row, switches


def check_complex_rule(series: pd.DataFrame, cfg: dict) -> dict[str, bool]:
    """Evaluate the Binary-Complex conditions from saved columns and check them against the run.

    Adds boolean columns C_noimp, C_plateau, C_progress and C_stag (False where the monitor
    is not ready). The comparisons use the saved thresholds; nothing else is recomputed.
    """
    ready = series.ready.to_numpy() == 1
    series["C_noimp"] = ready & (series.no_improve_len >= cfg["plateau_max"]).to_numpy()
    series["C_plateau"] = ready & (series.D_C <= series.theta_C).to_numpy()
    series["C_progress"] = ready & ((series.D_R >= series.theta_R)
                                     | (series.delta >= series.theta_delta)).to_numpy()
    series["C_stag"] = series.C_noimp & series.C_plateau & series.C_progress
    # Streak: +1 on each ready C_stag, 0 on a ready failure; a fresh monitor starts at 0.
    streak, prev = np.zeros(len(series)), 0
    for k, (r, c) in enumerate(zip(ready, series.C_stag)):
        prev = (prev + 1 if c else 0) if r else 0
        streak[k] = prev
    fire = streak >= cfg["patience"]
    # Decision after a ready observation: exploit -> explore on fire, explore -> exploit
    # while the no-improvement counter is below plateau_max; otherwise keep the mode.
    mode, s = series.mode_used.to_numpy(), series.s.to_numpy()
    expected = np.where(~ready, mode, np.where(mode == 0, fire.astype(float),
                                                (series.no_improve_len >= cfg["plateau_max"]).astype(float)))
    return {
        "thresholds present exactly when ready": bool((series[["D_R", "D_C", "delta", "theta_R", "theta_C",
                                                                "theta_delta"]].notna().all(axis=1)
                                                        == ready).all()),
        "delta equals D_R - D_C": bool(np.allclose((series.D_R - series.D_C)[ready], series.delta[ready],
                                                   rtol=0, atol=1e-8)),
        "streak equals saved trigger_streak": bool((streak[ready] == series.trigger_streak[ready]).all()),
        "fire equals saved fire": bool((fire[ready] == (series.fire[ready] == 1)).all()),
        "rule reproduces saved decisions": bool((expected == s).all()),
    }


def plot_gapped(ax: plt.Axes, t: np.ndarray, v: np.ndarray, color: str, ls: str, label: str) -> None:
    """Plot with NaN gaps left open; a ready sample with no neighbour is drawn as a dot."""
    ax.plot(t, v, color=color, lw=1.0, ls=ls, label=label)
    ok = ~np.isnan(v)
    lone = ok & ~np.r_[False, ok[:-1]] & ~np.r_[ok[1:], False]
    ax.plot(t[lone], v[lone], ls="none", marker="o", ms=2.5, color=color)


def plot_temporal_complex(series: pd.DataFrame, g: int, i: int, pol: str, out: Path) -> list[Path]:
    """Binary-Complex: profit, D_C/D_R with thresholds, Delta with threshold, rule conditions."""
    t = series.t.to_numpy()
    mode = series.mode_used.to_numpy().astype(int)
    ready = series.ready.to_numpy() == 1
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH_IN, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.5, 0.9, 1.25], "hspace": 0.14})
    for ax in axes:
        shade_profiles(ax, t, mode)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0]
    ax.step(t, series.f_star, where="post", color="black", lw=1.1)
    ax.set_ylabel("Best-so-far\nprofit")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    ax = axes[1]
    for col, color, ls, label in (("D_C", "#0072B2", "-", "$D_C$"), ("theta_C", "#0072B2", "--", r"$\theta_C$"),
                                  ("D_R", "#009E73", "-", "$D_R$"), ("theta_R", "#009E73", "--", r"$\theta_R$")):
        plot_gapped(ax, t, series[col].to_numpy(), color, ls, label)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(-1, 3000)
    ax.set_yticks([0, 10, 100, 1000])
    ax.set_yticklabels(["0", "10", "100", "1000"])
    ax.set_ylabel("Distance\n(symlog)")
    ax.legend(loc="upper right", ncol=4, frameon=True, framealpha=0.9, edgecolor="none",
              handlelength=1.8, borderpad=0.3, columnspacing=1.0)

    ax = axes[2]
    plot_gapped(ax, t, series.delta.to_numpy(), "#6A3D9A", "-", r"$\Delta$")
    plot_gapped(ax, t, series.theta_delta.to_numpy(), "#6A3D9A", "--", r"$\theta_\Delta$")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("$\\Delta$\n$(D_R - D_C)$")
    # Placed over generations 325-555, where every saved value of both series is near 78.
    ax.legend(loc="lower center", bbox_to_anchor=(0.44, 0.0), ncol=2, frameon=True, framealpha=0.9,
              edgecolor="none", handlelength=1.8, borderpad=0.3, columnspacing=1.0)

    # One row per condition: dark where it holds, light where the ready monitor rejects it,
    # blank while the monitor warms up (no evaluation).
    ax = axes[3]
    rows = [("Ready", ready, None),
            (r"$C_{\mathrm{noimp}}$", series.C_noimp.to_numpy(), ready),
            (r"$C_{\mathrm{plateau}}$", series.C_plateau.to_numpy(), ready),
            (r"$C_{\mathrm{progress}}$", series.C_progress.to_numpy(), ready),
            (r"$S_t \geq \tau_{\mathrm{pat}}$", series.fire.to_numpy() == 1, ready)]
    for k, (_, holds, evaluated) in enumerate(rows):
        y = len(rows) - 1 - k
        layers = [("#BBBBBB", evaluated & ~holds)] if evaluated is not None else []
        for color, mask in layers + [("#333333", holds)]:
            edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
            spans = [(t[a] - 0.5, t[b - 1] - t[a] + 1.0) for a, b in zip(edges[::2], edges[1::2])]
            ax.broken_barh(spans, (y - 0.3, 0.6), facecolors=color, lw=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _, _ in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, 1000.5)
    ax.set_xlabel("Generation")

    handles = [Patch(color=EXPLOIT_COLOR, label="Exploitation profile"),
               Patch(color=EXPLORE_COLOR, label="Exploration profile"),
               Patch(color="#333333", label="Holds"),
               Patch(color="#BBBBBB", label="Fails (ready)")]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=len(handles),
                   frameon=False, handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.08)
    return save(fig, out, f"mkp-temporal-mknapcb{g}-{i}-{pol.replace('_', '-')}")


def plot_temporal(series: pd.DataFrame, g: int, i: int, pol: str, label: str, seed: int,
                  out: Path) -> list[Path]:
    t = series.t.to_numpy()
    mode = series.mode_used.to_numpy().astype(int)
    continuous = pol.startswith("continuous")
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH_IN, 4.1), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.5, 1.0], "hspace": 0.12})
    for ax in axes:
        shade_profiles(ax, t, mode)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0]
    ax.step(t, series.f_star, where="post", color="black", lw=1.1)
    ax.set_ylabel("Best-so-far\nprofit")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    ax = axes[1]
    # NaN marks monitor warm-up; matplotlib leaves those gaps open.
    ax.plot(t, series.D_C, color="#0072B2", lw=1.1, label="$D_C$ (DDTW to constant)")
    ax.plot(t, series.theta_C, color="#D55E00", lw=1.0, ls="--", label=r"$\theta_C$")
    # A ready sample between two warm-up gaps has no segment to draw; mark it.
    for col, color in (("theta_C", "#D55E00"), ("D_C", "#0072B2")):
        v = series[col].to_numpy()
        ok = ~np.isnan(v)
        lone = ok & ~np.r_[False, ok[:-1]] & ~np.r_[ok[1:], False]
        ax.plot(t[lone], v[lone], ls="none", marker="o", ms=2.5, color=color)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(-1, 3000)
    ax.set_yticks([0, 10, 100, 1000])
    ax.set_yticklabels(["0", "10", "100", "1000"])
    ax.set_ylabel("Distance\n(symlog)")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9, edgecolor="none",
              handlelength=1.8, borderpad=0.3)

    ax = axes[2]
    if continuous:
        ax.plot(t, series.intensity, color="#6A3D9A", lw=1.1)
        ax.axhline(0.5, color="#555555", lw=0.7, ls=":")
        ax.set_ylim(-0.08, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Intensity\n$I_t$")
    else:
        ax.step(t, mode, where="post", color="black", lw=1.1)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Exploit", "Explore"])
        ax.set_ylabel("Profile")
    ax.set_xlim(0.5, 1000.5)
    ax.set_xlabel("Generation")

    if continuous:
        # mode_used is the saved I_t > 0.5 label (checked in check_temporal).
        handles = [Patch(color=EXPLOIT_COLOR, label=r"Exploitation-oriented ($I_t \leq 0.5$)"),
                   Patch(color=EXPLORE_COLOR, label=r"Exploration-oriented ($I_t > 0.5$)")]
    else:
        handles = [Patch(color=EXPLOIT_COLOR, label="Exploitation profile in use"),
                   Patch(color=EXPLORE_COLOR, label="Exploration profile in use")]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                   frameon=False, handlelength=1.4, columnspacing=1.2, borderaxespad=0.2)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.10)
    return save(fig, out, f"mkp-temporal-mknapcb{g}-{i}-{pol.replace('_', '-')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="root of the saved campaigns (contains SHA256SUMS)")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX,
                        help="results.tex used for the f* and two-decimal gap parity checks")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for PDF/PNG")
    parser.add_argument("--check-only", action="store_true", help="run every check, write nothing")
    stems = [f"mkp-gap-n{n}" for n in SIZES] + [f"mkp-temporal-mknapcb{g}-{i}-{pol.replace('_', '-')}"
                                                 for g, i, pol, _, _ in TEMPORAL]
    parser.add_argument("--only", action="append", choices=stems, metavar="STEM",
                        help="after all checks, write only this figure (repeatable); one of: "
                             + ", ".join(stems))
    args = parser.parse_args()
    wanted = set(args.only or stems)

    plt.rcParams.update(STYLE)
    df = compute_gaps(args.results, args.tex)
    for n, files in SIZES.items():
        sub = df[df.file.isin(files)]
        print(f"n={n}: {len(sub)} points, gap {sub.gap.min():.4f}-{sub.gap.max():.4f} %")
    print(f"total points: {len(df)}; all two-decimal gaps match tab:mkp_gap")

    sums = load_checksums(args.results)
    temporal = []
    for g, i, pol, label, seed in TEMPORAL:
        series, row, switches = check_temporal(args.results, sums, g, i, pol, seed)
        temporal.append((series, g, i, pol, label, seed))
        print(f"mknapcb{g}[{i}] {label} seed {seed}: best {row.best_fitness:.0f}, "
              f"gap {row.gap_pct:.2f} %, {switches} profile changes")
    if args.check_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for n in SIZES:
        if f"mkp-gap-n{n}" in wanted:
            written += plot_gap(df, n, args.out)
    for series, g, i, pol, label, seed in temporal:
        if f"mkp-temporal-mknapcb{g}-{i}-{pol.replace('_', '-')}" not in wanted:
            continue
        if pol == "binary_complex":
            written += plot_temporal_complex(series, g, i, pol, args.out)
        else:
            written += plot_temporal(series, g, i, pol, label, seed, args.out)
    for p in written:
        print(f"{p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p}  {sha256(p)[:16]}")


if __name__ == "__main__":
    main()
