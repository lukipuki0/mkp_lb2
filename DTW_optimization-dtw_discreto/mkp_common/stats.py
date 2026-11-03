"""
mkp_common.stats — Statistical helpers for comparing metaheuristic variants.

Provides Wilcoxon signed-rank tests (paired by seed/epoch) and Shapiro-Wilk
normality checks on paired differences for comparing DTW adaptations against
a vanilla baseline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _check_scipy():
    """Return scipy.stats if available, otherwise None."""
    try:
        from scipy import stats
        return stats
    except Exception:  # pragma: no cover
        return None


def _shapiro_on_differences(
    baseline: np.ndarray,
    version: np.ndarray,
) -> Dict:
    """
    Shapiro-Wilk normality test on paired differences d = version - baseline.

    Returns a dict with statistic, p_value, and a boolean ``normal`` that is
    True when normality cannot be rejected at α = 0.05.
    """
    stats = _check_scipy()
    diffs = version - baseline

    # All-zero differences: technically not normal, but degenerate.
    if np.all(diffs == 0):
        return {"statistic": 1.0, "p_value": 1.0, "normal": True,
                "note": "degenerate (all differences zero)"}

    try:
        stat, p = stats.shapiro(diffs)
    except Exception:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "normal": False, "note": "shapiro failed"}

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "normal": p >= 0.05,
    }


def wilcoxon_test(
    baseline_fits: Sequence[float],
    version_fits: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "greater",
) -> Dict:
    """
    Paired Wilcoxon signed-rank test comparing a version against a baseline.

    H1 with alternative='greater' is: version > baseline.

    Parameters
    ----------
    baseline_fits, version_fits : sequence of float
        Paired fitness values (same length, same seed/epoch order).
    alpha : float, optional
        Significance level for the raw test (default 0.05).
    alternative : str, optional
        SciPy alternative hypothesis for ``version - baseline``.

    Returns
    -------
    dict
        {"p_value": float, "significant": bool, "statistic": float,
         "median_diff": float, "effect_size": float}. ``median_diff`` is the
         median of the paired per-epoch differences ``version - baseline``.
    """
    stats = _check_scipy()
    if stats is None:
        raise RuntimeError("scipy is required for statistical tests. Run: pip install scipy")

    baseline = np.asarray(baseline_fits, dtype=float)
    version = np.asarray(version_fits, dtype=float)

    if baseline.shape != version.shape:
        raise ValueError("baseline_fits and version_fits must have the same length")

    median_diff = float(np.median(version - baseline))
    n_pairs = len(baseline)

    try:
        result = stats.wilcoxon(
            version,
            baseline,
            alternative=alternative,
            zero_method="zsplit",
            mode="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError:
        # All differences are zero (or otherwise degenerate).
        statistic = 0.0
        p_value = 1.0

    # Rank-biserial correlation as a simple effect-size estimate.
    effect_size = statistic / n_pairs if n_pairs > 0 else 0.0

    return {
        "p_value": p_value,
        "significant": p_value < alpha,
        "statistic": statistic,
        "median_diff": median_diff,
        "effect_size": effect_size,
    }


def _mean_verdict(mean_diff: float) -> str:
    """Return a fitness-maximization verdict from an aggregate mean difference."""
    if np.isnan(mean_diff):
        return "N/A"
    if mean_diff > 0:
        return "BETTER"
    if mean_diff < 0:
        return "WORSE"
    return "SAME"


def _finite_float(value) -> float:
    """Return a finite float or NaN for missing/non-numeric values."""
    if isinstance(value, (bool, np.bool_, str, bytes)):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _timing_stats(values, expected_length: Optional[int] = None) -> tuple:
    """Return mean and population standard deviation for raw timing values.

    A timing array is valid only when it is a one-dimensional, non-empty array
    of finite, non-negative numeric values.  When the result metadata exposes
    an epoch count, the array must contain exactly one value per epoch.
    """
    if not isinstance(values, (list, tuple, np.ndarray)):
        return np.nan, np.nan

    try:
        if len(values) == 0:
            return np.nan, np.nan
        if expected_length is not None and len(values) != expected_length:
            return np.nan, np.nan
        if any(
            isinstance(value, (bool, np.bool_, str, bytes))
            or not np.isscalar(value)
            for value in values
        ):
            return np.nan, np.nan
        timings = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return np.nan, np.nan

    if (
        timings.ndim != 1
        or timings.size == 0
        or not np.all(np.isfinite(timings))
        or np.any(timings < 0)
    ):
        return np.nan, np.nan

    return float(np.mean(timings)), float(np.std(timings))


def _time_verdict(time_diff: float) -> str:
    """Return a runtime verdict where a negative difference is better."""
    time_diff = _finite_float(time_diff)
    if np.isnan(time_diff):
        return "N/A"
    if time_diff < 0:
        return "FASTER"
    if time_diff > 0:
        return "SLOWER"
    return "SAME"


def _load_mh_json(directory: Optional[str], mh: str) -> Optional[dict]:
    """Load the first JSON file matching ``{mh}_*.json`` in *directory*."""
    if not directory:
        return None
    path = Path(directory)
    if not path.is_dir():
        return None

    candidates = sorted(path.glob(f"{mh}_*.json"))
    if not candidates:
        return None

    try:
        with open(candidates[0], encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def compare_versions(
    baseline_dir: str,
    version_dirs: Dict[str, str],
    mh_names: List[str],
    alpha: float = 0.05,
    alternative: str = "greater",
) -> Dict:
    """
    Compare every version in *version_dirs* against the vanilla baseline.

    Shapiro-Wilk normality is checked on paired differences for each
    (MH, version) pair.  Wilcoxon significance is the raw result
    ``p_value < alpha`` for each paired comparison.

    Parameters
    ----------
    baseline_dir : str
        Path to the vanilla JSON directory.
    version_dirs : dict[str, str]
        Mapping ``version_label -> directory_path``.
    mh_names : list[str]
        Metaheuristics to compare (e.g., ["PSO", "GA", "GWO", "DE"]).
    alpha : float, optional
        Significance level applied to each raw Wilcoxon result.
    alternative : str, optional
        SciPy alternative hypothesis (default "greater", use "two-sided" for
        detecting both improvement and degradation).

    Returns
    -------
    dict
        Structured results with summary, per-MH statistics, per-version
        Wilcoxon outcomes, Shapiro-Wilk results, the aggregate mean
        difference, the paired median difference, and descriptive runtime
        statistics from the raw ``tiempos`` arrays.
    """
    stats_mod = _check_scipy()
    if stats_mod is None:
        raise RuntimeError("scipy is required for statistical tests. Run: pip install scipy")

    n_versions = len(version_dirs)
    per_mh: Dict[str, Dict] = {}
    optimo_conocido: Optional[float] = None

    for mh in mh_names:
        baseline_data = _load_mh_json(baseline_dir, mh)
        if baseline_data is None:
            continue

        if optimo_conocido is None:
            optimo_conocido = baseline_data.get("optimo_conocido")

        baseline_fits = baseline_data.get("fitness", [])
        baseline_arr = np.asarray(baseline_fits, dtype=float)
        baseline_epoch_count = baseline_data.get("epochs")
        if not isinstance(baseline_epoch_count, int) or isinstance(baseline_epoch_count, bool):
            baseline_epoch_count = len(baseline_fits) if isinstance(baseline_fits, list) else None
        baseline_time_mean, baseline_time_std = _timing_stats(
            baseline_data.get("tiempos"), expected_length=baseline_epoch_count
        )
        mh_entry = {
            "baseline_mean": float(np.mean(baseline_fits)) if baseline_fits else np.nan,
            "baseline_std": float(np.std(baseline_fits)) if baseline_fits else np.nan,
            "baseline_time_mean": baseline_time_mean,
            "baseline_time_std": baseline_time_std,
            "baseline_fitness": baseline_fits,
            "versions": {},
        }

        for version_name, version_dir in version_dirs.items():
            version_data = _load_mh_json(version_dir, mh)
            if version_data is None:
                continue

            version_fits = version_data.get("fitness", [])
            if len(version_fits) != len(baseline_fits):
                continue

            version_arr = np.asarray(version_fits, dtype=float)

            # Shapiro-Wilk on paired differences.
            shapiro = _shapiro_on_differences(baseline_arr, version_arr)

            # Wilcoxon on paired per-epoch differences with raw alpha.
            test_result = wilcoxon_test(
                baseline_fits,
                version_fits,
                alpha=alpha,
                alternative=alternative,
            )

            version_mean = float(np.mean(version_fits)) if version_fits else np.nan
            baseline_mean = mh_entry["baseline_mean"]
            mean_diff = version_mean - baseline_mean
            version_epoch_count = version_data.get("epochs")
            if not isinstance(version_epoch_count, int) or isinstance(version_epoch_count, bool):
                version_epoch_count = len(version_fits) if isinstance(version_fits, list) else None
            version_time_mean, version_time_std = _timing_stats(
                version_data.get("tiempos"), expected_length=version_epoch_count
            )
            if np.isfinite(version_time_mean) and np.isfinite(baseline_time_mean):
                time_diff = version_time_mean - baseline_time_mean
            else:
                time_diff = np.nan
            if np.isfinite(time_diff) and baseline_time_mean > 0:
                time_percent_diff = 100.0 * time_diff / baseline_time_mean
            else:
                time_percent_diff = np.nan

            mh_entry["versions"][version_name] = {
                "mean": version_mean,
                "std": float(np.std(version_fits)) if version_fits else np.nan,
                "mean_diff": mean_diff,
                "verdict": _mean_verdict(mean_diff),
                "time_mean": version_time_mean,
                "time_std": version_time_std,
                "time_diff": time_diff,
                "time_percent_diff": time_percent_diff,
                "time_verdict": _time_verdict(time_diff),
                "version_fitness": version_fits,
                **test_result,
                "shapiro": shapiro,
            }

        per_mh[mh] = mh_entry

    return {
        "summary": {
            "alpha": alpha,
            "n_versions": n_versions,
            "test": "Wilcoxon signed-rank (paired)",
            "normality_check": "Shapiro-Wilk on paired differences",
            "significance_rule": "raw Wilcoxon p-value < alpha",
            "alternative": alternative,
            "optimo_conocido": optimo_conocido,
        },
        "mhs": per_mh,
    }


def _can_encode_unicode() -> bool:
    """Return True if stdout can encode common box-drawing characters."""
    import sys

    try:
        enc = sys.stdout.encoding
    except AttributeError:
        return False

    if enc is None:
        return False

    sample = "╔╗╚╝║═╤╧╪│╠╣╟╢"
    try:
        sample.encode(enc)
        return True
    except UnicodeEncodeError:
        return False


def format_table(
    results: Dict,
    title: str = "Statistical Comparison vs Exploration-only",
    ascii_only: Optional[bool] = None,
) -> str:
    """
    Format comparison results as a paper-ready table.

    The visible verdict is based on the aggregate ``mean_diff`` (version mean
    minus baseline mean).  The displayed p-value is the raw paired Wilcoxon
    p-value; paired per-epoch differences are summarized separately by their
    median in :func:`format_math_table`.

    Parameters
    ----------
    results : dict
        Output from :func:`compare_versions`.
    title : str, optional
        Table title.
    ascii_only : bool, optional
        If True, use plain ASCII characters. If None, auto-detect stdout
        encoding capability.

    Returns
    -------
    str
        Multi-line formatted table.
    """
    if ascii_only is None:
        ascii_only = not _can_encode_unicode()

    if ascii_only:
        title = title.replace("—", "--").replace("–", "-")

    summary = results.get("summary", {})
    mhs = results.get("mhs", {})
    alpha = summary.get("alpha", 0.05)

    version_names = []
    for mh_entry in mhs.values():
        version_names.extend(mh_entry.get("versions", {}).keys())
    version_names = list(dict.fromkeys(version_names))

    # Build rows: each MH has a primary row (mean ± std) and a secondary row
    # with p-value and direction for each version.
    rows = []
    for mh, mh_entry in mhs.items():
        baseline_mean = mh_entry.get("baseline_mean", np.nan)
        baseline_std = mh_entry.get("baseline_std", np.nan)

        pm_symbol = "+/-" if ascii_only else "±"
        primary_cells = [f"{baseline_mean:.1f} {pm_symbol} {baseline_std:.1f}"]
        secondary_cells = [""]

        for version_name in version_names:
            v = mh_entry.get("versions", {}).get(version_name)
            if v is None:
                primary_cells.append("-" if ascii_only else "—")
                secondary_cells.append("-" if ascii_only else "—")
                continue

            mean = v.get("mean", np.nan)
            std = v.get("std", np.nan)
            mean_diff = v.get("mean_diff", mean - baseline_mean)
            p_value = v.get("p_value", np.nan)
            verdict = _mean_verdict(mean_diff)

            primary_cells.append(f"{mean:.1f} {pm_symbol} {std:.1f}")

            secondary_cells.append(
                f"mean diff={mean_diff:+.1f}  raw p={p_value:.4f}  ({verdict})"
            )

        rows.append((mh, primary_cells))
        rows.append(("", secondary_cells))

    # Column headers.
    pm_symbol = "+/-" if ascii_only else "±"
    headers = ["MH", f"Exploration-only (mean {pm_symbol} std)"] + version_names

    # Column widths based on content.
    col_widths = [len(h) for h in headers]
    for _, cells in rows:
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    # Choose box-drawing charset.
    if ascii_only:
        HL = "="
        VL = "|"
        TL = TR = BL = BR = "+"
        LC = RC = "+"
        TS = BS = CROSS = "+"
        CS = "|"
        LEFT_T = RIGHT_T = "+"
        HBAR = "-"
    else:
        HL = "═"
        VL = "║"
        TL = "╔"
        TR = "╗"
        BL = "╚"
        BR = "╝"
        LC = "╠"
        RC = "╣"
        TS = "╤"
        BS = "╧"
        CS = "│"
        CROSS = "╪"
        LEFT_T = "╟"
        RIGHT_T = "╢"
        HBAR = "─"

    def sep(left: str, right: str, cross: str = TS, line: str = HL) -> str:
        parts = [line * (col_widths[0] + 2)]
        for w in col_widths[1:]:
            parts.extend([cross, line * (w + 2)])
        return left + "".join(parts) + right

    def row_line(label: str, cells: List[str]) -> str:
        parts = [f" {label:<{col_widths[0]}} "]
        for i, cell in enumerate(cells):
            width = col_widths[i + 1]
            parts.append(f" {cell:>{width}} ")
        line = CS.join(parts)
        return VL + line + VL

    title_text = f" {title} "
    test_text = f" Raw Wilcoxon signed-rank p-values | alpha = {alpha:.4f} "

    title_width = sum(col_widths) + 3 * len(col_widths) + 1
    title_line = title_text.center(title_width)
    test_line = test_text.center(title_width)

    lines = [
        TL + HL * (title_width - 2) + TR,
        VL + title_line + VL,
        VL + test_line + VL,
        sep(LC, RC),
        row_line(headers[0], headers[1:]),
        sep(LC, RC, cross=CROSS),
    ]

    for i, (label, cells) in enumerate(rows):
        lines.append(row_line(label, cells))
        if i < len(rows) - 1 and label == "":
            # Separator between MH blocks.
            lines.append(sep(LEFT_T, RIGHT_T, cross=CROSS, line=HBAR))

    lines.append(sep(BL, BR, cross=BS))

    # Legend.
    lines.append("")
    hbar = "-" if ascii_only else "─"
    lines.append(hbar * (title_width - 2))
    lines.append("INTERPRETATION GUIDE:")
    lines.append("")
    lines.append("Mean difference = version mean - baseline mean.")
    lines.append("Positive means BETTER because fitness is maximized; negative means WORSE.")
    lines.append("Wilcoxon uses paired per-epoch differences, not aggregate means.")
    lines.append("Wilcoxon significance uses the raw rule p-value < alpha.")
    lines.append("Its paired median difference (median of per-epoch differences) is shown separately.")

    return "\n".join(lines)


def format_math_table(
    results: Dict,
    title: str = "Numerical Results",
) -> str:
    """
    Format comparison results as a clean mathematical table with raw numbers.

    Shows mean, standard deviation, aggregate mean difference, paired median
    difference, raw Wilcoxon p-value, Shapiro-Wilk status, and a verdict per
    MH and version.  Shapiro-Wilk is applied to paired per-epoch differences.
    """
    summary = results.get("summary", {})
    mhs = results.get("mhs", {})
    alpha = summary.get("alpha", 0.05)
    alternative = summary.get("alternative", "greater")
    alt_text = "(two-sided)" if alternative == "two-sided" else "(one-tailed >)"

    version_names = []
    for mh_entry in mhs.values():
        version_names.extend(mh_entry.get("versions", {}).keys())
    version_names = list(dict.fromkeys(version_names))

    lines = []
    lines.append(f"{'=' * 90}")
    lines.append(f"  {title}")
    lines.append(
        f"  Raw Wilcoxon signed-rank {alt_text} | alpha = {alpha:.4f} | "
        "Shapiro-Wilk on paired differences"
    )
    lines.append(f"{'=' * 90}")
    lines.append("")

    for mh, mh_entry in mhs.items():
        b_mean = mh_entry["baseline_mean"]
        b_std = mh_entry["baseline_std"]

        lines.append(f"  [{mh}]")
        lines.append(f"    Baseline (Exploration-only): {b_mean:.1f} +/- {b_std:.1f}")
        lines.append(
            f"    {'Version':<20s} {'Mean':>10s} {'Std':>10s} "
            f"{'Mean diff':>11s} {'Paired median diff':>19s} {'p-value':>10s} "
            f"{'Shapiro':>10s} {'Verdict'}"
        )
        lines.append(
            f"    {'-'*20} {'-'*10} {'-'*10} {'-'*11} {'-'*19} "
            f"{'-'*10} {'-'*10} {'-'*12}"
        )

        for vn in version_names:
            v = mh_entry.get("versions", {}).get(vn)
            if v is None:
                continue
            mean = v["mean"]
            std = v["std"]
            mean_diff = v.get("mean_diff", mean - b_mean)
            paired_median_diff = v["median_diff"]
            p = v["p_value"]

            # Shapiro-Wilk normality verdict.
            shapiro = v.get("shapiro", {})
            normal = "OK" if shapiro.get("normal", False) else "NOT normal"
            if shapiro.get("note"):
                normal = shapiro["note"][:10]
            verdict = _mean_verdict(mean_diff)

            lines.append(
                f"    {vn:<20s} {mean:10.1f} {std:10.1f} "
                f"{mean_diff:+11.1f} {paired_median_diff:+19.1f} {p:10.4f} "
                f"{normal:>10s}  {verdict}"
            )

        # Best version for this MH by aggregate mean difference.
        best_v = None
        best_mean_diff = float("-inf")
        for vn in version_names:
            v = mh_entry.get("versions", {}).get(vn)
            if v:
                mean_diff = v.get("mean_diff", v["mean"] - b_mean)
                if mean_diff > best_mean_diff:
                    best_mean_diff = mean_diff
                    best_v = vn
        if best_v:
            lines.append(f"    -> Best by mean: {best_v}  (mean diff = {best_mean_diff:+.1f})")
        lines.append("    Shapiro: normality test on paired per-epoch differences (OK = not rejected at alpha=0.05).")
        lines.append("")

    return "\n".join(lines)


def format_time_table(
    results: Dict,
    title: str = "Descriptive Runtime Comparison vs Exploration-only",
    ascii_only: Optional[bool] = None,
) -> str:
    """Format descriptive execution-time statistics separately for every MH.

    The table uses only runtime statistics derived from raw JSON ``tiempos``
    arrays.  The difference is ``version mean - baseline mean``; therefore a
    negative value means that the version is faster.  No inferential or
    significance claim is made for runtime.
    """
    if ascii_only is None:
        ascii_only = not _can_encode_unicode()

    if ascii_only:
        title = title.replace("—", "--").replace("–", "-")

    mhs = results.get("mhs", {})
    if not isinstance(mhs, dict):
        mhs = {}

    version_names = []
    for mh_entry in mhs.values():
        if not isinstance(mh_entry, dict):
            continue
        versions = mh_entry.get("versions", {})
        if not isinstance(versions, dict):
            continue
        for version_name in versions:
            if version_name not in version_names:
                version_names.append(version_name)

    pm_symbol = "+/-" if ascii_only else "±"

    def format_value(value, signed: bool = False) -> str:
        number = _finite_float(value)
        if np.isnan(number):
            return "N/A"
        return f"{number:+.3f}" if signed else f"{number:.3f}"

    def format_mean_std(mean, std) -> str:
        return f"{format_value(mean)} {pm_symbol} {format_value(std)}"

    headers = [
        "Version",
        f"Mean {pm_symbol} Std (s)",
        "Difference vs baseline (s)",
        "Relative change (%)",
        "Interpretation",
    ]
    display_names = [str(name) for name in version_names]
    column_widths = [len(header) for header in headers]
    for display_name in display_names:
        column_widths[0] = max(column_widths[0], len(display_name))

    def table_row(cells) -> str:
        alignments = ["<", ">", ">", ">", "<"]
        formatted = [
            f"{cell:{alignments[index]}{column_widths[index]}}"
            for index, cell in enumerate(cells)
        ]
        return "    " + " | ".join(formatted)

    separator = "    " + "-+-".join("-" * width for width in column_widths)
    table_width = max(96, sum(column_widths) + 3 * (len(headers) - 1) + 4)

    lines = [
        "=" * table_width,
        f"  {title}",
        "  Descriptive runtime data only; no significance claim is made.",
        "  Difference = version mean - Exploration-only baseline mean; negative means FASTER.",
        "  Relative change = 100 * difference / Exploration-only baseline mean.",
        "=" * table_width,
        "",
    ]

    if not mhs:
        lines.append("  No runtime comparison data available.")
        return "\n".join(lines)

    for mh, mh_entry in mhs.items():
        if not isinstance(mh_entry, dict):
            mh_entry = {}

        baseline_mean = mh_entry.get("baseline_time_mean", np.nan)
        baseline_std = mh_entry.get("baseline_time_std", np.nan)
        lines.append(f"  [{mh}]")
        lines.append(
            "    Baseline (Exploration-only): "
            f"{format_mean_std(baseline_mean, baseline_std)} s"
        )
        lines.append(table_row(headers))
        lines.append(separator)

        versions = mh_entry.get("versions", {})
        if not isinstance(versions, dict):
            versions = {}

        for version_name, display_name in zip(version_names, display_names):
            version = versions.get(version_name)
            if not isinstance(version, dict):
                mean_std = f"N/A {pm_symbol} N/A"
                time_diff = np.nan
                time_percent_diff = np.nan
            else:
                mean_std = format_mean_std(
                    version.get("time_mean", np.nan),
                    version.get("time_std", np.nan),
                )
                time_diff = version.get("time_diff", np.nan)
                time_percent_diff = version.get("time_percent_diff", np.nan)

            percent_text = format_value(time_percent_diff, signed=True)
            if percent_text != "N/A":
                percent_text += "%"

            lines.append(
                table_row(
                    [
                        display_name,
                        mean_std,
                        format_value(time_diff, signed=True),
                        percent_text,
                        _time_verdict(time_diff),
                    ]
                )
            )

        if not display_names:
            lines.append("    No versions with runtime data available.")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_raw_table(
    results: Dict,
    title: str = "Raw Wilcoxon Results",
    ascii_only: Optional[bool] = None,
) -> str:
    """
    Format the raw paired Wilcoxon results.

    This formatter is retained as a separate CLI/file output for compatibility;
    it uses the same unadjusted p-values, mean-based verdicts, and explanatory
    notes as :func:`format_table`.
    """
    return format_table(results, title=title, ascii_only=ascii_only)
