"""
run_all_hpc.py — HPC-optimized parallel experiment runner.
===========================================================
Replaces ``run_all.py`` for SLURM/HPC execution. Parallelizes ALL
independent work units (strategy × MH × epoch) across available CPU cores
using ProcessPoolExecutor.

Usage (from project root):
    python run_all_hpc.py
    python run_all_hpc.py --instancia instances/mknapcb1.txt --indice 0
    python run_all_hpc.py --cpus 40

In your SLURM .sh script, just swap the python line:
    python run_all_hpc.py
"""

from __future__ import annotations

import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# Pre-parse --instancia / --indice BEFORE any imports.
# The config modules read os.environ at import time, so this must come first.
# ═══════════════════════════════════════════════════════════════════════════
_args = sys.argv[1:]
for i, arg in enumerate(_args):
    if arg == "--instancia" and i + 1 < len(_args):
        os.environ["MKP_INSTANCIA"] = _args[i + 1]
    elif arg == "--indice" and i + 1 < len(_args):
        os.environ["MKP_INDICE"] = _args[i + 1]

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Shared config (single source of truth — mkp_common.config)
# ---------------------------------------------------------------------------
from mkp_common import (
    BinaryPSO,
    GeneticAlgorithm,
    BinaryGWO,
    BinaryDE,
    cargar_instancia,
)
from mkp_common.config import (
    RUTA_INSTANCIA,
    INDICE_INSTANCIA,
    NUM_PARTICULAS,
    NUM_ITERACIONES,
    EPOCHS,
    VERBOSE,
    DTW_FIRE_BINARIO,
    DTW_FIRE_D2,
    DTW_SIGMOID_DELTA,
    DTW_B3_D2,
    B1_K,
    B1_CENTER,
    B3_SCALE,
)

# ---------------------------------------------------------------------------
# Strategy-specific imports
# ---------------------------------------------------------------------------
from mkp_common.runner import run_experiment as _generic_run
from binary_simple.config import fire_d2 as _fire_d2_fn
from continuous_complex.runner import run_experiment as _continuous_complex_run
from continuous_simple.runner import run_experiment as _continuous_simple_run
from vanilla.runner import run_experiment as _vanilla_run
from vanilla_explotacion.runner import run_experiment as _vanilla_explotacion_run
from vanilla_exploracion.runner import run_experiment as _vanilla_exploracion_run
from vanilla.resultados import generate_plots as _plots_vanilla
from vanilla_explotacion.resultados import generate_plots as _plots_vanilla_explotacion
from vanilla_exploracion.resultados import generate_plots as _plots_vanilla_exploracion
from binary_complex.resultados import generate_plots as _plots_binary_complex
from binary_simple.resultados import generate_plots as _plots_binary_simple
from continuous_complex.resultados import generate_plots as _plots_continuous_complex
from continuous_simple.resultados import generate_plots as _plots_continuous_simple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MHClass = type
TaskResult = Dict[str, Any]

# ---------------------------------------------------------------------------
# Strategy registry
#   Each entry defines:
#     - folder:     results/{folder}/todos/... (matches original estructura)
#     - label:      human name for logging
#     - runner:     callable to run ONE experiment (1 epoch)
#     - extra_info: metadata saved into JSON
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Named runner functions (must be module-level — picklable by multiprocessing)
# ---------------------------------------------------------------------------


def _runner_vanilla(mh_c, inst, seed):
    return _vanilla_run(
        mh_class=mh_c,
        inst=inst,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=seed,
    )


def _runner_vanilla_explotacion(mh_c, inst, seed):
    """Vanilla-Explotación: MHs en modo exploit (igual que vanilla estándar)."""
    return _vanilla_explotacion_run(
        mh_class=mh_c,
        inst=inst,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=seed,
    )


def _runner_vanilla_exploracion(mh_c, inst, seed):
    """Vanilla-Exploración: MHs forzadas a modo explore."""
    return _vanilla_exploracion_run(
        mh_class=mh_c,
        inst=inst,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=seed,
    )


def _runner_binary_complex(mh_c, inst, seed):
    """Binary-Complex: decisión multi-criterio (3 condiciones + patience)."""
    return _generic_run(
        mh_class=mh_c,
        inst=inst,
        monitor_cfg=DTW_FIRE_BINARIO,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=seed,
        verbose=VERBOSE,
        fire_fn=None,
    )


def _runner_binary_simple(mh_c, inst, seed):
    """Binary-Simple: decisión por umbral D2."""
    return _generic_run(
        mh_class=mh_c,
        inst=inst,
        monitor_cfg=DTW_FIRE_D2,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=seed,
        verbose=VERBOSE,
        fire_fn=_fire_d2_fn,
    )


def _runner_continuous_complex(mh_c, inst, seed):
    """Continuous-Complex: intensidad sigmoide desde delta."""
    return _continuous_complex_run(
        mh_class=mh_c,
        inst=inst,
        monitor_cfg=DTW_SIGMOID_DELTA,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=seed,
        verbose=VERBOSE,
        k=B1_K,
        center=B1_CENTER,
    )


def _runner_continuous_simple(mh_c, inst, seed):
    """Continuous-Simple: intensidad directa desde D2."""
    return _continuous_simple_run(
        mh_class=mh_c,
        inst=inst,
        monitor_cfg=DTW_B3_D2,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=seed,
        verbose=VERBOSE,
        scale=B3_SCALE,
    )


STRATEGIES: Dict[str, dict] = {
    "vanilla_explotacion": {
        "folder": "vanilla_explotacion",
        "label": "Vanilla-Explotación",
        "extra_info": {"estrategia": "vanilla_explotacion"},
        "runner": _runner_vanilla_explotacion,
    },
    "vanilla_exploracion": {
        "folder": "vanilla_exploracion",
        "label": "Vanilla-Exploración",
        "extra_info": {"estrategia": "vanilla_exploracion"},
        "runner": _runner_vanilla_exploracion,
    },
    """ "vanilla": {
        "folder": "vanilla",
        "label": "Vanilla (sin DTW)",
        "extra_info": {"estrategia": "vanilla"},
        "runner": _runner_vanilla,
    }, """
    "binary_simple": {
        "folder": "binary_simple",
        "label": "Binary-Simple",
        "extra_info": {
            "estrategia": "binary_simple",
            "decision_rule": "D2 <= theta_c",
            "dtw_window": DTW_FIRE_D2.window,
        },
        "runner": _runner_binary_simple,
    },
    "binary_complex": {
        "folder": "binary_complex",
        "label": "Binary-Complex",
        "extra_info": {
            "estrategia": "binary_complex",
            "dtw_window": DTW_FIRE_BINARIO.window,
            "dtw_patience": DTW_FIRE_BINARIO.patience,
        },
        "runner": _runner_binary_complex,
    },
    "continuous_simple": {
        "folder": "continuous_simple",
        "label": "Continuous-Simple",
        "extra_info": {
            "estrategia": "continuous_simple",
            "scale": B3_SCALE,
            "dtw_window": DTW_B3_D2.window,
        },
        "runner": _runner_continuous_simple,
    },
    "continuous_complex": {
        "folder": "continuous_complex",
        "label": "Continuous-Complex",
        "extra_info": {
            "estrategia": "continuous_complex",
            "k": B1_K,
            "center": B1_CENTER,
            "dtw_window": DTW_SIGMOID_DELTA.window,
        },
        "runner": _runner_continuous_complex,
    },
}

# MHs to evaluate — same 4 in every strategy
MHS: Dict[str, MHClass] = {
    "PSO": BinaryPSO,
    "GA": GeneticAlgorithm,
    "GWO": BinaryGWO,
    "DE": BinaryDE,
}


# =============================================================================
# Worker function (module-level → pickleable by multiprocessing)
# =============================================================================


def _run_one_epoch(
    strategy_key: str,
    mh_key: str,
    epoch: int,
    inst: dict,
) -> Tuple[str, str, int, TaskResult]:
    """
    Execute ONE epoch for one strategy × MH combination.

    Returns (strategy_key, mh_key, epoch, result_dict).
    The result_dict matches what ``run_experiment`` returns.
    """
    strategy = STRATEGIES[strategy_key]
    mh_class = MHS[mh_key]
    seed = epoch + 1  # same seed convention as run_epochs()

    t0 = time.perf_counter()
    runner = strategy["runner"]
    res = runner(mh_class, inst, seed)
    t1 = time.perf_counter()

    res["tiempo"] = t1 - t0
    res["epoch"] = epoch + 1
    res["semilla"] = seed

    return (strategy_key, mh_key, epoch, res)


# =============================================================================
# Helpers
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HPC-optimized parallel experiment runner for DTW strategies"
    )
    p.add_argument(
        "--instancia",
        default=RUTA_INSTANCIA,
        help=f"Path to MKP instance (default: {RUTA_INSTANCIA})",
    )
    p.add_argument(
        "--indice",
        type=int,
        default=INDICE_INSTANCIA,
        help=f"Instance index (default: {INDICE_INSTANCIA})",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Max workers for ProcessPoolExecutor (default: os.cpu_count())",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs per MH (default: {EPOCHS})",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=list(STRATEGIES),
        help="Strategies to skip (e.g., --skip vanilla fire_d2)",
    )
    return p.parse_args()


class _NoOpPbar:
    """Drop-in replacement for tqdm when it's not installed."""

    def __init__(self, total=0, desc="", unit=""):
        self.total = total
        self.n = 0

    def update(self, n=1):
        self.n += n

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _try_import_tqdm():
    """Return tqdm if installed, else a no-op wrapper."""
    try:
        from tqdm import tqdm as _tqdm

        return _tqdm
    except ImportError:
        return _NoOpPbar


def _save_strategy_results(
    strategy_key: str,
    mh_results: Dict[str, List[TaskResult]],
    inst: dict,
    instancia: str,
    indice: int,
    run_id: str,
) -> str:
    """
    Save JSON files for one strategy (all MHs) in the expected directory
    structure and return the output directory path.
    """
    from mkp_common.results import save_results

    strategy = STRATEGIES[strategy_key]
    inst_name = Path(instancia).stem
    save_dir = (
        Path("results")
        / strategy["folder"]
        / "todos"
        / f"{inst_name}_{indice}"
        / f"comparacion_mhs_{run_id}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    for mh_key, resultados in mh_results.items():
        # Sort by epoch to ensure correct order
        resultados.sort(key=lambda r: r["epoch"])
        extra = dict(strategy["extra_info"])
        extra.update(
            {
                "instancia": instancia,
                "idx": indice,
                "poblacion": NUM_PARTICULAS,
                "iteraciones": NUM_ITERACIONES,
            }
        )
        save_results(
            resultados,
            path=str(save_dir / f"{mh_key}_{inst_name}_{indice}.json"),
            mh_name=mh_key,
            optimo_conocido=inst.get("optimo") if inst.get("optimo", 0) > 0 else None,
            extra_info=extra,
        )

    return str(save_dir)


def _summarise_strategy(
    strategy_key: str, mh_results: Dict[str, List[TaskResult]]
) -> str:
    """Return a one-line summary per MH for this strategy."""
    lines = []
    for mh_key, resultados in mh_results.items():
        fits = [r["mejor_fitness"] for r in resultados]
        times = [r["tiempo"] for r in resultados]
        fire_count = sum(
            r.get("fire_count", 0) for r in resultados
        )
        lines.append(
            f"    {mh_key:<6s}  "
            f"best={max(fits):.1f}  avg={np.mean(fits):.1f}  "
            f"std={np.std(fits):.1f}  "
            f"fires={fire_count}  "
            f"T={sum(times):.1f}s"
        )
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = _parse_args()

    # Instance
    instancia = args.instancia
    indice = args.indice
    inst = cargar_instancia(instancia, idx=indice)

    # Active strategies
    active_strategies = {
        k: v for k, v in STRATEGIES.items() if k not in args.skip
    }
    if not active_strategies:
        print("ERROR: all strategies were skipped (--skip). Nothing to do.")
        return 1

    n_epochs = args.epochs
    max_workers = args.cpus or os.cpu_count() or 1

    # Total task count
    total_tasks = len(active_strategies) * len(MHS) * n_epochs

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    inst_name = Path(instancia).stem
    instance_label = f"{inst_name}[{indice}]"

    # ── Header ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  HPC PARALLEL RUNNER — DTW Strategy Comparison")
    print("=" * 70)
    print(f"  Instance:      {instance_label}")
    print(f"  n={inst['n']}, m={inst['m']}, optimum={inst.get('optimo', '?')}")
    print(f"  Population:    {NUM_PARTICULAS}")
    print(f"  Iterations:    {NUM_ITERACIONES}")
    print(f"  Epochs:        {n_epochs}")
    print(f"  Strategies:    {', '.join(active_strategies)}")
    print(f"  MHs:           {', '.join(MHS)}")
    print(f"  Total tasks:   {total_tasks}")
    print(f"  Workers:       {max_workers}")
    print(f"  Run ID:        {run_id}")
    print()

    # ── Generate task list ───────────────────────────────────────────────
    tasks: List[Tuple[str, str, int]] = []
    for sk in active_strategies:
        for mk in MHS:
            for ep in range(n_epochs):
                tasks.append((sk, mk, ep))

    # ── Execute in parallel ──────────────────────────────────────────────
    tqdm = _try_import_tqdm()
    results_bucket: Dict[str, Dict[str, List[TaskResult]]] = {
        sk: {mk: [] for mk in MHS} for sk in active_strategies
    }

    t_start = time.perf_counter()
    completed = 0
    errors = 0

    print(f"  Dispatching {total_tasks} tasks across {max_workers} workers...")
    print()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one_epoch, sk, mk, ep, inst): (sk, mk, ep)
            for sk, mk, ep in tasks
        }

        pbar = tqdm(total=total_tasks, desc="  Progress", unit="task")

        for future in as_completed(futures):
            try:
                sk, mk, ep, res = future.result()
                results_bucket[sk][mk].append(res)
                completed += 1
            except Exception as exc:
                errors += 1
                sk, mk, ep = futures[future]
                print(f"\n  [ERROR] {sk}/{mk} epoch={ep+1}: {exc}", flush=True)

            pbar.update(1)
        pbar.close()

    elapsed = time.perf_counter() - t_start
    elapsed_str = str(timedelta(seconds=round(elapsed)))

    # ── Save results ─────────────────────────────────────────────────────
    print(f"\n  Completed: {completed}/{total_tasks}  Errors: {errors}")
    print(f"  Wall time: {elapsed_str}")
    print()

    # Plot functions — one per strategy (same signature everywhere)
    _PLOTTERS = {
        "vanilla_explotacion": _plots_vanilla_explotacion,
        "vanilla_exploracion": _plots_vanilla_exploracion,
        "vanilla": _plots_vanilla,
        "binary_simple": _plots_binary_simple,
        "binary_complex": _plots_binary_complex,
        "continuous_simple": _plots_continuous_simple,
        "continuous_complex": _plots_continuous_complex,
    }

    saved_dirs: Dict[str, str] = {}
    for sk in active_strategies:
        label = STRATEGIES[sk]["label"]
        mh_res = results_bucket[sk]

        # Skip strategies where ALL epochs failed
        if all(len(v) == 0 for v in mh_res.values()):
            print(f"  [{label}] SKIP — all epochs failed")
            continue

        print(f"  [{label}]")
        print(_summarise_strategy(sk, mh_res))
        print()

        saved = _save_strategy_results(sk, mh_res, inst, instancia, indice, run_id)
        saved_dirs[sk] = saved

        # Generate plots (reuses existing per-strategy plot functions)
        plotter = _PLOTTERS.get(sk)
        if plotter:
            try:
                plotter(mh_res, inst, saved)
                print(f"    Plots saved to {saved}/")
            except Exception as exc:
                print(f"    [WARN] Plot generation failed for {sk}: {exc}")

    # ── Final summary ────────────────────────────────────────────────────
    print("=" * 70)
    print("  SAVED RESULTS")
    print("=" * 70)
    for sk, d in saved_dirs.items():
        print(f"  {STRATEGIES[sk]['label']:<25s} → {d}")
    print(f"\n  Total wall time: {elapsed_str}")
    print(f"  Tasks: {completed} ok, {errors} errors")
    print("=" * 70)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
