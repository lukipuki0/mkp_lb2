"""Ejecuta las siete variantes WOA--ABC sobre HRES2-H2/WPEB.

Los resultados se escriben exclusivamente en ``woa_abc/resultados/hres2``.
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dtw_stagnation import StagnationConfig
from woa_abc.adaptive import M0_NO_DTW, VARIANT_NAMES, resolve_variants
from woa_abc.config_experimentos import HRES2_DTW, HRES2_MH
from woa_abc.hres_reporting import (
    save_epoch_plots,
    save_global_report,
    save_problem_report,
    save_run_summary,
)
from woa_abc.cooperativo_hres2_dtw import (
    CooperativeHRES2Params,
    ejecutar_hres2_cooperativo,
)
from woa_abc.result_io import (
    create_output_dir,
    save_configuration,
    save_run_details,
    write_csv,
    write_json,
)


def _load_hres2_model() -> tuple[type, Any]:
    module = importlib.import_module("HRES2-H2.wpeb_model")
    return module.HRES2Function, module.decode_solution


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark WOA--ABC adaptativo para HRES2")
    parser.add_argument("--variants", nargs="+", default=["all"], help="all o nombres M0/M1/M2/M4/M5/M6/M8")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=HRES2_DTW.window)
    parser.add_argument("--band", type=int, default=HRES2_DTW.band)
    parser.add_argument("--min-slope", type=float, default=HRES2_DTW.min_slope)
    parser.add_argument("--plateau-max", type=int, default=HRES2_DTW.plateau_max)
    parser.add_argument("--patience", type=int, default=HRES2_DTW.patience)
    parser.add_argument("--ddtw", action="store_true", default=HRES2_DTW.use_ddtw)
    parser.add_argument("--fixed-thresholds", action="store_true", default=not HRES2_DTW.adapt_thresholds)
    parser.add_argument("--p-low", type=float, default=HRES2_DTW.p_low)
    parser.add_argument("--p-high", type=float, default=HRES2_DTW.p_high)
    parser.add_argument("--improvement-tol", type=float, default=HRES2_DTW.improvement_tol)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verbose-engine", action="store_true")
    return parser


def _summary_markdown(rows: list[dict]) -> str:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(float(row["best_value"]))
    lines = [
        "# Resumen WOA--ABC HRES2",
        "",
        "Este archivo contiene estadística descriptiva. Las pruebas inferenciales quedan para la fase posterior.",
        "",
        "| Variante | Runs | Mejor LCOE | Media | Mediana | Desv. estándar | Factibles |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, values in grouped.items():
        variant_rows = [row for row in rows if row["variant"] == variant]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        feasible = sum(bool(row["feasible"]) for row in variant_rows)
        lines.append(
            f"| {variant} | {len(values)} | {min(values):.12g} | "
            f"{statistics.mean(values):.12g} | {statistics.median(values):.12g} | "
            f"{std:.12g} | {feasible} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    if args.runs < 1:
        raise ValueError("--runs debe ser positivo")
    variants = resolve_variants(args.variants)
    dtw_mode = "ddtw" if args.ddtw else "dtw"
    output_dir = create_output_dir(
        "hres2",
        args.output_dir,
        experiment_label=f"run_HRES2_{dtw_mode}",
    )
    HRES2Function, decode_solution = _load_hres2_model()
    func = HRES2Function()
    stag_cfg = StagnationConfig(
        window=args.window,
        band=args.band,
        min_slope=args.min_slope,
        plateau_max=args.plateau_max,
        patience=args.patience,
        use_ddtw=args.ddtw,
        adapt_thresholds=not args.fixed_thresholds,
        p_low=args.p_low,
        p_high=args.p_high,
        improvement_tol=args.improvement_tol,
    )
    config = {
        "domain": "hres2",
        "problem": func.name,
        "dtw_mode": dtw_mode,
        "variants": variants,
        "runs": args.runs,
        "iterations": args.iterations,
        "pop_size": args.pop_size,
        "base_seed": args.seed,
        "seed_policy": "run 1 uses base_seed; later runs use base_seed + run_index",
        "execution_order": "variant, run",
        "bounds": {
            "lower": np.asarray(func.lb_vector, dtype=float),
            "upper": np.asarray(func.ub_vector, dtype=float),
        },
        "stagnation": asdict(stag_cfg),
        "mh_base": asdict(HRES2_MH),
        "statistical_analysis": "deferred until 31 runs",
    }
    save_configuration(output_dir, config)
    rows: list[dict] = []
    problem_dir = output_dir / func.name
    problem_dir.mkdir(parents=True, exist_ok=False)
    problem_rows: list[dict] = []
    best_traces: dict[str, object] = {}
    best_values: dict[str, float] = {}
    total = len(variants) * args.runs
    completed = 0

    for variant in variants:
        for run_index in range(args.runs):
            seed = args.seed + run_index
            params = CooperativeHRES2Params(
                pop_size=args.pop_size,
                iterations=args.iterations,
                variant=variant,
                use_dtw=variant != M0_NO_DTW,
                stag_cfg=stag_cfg,
                woa_exploration=HRES2_MH.woa_exploration,
                exploration_probability=HRES2_MH.exploration_probability,
                step_initial=HRES2_MH.step_initial,
                step_final=HRES2_MH.step_final,
                momentum_factor=HRES2_MH.momentum_factor,
                abc_limit_divisor=HRES2_MH.abc_limit_divisor,
                seed=seed,
            )
            started = time.perf_counter()
            result = ejecutar_hres2_cooperativo(func, params, verbose=args.verbose_engine)
            elapsed = time.perf_counter() - started
            solution = np.asarray(result.mejor_sol_global, dtype=float)
            decoded = dict(decode_solution(solution, func.config))
            info = dict(result.mejor_info_global)
            alarms = sum(epoch.stagnation_fires for epoch in result.epochs)
            transitions = sum(epoch.mode_transitions for epoch in result.epochs)
            updates = sum(epoch.parameter_updates for epoch in result.epochs)
            row = {
                "domain": "hres2",
                "problem": func.name,
                "dimension": int(func.n_dim),
                "variant": variant,
                "run": run_index + 1,
                "seed": seed,
                "best_value": result.mejor_valor_global,
                "lcoe_cny_per_kwh": info.get("lcoe_cny_per_kwh"),
                "lcoh_cny_per_kg": info.get("lcoh_cny_per_kg"),
                "agsr": info.get("agsr"),
                "total_h2_kg": info.get("total_h2_kg"),
                "electrolyzer_cf": info.get("electrolyzer_cf"),
                "feasible": bool(info.get("feasible", False)),
                "wind_mw": decoded.get("wind_mw"),
                "pv_mw": decoded.get("pv_mw"),
                "n_el_units": decoded.get("n_el_units"),
                "electrolyzer_mw": decoded.get("electrolyzer_mw"),
                "battery_mw": decoded.get("battery_mw"),
                "battery_duration_h": decoded.get("battery_duration_h"),
                "iterations": args.iterations,
                "objective_evaluations": result.objective_evaluations,
                "time_seconds": elapsed,
                "dtw_alarm_count": alarms,
                "mode_transition_count": transitions,
                "parameter_update_count": updates,
            }
            rows.append(row)
            problem_rows.append(row)
            if variant not in best_values or result.mejor_valor_global < best_values[variant]:
                best_values[variant] = result.mejor_valor_global
                best_traces[variant] = result.epochs[0]
            detail_root = problem_dir / variant / f"run_{run_index + 1:02d}"
            for epoch in result.epochs:
                save_run_details(detail_root, epoch)
                save_epoch_plots(detail_root, epoch, None, variant, scale="linear")
            write_json(detail_root / "solucion_decodificada.json", decoded)
            write_csv(detail_root / "metricas_hres2.csv", [info])
            write_json(
                detail_root / "resultado.json",
                {
                    "summary": row,
                    "best_solution": result.mejor_sol_global,
                    "decoded_solution": decoded,
                    "hres2_metrics": info,
                },
            )
            save_run_summary(detail_root, row, decoded, info)
            write_csv(problem_dir / "runs_resultados.csv", problem_rows)
            write_csv(output_dir / "todos_los_runs.csv", rows)
            write_csv(output_dir / "resultados_variantes.csv", rows)
            write_json(output_dir / "resultados_variantes.json", rows)
            write_csv(output_dir / "results_variants.csv", rows)
            write_json(output_dir / "results_variants.json", rows)
            completed += 1
            print(
                f"[{completed}/{total}] HRES2 {variant} seed={seed} -> "
                f"LCOE={result.mejor_valor_global:.8g} factible={row['feasible']}",
                flush=True,
            )

    save_problem_report(problem_dir, problem_rows, best_traces)
    write_csv(output_dir / "todos_los_runs.csv", rows)
    write_csv(output_dir / "resultados_variantes.csv", rows)
    write_json(output_dir / "resultados_variantes.json", rows)
    save_global_report(output_dir, rows)
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        output = run(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Resultados HRES2 guardados en: {output}")


if __name__ == "__main__":
    main()


__all__ = ["VARIANT_NAMES", "main", "run"]
