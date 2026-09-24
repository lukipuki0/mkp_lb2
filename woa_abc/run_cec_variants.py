"""Ejecuta las siete variantes WOA--ABC sobre CEC2022.

Cada lanzamiento corresponde a un único experimento y una única dimensión.
Por defecto se ejecutan F1--F12 con D=10 y una corrida por variante.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from continuous_benchmark.funciones_cec2022 import get_test_functions
from dtw_stagnation import StagnationConfig
from woa_abc.adaptive import M0_NO_DTW, VARIANT_NAMES, resolve_variants
from woa_abc.cec_reporting import (
    save_epoch_plots,
    save_function_report,
    save_global_report,
    save_run_summary,
)
from woa_abc.config_experimentos import CEC_DTW, CEC_MH
from woa_abc.cooperativo_cec_dtw import CooperativeCECParams, ejecutar_cec_cooperativo
from woa_abc.result_io import (
    create_output_dir,
    save_configuration,
    save_run_details,
    write_csv,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark WOA--ABC adaptativo para CEC2022")
    parser.add_argument("--functions", nargs="+", default=["all"], help="all, 1, F1, ..., 12, F12")
    parser.add_argument(
        "--dimension",
        "--dimensions",
        dest="dimension",
        type=int,
        default=10,
        choices=[10, 20],
        help="Una sola dimensión por experimento (por defecto: 10)",
    )
    parser.add_argument("--variants", nargs="+", default=["all"], help="all o nombres M0/M1/M2/M4/M5/M6/M8")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=CEC_DTW.window)
    parser.add_argument("--band", type=int, default=CEC_DTW.band)
    parser.add_argument("--min-slope", type=float, default=CEC_DTW.min_slope)
    parser.add_argument("--plateau-max", type=int, default=CEC_DTW.plateau_max)
    parser.add_argument("--patience", type=int, default=CEC_DTW.patience)
    parser.add_argument("--ddtw", action="store_true", default=CEC_DTW.use_ddtw)
    parser.add_argument("--fixed-thresholds", action="store_true", default=not CEC_DTW.adapt_thresholds)
    parser.add_argument("--p-low", type=float, default=CEC_DTW.p_low)
    parser.add_argument("--p-high", type=float, default=CEC_DTW.p_high)
    parser.add_argument("--improvement-tol", type=float, default=CEC_DTW.improvement_tol)
    parser.add_argument("--output-dir", type=Path, help="Carpeta base donde crear run_D10_* o run_D20_*")
    parser.add_argument("--verbose-engine", action="store_true")
    return parser


def _function_ids(values: Sequence[str]) -> list[int]:
    if any(value.lower() == "all" for value in values):
        return list(range(1, 13))
    result: list[int] = []
    for value in values:
        normalized = value.upper().removeprefix("F")
        try:
            number = int(normalized)
        except ValueError as exc:
            raise ValueError(f"función CEC inválida: {value!r}") from exc
        if number not in range(1, 13):
            raise ValueError(f"función CEC fuera de rango: {number}")
        if number not in result:
            result.append(number)
    return result


def run(args: argparse.Namespace) -> Path:
    if args.runs < 1:
        raise ValueError("--runs debe ser positivo")

    variants = resolve_variants(args.variants)
    function_ids = _function_ids(args.functions)
    dimension = int(args.dimension)
    functions = get_test_functions(dimension)
    dtw_mode = "ddtw" if args.ddtw else "dtw"
    output_dir = create_output_dir(
        "cec",
        args.output_dir,
        experiment_label=f"run_D{dimension}_{dtw_mode}",
    )
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
        "domain": "cec",
        "functions": function_ids,
        "dimension": dimension,
        "dtw_mode": dtw_mode,
        "variants": variants,
        "runs": args.runs,
        "iterations": args.iterations,
        "pop_size": args.pop_size,
        "base_seed": args.seed,
        "seed_policy": "run 1 uses base_seed; later runs use base_seed + run_index",
        "execution_order": "function F1..F12, variant, run",
        "stagnation": asdict(stag_cfg),
        "mh_base": asdict(CEC_MH),
        "statistical_analysis": "deferred until 31 runs",
    }
    save_configuration(output_dir, config)

    rows: list[dict] = []
    total = len(function_ids) * len(variants) * args.runs
    completed = 0

    print("\n" + "=" * 70)
    print(f"  EXPERIMENTO WOA--ABC CEC2022 | D={dimension}")
    print(f"  Funciones: {len(function_ids)} | Variantes: {len(variants)} | Runs: {args.runs}")
    print(f"  Salida: {output_dir}")
    print("=" * 70)

    for function_position, function_id in enumerate(function_ids, 1):
        func = functions[function_id - 1]
        function_dir = output_dir / f"CEC_{function_id:02d}_{func.name}"
        function_dir.mkdir(parents=True, exist_ok=False)
        function_rows: list[dict] = []
        best_traces: dict[str, object] = {}
        best_values: dict[str, float] = {}

        print(f"\n{'=' * 70}")
        print(f"  [{function_position}/{len(function_ids)}] CEC {function_id}: {func.name} (D={dimension})")
        print(f"{'=' * 70}")

        for variant in variants:
            for run_index in range(args.runs):
                seed = args.seed + run_index
                params = CooperativeCECParams(
                    pop_size=args.pop_size,
                    iterations=args.iterations,
                    variant=variant,
                    use_dtw=variant != M0_NO_DTW,
                    stag_cfg=stag_cfg,
                    woa_exploration=CEC_MH.woa_exploration,
                    exploration_probability=CEC_MH.exploration_probability,
                    step_initial=CEC_MH.step_initial,
                    step_final=CEC_MH.step_final,
                    momentum_factor=CEC_MH.momentum_factor,
                    abc_limit_divisor=CEC_MH.abc_limit_divisor,
                    seed=seed,
                )
                started = time.perf_counter()
                result = ejecutar_cec_cooperativo(func, params, verbose=args.verbose_engine)
                elapsed = time.perf_counter() - started
                alarms = sum(epoch.stagnation_fires for epoch in result.epochs)
                transitions = sum(epoch.mode_transitions for epoch in result.epochs)
                updates = sum(epoch.parameter_updates for epoch in result.epochs)
                error = result.mejor_valor_global - result.valor_optimo
                row = {
                    "domain": "cec",
                    "function_id": function_id,
                    "problem": func.name,
                    "dimension": dimension,
                    "variant": variant,
                    "run": run_index + 1,
                    "seed": seed,
                    "best_value": result.mejor_valor_global,
                    "optimum_reference": result.valor_optimo,
                    "error": error,
                    "gap_pct": result.gap_pct,
                    "iterations": args.iterations,
                    "objective_evaluations": result.objective_evaluations,
                    "time_seconds": elapsed,
                    "dtw_alarm_count": alarms,
                    "mode_transition_count": transitions,
                    "parameter_update_count": updates,
                }
                rows.append(row)
                function_rows.append(row)
                if variant not in best_values or result.mejor_valor_global < best_values[variant]:
                    best_values[variant] = result.mejor_valor_global
                    best_traces[variant] = result.epochs[0]

                run_dir = function_dir / variant / f"run_{run_index + 1:02d}"
                run_dir.mkdir(parents=True, exist_ok=False)
                for epoch in result.epochs:
                    save_run_details(run_dir, epoch)
                    save_epoch_plots(run_dir, epoch, result.valor_optimo, variant, scale="log")
                write_json(
                    run_dir / "resultado.json",
                    {
                        "summary": row,
                        "best_solution": result.mejor_sol_global,
                    },
                )
                save_run_summary(run_dir, row)

                # Conservar lo ya terminado si el trabajo HPC se interrumpe.
                write_csv(function_dir / "runs_resultados.csv", function_rows)
                write_csv(output_dir / "todos_los_runs.csv", rows)
                write_csv(output_dir / "resultados_variantes.csv", rows)
                write_json(output_dir / "resultados_variantes.json", rows)

                completed += 1
                print(
                    f"  [{completed}/{total}] {variant} | run={run_index + 1} | "
                    f"seed={seed} | mejor={result.mejor_valor_global:.8g}",
                    flush=True,
                )

        save_function_report(function_dir, function_rows, best_traces)
        print(f"  Resumen de F{function_id:02d}: {function_dir}")

    write_csv(output_dir / "todos_los_runs.csv", rows)
    write_csv(output_dir / "resultados_variantes.csv", rows)
    write_json(output_dir / "resultados_variantes.json", rows)
    save_global_report(output_dir, rows, dimension)
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        output = run(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"\nResultados CEC guardados en: {output}")


if __name__ == "__main__":
    main()


__all__ = ["VARIANT_NAMES", "main", "run"]
