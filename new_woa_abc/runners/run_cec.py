"""Ejecuta las siete variantes nuevas sobre CEC2022."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from new_woa_abc.config import CEC_DTW
from new_woa_abc.core.woa_abc import WOAABCOptimizer
from new_woa_abc.domains.cec import load_cec_problems
from new_woa_abc.dtw.strategies import resolve_variants
from new_woa_abc.reporting import (
    create_experiment_dir,
    save_run_artifacts,
    save_run_plots,
    save_global_statistical_summary,
    save_paired_statistical_analysis,
    save_summary,
    write_csv,
    write_json,
)

from .common import (
    add_common_arguments,
    build_config,
    configuration_dict,
    save_all_run_artifacts,
    validate_statistical_setup,
)
from .checkpoints import (
    archive_existing_directory,
    load_typed_rows,
    resume_directory,
    validate_resume_config,
)


RESULT_ROOT = Path(__file__).resolve().parents[1] / "resultados" / "cec"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="New WOA--ABC para CEC2022")
    result.add_argument("--functions", nargs="+", default=["all"])
    result.add_argument("--dimension", type=int, choices=[10, 20], default=10)
    result.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    result.add_argument(
        "--workers",
        type=int,
        default=1,
        help="corridas independientes en paralelo",
    )
    result.add_argument(
        "--resume-dir",
        type=Path,
        help="reanuda una carpeta de campaña existente y omite corridas completas",
    )
    add_common_arguments(result, CEC_DTW)
    return result


def function_ids(values: Sequence[str]) -> list[int]:
    if any(value.lower() == "all" for value in values):
        return list(range(1, 13))
    selected: list[int] = []
    for value in values:
        normalized = value.upper().removeprefix("F")
        try:
            function_id = int(normalized)
        except ValueError as exc:
            raise ValueError(f"función inválida: {value}") from exc
        if not 1 <= function_id <= 12:
            raise ValueError("las funciones CEC válidas son F1..F12")
        if function_id not in selected:
            selected.append(function_id)
    return selected


@dataclass(frozen=True)
class _CECTask:
    dimension: int
    function_id: int
    variant: str
    run_index: int
    args: argparse.Namespace


_CEC_PROBLEM_CACHE: dict[int, list] = {}


def _run_cec_task(
    task: _CECTask,
) -> tuple[object, dict[str, Any], dict[str, Any]]:
    """Ejecuta una pareja función-variante-semilla en un worker."""

    problems = _CEC_PROBLEM_CACHE.get(task.dimension)
    if problems is None:
        problems = load_cec_problems(task.dimension)
        _CEC_PROBLEM_CACHE[task.dimension] = problems
    problem = problems[task.function_id - 1]
    seed = task.args.seed + task.run_index
    config = build_config(task.args, task.variant, seed)
    started = time.perf_counter()
    result = WOAABCOptimizer(problem, config).run(task.args.verbose_engine)
    elapsed = time.perf_counter() - started
    error = result.best_cost - float(problem.optimum)
    row = {
        "domain": "cec",
        "function_id": task.function_id,
        "problem": problem.name,
        "dimension": problem.dimension,
        "variant": task.variant,
        "run": task.run_index + 1,
        "seed": seed,
        "best_cost": result.best_cost,
        "optimum_reference": problem.optimum,
        "error": error,
        "absolute_error": abs(error),
        "optimum_reached": bool(
            np.isclose(
                result.best_cost,
                problem.optimum,
                rtol=1e-9,
                atol=1e-8,
            )
        ),
        "objective_evaluations": result.objective_evaluations,
        "time_seconds": elapsed,
        "fire_count": result.fire_count,
        "transition_count": result.transition_count,
        "parameter_update_count": result.parameter_update_count,
    }
    return result, row, asdict(config)


def _cec_row_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["function_id"]), str(row["variant"]), int(row["seed"])


def _load_cec_checkpoint(path: Path) -> list[dict[str, Any]]:
    return load_typed_rows(
        path,
        integer_fields={
            "function_id",
            "dimension",
            "run",
            "seed",
            "objective_evaluations",
            "fire_count",
            "transition_count",
            "parameter_update_count",
        },
        float_fields={
            "best_cost",
            "optimum_reference",
            "error",
            "absolute_error",
            "time_seconds",
        },
        boolean_fields={"optimum_reached"},
    )


def _best_artifact_matches(directory: Path, row: dict[str, Any]) -> bool:
    result_path = directory / "resultado.json"
    if not result_path.is_file():
        return False
    import json

    stored = json.loads(result_path.read_text(encoding="utf-8"))["summary"]
    return (
        str(stored["variant"]) == str(row["variant"])
        and np.isclose(
            float(stored["best_cost"]),
            float(row["best_cost"]),
            rtol=1e-12,
            atol=1e-12,
        )
    )


def run(args: argparse.Namespace) -> Path:
    if args.runs < 1:
        raise ValueError("--runs debe ser positivo")
    if args.workers < 1:
        raise ValueError("--workers debe ser positivo")
    variants = resolve_variants(args.variants)
    statistics_enabled = validate_statistical_setup(args, variants)
    keep_all_details = save_all_run_artifacts(args)
    selected_ids = function_ids(args.functions)
    problems = load_cec_problems(args.dimension)
    _CEC_PROBLEM_CACHE[args.dimension] = problems
    mode = "ddtw" if args.ddtw else "dtw"
    experiment_config = configuration_dict(args, "cec")
    experiment_config.update({
        "dimension": args.dimension,
        "functions": selected_ids,
        "variants": variants,
        "parallel_execution": {
            "unit": "one function-variant-seed run",
            "requested_workers": args.workers,
            "effective_workers": 0,
        },
    })
    if args.resume_dir is None:
        output = create_experiment_dir(args.output_dir, f"run_D{args.dimension}_{mode}")
        all_rows: list[dict[str, Any]] = []
    else:
        output, existing_config = resume_directory(args.resume_dir, "cec")
        resume_fields = [
            "runs",
            "iterations",
            "pop_size",
            "base_seed",
            "dtw",
            "strategy",
            "woa_abc",
            "dimension",
            "functions",
            "variants",
        ]
        if "runtime" in existing_config:
            resume_fields.append("runtime")
        validate_resume_config(
            existing_config,
            experiment_config,
            resume_fields,
        )
        all_rows = _load_cec_checkpoint(output / "todos_los_runs.csv")

    expected_keys = {
        (function_id, variant, args.seed + run_index)
        for function_id in selected_ids
        for variant in variants
        for run_index in range(args.runs)
    }
    completed_keys = {_cec_row_key(row) for row in all_rows}
    if len(completed_keys) != len(all_rows):
        raise ValueError("el checkpoint CEC contiene corridas duplicadas")
    unexpected = completed_keys - expected_keys
    if unexpected:
        raise ValueError(f"el checkpoint CEC contiene corridas inesperadas: {unexpected}")

    tasks = [
        _CECTask(args.dimension, function_id, variant, run_index, args)
        for function_id in selected_ids
        for variant in variants
        for run_index in range(args.runs)
        if (function_id, variant, args.seed + run_index) not in completed_keys
    ]
    workers = min(args.workers, len(tasks)) if tasks else 0
    experiment_config["parallel_execution"]["effective_workers"] = workers
    experiment_config["resume"] = {
        "enabled": args.resume_dir is not None,
        "rows_loaded": len(all_rows),
    }
    write_json(output / "configuracion.json", experiment_config)
    (output / "configuracion.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in experiment_config.items()) + "\n",
        encoding="utf-8",
    )

    problem_analyses: list[tuple[str, dict]] = []
    total = len(selected_ids) * len(variants) * args.runs
    variant_order = {variant: index for index, variant in enumerate(variants)}
    function_order = {function_id: index for index, function_id in enumerate(selected_ids)}
    problem_dirs = {
        function_id: output / f"CEC_{function_id:02d}_{problems[function_id - 1].name}"
        for function_id in selected_ids
    }
    for directory in problem_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    session_best: dict[tuple[int, str], tuple[object, dict, dict]] = {}

    def sort_rows() -> None:
        all_rows.sort(key=lambda row: (
            function_order[int(row["function_id"])],
            variant_order[str(row["variant"])],
            int(row["run"]),
        ))

    def checkpoint(function_id: int | None = None) -> None:
        sort_rows()
        write_csv(output / "todos_los_runs.csv", all_rows)
        selected_directories = (
            problem_dirs.items()
            if function_id is None
            else ((function_id, problem_dirs[function_id]),)
        )
        for selected_id, directory in selected_directories:
            write_csv(
                directory / "resultados.csv",
                [row for row in all_rows if int(row["function_id"]) == selected_id],
            )

    completed = len(all_rows)
    print(
        f"New WOA--ABC CEC | D={args.dimension} | workers={workers} | "
        f"reanudadas={completed} | salida={output}"
    )

    def accept(task: _CECTask, payload: tuple[object, dict, dict]) -> None:
        nonlocal completed
        result, row, configuration = payload
        all_rows.append(row)
        best_key = (task.function_id, task.variant)
        previous = session_best.get(best_key)
        if previous is None or float(row["absolute_error"]) <= float(
            previous[1]["absolute_error"]
        ):
            session_best[best_key] = (result, row, configuration)
        if keep_all_details:
            run_dir = (
                problem_dirs[task.function_id]
                / task.variant
                / f"run_{task.run_index + 1:02d}"
            )
            if not run_dir.exists():
                save_run_artifacts(run_dir, result, row, configuration)
                save_run_plots(
                    run_dir,
                    result,
                    problems[task.function_id - 1].optimum,
                )
        completed += 1
        checkpoint(task.function_id)
        print(
            f"[{completed}/{total}] F{task.function_id:02d} {task.variant} "
            f"seed={row['seed']} best={float(row['best_cost']):.10g}",
            flush=True,
        )

    if workers == 1:
        for task in tasks:
            accept(task, _run_cec_task(task))
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_cec_task, task): task for task in tasks}
            for future in as_completed(futures):
                accept(futures[future], future.result())

    checkpoint()
    for function_id in selected_ids:
        problem = problems[function_id - 1]
        problem_dir = problem_dirs[function_id]
        function_rows = [
            row for row in all_rows if int(row["function_id"]) == function_id
        ]
        if len(function_rows) != len(variants) * args.runs:
            raise RuntimeError(f"CEC F{function_id:02d} quedó incompleta")
        if not keep_all_details:
            for variant in variants:
                variant_rows = [
                    row for row in function_rows if str(row["variant"]) == variant
                ]
                best_row = min(variant_rows, key=lambda row: float(row["absolute_error"]))
                best_dir = problem_dir / variant / "mejor_run"
                if _best_artifact_matches(best_dir, best_row):
                    continue
                candidate = session_best.get((function_id, variant))
                if candidate is None or _cec_row_key(candidate[1]) != _cec_row_key(best_row):
                    run_index = int(best_row["seed"]) - args.seed
                    candidate = _run_cec_task(
                        _CECTask(args.dimension, function_id, variant, run_index, args)
                    )
                best_result, materialized_row, best_configuration = candidate
                archive_existing_directory(best_dir)
                save_run_artifacts(
                    best_dir,
                    best_result,
                    materialized_row,
                    best_configuration,
                )
                save_run_plots(best_dir, best_result, problem.optimum)
        title = f"CEC F{function_id:02d} — D={args.dimension}"
        save_summary(problem_dir, function_rows, title)
        if statistics_enabled:
            analysis = save_paired_statistical_analysis(
                problem_dir,
                function_rows,
                metric="absolute_error",
                metric_label="Error absoluto respecto al óptimo",
                title=title,
                reference_variant=args.reference_variant,
                minimize=True,
                alpha=args.alpha,
            )
            problem_analyses.append((f"CEC F{function_id:02d}", analysis))

    write_csv(output / "todos_los_runs.csv", all_rows)
    write_json(output / "todos_los_runs.json", all_rows)
    if statistics_enabled:
        save_global_statistical_summary(
            output,
            problem_analyses,
            title=f"CEC2022 D={args.dimension}",
        )
    lines = [
        f"# Resumen global CEC2022 D={args.dimension}",
        "",
        "Cada función conserva su resumen por variante. No se promedian costos de funciones diferentes.",
        "",
        f"- Funciones: {len(selected_ids)}",
        f"- Variantes: {len(variants)}",
        f"- Runs por variante: {args.runs}",
        f"- Registros totales: {len(all_rows)}",
        (
            "- Estadística inferencial: generada con semillas emparejadas."
            if statistics_enabled
            else "- Estadística inferencial: no generada; use --runs 31."
        ),
    ]
    (output / "resumen_global.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    output = run(parser().parse_args(argv))
    print(f"Resultados guardados en: {output}")


if __name__ == "__main__":
    main()
