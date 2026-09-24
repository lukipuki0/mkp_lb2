"""Ejecuta las siete variantes nuevas sobre HRES2-H2/WPEB."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from new_woa_abc.config import HRES2_DTW
from new_woa_abc.core.woa_abc import WOAABCOptimizer
from new_woa_abc.domains.hres2 import load_hres2_problem
from new_woa_abc.dtw.strategies import resolve_variants
from new_woa_abc.reporting import (
    create_experiment_dir,
    save_global_statistical_summary,
    save_paired_statistical_analysis,
    save_run_artifacts,
    save_run_plots,
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


RESULT_ROOT = Path(__file__).resolve().parents[1] / "resultados" / "hres2"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="New WOA--ABC para HRES2")
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
    add_common_arguments(result, HRES2_DTW)
    return result


@dataclass(frozen=True)
class _HRES2Task:
    variant: str
    run_index: int
    args: argparse.Namespace


_HRES2_PROBLEM_CACHE: tuple[object, object] | None = None


def _run_hres2_task(
    task: _HRES2Task,
) -> tuple[object, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Ejecuta una pareja variante-semilla en un worker independiente."""

    global _HRES2_PROBLEM_CACHE
    if _HRES2_PROBLEM_CACHE is None:
        _HRES2_PROBLEM_CACHE = load_hres2_problem()
    problem, _model = _HRES2_PROBLEM_CACHE
    seed = task.args.seed + task.run_index
    config = build_config(task.args, task.variant, seed)
    started = time.perf_counter()
    result = WOAABCOptimizer(problem, config).run(task.args.verbose_engine)
    elapsed = time.perf_counter() - started
    position = result.best_position
    decoded = problem.decode(position)
    metrics = problem.metrics(position)
    row = {
        "domain": "hres2",
        "problem": problem.name,
        "dimension": problem.dimension,
        "variant": task.variant,
        "run": task.run_index + 1,
        "seed": seed,
        "best_cost": result.best_cost,
        "objective_evaluations": result.objective_evaluations,
        "time_seconds": elapsed,
        "fire_count": result.fire_count,
        "transition_count": result.transition_count,
        "parameter_update_count": result.parameter_update_count,
        "feasible": bool(metrics.get("feasible", False)),
        "lcoe_cny_per_kwh": metrics.get("lcoe_cny_per_kwh"),
        "lcoh_cny_per_kg": metrics.get("lcoh_cny_per_kg"),
        "agsr": metrics.get("agsr"),
        "total_h2_kg": metrics.get("total_h2_kg"),
        **decoded,
    }
    return result, row, asdict(config), decoded, metrics


def _hres2_row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["variant"]), int(row["seed"])


def _load_hres2_checkpoint(path: Path) -> list[dict[str, Any]]:
    return load_typed_rows(
        path,
        integer_fields={
            "dimension",
            "run",
            "seed",
            "objective_evaluations",
            "fire_count",
            "transition_count",
            "parameter_update_count",
            "n_el_units",
        },
        float_fields={
            "best_cost",
            "time_seconds",
            "lcoe_cny_per_kwh",
            "lcoh_cny_per_kg",
            "agsr",
            "total_h2_kg",
            "wind_mw",
            "pv_mw",
            "electrolyzer_mw",
            "battery_mw",
            "battery_duration_h",
            "battery_mwh",
        },
        boolean_fields={"feasible"},
    )


def _best_artifact_has_cost(directory: Path, best_cost: float) -> bool:
    import json

    result_path = directory / "resultado.json"
    if not result_path.is_file():
        return False
    stored = json.loads(result_path.read_text(encoding="utf-8"))["summary"]
    return np.isclose(
        float(stored["best_cost"]),
        float(best_cost),
        rtol=1e-12,
        atol=1e-12,
    )


def run(args: argparse.Namespace) -> Path:
    global _HRES2_PROBLEM_CACHE

    if args.runs < 1:
        raise ValueError("--runs debe ser positivo")
    if args.workers < 1:
        raise ValueError("--workers debe ser positivo")
    variants = resolve_variants(args.variants)
    statistics_enabled = validate_statistical_setup(args, variants)
    keep_all_details = save_all_run_artifacts(args)
    problem, _model = load_hres2_problem()
    _HRES2_PROBLEM_CACHE = (problem, _model)
    mode = "ddtw" if args.ddtw else "dtw"
    experiment_config = configuration_dict(args, "hres2")
    experiment_config.update({
        "problem": problem.name,
        "dimension": problem.dimension,
        "lower": problem.lower.tolist(),
        "upper": problem.upper.tolist(),
        "variants": variants,
        "parallel_execution": {
            "unit": "one variant-seed run",
            "requested_workers": args.workers,
            "effective_workers": 0,
        },
    })
    if args.resume_dir is None:
        output = create_experiment_dir(args.output_dir, f"run_HRES2_{mode}")
        rows: list[dict[str, Any]] = []
    else:
        output, existing_config = resume_directory(args.resume_dir, "hres2")
        resume_fields = [
            "runs",
            "iterations",
            "pop_size",
            "base_seed",
            "dtw",
            "strategy",
            "woa_abc",
            "problem",
            "dimension",
            "lower",
            "upper",
            "variants",
        ]
        if "runtime" in existing_config:
            resume_fields.append("runtime")
        validate_resume_config(
            existing_config,
            experiment_config,
            resume_fields,
        )
        rows = _load_hres2_checkpoint(output / "todos_los_runs.csv")

    expected_keys = {
        (variant, args.seed + run_index)
        for variant in variants
        for run_index in range(args.runs)
    }
    completed_keys = {_hres2_row_key(row) for row in rows}
    if len(completed_keys) != len(rows):
        raise ValueError("el checkpoint HRES2 contiene corridas duplicadas")
    unexpected = completed_keys - expected_keys
    if unexpected:
        raise ValueError(f"el checkpoint HRES2 contiene corridas inesperadas: {unexpected}")
    tasks = [
        _HRES2Task(variant, run_index, args)
        for variant in variants
        for run_index in range(args.runs)
        if (variant, args.seed + run_index) not in completed_keys
    ]
    workers = min(args.workers, len(tasks)) if tasks else 0
    experiment_config["parallel_execution"]["effective_workers"] = workers
    experiment_config["resume"] = {
        "enabled": args.resume_dir is not None,
        "rows_loaded": len(rows),
    }
    write_json(output / "configuracion.json", experiment_config)
    (output / "configuracion.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in experiment_config.items()) + "\n",
        encoding="utf-8",
    )

    problem_dir = output / problem.name
    problem_dir.mkdir(parents=True, exist_ok=True)
    session_best: dict[str, tuple] = {}
    total = len(variants) * args.runs
    variant_order = {variant: index for index, variant in enumerate(variants)}

    def checkpoint() -> None:
        rows.sort(key=lambda row: (
            variant_order[str(row["variant"])],
            int(row["run"]),
        ))
        write_csv(problem_dir / "resultados.csv", rows)
        write_csv(output / "todos_los_runs.csv", rows)

    completed = len(rows)
    print(
        f"New WOA--ABC HRES2 | workers={workers} | reanudadas={completed} | "
        f"salida={output}"
    )

    def accept(task: _HRES2Task, payload: tuple) -> None:
        nonlocal completed
        result, row, configuration, decoded, metrics = payload
        rows.append(row)
        previous = session_best.get(task.variant)
        if previous is None or result.best_cost <= previous[0].best_cost:
            session_best[task.variant] = payload
        if keep_all_details:
            run_dir = problem_dir / task.variant / f"run_{task.run_index + 1:02d}"
            if not run_dir.exists():
                save_run_artifacts(
                    run_dir,
                    result,
                    row,
                    configuration,
                    decoded,
                    metrics,
                )
                write_csv(run_dir / "metricas_hres2.csv", [metrics])
                save_run_plots(run_dir, result, problem.optimum)
        completed += 1
        checkpoint()
        print(
            f"[{completed}/{total}] {task.variant} seed={row['seed']} "
            f"LCOE={result.best_cost:.10g} feasible={row['feasible']}",
            flush=True,
        )

    if workers == 1:
        for task in tasks:
            accept(task, _run_hres2_task(task))
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_hres2_task, task): task for task in tasks}
            for future in as_completed(futures):
                accept(futures[future], future.result())

    checkpoint()
    if len(rows) != total:
        raise RuntimeError("la campaña HRES2 quedó incompleta")

    if not keep_all_details:
        for variant in variants:
            variant_rows = [row for row in rows if str(row["variant"]) == variant]
            best_cost = min(float(row["best_cost"]) for row in variant_rows)
            best_dir = problem_dir / variant / "mejor_run"
            if _best_artifact_has_cost(best_dir, best_cost):
                continue
            record = session_best.get(variant)
            if record is None or not np.isclose(
                float(record[1]["best_cost"]), best_cost, rtol=1e-12, atol=1e-12
            ):
                best_row = min(variant_rows, key=lambda row: float(row["best_cost"]))
                record = _run_hres2_task(
                    _HRES2Task(variant, int(best_row["seed"]) - args.seed, args)
                )
            best_result, best_row, best_configuration, best_decoded, best_metrics = record
            archive_existing_directory(best_dir)
            save_run_artifacts(
                best_dir,
                best_result,
                best_row,
                best_configuration,
                best_decoded,
                best_metrics,
            )
            write_csv(best_dir / "metricas_hres2.csv", [best_metrics])
            save_run_plots(best_dir, best_result, problem.optimum)

    title = "HRES2-H2/WPEB"
    save_summary(problem_dir, rows, title)
    if statistics_enabled:
        analysis = save_paired_statistical_analysis(
            problem_dir,
            rows,
            metric="best_cost",
            metric_label="LCOE / costo objetivo",
            title=title,
            reference_variant=args.reference_variant,
            minimize=True,
            alpha=args.alpha,
        )
        save_global_statistical_summary(
            output,
            [(problem.name, analysis)],
            title=title,
        )
    write_csv(output / "todos_los_runs.csv", rows)
    write_json(output / "todos_los_runs.json", rows)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    output = run(parser().parse_args(argv))
    print(f"Resultados guardados en: {output}")


if __name__ == "__main__":
    main()
