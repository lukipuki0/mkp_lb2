"""Ejecuta las siete variantes WOA--ABC sobre instancias MKP."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from new_woa_abc.config import MKP_DTW
from new_woa_abc.core.binary_woa_abc import BinaryWOAABCConfig, BinaryWOAABCOptimizer
from new_woa_abc.domains.mkp import (
    MKPInstance,
    instance_group_name,
    parse_mkp_file,
    resolve_mkp_files,
)
from new_woa_abc.dtw.strategies import resolve_variants
from new_woa_abc.reporting import (
    create_experiment_dir,
    save_global_statistical_summary,
    save_mkp_group_summary,
    save_mkp_run_artifacts,
    save_mkp_run_plots,
    save_mkp_summary,
    save_paired_statistical_analysis,
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


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_INSTANCES_DIR = PROJECT_ROOT / "instancias"
RESULT_ROOT = PACKAGE_ROOT / "resultados" / "mkp"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="New WOA--ABC binaria para MKP")
    result.add_argument(
        "--files",
        nargs="+",
        default=["all"],
        help="all, 1..9, mknapcb1 o una ruta .txt",
    )
    result.add_argument(
        "--instances",
        nargs="+",
        default=["0-2"],
        help="índices 0-based: 0, 0-9 o all",
    )
    result.add_argument("--instances-dir", type=Path, default=DEFAULT_INSTANCES_DIR)
    result.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    result.add_argument(
        "--workers",
        type=int,
        default=1,
        help="procesos paralelos; run_mkp.sh usa SLURM_CPUS_PER_TASK (9 por defecto)",
    )
    result.add_argument("--v-max", type=float, default=6.0)
    result.add_argument("--g1-initial", type=float, default=0.5)
    result.add_argument("--g1-final", type=float, default=1.0)
    result.add_argument("--g2-initial", type=float, default=0.5)
    result.add_argument("--g2-final", type=float, default=7.2)
    result.add_argument("--g3-initial", type=float, default=0.5)
    result.add_argument("--g3-final", type=float, default=0.0)
    result.add_argument("--greedy-fraction", type=float, default=0.30)
    result.add_argument("--initialization-noise", type=float, default=0.35)
    result.add_argument("--repair-latent-weight", type=float, default=0.15)
    result.add_argument("--equality-acceptance", type=float, default=0.10)
    result.add_argument("--abc-limit-population-factor", type=float, default=1.0)
    result.add_argument("--local-search-interval", type=int, default=0)
    result.add_argument("--local-search-start", type=int, default=50)
    result.add_argument("--local-search-passes", type=int, default=2)
    result.add_argument("--local-search-ejections", type=int, default=15)
    result.add_argument("--local-search-depth", type=int, choices=[1, 2, 3], default=3)
    result.add_argument("--local-search-elites", type=int, default=3)
    result.add_argument(
        "--early-stop-at-best-known",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    add_common_arguments(result, MKP_DTW)
    result.set_defaults(iterations=3000)
    return result


def resolve_instance_indices(values: Sequence[str], count: int) -> list[int]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    if not tokens or any(token.lower() == "all" for token in tokens):
        return list(range(count))
    selected: list[int] = []
    for token in tokens:
        if "-" in token:
            pieces = token.split("-", 1)
            try:
                start, stop = int(pieces[0]), int(pieces[1])
            except ValueError as exc:
                raise ValueError(f"rango de instancias inválido: {token}") from exc
            if stop < start:
                raise ValueError(f"rango decreciente no permitido: {token}")
            candidates = range(start, stop + 1)
        else:
            try:
                candidates = (int(token),)
            except ValueError as exc:
                raise ValueError(f"índice de instancia inválido: {token}") from exc
        for index in candidates:
            if not 0 <= index < count:
                raise ValueError(f"índice {index} fuera de rango [0, {count - 1}]")
            if index not in selected:
                selected.append(index)
    return selected


def build_binary_config(
    args: argparse.Namespace,
    variant: str,
    seed: int,
) -> BinaryWOAABCConfig:
    return BinaryWOAABCConfig(
        algorithm=build_config(args, variant, seed),
        v_max=args.v_max,
        g1_initial=args.g1_initial,
        g1_final=args.g1_final,
        g2_initial=args.g2_initial,
        g2_final=args.g2_final,
        g3_initial=args.g3_initial,
        g3_final=args.g3_final,
        greedy_fraction=args.greedy_fraction,
        initialization_noise=args.initialization_noise,
        repair_latent_weight=args.repair_latent_weight,
        equality_acceptance=args.equality_acceptance,
        abc_limit_population_factor=args.abc_limit_population_factor,
        local_search_interval=args.local_search_interval,
        local_search_start=args.local_search_start,
        local_search_passes=args.local_search_passes,
        local_search_ejections=args.local_search_ejections,
        local_search_depth=args.local_search_depth,
        local_search_elites=args.local_search_elites,
        early_stop_at_best_known=args.early_stop_at_best_known,
    )


@dataclass(frozen=True)
class _InstanceTask:
    """Una instancia completa; unidad de paralelismo compatible con Slurm."""

    instance: MKPInstance
    instance_dir: Path
    args: argparse.Namespace
    variants: tuple[str, ...]
    statistics_enabled: bool
    keep_all_details: bool


def _run_instance_task(
    task: _InstanceTask,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    """Ejecuta todas las variantes/seeds de una instancia en un proceso."""

    instance = task.instance
    args = task.args
    rows: list[dict[str, Any]] = []
    best_records: dict[str, tuple] = {}
    total = len(task.variants) * args.runs
    completed = 0

    for variant in task.variants:
        for run_index in range(args.runs):
            seed = args.seed + run_index
            config = build_binary_config(args, variant, seed)
            started = time.perf_counter()
            result = BinaryWOAABCOptimizer(instance, config).run(args.verbose_engine)
            elapsed = time.perf_counter() - started
            bits = np.asarray(result.best_solution, dtype=np.int8)
            usage = instance.resource_use(bits)
            gap_absolute = (
                None
                if instance.best_known is None
                else instance.best_known - result.best_profit
            )
            row = {
                "domain": "mkp",
                "family": instance.family,
                "instance": instance.index,
                "problem": instance.name,
                "items": instance.dimension,
                "constraints": instance.constraints,
                "variant": variant,
                "run": run_index + 1,
                "seed": seed,
                "best_profit": result.best_profit,
                "best_known": instance.best_known,
                "absolute_gap": gap_absolute,
                "gap_percent": result.gap_percent,
                "optimum_reached": result.optimum_reached,
                "feasible": instance.is_feasible(bits),
                "selected_items": int(np.sum(bits)),
                "minimum_slack": float(np.min(instance.capacities - usage)),
                "iterations_completed": result.iterations_completed,
                "objective_evaluations": result.objective_evaluations,
                "time_seconds": elapsed,
                "stop_reason": result.stop_reason,
                "fire_count": result.fire_count,
                "transition_count": result.transition_count,
                "parameter_update_count": result.parameter_update_count,
                "local_search_improvements": result.local_search_improvements,
            }
            configuration = asdict(config)
            previous_best = best_records.get(variant)
            if previous_best is None or result.best_profit > previous_best[0].best_profit:
                best_records[variant] = (result, row, configuration)
            if task.keep_all_details:
                run_dir = task.instance_dir / variant / f"run_{run_index + 1:02d}"
                save_mkp_run_artifacts(
                    run_dir,
                    result,
                    instance,
                    row,
                    configuration,
                )
                save_mkp_run_plots(run_dir, result)
            rows.append(row)
            write_csv(task.instance_dir / "resultados.csv", rows)
            completed += 1
            gap_text = (
                "N/D"
                if result.gap_percent is None
                else f"{result.gap_percent:.4f}%"
            )
            print(
                f"[{instance.name} {completed}/{total}] {variant} seed={seed} "
                f"profit={result.best_profit:.0f} gap={gap_text} "
                f"feasible={row['feasible']}",
                flush=True,
            )

    if not task.keep_all_details:
        for variant, (best_result, best_row, best_configuration) in best_records.items():
            best_dir = task.instance_dir / variant / "mejor_run"
            save_mkp_run_artifacts(
                best_dir,
                best_result,
                instance,
                best_row,
                best_configuration,
            )
            save_mkp_run_plots(best_dir, best_result)

    title = f"MKP {instance.name}"
    save_mkp_summary(
        task.instance_dir,
        rows,
        title,
        instance.best_known,
    )
    analysis = None
    if task.statistics_enabled:
        analysis = save_paired_statistical_analysis(
            task.instance_dir,
            rows,
            metric="best_profit",
            metric_label="Beneficio",
            title=title,
            reference_variant=args.reference_variant,
            minimize=False,
            alpha=args.alpha,
        )
    return instance.name, rows, analysis


def run(args: argparse.Namespace) -> Path:
    if args.runs < 1:
        raise ValueError("--runs debe ser positivo")
    if args.workers < 1:
        raise ValueError("--workers debe ser positivo")
    variants = resolve_variants(args.variants)
    statistics_enabled = validate_statistical_setup(args, variants)
    keep_all_details = save_all_run_artifacts(args)
    files = resolve_mkp_files(args.files, args.instances_dir)
    parsed = [(path, parse_mkp_file(path)) for path in files]
    selections = [
        (path, instances, resolve_instance_indices(args.instances, len(instances)))
        for path, instances in parsed
    ]
    mode = "ddtw" if args.ddtw else "dtw"
    output = create_experiment_dir(args.output_dir, f"run_MKP_{mode}")

    experiment_config = configuration_dict(args, "mkp")
    experiment_config.update({
        "files": [str(path) for path in files],
        "instance_selection": list(args.instances),
        "result_grouping": {
            "by": "item count",
            "group_name_template": "grupo_<n>_items",
        },
        "variants": variants,
        "binary_woa_abc": asdict(build_binary_config(args, "M0_no_dtw", args.seed)),
        "objective": "maximize profit",
        "dtw_input": "best_profit (direct maximization signal)",
        "parallel_execution": {
            "unit": "one complete MKP instance",
            "requested_workers": args.workers,
        },
    })
    total_instances = sum(len(indices) for _, _, indices in selections)
    all_rows: list[dict] = []
    problem_analyses: list[tuple[str, dict]] = []
    tasks: list[_InstanceTask] = []
    for path, instances, indices in selections:
        for instance_index in indices:
            instance = instances[instance_index]
            group_dir = output / instance_group_name(instance.dimension)
            family_dir = group_dir / path.stem.lower()
            family_dir.mkdir(parents=True, exist_ok=True)
            instance_dir = family_dir / f"inst_{instance_index:02d}"
            instance_dir.mkdir(parents=True, exist_ok=False)
            tasks.append(_InstanceTask(
                instance=instance,
                instance_dir=instance_dir,
                args=args,
                variants=tuple(variants),
                statistics_enabled=statistics_enabled,
                keep_all_details=keep_all_details,
            ))

    workers = min(args.workers, len(tasks))
    experiment_config["parallel_execution"]["effective_workers"] = workers
    write_json(output / "configuracion.json", experiment_config)
    (output / "configuracion.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in experiment_config.items()) + "\n",
        encoding="utf-8",
    )
    print(
        f"New WOA--ABC MKP | instancias={total_instances} | "
        f"variantes={len(variants)} | workers={workers} | salida={output}"
    )

    if workers == 1:
        task_results = []
        for task in tasks:
            name, rows, analysis = _run_instance_task(task)
            task_results.append((name, rows, analysis))
            all_rows.extend(rows)
            write_csv(output / "todos_los_runs.csv", all_rows)
            print(
                f"[instancia completada] {name} "
                f"({len(task_results)}/{len(tasks)})",
                flush=True,
            )
    else:
        task_results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_instance_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                name, rows, analysis = future.result()
                task_results.append((name, rows, analysis))
                print(
                    f"[instancia completada] {name} "
                    f"({len(task_results)}/{len(tasks)})",
                    flush=True,
                )
                all_rows.extend(rows)
                write_csv(output / "todos_los_runs.csv", all_rows)

    variant_order = {variant: index for index, variant in enumerate(variants)}
    all_rows.sort(key=lambda row: (
        str(row["family"]),
        int(row["instance"]),
        variant_order[str(row["variant"])],
        int(row["run"]),
    ))
    for name, _rows, analysis in task_results:
        if analysis is not None:
            problem_analyses.append((name, analysis))
    problem_analyses.sort(key=lambda item: item[0])

    rows_by_size: dict[int, list[dict]] = {}
    analyses_by_size: dict[int, list[tuple[str, dict]]] = {}
    for name, rows, analysis in task_results:
        if not rows:
            continue
        item_count = int(rows[0]["items"])
        rows_by_size.setdefault(item_count, []).extend(rows)
        if analysis is not None:
            analyses_by_size.setdefault(item_count, []).append((name, analysis))

    for item_count, group_rows in sorted(rows_by_size.items()):
        group_rows.sort(key=lambda row: (
            str(row["family"]),
            int(row["instance"]),
            variant_order[str(row["variant"])],
            int(row["run"]),
        ))
        save_mkp_group_summary(
            output / instance_group_name(item_count),
            group_rows,
            sorted(analyses_by_size.get(item_count, []), key=lambda item: item[0]),
            item_count,
        )

    write_csv(output / "todos_los_runs.csv", all_rows)
    write_json(output / "todos_los_runs.json", all_rows)
    if statistics_enabled:
        save_global_statistical_summary(
            output,
            problem_analyses,
            title="MKP Chu--Beasley",
        )
    optimum_hits = sum(bool(row["optimum_reached"]) for row in all_rows)
    lines = [
        "# Resumen global MKP",
        "",
        "Los beneficios crudos no se promedian entre instancias distintas.",
        "",
        f"- Familias: {len(files)}",
        f"- Instancias: {total_instances}",
        f"- Tamaños de grupo: {', '.join(str(size) for size in sorted(rows_by_size))} ítems",
        "- Selección predeterminada por archivo: índices 0–2.",
        f"- Variantes: {len(variants)}",
        f"- Runs por variante: {args.runs}",
        f"- Registros totales: {len(all_rows)}",
        f"- Valores conocidos alcanzados: {optimum_hits}/{len(all_rows)}",
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
