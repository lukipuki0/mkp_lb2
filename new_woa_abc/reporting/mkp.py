"""Persistencia y gráficos específicos de maximización para MKP."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_woa_abc.core.binary_woa_abc import MKPOptimizationResult
from new_woa_abc.domains.mkp import MKPInstance

from .io import write_csv, write_json


COLORS = {
    "profit": "#1a5276",
    "woa": "#2471a3",
    "abc": "#17a589",
    "explore": "#e74c3c",
    "delta": "#6c3483",
    "d2": "#c0392b",
    "threshold": "#e67e22",
    "intensity": "#c0392b",
    "best_known": "#7f8c8d",
    "fire": "#f39c12",
}


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _is_explore(mode: str) -> bool:
    return str(mode).startswith("explore")


def _shade_exploration(axis: plt.Axes, modes: list[str]) -> None:
    start: int | None = None
    first = True
    for index in range(len(modes) + 1):
        explore = index < len(modes) and _is_explore(modes[index])
        if explore and start is None:
            start = index
        elif not explore and start is not None:
            axis.axvspan(
                start,
                max(start, index - 1),
                color=COLORS["explore"],
                alpha=0.08,
                label="Explore" if first else None,
            )
            first = False
            start = None


def save_mkp_run_artifacts(
    run_dir: Path,
    result: MKPOptimizationResult,
    instance: MKPInstance,
    summary: dict[str, Any],
    configuration: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    convergence_rows = [
        {
            "iteration": iteration,
            "best_profit": value,
            "best_after_woa": result.woa_history[iteration],
            "best_after_abc": result.abc_history[iteration],
            "mode": result.mode_history[iteration],
            "intensity": result.intensity_history[iteration],
        }
        for iteration, value in enumerate(result.convergence_history)
    ]
    write_csv(run_dir / "historial_convergencia.csv", convergence_rows)
    write_csv(run_dir / "historial_dtw.csv", result.dtw_history)
    write_csv(run_dir / "historial_parametros.csv", result.parameter_history)
    write_json(run_dir / "eventos_control.json", result.control_events)

    bits = np.asarray(result.best_solution, dtype=np.int8)
    usage = instance.resource_use(bits)
    slack = instance.capacities - usage
    resource_rows = [
        {
            "constraint": index + 1,
            "usage": usage[index],
            "capacity": instance.capacities[index],
            "slack": slack[index],
            "utilization_percent": 100.0 * usage[index] / instance.capacities[index],
        }
        for index in range(instance.constraints)
    ]
    write_csv(run_dir / "uso_recursos.csv", resource_rows)
    selected_zero_based = np.flatnonzero(bits).astype(int).tolist()
    selected_one_based = [index + 1 for index in selected_zero_based]
    write_json(
        run_dir / "resultado.json",
        {
            "summary": summary,
            "configuration": configuration,
            "best_solution": result.best_solution,
            "selected_items_zero_based": selected_zero_based,
            "selected_items_one_based": selected_one_based,
            "resource_use": usage,
            "resource_slack": slack,
        },
    )
    wrapped_bits = [
        "".join(str(int(bit)) for bit in bits[start : start + 100])
        for start in range(0, bits.size, 100)
    ]
    (run_dir / "solucion_binaria.txt").write_text(
        "\n".join([
            f"items_seleccionados_1_based: {' '.join(map(str, selected_one_based))}",
            "bits:",
            *wrapped_bits,
        ]) + "\n",
        encoding="utf-8",
    )
    gap = result.gap_percent
    lines = [
        "RESULTADO NEW WOA--ABC — MKP",
        f"Problema                 : {result.problem}",
        f"Variante                 : {result.variant}",
        f"Semilla                  : {result.seed}",
        f"Mejor beneficio          : {result.best_profit:.12g}",
        f"Mejor conocido           : {result.best_known if result.best_known is not None else 'N/D'}",
        f"Gap (%)                  : {gap:.8f}" if gap is not None else "Gap (%)                  : N/D",
        f"Óptimo alcanzado         : {result.optimum_reached}",
        f"Factible                 : {instance.is_feasible(bits)}",
        f"Ítems seleccionados      : {int(np.sum(bits))}",
        f"Iteraciones ejecutadas   : {result.iterations_completed}",
        f"Evaluaciones objetivo    : {result.objective_evaluations}",
        f"Motivo de detención      : {result.stop_reason}",
        f"Fires discretos          : {result.fire_count}",
        f"Transiciones de modo     : {result.transition_count}",
        f"Cambios de parámetros    : {result.parameter_update_count}",
        f"Mejoras por pulido local : {result.local_search_improvements}",
    ]
    (run_dir / "resumen.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_mkp_run_plots(
    run_dir: Path,
    result: MKPOptimizationResult,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 8,
    })
    iterations = np.arange(len(result.convergence_history))

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.step(iterations, result.woa_history, where="post", color=COLORS["woa"], alpha=0.6, label="Después de WOA")
    axis.step(iterations, result.abc_history, where="post", color=COLORS["abc"], alpha=0.7, label="Después de ABC/pulido")
    axis.step(iterations, result.convergence_history, where="post", color=COLORS["profit"], linewidth=1.8, label="Mejor global")
    _shade_exploration(axis, result.mode_history)
    if result.best_known is not None:
        axis.axhline(result.best_known, color=COLORS["best_known"], linestyle=":", label=f"Mejor conocido ({result.best_known:.0f})")
    axis.set(title=f"Convergencia MKP — {result.variant}", xlabel="Iteración", ylabel="Beneficio")
    axis.grid(alpha=0.2, linestyle=":")
    axis.legend()
    fig.tight_layout()
    _save_figure(fig, run_dir, "convergence")

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    top.step(iterations, result.convergence_history, where="post", color=COLORS["profit"], linewidth=1.7, label=f"Mejor ({result.best_profit:.0f})")
    _shade_exploration(top, result.mode_history)
    fire_iterations = [
        int(status["iteration"])
        for status in result.dtw_history
        if status.get("raw_fire")
    ]
    if fire_iterations:
        fire_values = [result.convergence_history[index] for index in fire_iterations]
        top.scatter(fire_iterations, fire_values, marker="v", s=30, color=COLORS["fire"], label="Fire", zorder=4)
    if result.best_known is not None:
        top.axhline(result.best_known, color=COLORS["best_known"], linestyle=":", label="Mejor conocido")
    top.set_ylabel("Beneficio")
    top.set_title(f"WOA--ABC binaria — {result.variant}")
    top.grid(alpha=0.2, linestyle=":")
    top.legend(loc="lower right")

    ready_iterations: list[int] = []
    signals: list[float] = []
    thresholds: list[float] = []
    applied_intensities: list[float] = []
    d2_view = result.variant in {"M1_fire_d2", "M4_d2_continuous"}
    for index, status in enumerate(result.dtw_history):
        if not status.get("ready"):
            continue
        ready_iterations.append(index)
        if d2_view:
            signals.append(float(status["D2_vs_const"]))
            thresholds.append(float(status["theta_c"]))
        else:
            signals.append(float(status["delta"]))
            thresholds.append(float(status["theta_delta"]))
        intensity = result.intensity_history[index]
        applied_intensities.append(0.0 if intensity is None else float(intensity))
    intensity_axis = bottom.twinx()
    if ready_iterations:
        bottom.plot(ready_iterations, signals, color=COLORS["d2" if d2_view else "delta"], label="D2" if d2_view else r"$\Delta=D1-D2$")
        bottom.plot(ready_iterations, thresholds, color=COLORS["threshold"], linestyle="--", label=r"$\theta_c$" if d2_view else r"$\theta_\Delta$")
        intensity_axis.plot(ready_iterations, applied_intensities, color=COLORS["intensity"], alpha=0.75, label="Intensidad aplicada")
    bottom.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    bottom.set(xlabel="Iteración", ylabel="D2" if d2_view else r"$\Delta$")
    bottom.set_title("Señal DTW/DDTW y adaptación")
    bottom.grid(alpha=0.2, linestyle=":")
    intensity_axis.set_ylabel("Intensidad")
    intensity_axis.set_ylim(-0.05, 1.05)
    lines_a, labels_a = bottom.get_legend_handles_labels()
    lines_b, labels_b = intensity_axis.get_legend_handles_labels()
    if lines_a or lines_b:
        bottom.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")
    fig.tight_layout(h_pad=1.5)
    _save_figure(fig, run_dir, "dtw_adaptation")

    rows = result.parameter_history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    if rows:
        axes[0, 0].plot(iterations, [row["woa_a_base"] for row in rows], linestyle=":", color="gray", label="base")
        axes[0, 0].plot(iterations, [row["woa_a_effective"] for row in rows], color=COLORS["woa"], label="efectivo")
        axes[0, 0].legend()
        axes[0, 1].plot(iterations, [row["abc_phi_effective"] for row in rows], color=COLORS["abc"])
        axes[1, 0].plot(iterations, [row["abc_limit_effective"] for row in rows], color=COLORS["abc"])
        axes[1, 1].plot(iterations, [row["abc_guide_strength"] for row in rows], color=COLORS["abc"])
    for axis, title in zip(
        axes.flat,
        ("WOA a: base y efectivo", "ABC phi efectivo", "ABC limit efectivo", "Guía ABC al élite"),
    ):
        axis.set_title(title)
        axis.grid(alpha=0.2, linestyle=":")
    axes[1, 0].set_xlabel("Iteración")
    axes[1, 1].set_xlabel("Iteración")
    fig.suptitle(f"Parámetros efectivos — {result.variant}")
    fig.tight_layout()
    _save_figure(fig, run_dir, "parameters")

    fig, axis = plt.subplots(figsize=(10, 4.8))
    if rows:
        for key, label in (("lb2_g1", "G1"), ("lb2_g2", "G2"), ("lb2_g3", "G3")):
            axis.plot(iterations, [row[key] for row in rows], label=label)
    axis.set(title="Calendario LB2 (independiente de DTW)", xlabel="Iteración", ylabel="Valor")
    axis.grid(alpha=0.2, linestyle=":")
    axis.legend()
    fig.tight_layout()
    _save_figure(fig, run_dir, "lb2_parameters")


def save_mkp_summary(
    directory: Path,
    rows: list[dict[str, Any]],
    title: str,
    best_known: float | None,
) -> None:
    """Resumen descriptivo correcto para maximización, sin mezclar instancias."""

    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "resultados.csv", rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for variant, variant_rows in grouped.items():
        values = [float(row["best_profit"]) for row in variant_rows]
        gaps = [float(row["gap_percent"]) for row in variant_rows if row.get("gap_percent") is not None]
        summary_rows.append({
            "variant": variant,
            "runs": len(values),
            "best": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "worst": min(values),
            "mean_gap_percent": statistics.mean(gaps) if gaps else None,
            "optimum_hits": sum(bool(row.get("optimum_reached")) for row in variant_rows),
        })
    write_csv(directory / "resumen.csv", summary_rows)
    lines = [
        f"# {title}",
        "",
        f"Mejor valor conocido: {best_known if best_known is not None else 'N/D'}.",
        "",
        "| Variante | Runs | Mejor | Media | Mediana | Desv. | Peor | Gap medio (%) | Óptimos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        gap = "N/D" if row["mean_gap_percent"] is None else f"{row['mean_gap_percent']:.6f}"
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['best']:.12g} | "
            f"{row['mean']:.12g} | {row['median']:.12g} | {row['std']:.12g} | "
            f"{row['worst']:.12g} | {gap} | {row['optimum_hits']} |"
        )
    lines.extend([
        "",
        "Con dos o más seeds, los archivos `analisis_estadistico_*` contienen las pruebas inferenciales pareadas.",
    ])
    (directory / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_mkp_group_summary(
    directory: Path,
    rows: list[dict[str, Any]],
    analyses: list[tuple[str, dict[str, Any]]],
    item_count: int,
) -> None:
    """Persiste los runs y el resumen estadístico de un tamaño MKP."""

    if not rows:
        raise ValueError("no hay resultados MKP para guardar en el grupo")
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "todos_los_runs.csv", rows)
    write_json(directory / "todos_los_runs.json", rows)

    problems = sorted({str(row["problem"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    variants = sorted({str(row["variant"]) for row in rows})
    optimum_hits = sum(
        row.get("optimum_reached") is True
        or str(row.get("optimum_reached", "")).strip().lower() == "true"
        for row in rows
    )
    lines = [
        f"# Grupo MKP — {item_count} ítems",
        "",
        f"- Tamaño del problema: {item_count} ítems.",
        f"- Familias: {len(families)} ({', '.join(families)}).",
        f"- Problemas: {len(problems)}.",
        f"- Variantes: {len(variants)}.",
        f"- Registros de runs: {len(rows)}.",
        f"- Runs que alcanzaron el mejor valor conocido: {optimum_hits}/{len(rows)}.",
        "",
        "Los beneficios crudos no se agregan entre problemas distintos. El ranking estadístico del grupo promedia los mean ranks pareados dentro de cada problema.",
        "",
        "## Problemas incluidos",
        "",
    ]
    lines.extend(f"- `{problem}`" for problem in problems)
    if analyses:
        from .statistics import save_global_statistical_summary

        save_global_statistical_summary(
            directory,
            analyses,
            title=f"MKP — grupo de {item_count} ítems",
        )
        lines.extend([
            "",
            "El ranking, sus tablas y el gráfico estadístico están en `analisis_estadistico_global.*` y `ranking_estadistico_global.*`.",
        ])
    else:
        lines.extend([
            "",
            "Análisis inferencial no generado (se requieren runs suficientes por variante).",
        ])
    (directory / "resumen_grupo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "save_mkp_group_summary",
    "save_mkp_run_artifacts",
    "save_mkp_run_plots",
    "save_mkp_summary",
]
