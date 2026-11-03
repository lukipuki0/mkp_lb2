"""Native hybrid text/CSV reports with iteration-based accounting."""
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

import numpy as np

HISTORY_FIELDS = [
    "iteracion", "epoch", "mh", "tipo", "fitness_best", "fitness_instantaneo",
    "dtw_ready", "dtw_fire", "D1_vs_ramp", "D2_vs_const", "dtw_delta",
    "theta_c", "theta_r", "theta_delta", "no_improve_len", "trigger_streak",
    "window_n", "estado_dtw",
]


def json_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {key: json_value(row.get(key)) for key in fields}
            writer.writerow({key: json.dumps(item, sort_keys=True, allow_nan=False)
                             if isinstance(item, (dict, list)) else item
                             for key, item in clean.items()})


def write_histories(result, output: Path):
    """Align details by each native SwitchLog, never compact missing block info."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, offset = [], 0
    for epoch, switch in enumerate(result.log_switches, 1):
        info_rows = switch.dtw_info_hist or []
        for local in range(switch.n_iters):
            index = offset + local
            if index >= len(result.historial_global):
                raise ValueError("Native switch lengths exceed the fitness history")
            info = info_rows[local] if local < len(info_rows) else {}
            available = bool(info)
            ready, fire = info.get("ready"), info.get("fire")
            delta = result.dtw_deltas_global[index] if index < len(result.dtw_deltas_global) else None
            instant = result.historial_inst_global[index] if index < len(result.historial_inst_global) else None
            state = ("DTW no disponible" if not available else
                     "Llenando ventana" if not ready else
                     "ESTANCAMIENTO (Fire)" if fire else "Explotacion activa")
            rows.append(dict(zip(HISTORY_FIELDS, [
                index + 1, epoch, switch.mh_nombre, switch.tipo,
                result.historial_global[index], instant,
                ready if available else None, fire if available else None,
                info.get("D1_vs_ramp"), info.get("D2_vs_const"), delta,
                info.get("theta_c"), info.get("theta_r"), info.get("theta_delta"),
                info.get("no_improve_len"), info.get("trigger_streak"), info.get("n"), state,
            ])))
        offset += switch.n_iters
    if offset != len(result.historial_global):
        raise ValueError("Native switch lengths must equal the fitness history length")
    write_csv(output / "historial_dtw.csv", rows, HISTORY_FIELDS)
    with (output / "historial_dtw_detalle.txt").open("w", encoding="utf-8") as handle:
        handle.write("HISTORIAL DETALLADO DTW POR ITERACION\n")
        handle.write(" | ".join(HISTORY_FIELDS) + "\n")
        for row in rows:
            handle.write(" | ".join("--" if json_value(row[key]) is None else str(row[key])
                                    for key in HISTORY_FIELDS) + "\n")


def write_run(result: dict, output: Path, *, name: str, instance, best=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    native, summary = result["native_result"], result["summary"]
    write_json(output / "summary.json", summary)
    write_json(output / "native_result.json", native)
    write_histories(native, output)
    filename = "resumen_pipeline_best_run.txt" if best else "resumen_pipeline.txt"
    with (output / filename).open("w", encoding="utf-8") as handle:
        handle.write(f"Instancia          : {name}\n")
        handle.write(f"Items / Restricc.  : {instance.n} / {instance.m}\n")
        handle.write(f"Mejor valor global : {native.mejor_valor_global:.1f}\n")
        handle.write(f"Optimo conocido    : {native.valor_optimo:.1f}\n")
        if summary["gap_pct"] is not None:
            handle.write(f"Gap relativo       : {summary['gap_pct']:.2f}%\n")
        handle.write(f"Iteraciones pedidas: {summary['requested_iterations']}\n")
        handle.write(f"Iteraciones reales : {summary['actual_iterations']}\n")
        handle.write(f"Sobrepaso de epoch : {summary['iteration_overshoot']}\n")
        handle.write(f"Total switches     : {native.n_switches}\n")
        handle.write(f"Transiciones MH    : {summary['transitions']}\n\n")
        for i, sw in enumerate(native.log_switches, 1):
            handle.write(f"{i}. {sw.mh_nombre} ({sw.tipo}) | mejor={sw.mejor_valor:.1f}"
                         f" | {sw.t_inicio:.1f}s-{sw.t_fin:.1f}s | iters={sw.n_iters}\n")


def write_batch(results: list[dict], output: Path, *, name: str, instance, plots=True):
    """Hybrid batch filenames; no extra standalone or statistical runs are launched."""
    output = Path(output)
    values = [result["summary"]["best_fitness"] for result in results]
    best_index = int(np.argmax(values))
    best_dir = output / f"best_run_{best_index + 1:02d}"
    write_run(results[best_index], best_dir, name=name, instance=instance, best=True)
    records = []
    for i, result in enumerate(results, 1):
        summary = result["summary"]
        records.append({"run": i, "mejor_valor": summary["best_fitness"], "gap_pct": summary["gap_pct"],
                        "n_switches": summary["n_blocks"], "tiempo_s": summary["elapsed_seconds"],
                        "iteraciones_solicitadas": summary["requested_iterations"],
                        "iteraciones_reales": summary["actual_iterations"],
                        "sobrepaso": summary["iteration_overshoot"]})
    write_csv(output / "runs_resultados.csv", records, list(records[0]))
    with (output / "resumen_pipeline.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Instancia          : {name}\nItems / Restricc.  : {instance.n} / {instance.m}\n")
        handle.write(f"Runs ejecutados    : {len(results)}\nOptimo conocido    : {instance.valor_optimo:.1f}\n")
        for label, value in (("Media valor", np.mean(values)), ("Desviacion est.", np.std(values)),
                             ("Mediana valor", np.median(values)), ("Mejor valor", max(values)), ("Peor valor", min(values))):
            handle.write(f"{label:<19}: {value:.2f}\n")
        handle.write(f"Mejor run          : {best_index + 1}\n")
        handle.write("Presupuesto        : iteraciones nativas; cierre al terminar cada epoch\n")
        for record in records:
            handle.write(f"Run {record['run']}: pedidas={record['iteraciones_solicitadas']}, "
                         f"reales={record['iteraciones_reales']}, sobrepaso={record['sobrepaso']}\n")
    if plots:
        from .plotting import plot_run, plot_distribution
        plot_run(results[best_index]["native_result"], best_dir, metadata=results[best_index]["summary"])
        plot_distribution(name, values, instance.valor_optimo, output)
    return best_dir
