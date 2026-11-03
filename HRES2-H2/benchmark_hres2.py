"""
HRES2-H2/benchmark_hres2.py
---------------------------
Benchmark de Ejecución Múltiple (31 Runs Independientes) para el sistema HRES2-H2
usando el Orquestador Híbrido de Metaheurísticas por Rotación DTW.
Pool Activo: PSO, GWO, WOA, EHO, ACO, ABC, ILS, SA, TS, VNS (Excluye GA).
"""

from __future__ import annotations

import os
import csv
import json
import math
import random
import argparse
import datetime
import platform
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Añadir primero la carpeta propia de HRES2-H2 (para cargar su módulo plots local)
sys.path.insert(0, os.path.dirname(__file__))
# Luego la raíz del proyecto
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

from dtw_stagnation import StagnationConfig
import importlib
wpeb_model = importlib.import_module("HRES2-H2.wpeb_model")
HRES2Function = wpeb_model.HRES2Function

import importlib.util as _ilu_orch
_orch_spec = _ilu_orch.spec_from_file_location(
    "hres2_orchestrator",
    os.path.join(os.path.dirname(__file__), "orchestrator.py")
)
_orch_mod = _ilu_orch.module_from_spec(_orch_spec)
_orch_spec.loader.exec_module(_orch_mod)
ejecutar_pipeline_hres2        = _orch_mod.ejecutar_pipeline_hres2
ejecutar_mh_standalone_hres2   = _orch_mod.ejecutar_mh_standalone_hres2
COLORES_MH                     = _orch_mod.COLORES_MH
import importlib
analisis_mod = importlib.import_module("HRES2-H2.analisis_estadistico_hres2")
realizar_analisis_estadistico_completo = analisis_mod.realizar_analisis_estadistico_completo

# ── Importar módulo plots LOCAL de HRES2-H2 (sin conflicto con raíz) ──────────
import importlib.util as _ilu

def _load_hres2_plot(module_name, filename):
    _dir = os.path.dirname(__file__)
    spec = _ilu.spec_from_file_location(
        module_name, os.path.join(_dir, "plots", filename)
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_conv_mod  = _load_hres2_plot("hres2_plots_conv",  "convergencia.py")
_dtw_mod   = _load_hres2_plot("hres2_plots_dtw",   "dtw_delta.py")
_gantt_mod = _load_hres2_plot("hres2_plots_gantt", "switches_gantt.py")

grafico_convergencia_hres2 = _conv_mod.grafico_convergencia_hres2
grafico_dtw_delta           = _dtw_mod.grafico_dtw_delta
grafico_switches            = _gantt_mod.grafico_switches





# ── Configuración ─────────────────────────────────────────────────────────────

N_RUNS                 = 31     # Número de ejecuciones independientes
MAX_ITERS_POR_RUN      = 1000   # Iteraciones máximas por run
RANDOM_SEED            = 42     # Semilla global fijada para reproducibilidad (42)
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "resultados")

# Parámetros del Monitor DTW
STAG_WINDOW      = 40
STAG_BAND        = 4
STAG_MIN_SLOPE   = 0.1
STAG_PLATEAU_MAX = 15
STAG_PATIENCE    = 8
STAG_USE_DDTW    = True
STAG_ADAPT       = True
STAG_P_LOW       = 30.0
STAG_P_HIGH      = 70.0
PAPER_BASELINES = ["PSO", "GWO", "WOA", "EHO", "ACO", "ILS", "SA", "TS", "VNS"]
SEED_NOTE = (
    "A repeated seed does not by itself prove paired initial conditions; "
    "solvers consume the RNG differently and weather is constructed first."
)
ITERATION_NOTE = (
    "nominal_iterations is the requested max_iters budget, not an evaluation budget; "
    "actual_iterations may overshoot because epoch length is unchanged and max_epoch_iters is not forced."
)


def parse_modes(text: str) -> list[str]:
    modes = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not modes or any(mode not in ("dtw", "ddtw") for mode in modes):
        raise ValueError("modes must be a comma list of dtw and/or ddtw")
    return modes


def parse_baselines(text: str) -> list[str]:
    names = [part.strip().upper() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in PAPER_BASELINES]
    if unknown:
        raise ValueError(
            "baselines must be chosen from " + ",".join(PAPER_BASELINES)
            + "; ABC stays in the hybrid portfolio and is not a paper baseline"
        )
    return names


def hybrid_method(mode: str) -> str:
    if mode == "dtw":
        return "Hybrid DTW"
    if mode == "ddtw":
        return "Hybrid DDTW"
    raise ValueError(f"unknown distance mode: {mode}")


def campaign_seed(seed_base: int, method: str, run: int) -> int:
    offset = 0 if method.startswith("Hybrid") else 1000 * (PAPER_BASELINES.index(method) + 1)
    return int(seed_base) + int(run) + offset


def _seed_globals(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def monitor_dict(use_ddtw: bool) -> dict:
    return {
        "window": STAG_WINDOW, "band": STAG_BAND, "min_slope": STAG_MIN_SLOPE,
        "plateau_max": STAG_PLATEAU_MAX, "patience": STAG_PATIENCE, "use_ddtw": use_ddtw,
        "adapt_thresholds": STAG_ADAPT, "p_low": STAG_P_LOW, "p_high": STAG_P_HIGH,
    }


def make_stag(use_ddtw: bool) -> StagnationConfig:
    return StagnationConfig(**monitor_dict(use_ddtw))


def finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return finite_or_none(value)
    if isinstance(value, (np.bool_, bool)) or value is None or isinstance(value, (str, int)):
        return bool(value) if isinstance(value, (np.bool_, bool)) else value
    return str(value)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _source_evidence() -> dict:
    sha, error = None, None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.join(os.path.dirname(__file__), ".."),
            capture_output=True, text=True, timeout=10, check=False,
        )
        sha = proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None
        error = None if sha else (proc.stderr.strip() or f"git exit {proc.returncode}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        error = str(exc)
    return {"benchmark_path": os.path.abspath(__file__), "git_sha": sha, "git_error": error}


class _CanonicalWriter:
    def __init__(self, path: str):
        self.handle = open(path, "a", encoding="utf-8", newline="\n")

    def write(self, record: dict) -> None:
        self.handle.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def resolve_output_dir(output_dir: str | None, label: str) -> str:
    if output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        candidate = os.path.join(OUTPUT_BASE, f"run_{label}_{stamp}")
        while os.path.exists(candidate):
            candidate += "_fresh"
        return candidate
    if os.path.exists(output_dir):
        raise FileExistsError("refusing to overwrite existing output destination: " + str(output_dir))
    return output_dir


def solution_from_result(result) -> list[float]:
    solution = getattr(result, "mejor_solucion", None)
    if solution is None:
        solution = getattr(result, "mejor_solucion_global", None)
    if solution is None:
        raise ValueError("optimization result has no mejor_solucion")
    return [float(item) for item in solution]


def objective_from_result(result) -> float:
    value = getattr(result, "mejor_valor", None)
    if value is None:
        value = result.mejor_valor_global
    return float(value)


def history_length(result):
    for name in ("historial_global", "historial"):
        history = getattr(result, name, None)
        if history is not None:
            return len(history)
    iterations = getattr(result, "iteraciones", None)
    return None if iterations is None else int(iterations)


def result_elapsed(result):
    for name in ("elapsed_seconds", "elapsed"):
        if getattr(result, name, None) is not None:
            return finite_or_none(getattr(result, name))
    switches = getattr(result, "log_switches", None) or []
    if switches and getattr(switches[-1], "t_fin", None) is not None:
        return finite_or_none(switches[-1].t_fin)
    return None


def build_run_record(func, result, *, run, method, mode, seed, role, nominal, elapsed):
    solution = solution_from_result(result)
    objective = objective_from_result(result)
    info = func.get_info(np.asarray(solution, dtype=float))
    physical_lcoe = finite_or_none(info.get("lcoe_cny_per_kwh"))
    physical_lcoh = finite_or_none(info.get("lcoh_cny_per_kg"))
    physical_agsr = finite_or_none(info.get("agsr"))
    feasible = None if info.get("feasible") is None else bool(info.get("feasible"))
    objective_finite = finite_or_none(objective)
    same_lcoe = (
        feasible is True and objective_finite is not None and physical_lcoe is not None
        and math.isclose(objective_finite, physical_lcoe, rel_tol=1e-12, abs_tol=0.0)
    )
    actual = history_length(result)
    decoded_keys = (
        "wind_mw", "pv_mw", "n_el_units", "electrolyzer_mw", "battery_mw",
        "battery_duration_h", "battery_mwh", "total_h2_kg", "electrolyzer_cf",
    )
    decoded = {
        key: info.get(key) if key == "n_el_units" else finite_or_none(info.get(key))
        for key in decoded_keys
    }
    return {
        "run": run, "method": method, "mode": mode, "seed": seed, "role": role,
        "objective": objective_finite, "objective_repr": repr(objective),
        "objective_is_physical_lcoe": same_lcoe, "feasible": feasible,
        "physical_lcoe": physical_lcoe,
        "physical_lcoe_repr": None if physical_lcoe is None else repr(physical_lcoe),
        "physical_lcoh": physical_lcoh,
        "physical_lcoh_repr": None if physical_lcoh is None else repr(physical_lcoh),
        "physical_agsr": physical_agsr,
        "physical_agsr_repr": None if physical_agsr is None else repr(physical_agsr),
        "solution": [finite_or_none(item) for item in solution],
        "solution_repr": [repr(float(item)) for item in solution],
        "decoded": decoded, "nominal_iterations": nominal, "actual_iterations": actual,
        "overshoot_iterations": None if actual is None else int(actual) - int(nominal),
        "elapsed_seconds": finite_or_none(elapsed), "result_elapsed_seconds": result_elapsed(result),
        "paired_initial_conditions": False, "seed_note": SEED_NOTE,
    }


def num_or_nan(value) -> float:
    number = finite_or_none(value)
    return float("nan") if number is None else number


def detalle_from_record(record: dict, result) -> dict:
    decoded = record["decoded"]
    return {
        "run": record["run"], "lcoe": num_or_nan(record["physical_lcoe"]),
        "lcoh": num_or_nan(record["physical_lcoh"]), "agsr": num_or_nan(record["physical_agsr"]),
        "total_h2_kg": num_or_nan(decoded.get("total_h2_kg")),
        "electrolyzer_cf": num_or_nan(decoded.get("electrolyzer_cf")),
        "switches": len(getattr(result, "log_switches", []) or []),
        "wind_mw": num_or_nan(decoded.get("wind_mw")), "pv_mw": num_or_nan(decoded.get("pv_mw")),
        "n_el_units": decoded.get("n_el_units"), "electrolyzer_mw": num_or_nan(decoded.get("electrolyzer_mw")),
        "battery_mw": num_or_nan(decoded.get("battery_mw")),
        "battery_duration_h": num_or_nan(decoded.get("battery_duration_h")),
        "objective": num_or_nan(record["objective"]),
    }


# ── Función Auxiliar para Graficar Boxplot de HRES2-H2 ─────────────────────────

def grafico_boxplot_hres2(valores: list[float], output_dir: str, etiqueta: str = "HRES2-H2 Hybrid DTW"):
    """Genera el gráfico de boxplot para las 31 ejecuciones de HRES2-H2."""
    finite = [value for value in valores if math.isfinite(value)]
    if not finite:
        print("  [boxplot] omitido: no hay LCOE fisico finito")
        return
    plt.figure(figsize=(7, 5))
    bp = plt.boxplot(finite, patch_artist=True, widths=0.4,
                     boxprops=dict(facecolor="#2196F3", color="#1565C0", alpha=0.7),
                     medianprops=dict(color="#D32F2F", linewidth=2.0),
                     whiskerprops=dict(color="#1565C0", linewidth=1.5),
                     capprops=dict(color="#1565C0", linewidth=1.5),
                     flierprops=dict(marker="o", color="#D32F2F", alpha=0.6))

    media = np.mean(finite)
    valores = finite
    plt.axhline(y=media, color="#388E3C", linestyle="--", linewidth=1.5, label=f"Mean: {media:.4f} CNY/kWh")

    plt.title("LCOE Distribution across 31 Runs - HRES2-H2 (WPEB)", fontsize=12, fontweight="bold")
    plt.ylabel("LCOE (CNY/kWh)", fontsize=11)
    plt.xticks([1], [etiqueta])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "lcoe_boxplot_31runs.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [boxplot] {out_path}")


# ── Orquestación del Benchmark HRES2-H2 ────────────────────────────────────────

def ejecutar_benchmark_hres2(
    n_runs    : int = N_RUNS,
    max_iters : int = MAX_ITERS_POR_RUN,
    output_dir: str | None = None,
    *,
    modes: list[str] | None = None,
    baselines: list[str] | None = None,
    nonpaper: bool = False,
    pipeline=None,
    standalone=None,
    function_factory=None,
    seed_globals=None,
    seed_base: int = RANDOM_SEED,
) -> dict:
    """Ejecuta el pipeline híbrido y baselines, guardando cada observación."""
    selected_modes = list(modes) if modes is not None else ["ddtw" if STAG_USE_DDTW else "dtw"]
    selected_baselines = list(PAPER_BASELINES if baselines is None else baselines)
    unknown = [name for name in selected_baselines if name not in PAPER_BASELINES]
    if unknown:
        raise ValueError("ABC stays in the hybrid portfolio and is not a paper baseline: " + ",".join(unknown))
    for mode in selected_modes:
        hybrid_method(mode)
    label = "_".join(selected_modes) if len(selected_modes) > 1 else selected_modes[0]
    output_dir = resolve_output_dir(output_dir, "nonpaper_" + label if nonpaper else label)
    os.makedirs(output_dir)
    pipeline_fn = pipeline or ejecutar_pipeline_hres2
    standalone_fn = standalone or ejecutar_mh_standalone_hres2
    seed_fn = seed_globals or _seed_globals
    factory = function_factory or HRES2Function
    func = factory()
    df = getattr(func, "df_year", None)
    columns = getattr(df, "columns", None)
    manifest = {
        "effective_monitor": {mode: monitor_dict(mode == "ddtw") for mode in selected_modes},
        "effective_configuration": {
            "n_runs": n_runs, "nominal_iterations": max_iters, "modes": selected_modes,
            "baselines": selected_baselines, "nonpaper": nonpaper, "seed_base": seed_base,
            "iteration_note": ITERATION_NOTE, "seed_note": SEED_NOTE,
            "hybrid_population": list(getattr(_orch_mod, "POOL_POBLACIONAL_HRES2", [])),
            "hybrid_trajectory": list(getattr(_orch_mod, "POOL_TRAYECTORIA_HRES2", [])),
        },
        "model": {"type": type(func).__name__, "config": getattr(func, "config", None)},
        "weather": {
            "constructed_before_campaign_seeds": True,
            "default_factory": function_factory is None,
            "default_generator": "generate_synthetic_tmy_8760" if function_factory is None else None,
            "rows": None if df is None else int(len(df)),
            "columns": None if columns is None else [str(column) for column in columns],
            "note": "Campaign seeds are applied only after weather construction.",
        },
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "source": _source_evidence(),
    }
    write_json(os.path.join(output_dir, "manifest.json"), manifest)
    writer = _CanonicalWriter(os.path.join(output_dir, "runs.jsonl"))
    all_records = []
    primary_label = hybrid_method(selected_modes[0])
    valores_lcoe, detalles_runs, resultados_runs = [], [], []
    resultados_dict = {}
    banner = "=" * 66
    print(f"\n{banner}\n  BENCHMARK HRES2-H2 - {primary_label}\n{banner}")
    print(f"  Runs: {n_runs} | nominal iters: {max_iters} | {SEED_NOTE}")
    try:
        for mode in selected_modes:
            method = hybrid_method(mode)
            stag_cfg = make_stag(mode == "ddtw")
            for run in range(1, n_runs + 1):
                seed = campaign_seed(seed_base, method, run)
                seed_fn(seed)
                print(f"  >>> {method} [{run:02d}/{n_runs:02d}] seed={seed}")
                started = time.perf_counter()
                result = pipeline_fn(func=func, max_iters=max_iters, stag_cfg=stag_cfg, verbose=False)
                record = build_run_record(
                    func, result, run=run, method=method, mode=mode, seed=seed,
                    role="hybrid", nominal=max_iters, elapsed=time.perf_counter() - started,
                )
                writer.write(record)
                all_records.append(record)
                if mode == selected_modes[0]:
                    valores_lcoe.append(num_or_nan(record["physical_lcoe"]))
                    detalles_runs.append(detalle_from_record(record, result))
                    resultados_runs.append(result)
            resultados_dict[method] = [
                num_or_nan(item["physical_lcoe"]) for item in all_records if item["method"] == method
            ]
        for name in selected_baselines:
            values = []
            for run in range(1, n_runs + 1):
                seed = campaign_seed(seed_base, name, run)
                seed_fn(seed)
                started = time.perf_counter()
                result = standalone_fn(func, name, max_iters)
                record = build_run_record(
                    func, result, run=run, method=name, mode="baseline", seed=seed,
                    role="baseline", nominal=max_iters, elapsed=time.perf_counter() - started,
                )
                writer.write(record)
                all_records.append(record)
                values.append(num_or_nan(record["physical_lcoe"]))
            resultados_dict[name] = values
            print(f"    {name:<4s} | registros: {len(values)} | no reejecutado por modo")
    finally:
        writer.close()

    # 3. Métricas del modo primario. min y mediana no se intercambian.
    finite_lcoe = [value for value in valores_lcoe if math.isfinite(value)]
    media_lcoe   = float(np.mean(finite_lcoe)) if finite_lcoe else float("nan")
    std_lcoe     = float(np.std(finite_lcoe)) if finite_lcoe else float("nan")
    mediana_lcoe = float(np.median(finite_lcoe)) if finite_lcoe else float("nan")
    mejor_lcoe   = float(np.min(finite_lcoe)) if finite_lcoe else float("nan")
    peor_lcoe    = float(np.max(finite_lcoe)) if finite_lcoe else float("nan")

    valores_lcoh = [d["lcoh"] for d in detalles_runs]
    valores_agsr = [d["agsr"] for d in detalles_runs]
    media_lcoh   = float(np.mean(valores_lcoh))
    media_agsr   = float(np.mean(valores_agsr))

    best_run_idx = int(np.argmin([item["objective"] for item in detalles_runs]))
    best_config = detalles_runs[best_run_idx]
    best_res = resultados_runs[best_run_idx]

    # Guardar gráficos e información EXCLUSIVAMENTE del MEJOR RUN
    best_run_dir = os.path.join(output_dir, f"best_run_{best_run_idx + 1:02d}")
    os.makedirs(best_run_dir, exist_ok=True)

    grafico_convergencia_hres2(
        historial_global = best_res.historial_global,
        log_switches     = best_res.log_switches,
        colores_mh       = COLORES_MH,
        output_dir       = best_run_dir,
    )
    grafico_dtw_delta(
        dtw_deltas_global = best_res.dtw_deltas_global,
        log_switches      = best_res.log_switches,
        colores_mh        = COLORES_MH,
        output_dir        = best_run_dir,
    )
    grafico_switches(
        log_switches = best_res.log_switches,
        colores_mh   = COLORES_MH,
        output_dir   = best_run_dir,
    )


    with open(os.path.join(best_run_dir, "detalle_run.txt"), "w", encoding="utf-8") as f_run:
        f_run.write(f"DETALLE MEJOR RUN (Run #{best_run_idx + 1:02d})\n")
        f_run.write(f"LCOE                : {best_config['lcoe']:.6f} CNY/kWh\n")
        f_run.write(f"LCOH                : {best_config['lcoh']:.6f} CNY/kg\n")
        f_run.write(f"AGSR (Grid Surplus) : {best_config['agsr'] * 100:.2f}%\n")
        f_run.write(f"H2 Anual Producido  : {best_config['total_h2_kg']:,.1f} kg\n")
        f_run.write(f"Factor Cap. Elz     : {best_config['electrolyzer_cf'] * 100:.2f}%\n")
        f_run.write(f"Total Switches      : {best_config['switches']}\n")
        f_run.write(f"Potencia Eólica     : {best_config['wind_mw']:.2f} MW\n")
        f_run.write(f"Potencia Solar PV   : {best_config['pv_mw']:.2f} MW\n")
        f_run.write(f"Electrolizador MW   : {best_config['electrolyzer_mw']:.1f} MW ({best_config['n_el_units']} unidades)\n")
        f_run.write(f"Batería MW          : {best_config['battery_mw']:.1f} MW ({best_config['battery_duration_h']} h)\n")

    # CSV y TXT Historial Detallado DTW del Mejor Run
    csv_best_path = os.path.join(best_run_dir, "historial_dtw.csv")
    txt_best_path = os.path.join(best_run_dir, "historial_dtw_detalle.txt")

    deltas          = best_res.dtw_deltas_global
    inst_hist       = getattr(best_res, 'historial_inst_global', []) or []
    dtw_info_global = getattr(best_res, 'dtw_info_global', []) or []

    with open(csv_best_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteracion", "epoch", "mh", "tipo", "lcoe_best", "lcoe_instantaneo",
            "dtw_ready", "dtw_fire", "D1_vs_ramp", "D2_vs_const", "dtw_delta",
            "theta_c", "theta_r", "theta_delta", "no_improve_len", "trigger_streak",
            "window_n", "estado_dtw"
        ])
        offset = 0
        for ep_idx, sw in enumerate(best_res.log_switches, 1):
            n_seg = sw.n_iters
            for i_local in range(n_seg):
                idx = offset + i_local
                if idx >= len(best_res.historial_global):
                    break
                fit  = best_res.historial_global[idx]
                fi   = inst_hist[idx] if idx < len(inst_hist) else float("nan")
                info = dtw_info_global[idx] if idx < len(dtw_info_global) else {}

                ready = info.get("ready", False)
                fire  = info.get("fire", False)
                d1    = info.get("D1_vs_ramp", float("nan"))
                d2    = info.get("D2_vs_const", float("nan"))
                delta = info.get("delta", deltas[idx] if idx < len(deltas) else float("nan"))
                tc    = info.get("theta_c", float("nan"))
                tr    = info.get("theta_r", float("nan"))
                td    = info.get("theta_delta", float("nan"))
                no_imp= info.get("no_improve_len", 0)
                streak= info.get("trigger_streak", 0)
                win_n = info.get("n", 0)

                if not ready:
                    estado_str = "Llenando ventana"
                elif fire:
                    estado_str = "ESTANCAMIENTO (Fire)"
                else:
                    estado_str = "Explotacion activa"

                writer.writerow([
                    idx + 1, ep_idx, sw.mh_nombre, sw.tipo, fit,
                    "" if (isinstance(fi, float) and np.isnan(fi)) else fi,
                    ready, fire,
                    "" if (isinstance(d1, float) and np.isnan(d1)) else d1,
                    "" if (isinstance(d2, float) and np.isnan(d2)) else d2,
                    "" if (isinstance(delta, float) and np.isnan(delta)) else delta,
                    "" if (isinstance(tc, float) and np.isnan(tc)) else tc,
                    "" if (isinstance(tr, float) and np.isnan(tr)) else tr,
                    "" if (isinstance(td, float) and np.isnan(td)) else td,
                    no_imp, streak, win_n, estado_str
                ])
            offset += n_seg

    with open(txt_best_path, "w", encoding="utf-8") as f_dtw:
        f_dtw.write("======================================================================================================================================================================================\n")
        f_dtw.write(f"  HISTORIAL DETALLADO DE TODAS LAS MÉTRICAS DTW POR ITERACIÓN — MEJOR RUN (#{best_run_idx + 1:02d})\n")
        f_dtw.write("======================================================================================================================================================================================\n\n")
        f_dtw.write(
            f"  {'Iter':>6}  {'Epoch':>5}  {'MH':<6}  {'Tipo':<12}  {'LCOE Best':>14}  {'LCOE Inst':>14}  "
            f"{'Ready':>6}  {'Fire':>6}  {'D1 (Ramp)':>14}  {'D2 (Const)':>14}  {'Delta DTW':>14}  "
            f"{'theta_c':>12}  {'theta_r':>12}  {'theta_delta':>14}  {'no_imp':>6}  {'streak':>6}  {'n_win':>6}  {'Estado DTW':<24}\n"
        )
        f_dtw.write("  " + "-" * 210 + "\n")

        offset = 0
        for ep_idx, sw in enumerate(best_res.log_switches, 1):
            n_seg = sw.n_iters
            for i_local in range(n_seg):
                idx = offset + i_local
                if idx >= len(best_res.historial_global):
                    break
                fit  = best_res.historial_global[idx]
                fi   = inst_hist[idx] if idx < len(inst_hist) else float("nan")
                info = dtw_info_global[idx] if idx < len(dtw_info_global) else {}

                ready = info.get("ready", False)
                fire  = info.get("fire", False)
                d1    = info.get("D1_vs_ramp", float("nan"))
                d2    = info.get("D2_vs_const", float("nan"))
                delta = info.get("delta", deltas[idx] if idx < len(deltas) else float("nan"))
                tc    = info.get("theta_c", float("nan"))
                tr    = info.get("theta_r", float("nan"))
                td    = info.get("theta_delta", float("nan"))
                no_imp= info.get("no_improve_len", 0)
                streak= info.get("trigger_streak", 0)
                win_n = info.get("n", 0)

                if not ready: estado_str = f"Llenando ventana (W={win_n})"
                elif fire:    estado_str = "[!] ESTANCAMIENTO (Fire)"
                else:         estado_str = "Explotacion activa"

                d1_s  = f"{d1:14.6f}" if (isinstance(d1, float) and not np.isnan(d1)) else f"{'--':>14}"
                d2_s  = f"{d2:14.6f}" if (isinstance(d2, float) and not np.isnan(d2)) else f"{'--':>14}"
                dl_s  = f"{delta:+14.6f}" if (isinstance(delta, float) and not np.isnan(delta)) else f"{'--':>14}"
                fi_s  = f"{fi:14.6f}" if (isinstance(fi, float) and not np.isnan(fi)) else f"{'--':>14}"
                tc_s  = f"{tc:12.6f}" if (isinstance(tc, float) and not np.isnan(tc)) else f"{'--':>12}"
                tr_s  = f"{tr:12.6f}" if (isinstance(tr, float) and not np.isnan(tr)) else f"{'--':>12}"
                td_s  = f"{td:14.6f}" if (isinstance(td, float) and not np.isnan(td)) else f"{'--':>14}"

                f_dtw.write(
                    f"  {idx + 1:6d}  {ep_idx:5d}  {sw.mh_nombre:<6}  {sw.tipo:<12}  {fit:14.6f}  {fi_s}  "
                    f"{str(ready):>6}  {str(fire):>6}  {d1_s}  {d2_s}  {dl_s}  "
                    f"{tc_s}  {tr_s}  {td_s}  {no_imp:6d}  {streak:6d}  {win_n:6d}  {estado_str:<24}\n"
                )
            offset += n_seg

    print(f"\n  [info] Gráficos y archivos DTW del mejor run guardados en: '{best_run_dir}'")


    print(f"\n{banner}")
    print(f"  RESUMEN ESTADÍSTICO GENERAL ({n_runs} RUNS) - HRES2-H2")
    print(banner)
    print(f"  Media LCOE   : {media_lcoe:.6f} CNY/kWh")
    print(f"  Std LCOE     : {std_lcoe:.6f}")
    print(f"  Media LCOH   : {media_lcoh:.6f} CNY/kg")
    print(f"  Media AGSR   : {media_agsr * 100:.2f}%")
    print(f"  Mediana LCOE fisico : {mediana_lcoe:.6f} CNY/kWh")
    print(f"  Mejor LCOE fisico (min) : {mejor_lcoe:.6f} CNY/kWh (Run #{best_run_idx + 1})")
    print("  El objetivo penalizado no se reporta como LCOE fisico.")
    print(f"  Peor LCOE    : {peor_lcoe:.6f} CNY/kWh")
    print(f"  Configuración Óptima:")
    print(f"    - Parque Eólico   : {best_config['wind_mw']:.2f} MW")
    print(f"    - Parque Solar PV : {best_config['pv_mw']:.2f} MW")
    print(f"    - Electrolizador  : {best_config['electrolyzer_mw']:.1f} MW ({best_config['n_el_units']} unidades)")
    print(f"    - Batería         : {best_config['battery_mw']:.1f} MW ({best_config['battery_duration_h']} horas)")
    print(f"    - Producción H2   : {best_config['total_h2_kg']:,.1f} kg/año (LCOH = {best_config['lcoh']:.4f} CNY/kg)")
    print(f"    - Ratio Excedente : AGSR = {best_config['agsr']*100:.2f}%")
    print(banner)

    # Boxplot individual del Hybrid DTW
    grafico_boxplot_hres2(valores_lcoe, output_dir, etiqueta=primary_label)

    # CSV de Runs del Hybrid DTW
    csv_runs_path = os.path.join(output_dir, "runs_resultados.csv")
    with open(csv_runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "lcoe_cny_kwh", "lcoh_cny_kg", "agsr_pct", "total_h2_kg", "electrolyzer_cf_pct",
                         "n_switches", "wind_mw", "pv_mw", "n_el_units", "electrolyzer_mw", "battery_mw", "battery_duration_h"])
        for d in detalles_runs:
            writer.writerow([d["run"], f"{d['lcoe']:.6f}", f"{d['lcoh']:.6f}", f"{d['agsr']*100:.4f}",
                             f"{d['total_h2_kg']:.2f}", f"{d['electrolyzer_cf']*100:.4f}",
                             d["switches"], d["wind_mw"], d["pv_mw"], d["n_el_units"],
                             d["electrolyzer_mw"], d["battery_mw"], d["battery_duration_h"]])
    print(f"  [csv] {csv_runs_path}")

    # Reporte TXT Resumen
    txt_path = os.path.join(output_dir, "resumen_hres2.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RESUMEN BENCHMARK HRES2-H2 (WPEB EXTENDIDO)\n")
        f.write(f"Runs independientes : {n_runs}\n")
        f.write(f"Max Iters / Run     : {max_iters}\n\n")
        f.write(f"Modo primario: {primary_label}\n")
        f.write(f"Mejor LCOE fisico (min): {mejor_lcoe:.6f}\n")
        f.write(f"Mediana LCOE fisico: {mediana_lcoe:.6f}\n")
        f.write("El objetivo de optimizacion no es LCOE fisico si es penalizacion o no finito.\n\n")
        f.write("Estadísticas Principales:\n")
        f.write(f"  Media LCOE       : {media_lcoe:.6f} CNY/kWh (± {std_lcoe:.6f})\n")
        f.write(f"  Media LCOH       : {media_lcoh:.6f} CNY/kg\n")
        f.write(f"  Media AGSR       : {media_agsr * 100:.2f}%\n")
        f.write(f"  Mejor LCOE       : {mejor_lcoe:.6f} CNY/kWh (Run #{best_run_idx + 1})\n")
        f.write(f"  Peor LCOE        : {peor_lcoe:.6f} CNY/kWh\n\n")
        f.write("Configuración Global Óptima:\n")
        f.write(f"  Parque Eólico    : {best_config['wind_mw']:.2f} MW\n")
        f.write(f"  Parque Solar PV  : {best_config['pv_mw']:.2f} MW\n")
        f.write(f"  Electrolizador   : {best_config['electrolyzer_mw']:.1f} MW ({best_config['n_el_units']} unidades)\n")
        f.write(f"  Batería          : {best_config['battery_mw']:.1f} MW ({best_config['battery_duration_h']} h)\n")
        f.write(f"  Producción H2    : {best_config['total_h2_kg']:,.1f} kg/año\n")
        f.write(f"  LCOH Óptimo      : {best_config['lcoh']:.6f} CNY/kg\n")
        f.write(f"  AGSR Óptimo      : {best_config['agsr'] * 100:.2f}%\n\n")
        f.write("Detalle por Run:\n")
        for d in detalles_runs:
            f.write(f"  Run {d['run']:02d}: LCOE={d['lcoe']:.6f} | LCOH={d['lcoh']:.4f} | AGSR={d['agsr']*100:.2f}% | "
                    f"Wind={d['wind_mw']:.1f}MW, PV={d['pv_mw']:.1f}MW, Elz={d['electrolyzer_mw']}MW, Bat={d['battery_mw']}MW ({d['battery_duration_h']}h) | "
                    f"Switches={d['switches']}\n")
    print(f"  [txt] {txt_path}")

    # Reporte Markdown Resumen
    md_path = os.path.join(output_dir, "resumen_hres2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Resumen Benchmark HRES2-H2 - {primary_label} ({n_runs} Runs)\n\n")
        f.write(f"- **Mejor LCOE fisico (min):** `{mejor_lcoe:.6f}`\n")
        f.write(f"- **Mediana LCOE fisico:** `{mediana_lcoe:.6f}`\n")
        f.write(f"- **Runs totales:** {n_runs}\n")
        f.write(f"- **Iteraciones por Run:** {max_iters}\n")
        f.write(f"- **LCOE Medio:** `{media_lcoe:.6f} CNY/kWh` (± {std_lcoe:.6f})\n")
        f.write(f"- **LCOH Medio:** `{media_lcoh:.6f} CNY/kg`\n")
        f.write(f"- **AGSR Medio (Excedente Red):** `{media_agsr * 100:.2f}%` (Límite max: 20.00%)\n")
        f.write(f"- **Mejor LCOE:** **`{mejor_lcoe:.6f} CNY/kWh`**\n\n")
        f.write("## Configuración del Sistema Óptimo Encontrado\n\n")
        f.write("| Componente / Métrica | Valor Óptimo |\n")
        f.write("|---|---|\n")
        f.write(f"| **LCOE (Coste Nivelado de Energía)** | `{best_config['lcoe']:.6f} CNY/kWh` |\n")
        f.write(f"| **LCOH (Coste Nivelado del Hidrógeno)** | `{best_config['lcoh']:.6f} CNY/kg` |\n")
        f.write(f"| **AGSR (Annual Green Hydrogen Surplus Ratio)** | `{best_config['agsr'] * 100:.2f}%` |\n")
        f.write(f"| **Producción de H2 Verde** | `{best_config['total_h2_kg']:,.1f} kg/año` |\n")
        f.write(f"| **Potencia Eólica** | `{best_config['wind_mw']:.2f} MW` |\n")
        f.write(f"| **Potencia Solar PV** | `{best_config['pv_mw']:.2f} MW` |\n")
        f.write(f"| **Electrolizador H2** | `{best_config['electrolyzer_mw']:.1f} MW` ({best_config['n_el_units']} módulos de 5 MW) |\n")
        f.write(f"| **Almacenamiento Baterías** | `{best_config['battery_mw']:.1f} MW` ({best_config['battery_duration_h']} h) |\n\n")
    print(f"  [md]  {md_path}")



    # Cada híbrido contra los baselines de paper elegidos. El otro híbrido no entra en la familia.
    stat_results_by_hybrid = {}
    for mode in selected_modes:
        method = hybrid_method(mode)
        family_names = [method, *selected_baselines]
        family_dict = {name: resultados_dict[name] for name in family_names}
        family_records = [
            record for record in all_records if str(record.get("method")) in family_dict
        ]
        analysis_dir = output_dir if len(selected_modes) == 1 else os.path.join(output_dir, "analisis_" + mode)
        stat_results_by_hybrid[method] = realizar_analisis_estadistico_completo(
            resultados_dict=family_dict,
            output_dir=analysis_dir,
            algoritmo_referencia=method,
            records=family_records,
            expected_n=n_runs,
        )
    stat_results = stat_results_by_hybrid[primary_label]
    write_json(os.path.join(output_dir, "completion.json"), {
        "completed": True, "records": len(all_records), "nonpaper": nonpaper,
        "reference": primary_label,
        "references": list(stat_results_by_hybrid),
    })

    return {
        "media_lcoe": media_lcoe,
        "std_lcoe": std_lcoe,
        "mejor_lcoe": mejor_lcoe,
        "best_config": best_config,
        "stat_results": stat_results,
        "stat_results_by_hybrid": stat_results_by_hybrid,
        "output_dir": output_dir,
        "mediana_lcoe": mediana_lcoe,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark HRES2-H2 con registros canonicos por run. --help no ejecuta simulaciones."
    )
    parser.add_argument("--output", default=None, help="Directorio nuevo. Un destino existente se rechaza sin sobrescribir.")
    parser.add_argument("--nonpaper", action="store_true", help="Smoke explicito: por defecto 1 run, 1 iteracion nominal, modo dtw y baseline PSO. Honors --runs/--iterations overrides, and also --modes/--baselines. n=1 statistical output is not meaningful.")
    parser.add_argument("--modes", default=None, help="Modos dtw,ddtw. Run sin argumentos ejecuta ambos; --nonpaper usa dtw por defecto.")
    parser.add_argument("--baselines", default=None, help="Subconjunto de PSO,GWO,WOA,EHO,ACO,ILS,SA,TS,VNS. ABC no es baseline.")
    parser.add_argument("--runs", type=int, default=None, help="Runs por metodo. Nominal de paper: 31.")
    parser.add_argument("--iterations", type=int, default=None, help="Presupuesto nominal max_iters. Nominal de paper: 1000.")
    args = parser.parse_args(argv)
    if args.nonpaper:
        n_runs = 1 if args.runs is None else args.runs
        max_iters = 1 if args.iterations is None else args.iterations
        selected_modes = parse_modes(args.modes) if args.modes else ["dtw"]
        selected_baselines = parse_baselines(args.baselines) if args.baselines else ["PSO"]
        nonpaper = True
    else:
        n_runs = N_RUNS if args.runs is None else args.runs
        max_iters = MAX_ITERS_POR_RUN if args.iterations is None else args.iterations
        selected_modes = parse_modes(args.modes) if args.modes else ["dtw", "ddtw"]
        selected_baselines = parse_baselines(args.baselines) if args.baselines else None
        nonpaper = False
    ejecutar_benchmark_hres2(
        n_runs, max_iters, args.output, modes=selected_modes,
        baselines=selected_baselines, nonpaper=nonpaper,
    )
    return 0



if __name__ == "__main__":
    main()
