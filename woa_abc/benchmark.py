"""Single-run exporter for the binary WOA-ABC hybrid.

One functional validation on a local OR-Library file. This is not the 31-run
batch writer and it does not compare algorithms or download data.

Detector settings are copied from ``hybrid_mkp/batch_benchmark.py``. A requested
Sakoe-Chiba band of 0 is the monitor's auto band, resolved when the config is
built. The manifest repeats the solver's effective config rather than a second
reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from . import solver as hybrid
from .mkp_plots import (
    BOX_NOTE,
    BOX_TITLE,
    CONVERGENCE_SIZE,
    DELTA_SIZE,
    DPI,
    EMPTY_DELTA_LABEL,
    PARAMETER_SIZE,
    draw_boxplot,
    draw_convergence,
    draw_delta,
    draw_parameters,
    plot_boxplot,
    plot_convergence,
    plot_delta,
    plot_parameters,
)
from .solver import HybridConfig, solve
from dtw_stagnation import StagnationConfig


# Copied from hybrid_mkp/batch_benchmark.py (STAG_*). That file does not set
# improvement_tol, so the copied value is the StagnationConfig default 0.
INHERITED_WINDOW = 40
INHERITED_BAND = 0
INHERITED_MIN_SLOPE = 0.1
INHERITED_PLATEAU_MAX = 15
INHERITED_PATIENCE = 8
INHERITED_USE_DDTW = True
INHERITED_ADAPT = True
INHERITED_P_LOW = 30.0
INHERITED_P_HIGH = 70.0
INHERITED_IMPROVEMENT_TOL = 0.0

DEFAULT_SEED = 43
DEFAULT_POPULATION_SIZE = 30
DEFAULT_ITERATIONS = 1000
DEFAULT_INSTANCE_INDEX = 0
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "resultados"

MH_LABEL = "WOA-ABC"
MH_KIND = "poblacional"
N_SWITCHES = 0
SUMMARY_BANNER = "Validación funcional de un solo run. No se afirma superioridad."

ESTADO_WARMUP = "Llenando ventana"
ESTADO_FIRE = "ESTANCAMIENTO (Fire)"
ESTADO_ACTIVE = "Explotacion activa"
ESTADO_PARTIAL = "Ciclo parcial sin diagnostico"
ESTADO_MISSING = "Sin diagnostico"

RUN_CSV_FIELDS = (
    "run",
    "mejor_valor",
    "gap_pct",
    "n_switches",
    "tiempo_s",
    "seed",
    "evaluations",
    "cycles_completed",
    "n_fires",
    "n_recoveries",
    "termination_reason",
)
HISTORY_CSV_FIELDS = (
    "iteracion",
    "epoch",
    "cycle",
    "mh",
    "tipo",
    "fitness_best",
    "fitness_instantaneo",
    "evaluations",
    "a",
    "rho",
    "schedule_index",
    "recovery_applied",
    "completeness",
    "dtw_ready",
    "dtw_fire",
    "D1_vs_ramp",
    "D2_vs_const",
    "dtw_delta",
    "theta_c",
    "theta_r",
    "theta_delta",
    "no_improve_len",
    "trigger_streak",
    "window_n",
    "estado_dtw",
)
EXPECTED_ARTIFACTS = (
    "manifest.json",
    "runs_resultados.csv",
    "resumen_pipeline.txt",
    "boxplot_runs.png",
    "runs/run_01.json",
    "best_run_01/resumen_pipeline_best_run.txt",
    "best_run_01/historial_dtw.csv",
    "best_run_01/historial_dtw_detalle.txt",
    "best_run_01/fitness_convergence.png",
    "best_run_01/dtw_delta.png",
    "best_run_01/parameter_recovery.png",
)


class ExportError(Exception):
    """Refused path, overwrite, or a diagnostic that cannot be joined safely."""


@dataclass
class ExportRequest:
    instance: Path
    instance_index: int = DEFAULT_INSTANCE_INDEX
    seed: int = DEFAULT_SEED
    population_size: int = DEFAULT_POPULATION_SIZE
    iterations: int = DEFAULT_ITERATIONS
    max_evaluations: int | None = None
    scout_limit: int | None = None
    output_root: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT)
    timestamp: str | None = None
    variant: str = "v0"
    destination: Path | None = None


@dataclass
class ExportOutcome:
    code: int
    output_dir: Path | None
    message: str = ""


@dataclass
class HistoryRow:
    """One history cycle joined to the diagnostic with the same cycle, if any."""

    cycle: int
    iteration: int
    evaluations: int
    a: float
    rho: float
    schedule_index: int
    best: float
    completeness: bool
    recovery_applied: bool | None
    diagnostic: dict | None


def inherited_stagnation_config() -> StagnationConfig:
    """Build the copied DDTW monitor. Band 0 becomes the auto band immediately."""

    return StagnationConfig(
        window=INHERITED_WINDOW,
        band=INHERITED_BAND,
        min_slope=INHERITED_MIN_SLOPE,
        plateau_max=INHERITED_PLATEAU_MAX,
        patience=INHERITED_PATIENCE,
        use_ddtw=INHERITED_USE_DDTW,
        adapt_thresholds=INHERITED_ADAPT,
        p_low=INHERITED_P_LOW,
        p_high=INHERITED_P_HIGH,
        improvement_tol=INHERITED_IMPROVEMENT_TOL,
    )


def build_hybrid_config(request: ExportRequest) -> HybridConfig:
    return HybridConfig(
        population_size=request.population_size,
        iterations=request.iterations,
        max_evaluations=request.max_evaluations,
        scout_limit=request.scout_limit,
        use_stagnation=True,
        stag_cfg=inherited_stagnation_config(),
        variant=request.variant,
    )


def _reject_url(value: str, label: str) -> None:
    """Reject remote paths, including a Windows Path that collapsed a slash."""

    text = str(value).strip().replace("\\", "/")
    lowered = text.lower()
    if "://" in text or lowered.startswith(("http:/", "https:/", "file:/", "ftp:/")):
        raise ExportError(f"{label} must be a local path, not a URL")


def _safe_timestamp(value: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in "\\/:."):
        raise ExportError("timestamp must be a single local directory name")
    return value


def _field(record: object, name: str):
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _number(value: float) -> str:
    return format(float(value), ".10g")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_fires(diagnostics: list) -> int:
    return sum(1 for item in diagnostics if item.get("fire"))


def count_recoveries(diagnostics: list) -> int:
    return sum(1 for item in diagnostics if item.get("recovery_applied"))


def join_history(history: list, diagnostics: list) -> list[HistoryRow]:
    """Join on cycle. A shorter diagnostic list must not slide onto earlier cycles."""

    indexed: dict[int, dict] = {}
    for item in diagnostics:
        if not isinstance(item, dict) or item.get("cycle") is None:
            raise ExportError("diagnostic is missing a cycle; refusing to zip by position")
        cycle = int(item["cycle"])
        if cycle in indexed:
            raise ExportError(f"duplicate diagnostic for cycle {cycle}")
        indexed[cycle] = item
    rows: list[HistoryRow] = []
    for record in history:
        cycle = int(_field(record, "cycle"))
        diagnostic = indexed.get(cycle)
        applied = None if diagnostic is None else bool(diagnostic.get("recovery_applied"))
        rows.append(
            HistoryRow(
                cycle=cycle,
                iteration=cycle + 1,
                evaluations=int(_field(record, "evaluations")),
                a=float(_field(record, "a")),
                rho=float(_field(record, "rho")),
                schedule_index=int(_field(record, "schedule_index")),
                best=float(_field(record, "best")),
                completeness=bool(_field(record, "completeness")),
                recovery_applied=applied,
                diagnostic=diagnostic,
            )
        )
    return rows


def estado_dtw(row: HistoryRow) -> str:
    if row.diagnostic is None:
        return ESTADO_PARTIAL if not row.completeness else ESTADO_MISSING
    if not row.diagnostic.get("ready"):
        return ESTADO_WARMUP
    if row.diagnostic.get("fire"):
        return ESTADO_FIRE
    return ESTADO_ACTIVE


def _diag_cell(diagnostic: dict | None, key: str, *, as_bool: bool = False) -> str:
    if diagnostic is None or diagnostic.get(key) is None:
        return ""
    value = diagnostic[key]
    if as_bool:
        return "True" if bool(value) else "False"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, float):
        return _number(value)
    return str(int(value) if isinstance(value, bool) else value)


def history_csv_row(row: HistoryRow) -> dict[str, str]:
    diagnostic = row.diagnostic
    applied = "" if row.recovery_applied is None else ("True" if row.recovery_applied else "False")
    return {
        "iteracion": str(row.iteration),
        "epoch": "",
        "cycle": str(row.cycle),
        "mh": MH_LABEL,
        "tipo": MH_KIND,
        "fitness_best": _number(row.best),
        "fitness_instantaneo": "",
        "evaluations": str(row.evaluations),
        "a": _number(row.a),
        "rho": _number(row.rho),
        "schedule_index": str(row.schedule_index),
        "recovery_applied": applied,
        "completeness": "True" if row.completeness else "False",
        "dtw_ready": _diag_cell(diagnostic, "ready", as_bool=True),
        "dtw_fire": _diag_cell(diagnostic, "fire", as_bool=True),
        "D1_vs_ramp": _diag_cell(diagnostic, "D1_vs_ramp"),
        "D2_vs_const": _diag_cell(diagnostic, "D2_vs_const"),
        "dtw_delta": _diag_cell(diagnostic, "delta"),
        "theta_c": _diag_cell(diagnostic, "theta_c"),
        "theta_r": _diag_cell(diagnostic, "theta_r"),
        "theta_delta": _diag_cell(diagnostic, "theta_delta"),
        "no_improve_len": _diag_cell(diagnostic, "no_improve_len"),
        "trigger_streak": _diag_cell(diagnostic, "trigger_streak"),
        "window_n": _diag_cell(diagnostic, "n"),
        "estado_dtw": estado_dtw(row),
    }


def run_csv_row(result, elapsed: float) -> dict[str, str]:
    gap = result.gap
    return {
        "run": "1",
        "mejor_valor": _number(result.best_value),
        "gap_pct": "" if gap is None else f"{float(gap):.2f}",
        "n_switches": str(N_SWITCHES),
        "tiempo_s": f"{elapsed:.2f}",
        "seed": str(result.metadata.get("seed", "")),
        "evaluations": str(result.evaluations),
        "cycles_completed": str(result.cycles_completed),
        "n_fires": str(count_fires(result.diagnostics)),
        "n_recoveries": str(count_recoveries(result.diagnostics)),
        "termination_reason": str(result.termination_reason),
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _labeled(label: str, value: object) -> str:
    return f"{label:<22}: {value}"


def _detector_line(result) -> str:
    config = result.metadata.get("config") or {}
    stag = config.get("stagnation")
    if not stag:
        return "detector off"
    mode = "DDTW" if stag.get("use_ddtw") else "DTW"
    return (
        f"{mode} window={stag.get('window')} band={stag.get('band')} "
        f"(pedido {INHERITED_BAND}, auto) min_slope={_number(float(stag['min_slope']))} "
        f"plateau={stag.get('plateau_max')} patience={stag.get('patience')} "
        f"p_low={_number(float(stag['p_low']))} p_high={_number(float(stag['p_high']))} "
        f"improvement_tol={_number(float(stag['improvement_tol']))}"
    )


def _gap_line(reference: float, gap: float | None) -> str:
    if reference <= 0.0 or gap is None:
        return "desconocido (referencia almacenada; no se reemplaza)"
    return f"{float(gap):.2f}%"


def summary_text(result, instance, display_name: str, elapsed: float) -> str:
    config = result.metadata.get("config") or {}
    ceiling = config.get("max_evaluations")
    reference = float(instance.valor_optimo)
    fires = count_fires(result.diagnostics)
    recoveries = count_recoveries(result.diagnostics)
    lines = [
        SUMMARY_BANNER,
        "No es una comparación de rendimiento.",
        "n_switches cuenta rotaciones de algoritmo. Este run no rota algoritmos.",
        "n_recoveries cuenta reinicios del reloj a/rho, no switches.",
        "Un solo run no tiene dispersión interpretable ni prueba inferencial.",
        "Tiempos de Gantt: no se inventan; no hay rotación de metaheurísticas.",
        "",
        _labeled("Instancia", display_name),
        _labeled("Items / Restricc.", f"{instance.n} / {instance.m}"),
        _labeled("Runs ejecutados", 1),
        _labeled("Optimo conocido", f"{reference:.1f}"),
        _labeled("Mejor valor", _number(result.best_value)),
        _labeled("Gap", _gap_line(reference, result.gap)),
        _labeled("n_switches", N_SWITCHES),
        _labeled("n_fires", fires),
        _labeled("n_recoveries", recoveries),
        _labeled("Ciclos completos", result.cycles_completed),
        _labeled("Ciclos registrados", len(result.history)),
        _labeled("Evaluaciones", result.evaluations),
        _labeled("Techo configurado", f"{ceiling} (no es el consumo)"),
        _labeled("Terminacion", result.termination_reason),
        _labeled("Semilla", result.metadata.get("seed")),
        _labeled("Tiempo de la corrida", f"{elapsed:.2f}s"),
        _labeled("Detector efectivo", _detector_line(result)),
        "",
    ]
    return "\n".join(lines)


def best_summary_text(result, instance, display_name: str, elapsed: float) -> str:
    body = summary_text(result, instance, f"{display_name} (Run #01)", elapsed)
    return body + "El único run es run 01. No hay selección entre campañas.\n"


def detail_text(rows: list[HistoryRow], display_name: str) -> str:
    preamble = [
        SUMMARY_BANNER,
        f"Historial detallado — {display_name}",
        "cycle es el ciclo híbrido del solver (0-based), no una iteración de una sola MH.",
        "epoch queda vacío: este solver no define épocas.",
        "fitness_instantaneo queda vacío: no se copia el mejor archivado.",
        "Los diagnósticos se unen por cycle. Un ciclo sin diagnóstico no hereda el siguiente.",
        "Umbrales: theta_c, theta_r, theta_delta. Contadores: no_improve_len, trigger_streak, window_n.",
        "",
        "\t".join(HISTORY_CSV_FIELDS),
    ]
    body = ["\t".join(history_csv_row(row)[name] for name in HISTORY_CSV_FIELDS) for row in rows]
    return "\n".join(preamble + body) + "\n"


def write_plots(output_dir: Path, rows: list[HistoryRow], reference: float, name: str, best_value: float) -> None:
    best = output_dir / "best_run_01"
    plot_convergence(rows, reference, best / "fitness_convergence.png")
    plot_delta(rows, best / "dtw_delta.png")
    plot_parameters(rows, best / "parameter_recovery.png")
    plot_boxplot(best_value, name, reference, output_dir / "boxplot_runs.png")


def write_tables(output_dir: Path, result, rows: list[HistoryRow], instance, display_name: str, elapsed: float) -> None:
    _write_csv(output_dir / "runs_resultados.csv", RUN_CSV_FIELDS, [run_csv_row(result, elapsed)])
    (output_dir / "resumen_pipeline.txt").write_text(
        summary_text(result, instance, display_name, elapsed),
        encoding="utf-8",
    )
    best = output_dir / "best_run_01"
    best.mkdir(parents=True, exist_ok=True)
    _write_csv(best / "historial_dtw.csv", HISTORY_CSV_FIELDS, [history_csv_row(row) for row in rows])
    (best / "historial_dtw_detalle.txt").write_text(detail_text(rows, display_name), encoding="utf-8")
    (best / "resumen_pipeline_best_run.txt").write_text(
        best_summary_text(result, instance, display_name, elapsed),
        encoding="utf-8",
    )


def output_completeness(output_dir: Path, *, manifest_present: bool) -> dict:
    present = [rel for rel in EXPECTED_ARTIFACTS if rel != "manifest.json" and (output_dir / rel).is_file()]
    if manifest_present:
        present.append("manifest.json")
    missing = [rel for rel in EXPECTED_ARTIFACTS if rel not in present]
    return {
        "complete": not missing,
        "expected": list(EXPECTED_ARTIFACTS),
        "present": present,
        "missing": missing,
    }


def build_manifest(
    result,
    instance,
    instance_path: Path,
    instance_index: int,
    elapsed: float,
    started_at: str,
    finished_at: str,
    status: str,
    plot_error: str | None,
    completeness: dict,
) -> dict:
    config = result.metadata.get("config")
    if not isinstance(config, dict):
        raise ExportError("solver metadata is missing the effective config")
    solver_instance = result.metadata.get("instance")
    reference = float(instance.valor_optimo)
    if solver_instance is not None and float(solver_instance.get("reference")) != reference:
        raise ExportError("solver instance reference does not match the loaded file")
    return {
        "status": status,
        "kind": "single_run_functional_validation",
        "runs": 1,
        "superiority_claim": False,
        "n_switches": N_SWITCHES,
        "solver_invocations": 1,
        "solver_rerun": False,
        "note": (
            "Single-run functional validation. n_switches is 0 because the "
            "algorithm does not rotate. n_recoveries counts a/rho clock restarts. "
            "evaluation_ceiling is the resolved cap, not evaluations consumed. "
            "A stored reference of 0 is unknown and is not replaced."
        ),
        "effective_config": json.loads(json.dumps(config)),
        "detector_request": {
            "window": INHERITED_WINDOW,
            "band": INHERITED_BAND,
            "band_mode": "auto",
            "min_slope": INHERITED_MIN_SLOPE,
            "plateau_max": INHERITED_PLATEAU_MAX,
            "patience": INHERITED_PATIENCE,
            "use_ddtw": INHERITED_USE_DDTW,
            "adapt_thresholds": INHERITED_ADAPT,
            "p_low": INHERITED_P_LOW,
            "p_high": INHERITED_P_HIGH,
            "improvement_tol": INHERITED_IMPROVEMENT_TOL,
        },
        "input": {
            "path": str(instance_path.resolve()),
            "sha256": file_sha256(instance_path),
            "index": instance_index,
            "n": instance.n,
            "m": instance.m,
            "reference": reference,
        },
        "seed": result.metadata.get("seed"),
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "solve_seconds": elapsed,
        },
        "counts": {
            "evaluations": result.evaluations,
            "evaluation_ceiling": config.get("max_evaluations"),
            "cycles_completed": result.cycles_completed,
            "cycles_recorded": len(result.history),
            "n_fires": count_fires(result.diagnostics),
            "n_recoveries": count_recoveries(result.diagnostics),
            "n_switches": N_SWITCHES,
            "termination_reason": result.termination_reason,
            "evaluations_by_phase": dict(result.evaluations_by_phase),
        },
        "output_completeness": completeness,
        "plot_error": plot_error,
    }


def allocate_run_dir(request: ExportRequest, instance_path: Path) -> Path:
    _reject_url(str(request.output_root), "output root")
    root = Path(request.output_root)
    if root.exists() and not root.is_dir():
        raise ExportError("output root must be a local directory")
    stamp = _safe_timestamp(request.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    parent = root / f"run_ddtw_{stamp}"
    if parent.exists():
        raise ExportError(f"refusing to overwrite existing directory: {parent}")
    target = parent / instance_path.stem / f"inst_{request.instance_index:02d}"
    parent.mkdir(parents=True, exist_ok=False)
    target.mkdir(parents=True, exist_ok=False)
    return target


def allocate_explicit_destination(destination: Path) -> Path:
    """Create one caller-chosen results directory after validation.

    The campaign uses this so a variant folder is not nested again under
    ``run_ddtw_<timestamp>/<file>/inst_<index>``. An existing path is refused.
    """

    _reject_url(str(destination), "destination")
    target = Path(destination)
    if target.exists():
        raise ExportError(f"refusing to overwrite existing path: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _display_name(instance_path: Path, index: int) -> str:
    return f"{instance_path.stem}/inst_{index:02d}"


def _failure_manifest(status: str, message: str, instance_path: Path, index: int, elapsed: float) -> dict:
    return {
        "status": status,
        "kind": "single_run_functional_validation",
        "runs": 1,
        "solver_invocations": 1,
        "solver_rerun": False,
        "plot_error": message,
        "timing": {"solve_seconds": elapsed},
        "input": {
            "path": str(instance_path.resolve()),
            "sha256": file_sha256(instance_path),
            "index": index,
        },
        "output_completeness": {
            "complete": False,
            "expected": list(EXPECTED_ARTIFACTS),
            "present": [],
            "missing": list(EXPECTED_ARTIFACTS),
        },
    }


def export_single_run(request: ExportRequest, solver=None) -> ExportOutcome:
    """Solve once, write raw JSON, then summaries and plots. Do not rerun."""

    runner = solve if solver is None else solver
    hybrid._validate_variant(request.variant)
    _reject_url(str(request.instance), "instance")
    _reject_url(str(request.output_root), "output root")
    if request.destination is not None:
        _reject_url(str(request.destination), "destination")
    hybrid._validate_seed(request.seed)
    config = build_hybrid_config(request)
    instance_path = hybrid._local_instance_file(str(request.instance))
    instance, info = hybrid._load_local_instance(instance_path, request.instance_index)
    if request.destination is None:
        output = allocate_run_dir(request, instance_path)
    else:
        output = allocate_explicit_destination(Path(request.destination))
    display = _display_name(instance_path, request.instance_index)
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    try:
        result = runner(instance, config, seed=request.seed, metadata={"instance": info})
    except Exception as exc:
        elapsed = time.perf_counter() - started
        _write_json(
            output / "manifest.json",
            _failure_manifest("solver_failed", f"{type(exc).__name__}: {exc}", instance_path, request.instance_index, elapsed),
        )
        return ExportOutcome(1, output, "solver failed; solver was not rerun")
    elapsed = time.perf_counter() - started
    finished_at = datetime.now().isoformat(timespec="seconds")
    raw_path = output / "runs" / "run_01.json"
    try:
        _write_json(raw_path, asdict(result))
    except (TypeError, ValueError) as exc:
        _write_json(
            output / "manifest.json",
            _failure_manifest("export_failed", f"{type(exc).__name__}: {exc}", instance_path, request.instance_index, elapsed),
        )
        return ExportOutcome(1, output, "raw JSON could not be written; solver was not rerun")

    status = "completed"
    plot_error = None
    try:
        rows = join_history(result.history, result.diagnostics)
        write_tables(output, result, rows, instance, display, elapsed)
    except Exception as exc:
        status = "export_failed"
        plot_error = "".join(traceback.format_exception(exc))
        traceback.print_exc()
    else:
        try:
            write_plots(output, rows, float(instance.valor_optimo), display, float(result.best_value))
        except Exception as exc:
            status = "plot_failed"
            plot_error = "".join(traceback.format_exception(exc))
            traceback.print_exc()
    try:
        manifest = build_manifest(
            result,
            instance,
            instance_path,
            request.instance_index,
            elapsed,
            started_at,
            finished_at,
            status,
            plot_error,
            output_completeness(output, manifest_present=True),
        )
        _write_json(output / "manifest.json", manifest)
    except Exception as exc:
        print(f"error: manifest update failed ({exc}); raw JSON kept", file=sys.stderr)
        return ExportOutcome(1, output, "manifest update failed; raw JSON kept; solver was not rerun")
    if status != "completed":
        return ExportOutcome(1, output, f"{status}: raw JSON kept; solver was not rerun")
    return ExportOutcome(0, output, str(output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="woa_abc.benchmark",
        description=(
            "Run exactly one binary WOA-ABC validation on a local OR-Library file "
            "and write CSV, TXT, and PNG artifacts. This is not a multi-run "
            "comparison and it does not download data."
        ),
    )
    parser.add_argument("--instance", required=True, help="Local OR-Library file. URLs are rejected.")
    parser.add_argument("--instance-index", type=int, default=DEFAULT_INSTANCE_INDEX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Hybrid-cycle horizon T.")
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--scout-limit", type=int, default=None)
    parser.add_argument(
        "--variant",
        default="v0",
        metavar="{v0,v1,v2,v3,v4}",
        help=(
            "One additive variant for this single run. v0 is the unchanged "
            "baseline. v1 uses scout limit 100 unless --scout-limit is set. "
            "v2, v3, and v4 change only post-fire amplitudes, the remaining-cycle "
            "recovery horizon, or the ABC coordinate mutation. Default: v0."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _request_from_args(args: argparse.Namespace) -> ExportRequest:
    hybrid._validate_variant(args.variant)
    _reject_url(str(args.instance), "instance")
    _reject_url(str(args.output_root), "output root")
    return ExportRequest(
        instance=Path(args.instance),
        instance_index=args.instance_index,
        seed=args.seed,
        population_size=args.population_size,
        iterations=args.iterations,
        max_evaluations=args.max_evaluations,
        scout_limit=args.scout_limit,
        output_root=Path(args.output_root),
        variant=args.variant,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    try:
        outcome = export_single_run(_request_from_args(args))
    except (ExportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    stream = sys.stdout if outcome.code == 0 else sys.stderr
    print(outcome.message, file=stream)
    if outcome.code != 0 and outcome.output_dir is not None:
        print(f"output: {outcome.output_dir}", file=sys.stderr)
    return outcome.code


if __name__ == "__main__":
    raise SystemExit(main())
