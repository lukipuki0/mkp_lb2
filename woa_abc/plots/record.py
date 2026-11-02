"""
Adapter from a WOA-ABC result to the run record the GWO-DE plots read.

plot_record(result) returns the keys of gwo_de_dtw's run_hybrid dict that the
plots use. `result` is a HybridResult or its asdict()/JSON form.

Alignment rules:
  - One entry per complete cycle, in cycle order: best_series, profile_series,
    parameter_series (a, rho, G3), intensity_series and monitor. A partial
    final cycle (budget stop) is left out, as it never updated the monitor;
    best_fitness is still the archived best, which may come from that cycle.
  - monitor[i] is the diagnostic of complete cycle i in GWO-DE form: the
    detector status, `changed`, and `profile`/`intensity` after the decision
    (the diagnostic's next_profile/next_intensity). With the detector on, a
    complete cycle without its diagnostic is an error.
  - With the detector off (use_stagnation=False) there is no monitor: every
    entry is a not-ready placeholder and monitor_config is absent, so
    dtw_delta.png and the temporal outputs are skipped, as GWO-DE skips them
    for missing data.
"""

from typing import Optional

STATUS_KEYS = ("ready", "fire", "no_improve_len", "trigger_streak", "n", "D1_vs_ramp",
               "D2_vs_const", "delta", "theta_c", "theta_r", "theta_delta")


def _get(record, name: str):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _monitor_entry(diagnostic: dict) -> dict:
    """GWO-DE monitor output: status keys present in the update, then the decision."""
    entry = {key: diagnostic[key] for key in STATUS_KEYS if diagnostic.get(key) is not None}
    entry["changed"] = bool(diagnostic["changed"])
    entry["profile"] = diagnostic["next_profile"]
    entry["intensity"] = diagnostic.get("next_intensity")
    return entry


def plot_record(result, instance: Optional[str] = None,
                known_optimum: Optional[float] = None) -> dict:
    """Run record of `result` for woa_abc.plots (see the module docstring)."""
    metadata = _get(result, "metadata") or {}
    config = metadata.get("config") or {}
    monitor_config = config.get("stagnation")
    if known_optimum is None:
        known_optimum = float((metadata.get("instance") or {}).get("reference", 0.0))
    diagnostics = {int(item["cycle"]): item for item in _get(result, "diagnostics")}

    best_series, profile_series, intensity_series, monitor = [], [], [], []
    parameter_series = {"a": [], "rho": [], "G3": []}
    for cycle in _get(result, "history"):
        if not _get(cycle, "completeness"):
            continue
        index = int(_get(cycle, "cycle"))
        name = str(_get(cycle, "profile"))
        best_series.append(float(_get(cycle, "best")))
        profile_series.append(name)
        for key in parameter_series:
            parameter_series[key].append(float(_get(cycle, key)))
        intensity_series.append(_get(cycle, "intensity"))
        if monitor_config is None:
            monitor.append({"ready": False, "changed": False, "profile": name})
        elif index in diagnostics:
            monitor.append(_monitor_entry(diagnostics[index]))
        else:
            raise ValueError(f"complete cycle {index} has no diagnostic")

    gap = _get(result, "gap")
    record = {
        "policy": config.get("policy"),
        "dtw_kirito": bool(config.get("dtw_kirito", False)),
        "seed": metadata.get("seed"),
        "binarization": "lb2",
        "instance": instance,
        "pop_size": config.get("population_size"),
        "iterations": config.get("iterations"),
        "evaluations": _get(result, "evaluations"),
        "cycles": len(best_series),
        "best_fitness": float(_get(result, "best_value")),
        "known_optimum": float(known_optimum),
        "gap_pct": gap,
        "best_series": best_series,
        "profile_series": profile_series,
        "parameter_series": parameter_series,
        "intensity_series": intensity_series,
        "changes": list(_get(result, "profile_changes")),
        "monitor": monitor,
    }
    if monitor_config is not None:
        record["monitor_config"] = dict(monitor_config)
        record["use_ddtw"] = bool(monitor_config["use_ddtw"])
        record["window"] = int(monitor_config["window"])
    return record
