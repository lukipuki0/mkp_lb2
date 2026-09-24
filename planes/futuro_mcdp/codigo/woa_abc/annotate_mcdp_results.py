"""Añade óptimos y gaps a una corrida MCDP ya generada."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from woa_abc.run_cooperative_mcdp_batch import (
    _default_paths,
    _format_gap_table,
    _load_optima,
    _make_params,
    _write_run_csv,
    _write_configuration_files,
)


def main() -> None:
    _, _, default_optima = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Carpeta run_YYYYMMDD_HHMMSS existente")
    parser.add_argument("--optima-file", type=Path, default=default_optima)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    json_path = run_dir / "results_cooperative_mcdp.json"
    records = json.loads(json_path.read_text(encoding="utf-8"))
    optima = _load_optima(args.optima_file.resolve())
    gaps: dict[int, dict[tuple[int, int], float]] = {i: {} for i in range(1, 11)}
    config_counts: dict[int, int] = {}
    seed_base = 42
    config_path = run_dir / "configuracion.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    old_parameters = configuration.get("parameters", {})
    seed_base = int(configuration.get("seed_base", old_parameters.get("seed", 42)))

    for record in records:
        key = (int(record["Instance_ID"]), int(record["Max_Cells"]), int(record["Machines_Per_Cell"]))
        optimum = optima[key]
        gap = (float(record["Cost"]) - optimum) / optimum * 100.0
        instance_id = key[0]
        config_counts[instance_id] = config_counts.get(instance_id, 0) + 1
        record["Seed"] = seed_base + (instance_id - 1) * 100 + config_counts[instance_id]
        record["Optimum"] = optimum
        record["Gap_Percent"] = gap
        gaps[key[0]][(key[1], key[2])] = gap

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_run_csv(records, run_dir / "results_cooperative_mcdp.csv")
    if config_path.exists():
        configuration["optima_file"] = str(args.optima_file.resolve())
        configuration["gap_added"] = True
        run_args = argparse.Namespace(
            iterations=old_parameters.get("iterations", 300),
            epochs=old_parameters.get("epochs", 1),
            pop_size=old_parameters.get("pop_size", 30),
            limit=old_parameters.get("limit"),
            window=old_parameters.get("window", 75),
            band=old_parameters.get("band", 0),
            min_slope=old_parameters.get("min_slope", 0.1),
            plateau_max=old_parameters.get("plateau_max", 15),
            patience=old_parameters.get("patience", 25),
            ddtw=old_parameters.get("ddtw", True),
            adapt_thresholds=old_parameters.get("adapt_thresholds", True),
        )
        configuration["parameters"] = asdict(_make_params(run_args, seed_base))
        configuration["seed_strategy"] = "seed = seed_base + (instance_id - 1) * 100 + configuration_number"
        configuration["seed_base"] = seed_base
        _write_configuration_files(configuration, run_dir)
    (run_dir / "resultados_tabla_gap_CooperativeMCDP.txt").write_text(
        _format_gap_table(gaps, 10) + "\n", encoding="utf-8"
    )
    print(f"Resultados anotados: {run_dir}")


if __name__ == "__main__":
    main()
