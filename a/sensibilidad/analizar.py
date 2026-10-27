"""Aggregate an OAT campaign into markdown tables for the revision response."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .grid import GRID, PAPER_BASE


MHS = ("PSO", "GA", "GWO", "DE")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Binary-Complex OAT sensitivity campaign"
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign ID or directory name (default: latest campaign)",
    )
    return parser.parse_args(argv)


def _campaign_dir(campaign: Optional[str]) -> Optional[Path]:
    root = PROJECT_ROOT / "results" / "sensibilidad"
    if campaign:
        dirname = campaign if campaign.startswith("campaign_") else f"campaign_{campaign}"
        return root / dirname

    candidates = sorted(path for path in root.glob("campaign_*") if path.is_dir())
    return candidates[-1] if candidates else None


def _load_manifest(campaign_dir: Path) -> Dict[str, Any]:
    manifest_path = campaign_dir / "configs.json"
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def _config_campaign_id(config: Dict[str, Any]) -> Optional[str]:
    """Return the manifest's per-config run ID."""
    return (
        config.get("per_config_campaign_id")
        or config.get("campaign_id")
        or config.get("run_id")
    )


def _fitness_values(data: Dict[str, Any]) -> List[float]:
    """Read the serialized per-epoch fitness values from save_results output."""
    # mkp_common.results.save_results serializes each epoch's mejor_fitness
    # list under the top-level ``fitness`` key.
    values = data.get("fitness")
    if values is None:
        values = data.get("mejor_fitness")
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    parsed: List[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return parsed


def _mh_file(directory: Path, mh: str, instance_name: str, indice: int) -> Optional[Path]:
    expected = directory / f"{mh}_{instance_name}_{indice}.json"
    if expected.exists():
        return expected

    # Keep analysis tolerant of a harmless filename variation while still
    # requiring exactly one candidate for that MH in the directory.
    candidates = sorted(directory.glob(f"{mh}_*.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _config_metrics(
    config: Dict[str, Any], manifest: Dict[str, Any]
) -> Dict[str, Optional[float]]:
    """Return one mean fitness per MH across all available indices/epochs."""
    config_id = config.get("config_id", "<unknown>")
    per_config_id = _config_campaign_id(config)
    metrics: Dict[str, Optional[float]] = {mh: None for mh in MHS}
    if not per_config_id:
        _warn(f"{config_id}: no per-config campaign ID in manifest; skipping")
        return metrics

    values_by_mh: Dict[str, List[float]] = {mh: [] for mh in MHS}
    instancias = manifest.get("instancias", [])
    indices = manifest.get("indices", [])
    for instancia in instancias:
        instance_name = Path(instancia).stem
        for indice in indices:
            directory = (
                PROJECT_ROOT
                / "results"
                / "binary_complex"
                / "todos"
                / f"{instance_name}_{indice}"
                / f"comparacion_mhs_{per_config_id}"
            )
            if not directory.is_dir():
                _warn(
                    f"{config_id}: missing results directory for "
                    f"{instance_name}[{indice}]: {directory}"
                )
                continue

            for mh in MHS:
                path = _mh_file(directory, mh, instance_name, int(indice))
                if path is None:
                    _warn(
                        f"{config_id}: missing {mh} JSON for "
                        f"{instance_name}[{indice}] in {directory}"
                    )
                    continue
                try:
                    with path.open(encoding="utf-8") as handle:
                        data = json.load(handle)
                except (OSError, json.JSONDecodeError) as exc:
                    _warn(f"{config_id}: cannot read {path}: {exc}")
                    continue
                values_by_mh[mh].extend(_fitness_values(data))

    for mh, values in values_by_mh.items():
        if values:
            metrics[mh] = statistics.fmean(values)
        else:
            _warn(f"{config_id}: no usable fitness values for {mh}")
    return metrics


def _format_value(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_delta(
    values: Dict[str, Optional[float]], base: Dict[str, Optional[float]]
) -> str:
    gaps = [
        values[mh] - base[mh]
        for mh in MHS
        if values[mh] is not None and base[mh] is not None
    ]
    if not gaps:
        return "—"
    return f"{statistics.fmean(gaps):+.3f}"


def _config_id_for(param: str, value: Any) -> str:
    if value == PAPER_BASE[param]:
        return "base"
    return f"{param}={value}"


def _markdown_tables(
    configs: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Optional[float]]],
    campaign_id: str,
) -> str:
    base_metrics = metrics.get("base", {mh: None for mh in MHS})
    lines = [
        f"# OAT sensitivity results — campaign `{campaign_id}`",
        "",
        "Mean best fitness is computed over all available epochs and selected "
        "indices for each MH. `Δ vs base` is the signed absolute gap averaged "
        "over the available MHs; positive values indicate higher fitness.",
        "",
    ]

    for param, values in GRID.items():
        lines.extend(
            [
                f"## `{param}`",
                "",
                "| value | PSO | GA | GWO | DE | Δ vs base |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for value in values:
            config_id = _config_id_for(param, value)
            row_metrics = metrics.get(
                config_id, {mh: None for mh in MHS}
            )
            label = f"{value}"
            if value == PAPER_BASE[param]:
                label += "*"
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        *(_format_value(row_metrics[mh]) for mh in MHS),
                        _format_delta(row_metrics, base_metrics),
                    ]
                )
                + " |"
            )
        lines.extend(["", "\\* Paper value.", ""])

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    campaign_dir = _campaign_dir(args.campaign)
    if campaign_dir is None:
        print("ERROR: no sensitivity campaigns found", file=sys.stderr)
        return 1
    if not campaign_dir.is_dir():
        print(f"ERROR: campaign directory not found: {campaign_dir}", file=sys.stderr)
        return 1

    try:
        manifest = _load_manifest(campaign_dir)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {campaign_dir / 'configs.json'}: {exc}", file=sys.stderr)
        return 1

    configs = manifest.get("configs", [])
    if not configs:
        print(f"ERROR: no configurations in {campaign_dir / 'configs.json'}", file=sys.stderr)
        return 1

    metrics = {
        config["config_id"]: _config_metrics(config, manifest)
        for config in configs
    }
    markdown = _markdown_tables(
        configs,
        metrics,
        str(
            manifest.get(
                "campaign_id", campaign_dir.name[len("campaign_") :]
            )
        ),
    )
    output_path = campaign_dir / "tabla_oat.md"
    try:
        output_path.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot write {output_path}: {exc}", file=sys.stderr)
        return 1

    print(markdown, end="")
    print(f"Wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
