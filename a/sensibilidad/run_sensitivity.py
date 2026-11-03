"""Run the Binary-Complex one-factor-at-a-time sensitivity campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .grid import ENV_MAP, PAPER_BASE, build_oat_configs


DEFAULT_INSTANCE = "instances/mknapcb1.txt"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Binary-Complex one-factor-at-a-time sensitivity campaign"
    )
    parser.add_argument(
        "--instancias",
        nargs="+",
        default=[DEFAULT_INSTANCE],
        help=f"Instance files (default: {DEFAULT_INSTANCE})",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0, 15, 29],
        help="Instance indices (default: 0 15 29)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=31,
        help="Epochs per Binary-Complex run (default: 31)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Maximum workers passed to run_all_hpc.py",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help=(
            "Resume an existing campaign id (loads its manifest and reuses "
            "per-config campaign ids); combined with --desde-config/--solo, "
            "or defaults to running only incomplete configs"
        ),
    )
    parser.add_argument(
        "--desde-config",
        default=None,
        help="Resume at this config_id, including that configuration",
    )
    parser.add_argument(
        "--solo",
        default=None,
        help="Run only this config_id",
    )
    return parser.parse_args(argv)


def _select_configs(
    configs: List[Dict[str, Any]],
    desde_config: Optional[str],
    solo: Optional[str],
) -> List[Dict[str, Any]]:
    """Apply inclusive resume and exact single-configuration filters."""
    selected = configs
    if desde_config is not None:
        config_ids = [config["config_id"] for config in configs]
        try:
            start = config_ids.index(desde_config)
        except ValueError as exc:
            raise ValueError(
                f"Unknown --desde-config value: {desde_config!r}"
            ) from exc
        selected = configs[start:]

    if solo is not None:
        if solo not in {config["config_id"] for config in configs}:
            raise ValueError(f"Unknown --solo value: {solo!r}")
        selected = [config for config in selected if config["config_id"] == solo]

    if not selected:
        raise ValueError("The selected filters leave no configurations to run")
    return selected


def _load_existing_manifest(campaign_dir: Path) -> Optional[Dict[str, Any]]:
    """Load an existing campaign manifest, or None when it does not exist."""
    manifest_path = campaign_dir / "configs.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _config_is_complete(
    config: Dict[str, Any], manifest: Dict[str, Any]
) -> bool:
    """True when every (instancia, indice) result dir has the four MH JSONs."""
    per_config_id = config.get("per_config_campaign_id")
    if not per_config_id:
        return False
    instancias = manifest.get("instancias", [])
    indices = manifest.get("indices", [])
    if not instancias or not indices:
        # Nothing verifiable: treat as incomplete so a resume never skips work.
        return False
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
                return False
            for mh in ("PSO", "GA", "GWO", "DE"):
                if not list(directory.glob(f"{mh}_*.json")):
                    return False
    return True


def _campaign_config(
    config: Dict[str, Any], campaign_id: str, position: int
) -> Dict[str, Any]:
    """Copy a grid config and attach its safe per-config campaign identifier."""
    manifest_config = dict(config)
    manifest_config["overrides"] = dict(config["overrides"])
    manifest_config["config_index"] = position
    manifest_config["per_config_campaign_id"] = f"{campaign_id}_{position:02d}"
    return manifest_config


def _build_command(
    instancia: str, indice: int, epochs: int, cpus: Optional[int]
) -> List[str]:
    command = [
        sys.executable,
        "run_all_hpc.py",
        "--instancia",
        instancia,
        "--indice",
        str(indice),
        "--epochs",
        str(epochs),
        "--skip",
        "vanilla_explotacion",
        "vanilla_exploracion",
        "binary_simple",
        "binary_patient",
        "--no-stats",
    ]
    if cpus is not None:
        command.extend(["--cpus", str(cpus)])
    return command


def _environment(config: Dict[str, Any], per_config_campaign_id: str) -> Dict[str, str]:
    """Build an explicit environment so each run is independent of local config."""
    environment = os.environ.copy()
    for param, env_name in ENV_MAP.items():
        environment[env_name] = str(PAPER_BASE[param])
    for param, value in config["overrides"].items():
        environment[ENV_MAP[param]] = str(value)
    environment["MKP_CAMPAIGN_ID"] = per_config_campaign_id
    return environment


def _write_manifest(
    campaign_dir: Path,
    campaign_id: str,
    instancias: List[str],
    indices: List[int],
    epochs: int,
    configs: List[Dict[str, Any]],
) -> Path:
    manifest_path = campaign_dir / "configs.json"
    manifest = {
        "campaign_id": campaign_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": "binary_complex",
        "instancias": instancias,
        "indices": indices,
        "epochs": epochs,
        "configs": configs,
    }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    all_configs = build_oat_configs()
    campaign_root = PROJECT_ROOT / "results" / "sensibilidad"

    if args.campaign:
        # Resume semantics: reuse the existing manifest and per-config
        # campaign ids so analizar.py can always compare against the base
        # configuration of the SAME campaign.
        campaign_id = args.campaign
        campaign_dir = campaign_root / f"campaign_{campaign_id}"
        manifest = _load_existing_manifest(campaign_dir)
        if manifest is None:
            print(f"ERROR: campaign not found: {campaign_dir}", file=sys.stderr)
            return 2
        manifest_configs = manifest.get("configs", [])
        if not manifest_configs:
            print("ERROR: manifest has no configurations", file=sys.stderr)
            return 2
        known_ids = {config["config_id"] for config in manifest_configs}
        expected_ids = {config["config_id"] for config in all_configs}
        if known_ids != expected_ids:
            print(
                "ERROR: manifest configs do not match the current grid; "
                "resume is only supported for the same grid",
                file=sys.stderr,
            )
            return 2
        try:
            selected_configs = _select_configs(
                manifest_configs, args.desde_config, args.solo
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.desde_config is None and args.solo is None:
            incomplete = [
                config
                for config in selected_configs
                if not _config_is_complete(config, manifest)
            ]
            skipped = len(selected_configs) - len(incomplete)
            if skipped:
                print(
                    f"  [RESUME] {skipped} already-complete config(s) skipped"
                )
            selected_configs = incomplete
        if not selected_configs:
            print("Nothing to run: every selected config is already complete.")
            return 0
        manifest_path = campaign_dir / "configs.json"
    else:
        campaign_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        campaign_dir = campaign_root / f"campaign_{campaign_id}"
        try:
            selected_configs = _select_configs(
                all_configs, args.desde_config, args.solo
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        selected_configs = [
            _campaign_config(config, campaign_id, position)
            for position, config in enumerate(selected_configs)
        ]
        manifest_path = _write_manifest(
            campaign_dir,
            campaign_id,
            list(args.instancias),
            list(args.indices),
            args.epochs,
            selected_configs,
        )

    print("=" * 72)
    print("  OAT SENSITIVITY CAMPAIGN — Binary-Complex")
    print("=" * 72)
    print(f"  Campaign:    {campaign_id}")
    print(f"  Mode:        {'resume' if args.campaign else 'new'}")
    print(f"  Configs:     {len(selected_configs)}")
    print(f"  Instances:   {', '.join(args.instancias)}")
    print(f"  Indices:     {', '.join(str(index) for index in args.indices)}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Manifest:     {manifest_path}")

    failures: List[Tuple[str, str, int, int]] = []
    total_runs = len(args.instancias) * len(args.indices)

    for position, config in enumerate(selected_configs):
        config_id = config["config_id"]
        per_config_campaign_id = config["per_config_campaign_id"]
        environment = _environment(config, per_config_campaign_id)
        config_failures = 0

        print("\n" + "-" * 72)
        print(
            f"  CONFIG {position + 1}/{len(selected_configs)}: {config_id} "
            f"(campaign {per_config_campaign_id})"
        )
        print(f"  Overrides: {config['overrides'] or 'paper base'}")
        print("-" * 72)

        for instancia in args.instancias:
            for indice in args.indices:
                command = _build_command(
                    instancia, indice, args.epochs, args.cpus
                )
                print(
                    f"  [RUN] {config_id} — "
                    f"{Path(instancia).stem}[{indice}]"
                )
                try:
                    result = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=environment,
                        check=False,
                    )
                    return_code = result.returncode
                except OSError as exc:
                    print(f"  [FAIL] Could not launch subprocess: {exc}")
                    return_code = 1

                if return_code != 0:
                    config_failures += 1
                    failures.append((config_id, instancia, indice, return_code))
                    print(
                        f"  [FAIL] {config_id} {instancia}[{indice}] "
                        f"(return code {return_code})"
                    )

        if config_failures:
            print(
                f"  [SUMMARY] {config_id}: "
                f"{total_runs - config_failures}/{total_runs} succeeded, "
                f"{config_failures} failed"
            )
        else:
            print(f"  [SUMMARY] {config_id}: {total_runs}/{total_runs} succeeded")

    print("\n" + "=" * 72)
    if failures:
        print(
            f"  OAT CAMPAIGN FAILED: {len(failures)} of "
            f"{len(selected_configs) * total_runs} runs failed"
        )
        for config_id, instancia, indice, return_code in failures:
            print(
                f"  - {config_id} {instancia}[{indice}]: "
                f"return code {return_code}"
            )
        return 1

    print(
        f"  OAT CAMPAIGN COMPLETE: "
        f"{len(selected_configs) * total_runs} runs succeeded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
