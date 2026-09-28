"""Execute isolated, provenance-checked rotational sensitivity campaigns."""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import warnings

from sensitivity.design import (
    REPO_ROOT, canonical_json, expand_tasks, load_protocol, protocol_hash,
    validate_protocol,
)


SOURCE_DIRS = ("hybrid_mkp", "continuous_benchmark", "HRES2-H2", "mkp_core", "mh", "lb2", "woa_abc")
POSTPROCESSING = {"analysis.py", "plots.py"}


def provenance() -> dict:
    """Hash local execution sources, including untracked new tooling.

    Manuscript, documentation, tests, configs and postprocessing are excluded.
    The exact protocol is hashed separately. No remote git operation is used.
    """
    sources = {REPO_ROOT / "dtw_stagnation.py"}
    for directory in SOURCE_DIRS:
        sources.update((REPO_ROOT / directory).rglob("*.py"))
    sources.update(path for path in (REPO_ROOT / "sensitivity").rglob("*.py")
                   if path.name not in POSTPROCESSING)
    entries = {}
    for path in sorted(sources):
        if path.is_file() and "__pycache__" not in path.parts:
            entries[path.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    sha, git_error = None, None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True)
        sha = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        git_error = str(exc)
    versions = {}
    for package in ("numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "git_sha": sha, "git_error": git_error,
            "source_files": entries,
            "source_fingerprint": hashlib.sha256(canonical_json(entries).encode()).hexdigest()}


def _output_path(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    allowed = (REPO_ROOT / "resultados" / "sensitivity").resolve()
    if (not allowed.is_relative_to(REPO_ROOT.resolve()) or output == allowed
            or not output.is_relative_to(allowed)):
        raise ValueError(f"Campaign output must be a subdirectory of {allowed}")
    return output


def input_fingerprints(protocol: dict) -> dict:
    """Record local data bytes, independently of source/protocol identity."""
    paths = set()
    for domain in protocol["domains"]:
        if domain["domain"] == "mkp":
            paths.update(REPO_ROOT / case["file"] for case in domain["cases"])
        elif domain["domain"] == "cec":
            # Include bundled data, not remote data and never trigger downloads.
            paths.update((REPO_ROOT / "continuous_benchmark" / "input_data").glob("*.txt"))
    return {path.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _atomic_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        # fsync protects completed records if a process or host is interrupted.
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def _campaign_lock(output: Path):
    lock = output / ".campaign.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            import os
            handle.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise ValueError(f"Campaign is locked: {lock}. Verify no worker is active before removing it.") from exc
    try:
        yield
    finally:
        lock.unlink()


def load_records(output_dir: str | Path) -> list[dict]:
    """Load only committed records; reject corrupted final records."""
    directory = Path(output_dir) / "records"
    temporary = list(directory.glob("*.json.tmp"))
    if temporary:
        warnings.warn(f"{len(temporary)} incomplete temporary record(s); not counted as completed", RuntimeWarning)
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or record.get("status") not in ("success", "failed")
                    or record.get("task_id") != path.stem):
                raise ValueError("invalid task identity or status")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Corrupt committed record: {path}: {exc}") from exc
        records.append(record)
    return records


def _export(output: Path, records: list[dict]) -> None:
    """CSV/JSONL are replaceable exports; per-task JSON files are authoritative."""
    records = sorted(records, key=lambda row: row["task_id"])
    fields = sorted({key for record in records for key in record})
    temporary = output / "runs.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list, bool))
                             else value for key, value in record.items()})
    temporary.replace(output / "runs.csv")
    temporary = output / "runs.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    temporary.replace(output / "runs.jsonl")


def _validate_result(result: object) -> None:
    if not isinstance(result, dict):
        raise ValueError("Adapter result must be an object")
    required = {"metric", "metric_name", "objective", "feasible", "actual_iters",
                "n_epochs", "n_switches", "elapsed_seconds", "effective_monitor", "solution"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Adapter result lacks {sorted(missing)}")
    if type(result["feasible"]) is not bool:
        raise ValueError("Adapter feasibility must be boolean")
    if not isinstance(result["metric_name"], str) or not result["metric_name"]:
        raise ValueError("Adapter metric_name must be nonempty")
    for field in ("metric", "objective", "elapsed_seconds"):
        value = result[field]
        if field == "metric" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Adapter {field} must be numeric")
    for field in ("actual_iters", "n_epochs", "n_switches"):
        if type(result[field]) is not int or result[field] < 0:
            raise ValueError(f"Adapter {field} must be a nonnegative integer")
    if result["elapsed_seconds"] < 0 or not isinstance(result["effective_monitor"], dict):
        raise ValueError("Invalid elapsed_seconds/effective_monitor")
    canonical_json(result)  # Reject NaN, infinity and non-JSON-native values.


def run_campaign(protocol: dict, output_dir: str | Path, *, resume: bool = False,
                 limit: int | None = None, adapter=None) -> list[dict]:
    """Run pending tasks sequentially; failures are recorded and retried on resume."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be positive")
    protocol = validate_protocol(protocol)
    tasks = expand_tasks(protocol)
    task_map = {task["task_id"]: task for task in tasks}
    output = _output_path(output_dir)
    manifest_path = output / "manifest.json"
    current_provenance = provenance()
    current_inputs = input_fingerprints(protocol)
    if resume:
        if not manifest_path.is_file():
            raise ValueError("Cannot resume without manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration_hash") != protocol_hash(protocol):
            raise ValueError("Cannot resume: protocol changed")
        if manifest.get("protocol") != protocol or manifest.get("task_count") != len(tasks):
            raise ValueError("Cannot resume: manifest protocol/task count corrupted")
        if manifest.get("provenance") != current_provenance:
            raise ValueError("Cannot resume: execution source/environment provenance changed")
        if manifest.get("input_fingerprints") != current_inputs:
            raise ValueError("Cannot resume: local scientific input data changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output already contains files; use --resume or a new campaign directory")
        manifest = {"schema_version": 1, "protocol": protocol, "task_count": len(tasks),
                    "configuration_hash": protocol_hash(protocol), "provenance": current_provenance,
                    "input_fingerprints": current_inputs,
                    "created_at": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=True)
    with _campaign_lock(output):
        if not resume:
            _atomic_json(manifest_path, manifest)
        (output / "records").mkdir(exist_ok=True)
        stored = load_records(output)
        records = {}
        for record in stored:
            task = task_map.get(record["task_id"])
            if task is None or any(record.get(key) != value for key, value in task.items()):
                raise ValueError("Saved record metadata does not match the protocol")
            if record["status"] == "success":
                _validate_result(record)
            records[record["task_id"]] = record
        pending = [task for task in tasks if records.get(task["task_id"], {}).get("status") != "success"]
        if limit is not None:
            pending = pending[:limit]
        if pending and adapter is None:
            from sensitivity.adapters import run_case
            adapter = run_case
        for task in pending:
            try:
                result = adapter(task["domain"], task["case"], task["monitor"],
                                 task["seed"], task["budget"], max_epoch_iters=task["max_epoch_iters"])
                _validate_result(result)
                if set(result) & (set(task) | {"status", "error", "traceback"}):
                    raise ValueError("Adapter attempted to overwrite task metadata")
                record = {**task, **result, "status": "success"}
            except Exception as exc:
                record = {**task, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}
            _atomic_json(output / "records" / f"{task['task_id']}.json", record)
            records[task["task_id"]] = record
            _export(output, list(records.values()))
        _export(output, list(records.values()))
        manifest.update(successful_tasks=sum(r["status"] == "success" for r in records.values()),
                        failed_tasks=sum(r["status"] == "failed" for r in records.values()),
                        pending_tasks=len(tasks) - sum(r["status"] == "success" for r in records.values()),
                        updated_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(manifest_path, manifest)
        return list(records.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol(args.config)
        tasks = expand_tasks(protocol)
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.dry_run:
            print(json.dumps({"name": protocol["name"], "task_count": len(tasks),
                              "configuration_hash": protocol_hash(protocol),
                              "domains": sorted({task["domain"] for task in tasks}),
                              "budget_unit": "orchestrator iterations, not objective evaluations",
                              "execution_limit": args.limit}, indent=2))
            return 0
        output = args.output_dir or REPO_ROOT / "resultados" / "sensitivity" / protocol["name"]
        records = run_campaign(protocol, output, resume=args.resume, limit=args.limit)
        failed = sum(row["status"] == "failed" for row in records)
        success = sum(row["status"] == "success" for row in records)
        print(f"{output}: {success}/{len(tasks)} successful, {failed} failed; remaining tasks are pending")
        return 1 if failed else 0
    except (ValueError, OSError) as exc:
        print(f"Campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
