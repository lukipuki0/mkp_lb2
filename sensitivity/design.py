"""Validate pilot protocols and expand paired one-factor-at-a-time tasks.

This module deliberately imports no numerical engines. Planning is inexpensive
and does not mutate solver defaults or create campaign output.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR_KEYS = {
    "window", "band_ratio", "min_slope", "plateau_max", "patience",
    "adapt_thresholds", "p_low", "p_high", "improvement_tol",
}
DEFAULT_MONITOR = {
    "window": 40, "band_ratio": 0.1, "min_slope": 0.0,
    "plateau_max": 15, "patience": 8, "adapt_thresholds": True,
    "p_low": 30.0, "p_high": 70.0, "improvement_tol": 0.0,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _keys(value: dict, allowed: set, where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {sorted(unknown)}")


def _integer(value: object, where: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")


def _number(value: object, where: str, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{where} must be finite and >= {minimum}")


def _identifier(value: object, where: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{where} must be a safe nonempty identifier")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}
    if value.endswith(".") or value.split(".")[0].upper() in reserved:
        raise ValueError(f"{where} cannot be a reserved Windows filename")


def _monitor(requested: dict) -> dict:
    if not isinstance(requested, dict):
        raise ValueError("baseline must be an object")
    _keys(requested, MONITOR_KEYS, "monitor")
    monitor = {**DEFAULT_MONITOR, **requested}
    for key in ("window", "plateau_max", "patience"):
        _integer(monitor[key], key, 2 if key == "window" else 1)
    for key in ("min_slope", "improvement_tol", "p_low", "p_high"):
        _number(monitor[key], key)
    _number(monitor["band_ratio"], "band_ratio")
    if not 0 < monitor["band_ratio"] <= 1:
        raise ValueError("band_ratio must be in (0, 1]")
    if type(monitor["adapt_thresholds"]) is not bool:
        raise ValueError("adapt_thresholds must be boolean")
    if not 0 <= monitor["p_low"] < monitor["p_high"] <= 100:
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 100")
    for key in ("band_ratio", "min_slope", "improvement_tol", "p_low", "p_high"):
        monitor[key] = float(monitor[key])
    return monitor


def resolve_monitor(requested: dict, mode: str) -> dict:
    resolved = _monitor(requested)
    ratio = resolved.pop("band_ratio")
    # Match legacy StagnationConfig's positive integer band convention.
    resolved["band"] = max(1, int(resolved["window"] * ratio))
    resolved["use_ddtw"] = mode == "ddtw"
    return resolved


def _case(case: dict, domain: str) -> None:
    if not isinstance(case, dict):
        raise ValueError("Each case must be an object")
    allowed = {"id"}
    if domain == "mkp":
        allowed |= {"file", "index", "reference_value", "reference_source"}
    elif domain == "cec":
        allowed |= {"function", "dimension"}
    _keys(case, allowed, f"{domain} case")
    _identifier(case.get("id"), "case.id")
    if domain == "mkp":
        path = case.get("file")
        if not isinstance(path, str) or not path or Path(path).is_absolute() or "://" in path:
            raise ValueError("MKP case.file must be a repository-relative path")
        candidate = (REPO_ROOT / path).resolve()
        if not candidate.is_relative_to(REPO_ROOT.resolve()) or not candidate.is_file():
            raise ValueError("MKP case.file must identify a file inside this repository")
        _integer(case.get("index"), "index", 0)
        with candidate.open(encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), "")
        try:
            count = int(first_line.strip())
        except ValueError as exc:
            raise ValueError("MKP case.file lacks an OR-Library instance-count header") from exc
        if case["index"] >= count:
            raise ValueError("MKP index exceeds the file's instance count")
        if "reference_value" in case:
            _number(case["reference_value"], "reference_value")
            if case["reference_value"] <= 0 or not case.get("reference_source"):
                raise ValueError("A positive reference_value requires reference_source")
        if "reference_source" in case and not isinstance(case["reference_source"], str):
            raise ValueError("reference_source must be a string")
    elif domain == "cec":
        _integer(case.get("function"), "function")
        if (case["function"] > 12 or type(case.get("dimension")) is not int
                or case["dimension"] not in (10, 20)):
            raise ValueError("CEC2022 requires function 1..12 and dimension 10 or 20")


def validate_protocol(protocol: dict) -> dict:
    """Reject ambiguous/invalid protocols; return an independent copy."""
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    _keys(protocol, {"schema_version", "name", "notes", "seeds", "modes",
                     "max_epoch_iters", "domains"}, "protocol")
    if type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    _identifier(protocol.get("name"), "name")
    if "notes" in protocol and not isinstance(protocol["notes"], str):
        raise ValueError("notes must be a string")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Explicit nonempty seeds are required")
    for seed in seeds:
        _integer(seed, "seed", 0)
        if seed > 2**32 - 1:
            raise ValueError("seed must fit numpy's unsigned 32-bit range")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    modes = protocol.get("modes")
    if not isinstance(modes, list) or not modes or any(m not in ("dtw", "ddtw") for m in modes):
        raise ValueError("modes must contain dtw and/or ddtw")
    if len(modes) != len(set(modes)):
        raise ValueError("modes must be unique")
    cap = protocol.get("max_epoch_iters")
    if cap is not None:
        _integer(cap, "max_epoch_iters")
    domains = protocol.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("domains must be a nonempty list")
    identities = set()
    for spec in domains:
        if not isinstance(spec, dict):
            raise ValueError("Domain specifications must be objects")
        _keys(spec, {"domain", "budget", "baseline", "cases", "factors"}, "domain")
        domain = spec.get("domain")
        if domain not in ("mkp", "cec", "hres"):
            raise ValueError("domain must be mkp, cec or hres")
        _integer(spec.get("budget"), "budget")
        identity = (domain, spec["budget"])
        if identity in identities:
            raise ValueError("Duplicate domain/budget specification")
        identities.add(identity)
        baseline = _monitor(spec.get("baseline"))
        cases = spec.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("cases must be a nonempty list")
        for case in cases:
            _case(case, domain)
        if len({case["id"] for case in cases}) != len(cases):
            raise ValueError("case IDs must be unique within a domain/budget")
        factors = spec.get("factors", {})
        if not isinstance(factors, dict):
            raise ValueError("factors must be an object")
        _keys(factors, (MONITOR_KEYS - {"p_low", "p_high"}) | {"percentiles"}, "factors")
        for factor, levels in factors.items():
            if not isinstance(levels, list) or not levels:
                raise ValueError(f"{factor} requires a nonempty level list")
            for level in levels:
                _monitor(_override(baseline, factor, level))
    return copy.deepcopy(protocol)


def _override(baseline: dict, factor: str, level: object) -> dict:
    requested = {**baseline}
    if factor == "percentiles":
        if not isinstance(level, list) or len(level) != 2:
            raise ValueError("Each percentile level must be a [low, high] pair")
        requested.update(p_low=level[0], p_high=level[1])
    else:
        requested[factor] = level
    return requested


def load_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return validate_protocol(json.load(handle))


def protocol_hash(protocol: dict) -> str:
    return digest(validate_protocol(protocol))


def expand_tasks(protocol: dict) -> list[dict]:
    """Resolve and deduplicate OAT settings, preserving identical paired seeds."""
    protocol = validate_protocol(protocol)
    tasks = []
    for spec in protocol["domains"]:
        baseline = _monitor(spec["baseline"])
        variants = [(baseline, "baseline", None, True)]
        for factor in sorted(spec.get("factors", {})):
            for level in spec["factors"][factor]:
                variants.append((_override(baseline, factor, level), factor, level, False))
        for mode in protocol["modes"]:
            seen = set()
            for requested, factor, level, is_baseline in variants:
                resolved = resolve_monitor(requested, mode)
                config_id = digest(resolved)[:16]
                if config_id in seen:
                    continue
                seen.add(config_id)
                for case in spec["cases"]:
                    for seed in protocol["seeds"]:
                        task = dict(domain=spec["domain"], case=copy.deepcopy(case), mode=mode,
                                    config_id=config_id, monitor=resolved.copy(),
                                    requested_monitor=copy.deepcopy(requested), factor=factor,
                                    level=copy.deepcopy(level), is_baseline=is_baseline,
                                    budget=spec["budget"], seed=seed,
                                    max_epoch_iters=protocol.get("max_epoch_iters"))
                        # Scientific identity excludes descriptive factor labels.
                        identity = {k: task[k] for k in ("domain", "case", "config_id", "budget",
                                                       "seed", "max_epoch_iters")}
                        task["task_id"] = digest(identity)[:24]
                        tasks.append(task)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Protocol expansion produced duplicate tasks")
    return tasks
