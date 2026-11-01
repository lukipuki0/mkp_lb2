"""Local OR-Library loader and greedy repair.

This parser is implemented here on purpose. The sequential package does not
import another optimizer's loader. Best-known values, when the instance header
is 0, come from ``mkcbres.txt`` beside the instance file.
"""

from pathlib import Path

import numpy as np

FAMILY_PREFIX = {
    "mknapcb1": "5.100", "mknapcb2": "5.250", "mknapcb3": "5.500",
    "mknapcb4": "10.100", "mknapcb5": "10.250", "mknapcb6": "10.500",
    "mknapcb7": "30.100", "mknapcb8": "30.250", "mknapcb9": "30.500",
}
BEST_KNOWN_FILE = "mkcbres.txt"


def parse_instances(text: str) -> list:
    """Parse OR-Library text into raw instance dicts."""
    rows = [list(map(float, line.split())) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty instance file")
    count = int(rows[0][0])
    ptr = 1
    parsed = []
    for _ in range(count):
        if ptr >= len(rows):
            raise ValueError("truncated instance file")
        n, m, optimum = rows[ptr]
        n, m = int(n), int(m)
        ptr += 1
        profits = _take(rows, ptr, n)
        ptr = profits[1]
        weights = []
        for _constraint in range(m):
            row, ptr = _take(rows, ptr, n)
            weights.append(row)
        capacities, ptr = _take(rows, ptr, m)
        parsed.append({
            "n": n,
            "m": m,
            "optimum": float(optimum),
            "p": np.asarray(profits[0], dtype=float),
            "r": np.asarray(weights, dtype=float),
            "b": np.asarray(capacities, dtype=float),
        })
    return parsed


def _take(rows, ptr, needed):
    values = []
    while len(values) < needed:
        if ptr >= len(rows):
            raise ValueError("truncated instance file")
        values.extend(rows[ptr])
        ptr += 1
    if len(values) != needed:
        raise ValueError("instance row does not match the declared size")
    return values, ptr


def density(profits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Item density min_i(p_j / r_ij). Infinite ratios become 10x the finite max."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = profits / weights
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return np.ones(profits.shape[0], dtype=float)
    cleaned = np.array(ratios, dtype=float, copy=True)
    cleaned[~np.isfinite(cleaned)] = float(finite.max()) * 10.0
    return cleaned.min(axis=0)


def read_best_known(path: Path) -> dict:
    found = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Problem Name" in line and found:
            break
        parts = line.split()
        if len(parts) == 2 and "-" in parts[0] and parts[0][:1].isdigit():
            found[parts[0]] = float(parts[1])
    return found


def best_known(path: Path, index: int) -> float:
    table = path.parent / BEST_KNOWN_FILE
    prefix = FAMILY_PREFIX.get(path.stem.lower())
    if prefix is None or not table.is_file():
        return 0.0
    return read_best_known(table).get(f"{prefix}-{index:02d}", 0.0)


def load_instance(path, index: int = 0) -> dict:
    """Load one 0-based instance. Header optimum 0 falls back to mkcbres, else 0."""
    file_path = Path(path)
    parsed = parse_instances(file_path.read_text(encoding="utf-8", errors="replace"))
    if not 0 <= index < len(parsed):
        raise ValueError(f"index {index} out of range ({len(parsed)} instances)")
    item = parsed[index]
    optimum = float(item["optimum"])
    if optimum <= 0:
        optimum = best_known(file_path, index)
    return {
        "n": item["n"],
        "m": item["m"],
        "p": item["p"],
        "r": item["r"],
        "b": item["b"],
        "optimo": optimum,
        "densidad": density(item["p"], item["r"]),
    }


def reparar(bits: np.ndarray, inst: dict):
    """Drop low-density items until feasible, then insert high-density items that fit."""
    chosen = np.array(bits, dtype=int, copy=True)
    profits, weights, capacity = inst["p"], inst["r"], inst["b"]
    order = np.argsort(inst["densidad"])
    used = weights @ chosen
    for item in order:
        if np.all(used <= capacity):
            break
        if chosen[item] == 1:
            chosen[item] = 0
            used = used - weights[:, item]
    for item in order[::-1]:
        if chosen[item] == 0 and np.all(used + weights[:, item] <= capacity):
            chosen[item] = 1
            used = used + weights[:, item]
    return chosen, float(chosen @ profits)
