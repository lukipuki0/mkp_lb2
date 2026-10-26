"""Best-feasible references for the bundled OR-Library mknapcb instances.

The published values are best feasible values found, not certified optima.
Only use them when the local instance file hash matches the bundled benchmark.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


_SOURCE = Path(__file__).with_name("data") / "mkcbres.txt"
_FAMILY_HASHES = {
    1: "05e733693d321e296ee0f0ec2bd045688b924313a883d1015f43bdef00417fd6",
    2: "3ca68e281b0368617849c8346575a18f04ec699ab047f28c331f86e3c296a8c9",
    3: "56ed69e1826ae5d339838d876044e44ec26f65b131826ed09da787357a42bcf1",
    4: "96e2e238d232063e5dee94590e5935fe07d2c4626dbc8755e2bf604495424dee",
    5: "a3c7f08a826ed0150f3fa0f3d15d7829326d63f4828b7bcded1d9a172b2ab116",
    6: "014a584817b460221244d7c513378f62694f5eb4642a09fa1fc13aebf0fdce31",
    7: "06c270fb2e2ba331862f51c6ae3a6ce5341287211214b7149e90cad56dd57a81",
    8: "4967487248b59de82afd7f5e4768ce5dde3946427c242c459c24fccd50c15d24",
    9: "8b03881b6c98c99aadd4dc96d288f61bf748058e34e2f80e6b7905ae1f6135bf",
}
_FAMILY_SHAPE = {
    1: (5, 100), 2: (5, 250), 3: (5, 500),
    4: (10, 100), 5: (10, 250), 6: (10, 500),
    7: (30, 100), 8: (30, 250), 9: (30, 500),
}
_ROW = re.compile(r"^(\d+)\.(\d+)-(\d+)\s+(\d+)\s*$")


def _references():
    values = {}
    for line in _SOURCE.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line.strip())
        if not match:
            if values and "LP optimal" in line:
                break
            continue
        constraints, items, index, value = map(int, match.groups())
        values[(constraints, items, index)] = value
    if len(values) != 270:
        raise ValueError(f"Expected 270 OR-Library best-feasible references, found {len(values)}")
    return values


_REFERENCES = _references()


def lookup_best_feasible(family: int, index: int, instance_file_sha256: str) -> int | None:
    """Return the published best-feasible value only for the exact bundled file."""
    if _FAMILY_HASHES.get(family) != instance_file_sha256 or not 0 <= index < 30:
        return None
    constraints, items = _FAMILY_SHAPE[family]
    return _REFERENCES.get((constraints, items, index))


def reference_source() -> str:
    return "OR-Library mkcbres.txt: best feasible value found (not certified optimum)"
