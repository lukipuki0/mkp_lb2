"""Joint parameter profile used by both sweeps of one generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    a: float
    F: float
    CR: float
    G3: float
