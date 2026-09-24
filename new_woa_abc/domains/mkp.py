"""Dominio MKP Chu--Beasley autocontenido para ``new_woa_abc``.

El archivo de referencia ``DTW_optimization-main`` no se importa.  Este
módulo contiene el parser OR-Library, la tabla de mejores valores conocidos y
la reparación binaria necesaria para que toda solución evaluada sea factible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


# Mejores valores conocidos de las 270 instancias Chu--Beasley (mkcbres).
# Cada fila contiene las 30 instancias del archivo correspondiente.
KNOWN_OPTIMA: dict[str, tuple[float, ...]] = {
    "mknapcb1": (
        24381, 24274, 23551, 23534, 23991, 24613, 25591, 23410, 24216, 24411,
        42757, 42545, 41968, 45090, 42218, 42927, 42009, 45020, 43441, 44554,
        59822, 62081, 59802, 60479, 61091, 58959, 61538, 61520, 59453, 59965,
    ),
    "mknapcb2": (
        59312, 61472, 62130, 59446, 58951, 60056, 60414, 61472, 61885, 58959,
        109109, 109841, 108489, 109383, 110720, 110256, 109016, 109037, 109957, 107038,
        149659, 155940, 149316, 152130, 150353, 150045, 148607, 149772, 155075, 154662,
    ),
    "mknapcb3": (
        120130, 117837, 121109, 120798, 122319, 122007, 119113, 120568, 121575, 120699,
        218422, 221191, 217534, 223558, 218962, 220514, 219987, 218194, 216976, 219693,
        295828, 308077, 299796, 306476, 300342, 302560, 301322, 306430, 302814, 299904,
    ),
    "mknapcb4": (
        23064, 22801, 22131, 22772, 22751, 22777, 21875, 22635, 22511, 22702,
        41395, 42344, 42401, 45624, 41884, 42995, 43559, 42970, 42212, 41207,
        57375, 58978, 58391, 61966, 60803, 61437, 56377, 59391, 60205, 60633,
    ),
    "mknapcb5": (
        59187, 58662, 58094, 61000, 58092, 58803, 58607, 58917, 59384, 59193,
        110863, 108659, 108932, 110037, 108423, 110841, 106075, 106686, 109825, 106723,
        151790, 148772, 151900, 151275, 151948, 152109, 153131, 153520, 149155, 149704,
    ),
    "mknapcb6": (
        117726, 119139, 119159, 118802, 116434, 119454, 119749, 118288, 117779, 119125,
        217318, 219022, 217772, 216802, 213809, 215013, 217896, 219949, 214332, 220833,
        304344, 302332, 302354, 300743, 304344, 301730, 304949, 296437, 301313, 307014,
    ),
    "mknapcb7": (
        21946, 21716, 20754, 21464, 21814, 22176, 21799, 21397, 22493, 20983,
        40767, 41304, 41560, 41041, 40872, 41058, 41062, 42719, 42230, 41700,
        57494, 60027, 58025, 60776, 58884, 60011, 58132, 59064, 58975, 60603,
    ),
    "mknapcb8": (
        56693, 58318, 56553, 56863, 56629, 57119, 56292, 56403, 57442, 56447,
        107689, 108338, 106385, 106796, 107396, 107246, 106308, 103993, 106835, 105751,
        150083, 149907, 152993, 153169, 150287, 148544, 147471, 152841, 149568, 149572,
    ),
    "mknapcb9": (
        115868, 114667, 116661, 115237, 116353, 115604, 113952, 114199, 115247, 116947,
        217995, 214534, 215854, 217836, 215566, 215762, 215772, 216336, 217290, 214624,
        301627, 299985, 304995, 301935, 304404, 296894, 303233, 306944, 303057, 300460,
    ),
}


def known_optimum(family: str, index: int) -> float | None:
    """Devuelve el valor de mkcbres o ``None`` si la instancia es externa."""

    values = KNOWN_OPTIMA.get(Path(family).stem.lower())
    if values is None or not 0 <= index < len(values):
        return None
    return float(values[index])


def instance_group_name(item_count: int) -> str:
    """Nombre de grupo por tamaño, usado para reunir resultados MKP."""

    if item_count < 1:
        raise ValueError("item_count debe ser positivo")
    return f"grupo_{int(item_count)}_items"


@dataclass
class MKPInstance:
    """Una instancia binaria de maximización con restricciones múltiples."""

    family: str
    index: int
    profits: NDArray[np.float64]
    weights: NDArray[np.float64]
    capacities: NDArray[np.float64]
    best_known: float | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        self.family = str(self.family).lower()
        self.profits = np.asarray(self.profits, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)
        if self.profits.ndim != 1:
            raise ValueError("profits debe ser un vector")
        if self.weights.ndim != 2 or self.weights.shape[1] != self.profits.size:
            raise ValueError("weights debe tener forma (restricciones, items)")
        if self.capacities.shape != (self.weights.shape[0],):
            raise ValueError("capacities no coincide con weights")
        if np.any(self.profits < 0) or np.any(self.weights < 0):
            raise ValueError("MKP requiere ganancias y pesos no negativos")
        if np.any(self.capacities <= 0):
            raise ValueError("todas las capacidades deben ser positivas")

        normalized_weight = self.weights / self.capacities[:, None]
        mean_consumption = np.mean(normalized_weight, axis=0)
        self.efficiency = self.profits / np.maximum(mean_consumption, 1e-12)
        self.drop_order = np.argsort(self.efficiency, kind="stable")
        self.add_order = self.drop_order[::-1]

    @property
    def name(self) -> str:
        return f"{self.family}_inst{self.index:02d}"

    @property
    def dimension(self) -> int:
        return int(self.profits.size)

    @property
    def constraints(self) -> int:
        return int(self.capacities.size)

    def evaluate(self, solution: Sequence[int] | NDArray[np.integer]) -> float:
        bits = np.asarray(solution, dtype=np.int8)
        if bits.shape != (self.dimension,):
            raise ValueError(f"la solución debe tener {self.dimension} bits")
        return float(self.profits @ bits)

    def resource_use(self, solution: Sequence[int] | NDArray[np.integer]) -> NDArray[np.float64]:
        bits = np.asarray(solution, dtype=np.int8)
        return np.asarray(self.weights @ bits, dtype=float)

    def is_feasible(self, solution: Sequence[int] | NDArray[np.integer]) -> bool:
        bits = np.asarray(solution, dtype=np.int8)
        return bool(
            bits.shape == (self.dimension,)
            and np.all((bits == 0) | (bits == 1))
            and np.all(self.resource_use(bits) <= self.capacities + 1e-9)
        )

    def gap(self, value: float) -> float | None:
        if self.best_known is None or self.best_known == 0:
            return None
        return 100.0 * (self.best_known - float(value)) / self.best_known

    def _priority_order(
        self,
        priority: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if priority is None:
            return self.drop_order, self.add_order
        scores = np.asarray(priority, dtype=float)
        if scores.shape != (self.dimension,):
            raise ValueError("priority debe tener un valor por ítem")
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max)
        drop = np.argsort(scores, kind="stable")
        return drop, drop[::-1]

    def repair(
        self,
        solution: Sequence[int] | NDArray[np.integer],
        priority: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.int8], float]:
        """Expulsa ítems hasta factibilidad y luego completa codiciosamente.

        ``priority`` permite que la posición latente de WOA--ABC cambie el
        orden sin abandonar la densidad normalizada del problema.
        """

        bits = (np.asarray(solution) > 0).astype(np.int8, copy=True)
        if bits.shape != (self.dimension,):
            raise ValueError(f"la solución debe tener {self.dimension} elementos")
        drop, add = self._priority_order(priority)
        usage = self.weights @ bits

        for item in drop:
            if np.all(usage <= self.capacities + 1e-12):
                break
            if bits[item]:
                bits[item] = 0
                usage -= self.weights[:, item]

        if np.any(usage > self.capacities + 1e-9):
            raise RuntimeError("la fase de expulsión no pudo reparar la solución")

        for item in add:
            if not bits[item] and np.all(
                usage + self.weights[:, item] <= self.capacities + 1e-12
            ):
                bits[item] = 1
                usage += self.weights[:, item]

        return bits, float(self.profits @ bits)

    def greedy_solution(
        self,
        rng: np.random.Generator | None = None,
        noise: float = 0.0,
    ) -> tuple[NDArray[np.int8], float]:
        """Construye una solución maximal; ``noise`` aporta diversidad."""

        priority = self.efficiency.copy()
        if rng is not None and noise > 0:
            priority *= np.exp(np.clip(rng.normal(0.0, noise, self.dimension), -3.0, 3.0))
        return self.repair(np.zeros(self.dimension, dtype=np.int8), priority)

    def improve(
        self,
        solution: Sequence[int] | NDArray[np.integer],
        rng: np.random.Generator,
        passes: int = 2,
        ejection_candidates: int = 15,
        exchange_depth: int = 3,
    ) -> tuple[NDArray[np.int8], float, int]:
        """Pulido factible por intercambios acotados dentro de un core.

        Se usan los seleccionados de menor densidad y los no seleccionados de
        mayor densidad. Todas las combinaciones hasta exchange_depth se
        comprueban de forma vectorizada. Es búsqueda local del élite, no otra
        metaheurística.
        """

        current, current_value = self.repair(solution)
        evaluations = 1
        if passes <= 0 or ejection_candidates <= 0 or exchange_depth <= 0:
            return current, current_value, evaluations

        for _ in range(passes):
            selected = np.flatnonzero(current)
            unselected = np.flatnonzero(1 - current)
            if selected.size == 0 or unselected.size == 0:
                break

            removal_pool = selected[
                np.argsort(self.efficiency[selected], kind="stable")
            ][: min(ejection_candidates, selected.size)]
            addition_pool = unselected[
                np.argsort(-self.efficiency[unselected], kind="stable")
            ][: min(ejection_candidates, unselected.size)]
            removal_sets = [
                tuple(int(item) for item in group)
                for size in range(1, min(exchange_depth, removal_pool.size) + 1)
                for group in combinations(removal_pool.tolist(), size)
            ]
            addition_sets = [
                tuple(int(item) for item in group)
                for size in range(1, min(exchange_depth, addition_pool.size) + 1)
                for group in combinations(addition_pool.tolist(), size)
            ]
            if not removal_sets or not addition_sets:
                break

            addition_weights = np.asarray(
                [np.sum(self.weights[:, group], axis=1) for group in addition_sets],
                dtype=float,
            )
            addition_profits = np.asarray(
                [np.sum(self.profits[list(group)]) for group in addition_sets],
                dtype=float,
            )
            usage = self.resource_use(current)
            best_value = current_value
            best_exchange: tuple[tuple[int, ...], tuple[int, ...]] | None = None

            # La permutación solo resuelve empates de forma reproducible.
            for removal_index in rng.permutation(len(removal_sets)):
                removed = removal_sets[int(removal_index)]
                remaining_usage = usage - np.sum(self.weights[:, removed], axis=1)
                available = self.capacities - remaining_usage
                feasible = np.all(addition_weights <= available + 1e-12, axis=1)
                evaluations += int(np.count_nonzero(feasible))
                if not feasible.any():
                    continue
                removed_profit = float(np.sum(self.profits[list(removed)]))
                values = current_value - removed_profit + addition_profits
                values = np.where(feasible, values, -np.inf)
                addition_index = int(np.argmax(values))
                value = float(values[addition_index])
                if value > best_value + 1e-12:
                    best_value = value
                    best_exchange = (removed, addition_sets[addition_index])

            if best_exchange is None:
                break
            removed, added = best_exchange
            candidate = current.copy()
            candidate[list(removed)] = 0
            candidate[list(added)] = 1
            if not self.is_feasible(candidate):
                raise RuntimeError("el intercambio local calculado no es factible")
            current, current_value = self.repair(candidate)
            evaluations += 1

        if not self.is_feasible(current):
            raise RuntimeError("el pulido local produjo una solución infactible")
        return current, float(current_value), evaluations


def parse_mkp_file(path: str | Path) -> list[MKPInstance]:
    """Lee todas las instancias de un archivo OR-Library Chu--Beasley."""

    source = Path(path).expanduser().resolve()
    try:
        tokens = source.read_text(encoding="utf-8").split()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"no existe el archivo MKP: {source}") from exc
    if not tokens:
        raise ValueError(f"archivo MKP vacío: {source}")

    values = np.asarray(tokens, dtype=float)
    cursor = 0

    def take(count: int, label: str) -> NDArray[np.float64]:
        nonlocal cursor
        stop = cursor + count
        if stop > values.size:
            raise ValueError(f"archivo truncado al leer {label} en {source.name}")
        result = values[cursor:stop]
        cursor = stop
        return result

    instance_count = int(take(1, "cantidad de instancias")[0])
    instances: list[MKPInstance] = []
    family = source.stem.lower()
    for index in range(instance_count):
        header = take(3, f"cabecera de instancia {index}")
        n, m = int(header[0]), int(header[1])
        embedded_optimum = float(header[2])
        if n < 1 or m < 1:
            raise ValueError(f"dimensiones inválidas en instancia {index}: n={n}, m={m}")
        profits = take(n, f"ganancias de instancia {index}").copy()
        weights = take(m * n, f"pesos de instancia {index}").reshape(m, n).copy()
        capacities = take(m, f"capacidades de instancia {index}").copy()
        optimum = embedded_optimum if embedded_optimum > 0 else known_optimum(family, index)
        instances.append(
            MKPInstance(
                family=family,
                index=index,
                profits=profits,
                weights=weights,
                capacities=capacities,
                best_known=optimum,
                source=source,
            )
        )

    if cursor != values.size:
        raise ValueError(
            f"sobran {values.size - cursor} números después de parsear {source.name}"
        )
    return instances


def discover_mkp_files(instances_dir: str | Path) -> list[Path]:
    directory = Path(instances_dir).expanduser().resolve()
    files = sorted(directory.glob("mknapcb*.txt"), key=lambda path: path.stem)
    if not files:
        raise FileNotFoundError(f"no se encontraron mknapcb*.txt en {directory}")
    return files


def resolve_mkp_files(values: Iterable[str], instances_dir: str | Path) -> list[Path]:
    """Resuelve ``all``, números de familia, nombres o rutas explícitas."""

    requested = list(values)
    directory = Path(instances_dir).expanduser().resolve()
    if not requested or any(value.lower() == "all" for value in requested):
        return discover_mkp_files(directory)

    result: list[Path] = []
    for value in requested:
        candidate = Path(value).expanduser()
        if candidate.exists():
            path = candidate.resolve()
        else:
            name = value.lower()
            if name.isdigit():
                name = f"mknapcb{name}"
            if not name.endswith(".txt"):
                name += ".txt"
            path = (directory / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"archivo MKP no encontrado: {path}")
        if path not in result:
            result.append(path)
    return result


__all__ = [
    "KNOWN_OPTIMA",
    "MKPInstance",
    "discover_mkp_files",
    "instance_group_name",
    "known_optimum",
    "parse_mkp_file",
    "resolve_mkp_files",
]
