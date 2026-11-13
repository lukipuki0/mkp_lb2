"""Repetición ASOC: 31 runs DDTW sin baselines ni ABC en el híbrido."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
benchmark = importlib.import_module("HRES2-H2.benchmark_hres2")

# Pool del manuscrito; no mutar los defaults del orquestador genérico.
ASOC_POPULATION = ("PSO", "GWO", "WOA", "EHO", "ACO")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=("Campaña ASOC DDTW sin ABC: 31 runs por método, 1000 iteraciones "
                     "nominales, sin baselines. --help no ejecuta simulaciones.")
    )
    parser.add_argument(
        "--output", default=None,
        help="Directorio nuevo; un destino existente se rechaza sin sobrescribir.",
    )
    args = parser.parse_args(argv)
    benchmark.ejecutar_benchmark_hres2(
        n_runs=31, max_iters=1000, output_dir=args.output,
        modes=["ddtw"], baselines=[],
        pool_poblacional=list(ASOC_POPULATION),
    )
    return 0


if __name__ == "__main__":
    main()
