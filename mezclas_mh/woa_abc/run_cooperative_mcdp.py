"""Run cooperative WOA--ABC + DTW on the bundled MCDP matrices.

The MCDP input contains matrices only; the number of cells and the cell
capacity are therefore explicit command-line parameters.

Example:
    python -m mezclas_mh.woa_abc.run_cooperative_mcdp --iterations 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el archivo directamente con una ruta absoluta en Windows.
# En modo módulo, Python ya tiene la raíz del repositorio en sys.path.
if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
else:
    _repo_root = Path(__file__).resolve().parents[2]

from dtw_stagnation import StagnationConfig
from mcdp_core.data import load_mcdp_instances

from mezclas_mh.woa_abc import CooperativeMCDPParams, ejecutar_mcdp_cooperativo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=str(_repo_root / "mcdp_core" / "instances" / "instancias.txt"),
        help="Archivo con matrices MCDP",
    )
    parser.add_argument("--instance", type=int, default=1, help="Índice humano de la matriz")
    parser.add_argument("--cells", type=int, default=3, help="Número de celdas")
    parser.add_argument(
        "--capacity",
        type=int,
        default=6,
        help="Máximo de máquinas por celda",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Límite de scouts ABC")
    parser.add_argument("--abc-guide-strength", type=float, default=0.20)
    parser.add_argument("--abc-phi-scale", type=float, default=1.0)
    parser.add_argument("--exploration-probability", type=float, default=0.50,
                        help="Probabilidad de exploración WOA")
    parser.add_argument("--step-initial", type=float, default=1.0)
    parser.add_argument("--step-final", type=float, default=0.05)
    parser.add_argument("--momentum-factor", type=float, default=0.20)
    parser.add_argument("--window", type=int, default=75, help="STAG_WINDOW")
    parser.add_argument("--band", type=int, default=0, help="STAG_BAND; 0 = automático")
    parser.add_argument("--min-slope", type=float, default=0.1, help="STAG_MIN_SLOPE")
    parser.add_argument("--plateau-max", type=int, default=15, help="STAG_PLATEAU_MAX")
    parser.add_argument("--patience", type=int, default=25, help="STAG_PATIENCE")
    parser.add_argument("--ddtw", action=argparse.BooleanOptionalAction, default=True,
                        help="STAG_USE_DDTW (usar DDTW)")
    parser.add_argument("--adapt", dest="adapt_thresholds", action=argparse.BooleanOptionalAction,
                        default=True, help="STAG_ADAPT (umbrales adaptativos)")
    parser.add_argument("--p-low", type=float, default=30.0, help="STAG_P_LOW")
    parser.add_argument("--p-high", type=float, default=70.0, help="STAG_P_HIGH")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    instances = load_mcdp_instances(args.file, args.cells, args.capacity)
    if not 1 <= args.instance <= len(instances):
        parser.error(f"--instance debe estar entre 1 y {len(instances)}")
    instance = instances[args.instance - 1]

    dtw_config = StagnationConfig(
        window=args.window,
        band=args.band,
        patience=args.patience,
        plateau_max=args.plateau_max,
        min_slope=args.min_slope,
        use_ddtw=args.ddtw,
        adapt_thresholds=args.adapt_thresholds,
        p_low=args.p_low,
        p_high=args.p_high,
    )
    params = CooperativeMCDPParams(
        pop_size=args.pop_size,
        iterations=args.iterations,
        epochs=args.epochs,
        limit=args.limit,
        abc_guide_strength=args.abc_guide_strength,
        abc_phi_scale=args.abc_phi_scale,
        exploration_probability=args.exploration_probability,
        step_initial=args.step_initial,
        step_final=args.step_final,
        momentum_factor=args.momentum_factor,
        stag_cfg=dtw_config,
        seed=args.seed,
    )

    result = ejecutar_mcdp_cooperativo(instance, params, verbose=True)
    evaluation = result.evaluacion_global
    print("\n=== Resumen cooperativo WOA--ABC para MCDP ===")
    print(
        f"Instancia: {args.instance} "
        f"({instance.num_machines} máquinas, {instance.num_parts} piezas)"
    )
    print(f"Celdas/capacidad: {instance.max_cells}/{instance.max_machines_per_cell}")
    print(f"Mejor costo: {result.mejor_costo_global:.1f}")
    print(f"Elementos excepcionales: {evaluation.exceptional_elements}")
    print(f"Factible: {evaluation.feasible}")
    for epoch in result.epochs:
        print(
            f"Epoch {epoch.epoch_idx + 1}: "
            f"DTW fires={epoch.stagnation_fires}, "
            f"handoffs WOA→ABC={len(epoch.eventos_cooperacion)}, "
            f"adaptaciones={len(epoch.eventos_adaptacion)}"
        )


if __name__ == "__main__":
    main()
