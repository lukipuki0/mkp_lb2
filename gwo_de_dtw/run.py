"""
Run the GWO-DE hybrid on one OR-Library MKP instance.

Usage (from the repository root):
    python -m gwo_de_dtw.run --instance instancias/mknapcb1.txt --index 0 \
        --binarization lb2 --signal ddtw --window 40 --pop 30 --iterations 1000 \
        --seed 1 --out gwo_de_dtw/results/mknapcb1_0_ddtw_s1.json

Defaults come from config.py. After a profile change the monitor is discarded
(fresh window): no decision until W new observations. With --out, the plots
are saved next to the JSON in <out without suffix>_plots/
(e.g. results/x.json -> results/x_plots/).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .problem import cargar_instancia
from dtw_stagnation import StagnationConfig

from .controller import run_hybrid
from .plots import save_all_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gwo_de_dtw.run",
        description="GWO-DE hybrid for the binary MKP with DTW/DDTW profile control.",
        allow_abbrev=False,  # otherwise --g3 is ambiguous with --g3-exploit/--g3-explore
    )
    parser.add_argument("--instance", default=config.INSTANCE,
                        help="OR-Library mknapcb file (default: %(default)s)")
    parser.add_argument("--index", type=int, default=config.INDEX,
                        help="instance index inside the file (default: %(default)s)")
    parser.add_argument("--binarization", choices=("lb2", "sigmoid"),
                        default=config.BINARIZATION,
                        help=f"lb2 works in [-{config.V_MAX:g}, {config.V_MAX:g}], sigmoid in "
                             f"[-{config.SIGMOID_V_MAX:g}, {config.SIGMOID_V_MAX:g}] "
                             "(default: %(default)s)")
    parser.add_argument("--signal", choices=("dtw", "ddtw"),
                        default="ddtw" if config.USE_DDTW else "dtw",
                        help="distance used by the monitor (default: %(default)s)")
    parser.add_argument("--window", type=int, default=config.WINDOW,
                        help="W, generations per window (default: %(default)s)")
    parser.add_argument("--band", type=int, default=config.BAND,
                        help="Sakoe-Chiba band, 0 = 10%% of W (default: %(default)s)")
    parser.add_argument("--plateau-max", type=int, default=config.PLATEAU_MAX,
                        help="generations without improvement for a plateau (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=config.PATIENCE,
                        help="consecutive triggers needed to fire (default: %(default)s)")
    parser.add_argument("--pop", type=int, default=config.POP_SIZE,
                        help="population size N (default: %(default)s)")
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS,
                        help="generations to run; evaluations (N initial + 4N per "
                             "generation with lb2, 2N with sigmoid) are only reported "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="random seed (default: %(default)s)")
    parser.add_argument("--g1", type=float, default=config.G1,
                        help="LB2 slope factor G1 (default: %(default)s)")
    parser.add_argument("--g2", type=float, default=config.G2,
                        help=f"LB2 G2, must be below v_max={config.V_MAX:g} with lb2 "
                             "(default: %(default)s)")
    parser.add_argument("--g3-exploit", type=float, default=config.EXPLOIT_G3,
                        help="LB2 base flip probability G3 while exploiting "
                             "(default: %(default)s; same value as --g3-explore = fixed G3)")
    parser.add_argument("--g3-explore", type=float, default=config.EXPLORE_G3,
                        help="LB2 base flip probability G3 while exploring "
                             "(default: %(default)s; same value as --g3-exploit = fixed G3)")
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON file with the full run record; plots go to "
                             "<out without suffix>_plots/, including temporal PNG/PDF "
                             "and NPZ/CSV diagnostics")
    return parser


def _to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.binarization == "lb2" and args.g2 >= config.V_MAX:
        parser.error(f"--g2 must be below v_max={config.V_MAX:g} "
                     "(otherwise the LB2 slope changes sign)")
    cfg = StagnationConfig(
        window=args.window, band=args.band, min_slope=config.MIN_SLOPE,
        plateau_max=args.plateau_max, patience=args.patience,
        use_ddtw=(args.signal == "ddtw"), adapt_thresholds=config.ADAPT_THRESHOLDS,
        p_low=config.P_LOW, p_high=config.P_HIGH, improvement_tol=config.IMPROVEMENT_TOL,
    )
    inst = cargar_instancia(args.instance, args.index)
    res = run_hybrid(inst, cfg, pop_size=args.pop,
                     iterations=args.iterations, seed=args.seed,
                     G1=args.g1, G2=args.g2,
                     g3_exploit=args.g3_exploit, g3_explore=args.g3_explore,
                     binarization=args.binarization)
    res["instance"] = f"{Path(args.instance).name}[{args.index}]"
    res["signal"] = args.signal

    print(f"instance={res['instance']} binarization={res['binarization']} "
          f"signal={args.signal} W={args.window} "
          f"pop={args.pop} iterations={args.iterations} seed={args.seed}")
    gap = res["gap_pct"]
    gap_txt = f"{gap:.2f}%" if gap is not None else "n/a"
    print(f"best={res['best_fitness']:.0f} known_optimum={res['known_optimum']:.0f} "
          f"gap={gap_txt} evaluations={res['evaluations']} "
          f"generations={res['generations']} changes={len(res['changes'])}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, default=_to_json, indent=1))
        save_all_plots(res, args.out.with_name(args.out.stem + "_plots"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
