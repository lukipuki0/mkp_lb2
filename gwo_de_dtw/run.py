"""
Run the GWO-DE hybrid on one OR-Library MKP instance with the four policies.

Usage (from the repository root):
    python -m gwo_de_dtw.run --instance instancias/mknapcb1.txt --index 0 \
        --binarization lb2 --signal ddtw --window 40 --pop 30 --iterations 1000 \
        --seed 1 --out gwo_de_dtw/results/mknapcb1_0_ddtw_s1.json

Defaults come from config.py. The policies of policies.POLICIES run one after
the other with the same settings, each with its own hybrid seeded with --seed.
Every invocation writes one group folder: <policy>/result.json and its plots
for each policy, plus convergence_comparison.png/.pdf with the four policies.
--out X.json uses the folder X/ (results/x.json -> results/x/binary_simple/...),
any other --out is the folder itself; without --out a new folder
<instance>_<index>_s<seed>_<YYYYmmdd-HHMMSS> is created in gwo_de_dtw/results/
(relative to the working directory). A non-empty or non-folder destination is
rejected before running.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from . import config
from .problem import cargar_instancia
from dtw_stagnation import StagnationConfig

from .controller import run_hybrid
from .policies import POLICIES
from .plots import plot_convergence_comparison, save_all_plots

# Parent of the default groups, relative to the working directory (the repository root).
RESULTS_DIR = Path("gwo_de_dtw") / "results"


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
    parser.add_argument("--dtw-kirito", action=argparse.BooleanOptionalAction,
                        default=config.DTW_KIRITO,
                        help="binary policies use the Kirito rolling control: one "
                             "monitor for the whole run, explore iff the trigger holds on "
                             "each ready observation, exploit otherwise (default: off = "
                             "fresh window after every change); continuous policies are "
                             "unchanged")
    parser.add_argument("--hybrid-mode", choices=("competitive", "sequential"),
                        default=config.HYBRID_MODE,
                        help="how GWO and DE are combined each generation: competitive = "
                             "GWO and DE candidates compete per individual; sequential = "
                             "full GWO sweep, then full DE sweep over the resulting "
                             "population (experimental) (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None,
                        help="group folder of this run: X.json means folder X/, any other "
                             "path is the folder itself; it must not exist or be empty. "
                             "Each policy writes <policy>/result.json (full run record) "
                             "and its plots, including temporal PNG/PDF and NPZ/CSV "
                             "diagnostics; the folder also gets "
                             "convergence_comparison.png/.pdf with the four policies "
                             "(default: new folder <instance>_<index>_s<seed>_<timestamp> "
                             "in gwo_de_dtw/results/)")
    return parser


def output_group(out: Path) -> Path:
    """Group folder named by --out: X.json -> X/, any other path is the folder itself."""
    return out.with_suffix("") if out.suffix.lower() == ".json" else out


def new_default_group(instance: str, index: int, seed: int) -> Path:
    """Creates and returns a new timestamp folder in RESULTS_DIR (never an existing one)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{Path(instance).stem}_{index}_s{seed}_{stamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        group = RESULTS_DIR / (base if n == 1 else f"{base}_{n}")
        try:
            group.mkdir()
            return group
        except FileExistsError:
            n += 1


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
    group = None
    if args.out is not None:
        group = output_group(args.out)
        if group.exists() and not group.is_dir():
            parser.error(f"output path is not a directory: {group}")
        if group.is_dir() and any(group.iterdir()):
            parser.error(f"output directory is not empty: {group}")
    cfg = StagnationConfig(
        window=args.window, band=args.band, min_slope=config.MIN_SLOPE,
        plateau_max=args.plateau_max, patience=args.patience,
        use_ddtw=(args.signal == "ddtw"), adapt_thresholds=config.ADAPT_THRESHOLDS,
        p_low=config.P_LOW, p_high=config.P_HIGH, improvement_tol=config.IMPROVEMENT_TOL,
    )
    inst = cargar_instancia(args.instance, args.index)
    instance = f"{Path(args.instance).name}[{args.index}]"
    if group is None:
        group = new_default_group(args.instance, args.index, args.seed)
    else:
        group.mkdir(parents=True, exist_ok=True)

    print(f"instance={instance} binarization={args.binarization} "
          f"signal={args.signal} W={args.window} "
          f"pop={args.pop} iterations={args.iterations} seed={args.seed}"
          + (" dtw_kirito=True" if args.dtw_kirito else "")
          + (f" hybrid_mode={args.hybrid_mode}"
             if args.hybrid_mode != "competitive" else ""))
    records = []
    for policy in POLICIES:
        res = run_hybrid(inst, cfg, pop_size=args.pop,
                         iterations=args.iterations, seed=args.seed,
                         G1=args.g1, G2=args.g2,
                         g3_exploit=args.g3_exploit, g3_explore=args.g3_explore,
                         binarization=args.binarization, policy=policy,
                         dtw_kirito=args.dtw_kirito, hybrid_mode=args.hybrid_mode)
        res["instance"] = instance
        res["signal"] = args.signal

        gap = res["gap_pct"]
        gap_txt = f"{gap:.2f}%" if gap is not None else "n/a"
        print(f"policy={policy} best={res['best_fitness']:.0f} "
              f"known_optimum={res['known_optimum']:.0f} "
              f"gap={gap_txt} evaluations={res['evaluations']} "
              f"generations={res['generations']} changes={len(res['changes'])}")

        folder = group / policy
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps(res, default=_to_json, indent=1))
        save_all_plots(res, folder)
        records.append(res)

    plot_convergence_comparison(records, group)
    print(f"output={group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
