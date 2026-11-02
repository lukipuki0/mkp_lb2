"""
Run the WOA-ABC hybrid on one OR-Library MKP instance with the four policies.

Usage (from the repository root):
    python -m woa_abc.run --instance instancias/mknapcb1.txt --index 0 \
        --signal ddtw --window 40 --pop 30 --iterations 1000 \
        --seed 1 --out woa_abc/resultados/mknapcb1_0_ddtw_s1.json

Mirrors gwo_de_dtw/run.py on feat/gwo-de-dtw-hybrid. Monitor defaults are the
single-run exporter's (benchmark.inherited_stagnation_config). The policies of
profiles.POLICIES run one after the other with the same settings, each with
its own solve seeded with --seed. Every invocation writes one group folder:
<policy>/result.json (the full WOA-ABC result) and its plots for each policy,
plus convergence_comparison.png/.pdf with the four policies. dtw_delta.png is
always written, as an empty diagnostic when no window was ready. A header
reference of 0 is the best known value from mkcbres.txt next to the instance
file (woa_abc.best_known), so gap and the optimum line appear.
--out X.json uses the folder X/ (resultados/x.json -> resultados/x/binary_simple/...),
any other --out is the folder itself; without --out a new folder
<instance>_<index>_s<seed>_<YYYYmmdd-HHMMSS> is created in woa_abc/resultados/
(beside this package, as the other woa_abc commands). A non-empty or
non-folder destination is rejected before running.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from . import solver as hybrid
from .benchmark import DEFAULT_OUTPUT_ROOT, _write_json, inherited_stagnation_config
from .plots import plot_convergence_comparison, plot_record, save_all_plots
from .profiles import POLICIES
from .solver import HybridConfig, solve

# Parent of the default groups.
RESULTS_DIR = DEFAULT_OUTPUT_ROOT


def build_parser() -> argparse.ArgumentParser:
    defaults = inherited_stagnation_config()
    parser = argparse.ArgumentParser(
        prog="python -m woa_abc.run",
        description="WOA-ABC hybrid for the binary MKP with the four DTW/DDTW profile policies.",
        allow_abbrev=False,
    )
    parser.add_argument("--instance", required=True,
                        help="local OR-Library mknapcb file; URLs are rejected")
    parser.add_argument("--index", "--instance-index", dest="index", type=int, default=0,
                        help="instance index inside the file (default: %(default)s)")
    parser.add_argument("--signal", choices=("dtw", "ddtw"),
                        default="ddtw" if defaults.use_ddtw else "dtw",
                        help="distance used by the monitor (default: %(default)s)")
    parser.add_argument("--window", type=int, default=defaults.window,
                        help="W, cycles per window (default: %(default)s)")
    parser.add_argument("--band", type=int, default=0,
                        help="Sakoe-Chiba band, 0 = 10%% of W (default: %(default)s)")
    parser.add_argument("--plateau-max", type=int, default=defaults.plateau_max,
                        help="cycles without improvement for a plateau (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=defaults.patience,
                        help="consecutive triggers needed to fire (default: %(default)s)")
    parser.add_argument("--pop", "--population-size", dest="pop", type=int, default=30,
                        help="population size N (default: %(default)s)")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="cycles to run (default: %(default)s)")
    parser.add_argument("--max-evaluations", type=int, default=None,
                        help="evaluation cap; a partial final cycle is left out of the plots "
                             "(default: none beyond the cycle horizon)")
    parser.add_argument("--scout-limit", type=int, default=None,
                        help="ABC scout limit (default: the variant's rule)")
    parser.add_argument("--variant", default="v0", choices=hybrid.VARIANTS,
                        help="additive variant (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: %(default)s)")
    parser.add_argument("--dtw-kirito", action=argparse.BooleanOptionalAction, default=False,
                        help="binary policies use the Kirito rolling control: one "
                             "monitor for the whole run, explore iff the trigger holds on "
                             "each ready observation, exploit otherwise (default: off = "
                             "fresh window after every change); continuous policies are "
                             "unchanged")
    parser.add_argument("--out", type=Path, default=None,
                        help="group folder of this run: X.json means folder X/, any other "
                             "path is the folder itself; it must not exist or be empty. "
                             "Each policy writes <policy>/result.json (full run record) "
                             "and its plots, including temporal PNG/PDF and NPZ/CSV "
                             "diagnostics; the folder also gets "
                             "convergence_comparison.png/.pdf with the four policies "
                             "(default: new folder <instance>_<index>_s<seed>_<timestamp> "
                             "in woa_abc/resultados/)")
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


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    group = None
    if args.out is not None:
        group = output_group(args.out)
        if group.exists() and not group.is_dir():
            parser.error(f"output path is not a directory: {group}")
        if group.is_dir() and any(group.iterdir()):
            parser.error(f"output directory is not empty: {group}")
    try:
        cfg = replace(inherited_stagnation_config(), window=args.window, band=args.band,
                      plateau_max=args.plateau_max, patience=args.patience,
                      use_ddtw=(args.signal == "ddtw"))
        configs = {policy: HybridConfig(
            population_size=args.pop, iterations=args.iterations,
            max_evaluations=args.max_evaluations, scout_limit=args.scout_limit,
            use_stagnation=True, stag_cfg=cfg, variant=args.variant,
            policy=policy, dtw_kirito=args.dtw_kirito) for policy in POLICIES}
        path = hybrid._local_instance_file(args.instance)
        inst, info = hybrid._load_local_instance(path, args.index)
    except ValueError as exc:
        parser.error(str(exc))
    instance = f"{path.name}[{args.index}]"
    if group is None:
        group = new_default_group(args.instance, args.index, args.seed)
    else:
        group.mkdir(parents=True, exist_ok=True)

    print(f"instance={instance} signal={args.signal} W={args.window} "
          f"pop={args.pop} iterations={args.iterations} seed={args.seed} "
          f"reference={info['reference']:.0f} reference_source={info['reference_source']}"
          + (" dtw_kirito=True" if args.dtw_kirito else "")
          + (f" variant={args.variant}" if args.variant != "v0" else ""))
    records = []
    for policy in POLICIES:
        result = solve(inst, configs[policy], seed=args.seed, metadata={"instance": info})
        res = plot_record(result, instance=instance)

        gap = res["gap_pct"]
        gap_txt = f"{gap:.2f}%" if gap is not None else "n/a"
        print(f"policy={policy} best={res['best_fitness']:.0f} "
              f"known_optimum={res['known_optimum']:.0f} "
              f"gap={gap_txt} evaluations={res['evaluations']} "
              f"cycles={res['cycles']} changes={len(res['changes'])}")

        folder = group / policy
        folder.mkdir()
        _write_json(folder / "result.json", asdict(result))
        save_all_plots(res, folder)
        records.append(res)

    plot_convergence_comparison(records, group)
    print(f"output={group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
