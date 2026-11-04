"""
CLI: GWO-DE DTW/DDTW policies on HRES2, one seed.

    python -m gwo_de_hres2.run [--policies p1,p2] [--signal ddtw|dtw]
        [--pop-size N] [--iterations T] [--seed S] [--output-dir DIR]

Runs every selected policy (default: the six of gwo_de_dtw.CAMPAIGN_POLICIES)
on the same HRES2Function (synthetic TMY weather, built once) and writes
runs.jsonl (one record per policy, flushed after each run) and manifest.json
into a fresh directory, plus <policy>/result.json and individual diagnostics
and convergence_comparison.png/.pdf at the root. A non-empty output directory is rejected before any
run. Run from the repository root.

Spec: odd/tasks/gwo-de-hres2.md (S5, S9).
"""

import argparse
import datetime
import importlib
import json
import os
import platform
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gwo_de_dtw.policies import CAMPAIGN_POLICIES  # noqa: E402

from . import config  # noqa: E402
from .control import EXPLOIT, EXPLORE, run_search  # noqa: E402
from .diagnostics import export_run, plot_outcomes  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_policies(text: str) -> list:
    names = [part.strip() for part in text.split(",") if part.strip()]
    for name in names:
        if name not in CAMPAIGN_POLICIES:
            raise argparse.ArgumentTypeError(f"unknown policy: {name}")
    if not names:
        raise argparse.ArgumentTypeError("no policy given")
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gwo_de_hres2.run",
                                description="GWO-DE DTW/DDTW profile policies on HRES2 (WPEB).")
    p.add_argument("--policies", type=parse_policies, default=list(CAMPAIGN_POLICIES),
                   help="comma list (default: all six: " + ",".join(CAMPAIGN_POLICIES) + ")")
    p.add_argument("--signal", choices=("ddtw", "dtw"), default=config.SIGNAL)
    p.add_argument("--pop-size", type=int, default=config.POP_SIZE)
    p.add_argument("--iterations", type=int, default=config.ITERATIONS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--output-dir", default=None,
                   help="fresh directory for full records, individual diagnostics and comparison "
                        "(default: gwo_de_hres2/results/<signal>_s<seed>_<timestamp>)")
    return p


def load_function():
    wpeb_model = importlib.import_module("HRES2-H2.wpeb_model")
    return wpeb_model.HRES2Function()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = args.output_dir
    if out is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(RESULTS_DIR, f"{args.signal}_s{args.seed}_{stamp}")
    if os.path.isdir(out) and os.listdir(out):
        parser.exit(2, f"error: output directory is not empty: {out}\n")
    os.makedirs(out, exist_ok=True)

    monitor_cfg = config.monitor_config(use_ddtw=args.signal == "ddtw")
    func = load_function()
    manifest = {
        "configuration": {"policies": args.policies, "signal": args.signal,
                          "pop_size": args.pop_size, "iterations": args.iterations,
                          "seed": args.seed, "hybrid_mode": "competitive",
                          "profiles": {p.name: {"a": p.a, "F": p.F, "CR": p.CR}
                                       for p in (EXPLOIT, EXPLORE)}},
        "monitor": {k: getattr(monitor_cfg, k) for k in monitor_cfg.__dataclass_fields__},
        "model": {"type": type(func).__name__, "config": func.config,
                  "weather": "generate_synthetic_tmy_8760",
                  "lb": list(map(float, func.lb_vector)), "ub": list(map(float, func.ub_vector))},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"GWO-DE on HRES2 signal={args.signal} N={args.pop_size} T={args.iterations} "
          f"seed={args.seed} -> {out}")
    records = []
    with open(os.path.join(out, "runs.jsonl"), "w", encoding="utf-8") as fh:
        for policy in args.policies:
            rec = run_search(func, monitor_cfg, policy=policy, pop_size=args.pop_size,
                             iterations=args.iterations, seed=args.seed)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            records.append(rec)
            export_run(rec, os.path.join(out, policy))
            print(f"  {policy:<22s} objective={rec['objective']:.6f} feasible={rec['feasible']} "
                  f"changes={len(rec['changes'])} evals={rec['evaluations']} "
                  f"time={rec['elapsed_seconds']:.1f}s")
    plot_outcomes(records, os.path.join(out, "convergence_comparison.png"),
                  f"HRES2 {args.signal.upper()}: one seed, selected policies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
