"""
Multi-seed campaign: every selected instance x seed x policy, keeping only the
best run of each policy per instance.

Usage (from the repository root):
    python -m gwo_de_dtw.campaign --out gwo_de_dtw/results/campaign_full

Defaults: mknapcb1..9.txt from instancias/, indices 0 1, seeds 1..31, the six
policies of policies.CAMPAIGN_POLICIES (the four adaptive rules plus the two
vanilla baselines) and the algorithm settings of config.py, i.e.
9 x 2 x 31 x 6 = 3348 runs. Every run calls controller.run_hybrid with the same
arguments as run.py (run.py stays the one-seed runner of the same six policies).

Everything is checked before the first run or write: instance file names
(basenames inside --instances-dir), indices, seeds, budgets and the output
folder (must not exist or be empty; default: new campaign_<YYYYmmdd-HHMMSS> in
gwo_de_dtw/results/). --dry-run stops after these checks.

Layout of the campaign folder:
    campaign_config.json        settings, selection, total/completed runs, status
    resumen.txt                 Spanish readable summary, rewritten after each instance
    summary.csv                 one row per run (all instances), written after each run
    statistics.csv              one row per instance and policy
    <file stem>/inst_<NN>/runs.csv, statistics.csv
    <file stem>/inst_<NN>/convergence_comparison.png/.pdf   best run of each policy
    <file stem>/inst_<NN>/<policy>/best_run/result.json + plots of that run

Best run = highest best_fitness; ties go to the lower seed. Only the current
best record of each policy is kept in memory; the other runs keep their
metrics in the CSV files but no record or plots. A failing run stops the
campaign (status failed, exit 1); finished rows and instances stay on disk.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time

from . import config
from .problem import cargar_instancia, parsear_instancias
from dtw_stagnation import StagnationConfig

from .controller import run_hybrid
from .policies import CAMPAIGN_POLICIES as POLICIES
from .plots import plot_convergence_comparison, save_all_plots
from .run import RESULTS_DIR, _to_json

FILES = tuple(f"mknapcb{i}.txt" for i in range(1, 10))
INDICES = (0, 1)
SEEDS = tuple(range(1, 32))

CONFIG_FILE = "campaign_config.json"
RESUMEN_FILE = "resumen.txt"
RUN_FIELDS = ["file", "index", "policy", "seed", "status", "best_fitness", "known_optimum",
              "gap_pct", "evaluations", "generations", "changes", "elapsed_s"]
STAT_FIELDS = ["file", "index", "policy", "runs", "best_fitness", "worst_fitness",
               "mean_fitness", "median_fitness", "std_fitness", "best_seed",
               "known_optimum", "best_gap_pct", "mean_gap_pct", "mean_elapsed_s"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gwo_de_dtw.campaign",
        description="Multi-seed campaign of the GWO-DE hybrid: every file x index x seed "
                    "x policy; only the best run of each policy keeps its record and plots.",
        allow_abbrev=False,
    )
    parser.add_argument("--instances-dir", type=Path, default=Path(config.INSTANCE).parent,
                        help="folder of the instance files (default: %(default)s)")
    parser.add_argument("--files", nargs="+", default=list(FILES),
                        help="instance file names inside --instances-dir, basenames only "
                             "(default: mknapcb1.txt .. mknapcb9.txt)")
    parser.add_argument("--indices", nargs="+", type=int, default=list(INDICES),
                        help="instance indices of every file (default: 0 1)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                        help="seeds, nonnegative and unique (default: 1 .. 31)")
    parser.add_argument("--binarization", choices=("lb2", "sigmoid"),
                        default=config.BINARIZATION, help="(default: %(default)s)")
    parser.add_argument("--signal", choices=("dtw", "ddtw"),
                        default="ddtw" if config.USE_DDTW else "dtw",
                        help="(default: %(default)s)")
    parser.add_argument("--window", type=int, default=config.WINDOW,
                        help="W, generations per window (default: %(default)s)")
    parser.add_argument("--band", type=int, default=config.BAND,
                        help="Sakoe-Chiba band, 0 = 10%% of W (default: %(default)s)")
    parser.add_argument("--plateau-max", type=int, default=config.PLATEAU_MAX,
                        help="(default: %(default)s)")
    parser.add_argument("--patience", type=int, default=config.PATIENCE,
                        help="(default: %(default)s)")
    parser.add_argument("--pop", type=int, default=config.POP_SIZE,
                        help="population size N, at least 4 (default: %(default)s)")
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS,
                        help="generations per run (default: %(default)s)")
    parser.add_argument("--g1", type=float, default=config.G1, help="(default: %(default)s)")
    parser.add_argument("--g2", type=float, default=config.G2, help="(default: %(default)s)")
    parser.add_argument("--g3-exploit", type=float, default=config.EXPLOIT_G3,
                        help="(default: %(default)s)")
    parser.add_argument("--g3-explore", type=float, default=config.EXPLORE_G3,
                        help="(default: %(default)s)")
    parser.add_argument("--dtw-kirito", action=argparse.BooleanOptionalAction,
                        default=config.DTW_KIRITO,
                        help="Kirito rolling control for the binary policies (default: off)")
    parser.add_argument("--out", type=Path, default=None,
                        help="campaign folder; it must not exist or be empty. It gets "
                             f"{CONFIG_FILE}, {RESUMEN_FILE}, summary.csv, statistics.csv and "
                             "<file stem>/inst_<NN>/ with runs.csv, statistics.csv, "
                             "convergence_comparison.png/.pdf and <policy>/best_run/ "
                             "(default: new folder campaign_<timestamp> in gwo_de_dtw/results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the selection and output, print the number of runs "
                             "and stop without running or writing anything")
    return parser


def monitor_config(args) -> StagnationConfig:
    """Same StagnationConfig as run.py for these options."""
    return StagnationConfig(
        window=args.window, band=args.band, min_slope=config.MIN_SLOPE,
        plateau_max=args.plateau_max, patience=args.patience,
        use_ddtw=(args.signal == "ddtw"), adapt_thresholds=config.ADAPT_THRESHOLDS,
        p_low=config.P_LOW, p_high=config.P_HIGH, improvement_tol=config.IMPROVEMENT_TOL,
    )


def _duplicate(values):
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def validate(parser, args) -> None:
    """Every check before the first run or write; parser.error exits with code 2."""
    for name in args.files:
        if name in ("", ".", "..") or Path(name).name != name or "/" in name or "\\" in name:
            parser.error(f"invalid instance file name (basename only): {name}")
    if _duplicate(args.files) is not None:
        parser.error(f"duplicate instance file: {_duplicate(args.files)}")
    if _duplicate(args.indices) is not None:
        parser.error(f"duplicate index: {_duplicate(args.indices)}")
    for seed in args.seeds:
        if seed < 0:
            parser.error(f"seeds must be nonnegative: {seed}")
    if _duplicate(args.seeds) is not None:
        parser.error(f"duplicate seed: {_duplicate(args.seeds)}")
    if args.pop < 4:
        parser.error("--pop must be >= 4 (DE/rand/1 needs three other individuals)")
    for option in ("iterations", "window", "plateau_max", "patience"):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.band < 0:
        parser.error("--band must be nonnegative")
    if args.binarization == "lb2" and args.g2 >= config.V_MAX:
        parser.error(f"--g2 must be below v_max={config.V_MAX:g} "
                     "(otherwise the LB2 slope changes sign)")
    for name in args.files:
        path = args.instances_dir / name
        if not path.is_file():
            parser.error(f"instance file not found: {path}")
        try:
            count = len(parsear_instancias(path.read_text()))
        except (OSError, ValueError, IndexError) as exc:
            parser.error(f"cannot read instance file {path}: {exc}")
        for index in args.indices:
            if not 0 <= index < count:
                parser.error(f"index {index} out of range for {name} ({count} instances)")
    if args.out is not None:
        if args.out.exists() and not args.out.is_dir():
            parser.error(f"output path is not a directory: {args.out}")
        if args.out.is_dir() and any(args.out.iterdir()):
            parser.error(f"output directory is not empty: {args.out}")


def new_default_root() -> Path:
    """Creates and returns a new campaign_<timestamp> folder in RESULTS_DIR."""
    base = f"campaign_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        root = RESULTS_DIR / (base if n == 1 else f"{base}_{n}")
        try:
            root.mkdir()
            return root
        except FileExistsError:
            n += 1


def _blank(value):
    return "" if value is None else value


def _start_csv(path: Path, fields) -> None:
    with open(path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()


def _append_csv(paths, fields, rows) -> None:
    for path in paths:
        with open(path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerows(rows)


def _write_json(path: Path, data: dict) -> None:
    """Replaces `path` in one step so an interruption leaves the previous version."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _replace_text(path: Path, text: str) -> None:
    """Replaces `path` in one step so an interruption leaves the previous version.

    The temporary file is removed if the replace does not finish. After a
    successful replace the temporary name no longer exists, so it is not deleted.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def is_better(res: dict, best) -> bool:
    """Higher best_fitness wins; on a tie the lower seed (independent of run order)."""
    if best is None:
        return True
    if res["best_fitness"] != best["best_fitness"]:
        return res["best_fitness"] > best["best_fitness"]
    return res["seed"] < best["seed"]


def policy_statistics(name: str, index: int, policy: str, rows, best: dict) -> dict:
    """Statistics row of one instance and policy from its run rows.

    std_fitness is the sample standard deviation, blank with a single run; gap
    columns are blank when the reference is unknown.
    """
    fitness = [row["best_fitness"] for row in rows]
    gaps = [row["gap_pct"] for row in rows if row["gap_pct"] != ""]
    return {
        "file": name, "index": index, "policy": policy, "runs": len(rows),
        "best_fitness": max(fitness), "worst_fitness": min(fitness),
        "mean_fitness": statistics.mean(fitness),
        "median_fitness": statistics.median(fitness),
        "std_fitness": statistics.stdev(fitness) if len(fitness) > 1 else "",
        "best_seed": best["seed"], "known_optimum": rows[0]["known_optimum"],
        "best_gap_pct": min(gaps) if gaps else "",
        "mean_gap_pct": statistics.mean(gaps) if gaps else "",
        "mean_elapsed_s": statistics.mean(row["elapsed_s"] for row in rows),
    }


_STATUS_ES = {
    "running": "en curso",
    "completed": "completada",
    "failed": "fallida",
    "interrupted": "interrumpida",
}
_SCOPE_ES = {
    "running": "Incluye solo las instancias completadas hasta ahora.",
    "completed": "Incluye todas las instancias completadas.",
    "failed": "Incluye solo las instancias completadas antes de detener la campa\u00f1a.",
    "interrupted": "Incluye solo las instancias completadas antes de detener la campa\u00f1a.",
}
_RULES_ES = (
    "El ranking ordena las mejores corridas: mayor fitness primero y, en empate, la semilla menor.\n"
    "Si el fitness y la semilla tambi\u00e9n empatan, se conserva el orden de los modos en la campa\u00f1a.\n"
    "La media se muestra, pero no determina el ranking.\n"
    "La desviaci\u00f3n est\u00e1ndar es muestral (n - 1). Con una sola corrida: no aplica.\n"
    "Si la referencia es desconocida, la brecha es no disponible (no cero)."
)


def _fmt_fitness(value) -> str:
    return f"{float(value):.4f}"


def _fmt_gap(value) -> str:
    """Unknown reference is 'no disponible', never zero."""
    if value is None or value == "":
        return "no disponible"
    return f"{float(value):.2f}%"


def _fmt_std(values) -> str:
    if len(values) < 2:
        return "no aplica"
    return f"{statistics.stdev(values):.4f}"


def _ranked_rows(rows):
    """Best fitness descending; equal fitness keeps the lower seed."""
    return sorted(rows, key=lambda row: (-float(row["best_fitness"]), int(row["seed"])))


def _mode_block(policy: str, rows, winner_seed) -> str:
    ranked = _ranked_rows(rows)
    fitness = [float(row["best_fitness"]) for row in rows]
    winner = next(row for row in rows if int(row["seed"]) == int(winner_seed))
    lines = [
        f"Modo {policy}",
        f"  Mejor fitness: {_fmt_fitness(max(fitness))}",
        f"  Peor fitness: {_fmt_fitness(min(fitness))}",
        f"  Media: {_fmt_fitness(statistics.mean(fitness))}",
        f"  Desviaci\u00f3n est\u00e1ndar muestral: {_fmt_std(fitness)}",
        f"  Semilla ganadora: {int(winner_seed)}",
        f"  Brecha de la semilla ganadora: {_fmt_gap(winner['gap_pct'])}",
        f"  Tiempo medio: {statistics.mean(float(row['elapsed_s']) for row in rows):.4f} s",
        "  Ranking de semillas (mejor a peor; por fitness, no por la media):",
    ]
    for place, row in enumerate(ranked, 1):
        lines.append(
            f"    {place}. semilla {int(row['seed'])}  fitness {_fmt_fitness(row['best_fitness'])}"
            f"  brecha {_fmt_gap(row['gap_pct'])}"
        )
    return "\n".join(lines)


def _instance_section(record) -> str:
    title = f"Instancia {record['file']} [{record['index']}]"
    blocks = [
        _mode_block(policy, record["rows"][policy], record["best"][policy]["seed"])
        for policy in POLICIES
    ]
    return title + "\n" + ("-" * len(title)) + "\n" + "\n\n".join(blocks)


def _mode_rank_key(policy: str, record):
    res = record["best"][policy]
    return (-float(res["best_fitness"]), int(res["seed"]), POLICIES.index(policy))


def _mode_ranking(record) -> str:
    """Four modes ordered by best run, not by mean. Mean is shown only."""
    lines = [f"Instancia {record['file']} [{record['index']}]"]
    for place, policy in enumerate(sorted(POLICIES, key=lambda p: _mode_rank_key(p, record)), 1):
        rows = record["rows"][policy]
        res = record["best"][policy]
        winner = next(row for row in rows if int(row["seed"]) == int(res["seed"]))
        mean = statistics.mean(float(row["best_fitness"]) for row in rows)
        lines.append(
            f"  {place}. {policy}  mejor {_fmt_fitness(res['best_fitness'])}"
            f"  semilla {int(res['seed'])}  brecha {_fmt_gap(winner['gap_pct'])}"
            f"  media {_fmt_fitness(mean)}"
        )
    return "\n".join(lines)


def _render_resumen(state: dict, args, completed) -> str:
    title = "Resumen de la campa\u00f1a GWO-DE"
    total_inst = len(args.files) * len(args.indices)
    error = state.get("error") or "ninguno"
    header = "\n".join([
        title,
        "=" * len(title),
        f"Estado: {_STATUS_ES[state['status']]}",
        f"Progreso: {len(completed)}/{total_inst} instancias completadas, "
        f"{state['completed_runs']}/{state['total_runs']} corridas",
        f"Error: {error}",
        f"Archivos: {', '.join(args.files)}",
        f"\u00cdndices: {', '.join(str(i) for i in args.indices)}",
        f"Semillas: {', '.join(str(s) for s in args.seeds)}",
        f"Modos: {', '.join(POLICIES)}",
        f"Binarizaci\u00f3n: {args.binarization}",
        f"Se\u00f1al: {args.signal}",
        f"Poblaci\u00f3n: {args.pop}",
        f"Iteraciones: {args.iterations}",
        f"Ventana W: {args.window}",
        f"Banda: {args.band}",
        f"Meseta m\u00e1xima: {args.plateau_max}",
        f"Paciencia: {args.patience}",
        f"G1: {args.g1:g}",
        f"G2: {args.g2:g}",
        f"G3 explotaci\u00f3n: {args.g3_exploit:g}",
        f"G3 exploraci\u00f3n: {args.g3_explore:g}",
        f"DTW Kirito: {'s\u00ed' if args.dtw_kirito else 'no'}",
    ])
    final_title = "Ranking final de los modos"
    final = "\n".join([
        final_title,
        "=" * len(final_title),
        "Orden por la mejor corrida de cada modo, dentro de cada instancia: mayor fitness "
        "y, en empate, semilla menor.",
        "No se mezclan fitness de instancias distintas. La media se muestra y no determina "
        "el orden.",
        _SCOPE_ES[state["status"]],
    ])
    parts = [header, _RULES_ES]
    parts.extend(_instance_section(record) for record in completed)
    parts.append(final)
    parts.extend(_mode_ranking(record) for record in completed)
    return "\n\n".join(parts) + "\n"


def _write_resumen(root: Path, state: dict, args, completed) -> None:
    _replace_text(root / RESUMEN_FILE, _render_resumen(state, args, completed))


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(parser, args)
    total = len(args.files) * len(args.indices) * len(args.seeds) * len(POLICIES)
    print(f"campaign files={len(args.files)} indices={len(args.indices)} "
          f"seeds={len(args.seeds)} policies={len(POLICIES)} runs={total} "
          f"binarization={args.binarization} signal={args.signal} W={args.window} "
          f"pop={args.pop} iterations={args.iterations}"
          + (" dtw_kirito=True" if args.dtw_kirito else ""))
    if args.dry_run:
        print("dry-run: selection and output validated, nothing executed or written")
        return 0

    cfg = monitor_config(args)
    if args.out is None:
        root = new_default_root()
    else:
        root = args.out
        root.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running", "total_runs": total, "completed_runs": 0, "error": None,
        "started": datetime.now().isoformat(timespec="seconds"), "finished": None,
        "selection": {"instances_dir": str(args.instances_dir), "files": args.files,
                      "indices": args.indices, "seeds": args.seeds,
                      "policies": list(POLICIES)},
        "config": {"binarization": args.binarization, "signal": args.signal,
                   "pop": args.pop, "iterations": args.iterations,
                   "g1": args.g1, "g2": args.g2, "g3_exploit": args.g3_exploit,
                   "g3_explore": args.g3_explore, "dtw_kirito": args.dtw_kirito,
                   "monitor_config": asdict(cfg)},
    }
    config_path = root / CONFIG_FILE
    _write_json(config_path, state)
    summary, overall = root / "summary.csv", root / "statistics.csv"
    _start_csv(summary, RUN_FIELDS)
    _start_csv(overall, STAT_FIELDS)

    completed = []
    where = None
    try:
        for name in args.files:
            for index in args.indices:
                folder = root / Path(name).stem / f"inst_{index:02d}"
                folder.mkdir(parents=True)
                runs = folder / "runs.csv"
                _start_csv(runs, RUN_FIELDS)
                inst = cargar_instancia(str(args.instances_dir / name), index)
                instance = f"{name}[{index}]"
                rows = {policy: [] for policy in POLICIES}
                best = {policy: None for policy in POLICIES}
                for seed in args.seeds:
                    for policy in POLICIES:
                        where = f"{instance} policy={policy} seed={seed}"
                        row = {"file": name, "index": index, "policy": policy, "seed": seed,
                               "status": "failed"}
                        start = time.perf_counter()
                        try:
                            res = run_hybrid(inst, cfg, pop_size=args.pop,
                                             iterations=args.iterations, seed=seed,
                                             G1=args.g1, G2=args.g2,
                                             g3_exploit=args.g3_exploit,
                                             g3_explore=args.g3_explore,
                                             binarization=args.binarization, policy=policy,
                                             dtw_kirito=args.dtw_kirito)
                        except BaseException:
                            _append_csv((runs, summary), RUN_FIELDS, [row])
                            raise
                        elapsed = time.perf_counter() - start
                        optimum = res["known_optimum"]
                        row.update(status="ok", best_fitness=res["best_fitness"],
                                   known_optimum=optimum if optimum and optimum > 0 else "",
                                   gap_pct=_blank(res["gap_pct"]),
                                   evaluations=res["evaluations"],
                                   generations=res["generations"],
                                   changes=len(res["changes"]), elapsed_s=round(elapsed, 4))
                        _append_csv((runs, summary), RUN_FIELDS, [row])
                        rows[policy].append(row)
                        if is_better(res, best[policy]):
                            best[policy] = res
                        state["completed_runs"] += 1
                        _write_json(config_path, state)
                        gap = res["gap_pct"]
                        print(f"[{state['completed_runs']}/{total}] {instance} seed={seed} "
                              f"policy={policy} best={res['best_fitness']:.0f} "
                              f"gap={f'{gap:.2f}%' if gap is not None else 'n/a'} "
                              f"elapsed={elapsed:.2f}s")

                where = f"{instance} export"
                stats = [policy_statistics(name, index, policy, rows[policy], best[policy])
                         for policy in POLICIES]
                winners = []
                for policy in POLICIES:
                    res = best[policy]
                    res["instance"] = instance
                    res["signal"] = args.signal
                    run_dir = folder / policy / "best_run"
                    run_dir.mkdir(parents=True)
                    (run_dir / "result.json").write_text(
                        json.dumps(res, default=_to_json, indent=1))
                    save_all_plots(res, run_dir)
                    winners.append(res)
                plot_convergence_comparison(winners, folder, seed_labels=True)
                _start_csv(folder / "statistics.csv", STAT_FIELDS)
                _append_csv((folder / "statistics.csv", overall), STAT_FIELDS, stats)
                completed.append({"file": name, "index": index, "rows": rows, "best": best})
                _write_resumen(root, state, args, completed)
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        state["status"] = "interrupted" if interrupted else "failed"
        state["error"] = f"{where}: {type(exc).__name__}: {exc}"
        state["finished"] = datetime.now().isoformat(timespec="seconds")
        try:
            _write_resumen(root, state, args, completed)
        except Exception:
            pass
        except BaseException:
            _write_json(config_path, state)
            raise
        _write_json(config_path, state)
        print(f"status={state['status']} runs={state['completed_runs']}/{total}")
        print(f"output={root}")
        if not isinstance(exc, Exception):
            raise
        print(f"error: campaign failed at {state['error']}", file=sys.stderr)
        return 1

    state["status"] = "completed"
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    _write_json(config_path, state)
    _write_resumen(root, state, args, completed)
    print(f"status=completed runs={state['completed_runs']}/{total}")
    print(f"output={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
