"""Separate many-seed variant campaign; only winning full exports survive success."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys

from .benchmark import ExportError, ExportOutcome, ExportRequest, _reject_url, export_single_run
from .campaign import (CampaignError, FILES, INDICES, VARIANTS, _selections,
                       _validate_settings, _outcome_row)
from .seed_campaign import (SEEDS, STAT_FIELDS as POLICY_STAT_FIELDS, _append_csv,
                            _duplicate, _read_json, _start_csv, _write_json,
                            is_better, policy_statistics, plot_comparison)
from . import solver as hybrid

RUN_FIELDS = ['file', 'index', 'variant', 'seed', 'status', 'best_fitness',
              'known_optimum', 'gap_pct', 'evaluations', 'generations', 'changes',
              'elapsed_s', 'recoveries', 'scouts', 'scout_limit', 'reference_source']
STAT_FIELDS = ['variant' if field == 'policy' else field for field in POLICY_STAT_FIELDS]


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m woa_abc.variant_seed_campaign',
        description='31 seeds x v0-v4; keep only the highest-fitness full export per variant '
                    '(ties: lower seed). Existing instance destinations are refused.', allow_abbrev=False)
    parser.add_argument('--instances-dir', type=Path, default=Path('instancias'))
    parser.add_argument('--files', nargs='+', default=list(FILES), help='default: mknapcb1.txt .. mknapcb9.txt')
    parser.add_argument('--indices', nargs='+', type=int, default=list(INDICES), help='default: 0 1 2')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS), help='unique nonnegative seeds; default: 1..31')
    parser.add_argument('--population-size', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--max-evaluations', type=int, default=180030)
    parser.add_argument('--out', required=True, type=Path,
                        help='root for <file>/inst_<II>/; existing root allowed, selected instances must not exist')
    parser.add_argument('--dry-run', action='store_true', help='validate without writing or solving')
    return parser


def folder_for(root, selection):
    return root / Path(selection.file_name).stem / f'inst_{selection.index:02d}'


def validate(parser, args):
    try:
        _reject_url(str(args.out), 'output root')
        if _duplicate(args.seeds) is not None:
            raise ValueError(f'duplicate seed: {_duplicate(args.seeds)}')
        for seed in args.seeds:
            hybrid._validate_seed(seed)
        _validate_settings(args.seeds[0], args.population_size, args.iterations, args.max_evaluations)
        selections = _selections(args.instances_dir, 'variant', files=args.files, indices=args.indices)
        if args.out.exists() and not args.out.is_dir():
            raise ValueError(f'output root is not a directory: {args.out}')
        for selection in selections:
            folder = folder_for(args.out, selection)
            if folder.exists() or folder.is_symlink():
                raise ValueError(f'refusing existing output: {folder}')
    except (CampaignError, ExportError, ValueError, OSError, IndexError) as exc:
        parser.error(str(exc))
    return selections


def persist(folder, config, stats):
    _write_json(folder / 'campaign_config.json', config)
    lines = ['Campaña WOA-ABC: v0-v4', f"Estado: {config['status']}",
             f"Corridas: {config['completed_runs']}/{config['total_runs']}",
             'Selección: mayor fitness; empate: menor semilla.',
             'Desviación estándar muestral (n - 1); vacía con una semilla.',
             'No se afirma superioridad.', f"Error: {config['error'] or 'ninguno'}"]
    for row in stats:
        lines.append(f"{row['variant']}: semilla {row['best_seed']}; mejor {row['best_fitness']}; "
                     f"peor {row['worst_fitness']}; media {row['mean_fitness']}; "
                     f"mediana {row['median_fitness']}; std {row['std_fitness']}")
    (folder / 'resumen.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_instance(args, members):
    first = members[0]
    folder = folder_for(args.out, first)
    # Exclusive creation, also protects against another task claiming the same instance.
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(status='running', started=datetime.now().isoformat(timespec='seconds'),
                  file=first.file_name, index=first.index, seeds=args.seeds,
                  variants=list(VARIANTS), reaction='variant', population_size=args.population_size,
                  iterations=args.iterations, max_evaluations=args.max_evaluations,
                  scout_limit_override=None, total_runs=len(members) * len(args.seeds),
                  completed_runs=0, error=None)
    stats = []
    rows = {m.variant: [] for m in members}
    best = {m.variant: None for m in members}
    runs_csv, stats_csv = folder / 'runs.csv', folder / 'statistics.csv'
    _start_csv(runs_csv, RUN_FIELDS)
    _start_csv(stats_csv, STAT_FIELDS)
    persist(folder, config, stats)
    staging = folder / '.runs'
    where = 'initialization'
    with (folder / 'run.log').open('w', encoding='utf-8', buffering=1) as log:
        def message(text):
            print(text, file=log, flush=True)
            print(text, flush=True)
        try:
            for seed in args.seeds:
                for member in members:
                    where = f'{member.variant} seed={seed}'
                    destination = staging / member.variant / f'seed_{seed:02d}'
                    message(f'start {where}')
                    request = ExportRequest(instance=member.path, instance_index=member.index,
                        seed=seed, population_size=args.population_size, iterations=args.iterations,
                        max_evaluations=args.max_evaluations, reaction='variant',
                        variant=member.variant, destination=destination)
                    try:
                        outcome = export_single_run(request)
                    except (ExportError, OSError, ValueError, RuntimeError) as exc:
                        outcome = ExportOutcome(1, destination, f'{type(exc).__name__}: {exc}')
                    metrics = _outcome_row(member, destination, outcome)
                    row = dict(file=member.file_name, index=member.index, variant=member.variant,
                        seed=seed, status=metrics['status'], best_fitness=metrics['fitness'],
                        known_optimum=member.reference if member.reference > 0 else None,
                        gap_pct=metrics['gap'], evaluations=metrics['evaluations'],
                        generations=metrics['cycles_completed'], changes=metrics['profile_changes'],
                        elapsed_s=metrics['elapsed'], recoveries=metrics['recoveries'],
                        scouts=metrics['scouts'], scout_limit=metrics['scout_limit'],
                        reference_source=member.reference_source)
                    if row['best_fitness'] is None:
                        row['status'] = 'failed'
                    _append_csv((runs_csv,), RUN_FIELDS, [row])
                    if row['status'] != 'completed':
                        raise RuntimeError(f'{outcome.message}; evidence: {destination}')
                    rows[member.variant].append(row)
                    config['completed_runs'] += 1
                    current = best[member.variant]
                    if is_better(row, None if current is None else current[0]):
                        if current is not None:
                            shutil.rmtree(current[1])
                        best[member.variant] = (row, destination)
                    else:
                        shutil.rmtree(destination)
                    persist(folder, config, stats)
                    message(f"done {where} fitness={row['best_fitness']}")
            raws = []
            where = 'winner export and comparison'
            for member in members:
                winner, source = best[member.variant]
                target = folder / member.variant / 'best_run'
                target.parent.mkdir()
                shutil.move(str(source), str(target))
                stat = policy_statistics(member.file_name, member.index, member.variant,
                                         rows[member.variant], winner)
                stat['variant'] = stat.pop('policy')
                stats.append(stat)
                raws.append(_read_json(target / 'runs/run_01.json'))
            _append_csv((stats_csv,), STAT_FIELDS, stats)
            plot_comparison(raws, folder, f'{Path(first.file_name).stem}/inst_{first.index:02d}', first.reference)
            # Only scratch created inside this newly claimed instance is removed.
            shutil.rmtree(staging)
            config.update(status='completed', finished=datetime.now().isoformat(timespec='seconds'))
            persist(folder, config, stats)
            message(f"completed {config['completed_runs']}/{config['total_runs']} {folder}")
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            config.update(status='failed', error=f'{where}: {type(exc).__name__}: {exc}')
            persist(folder, config, stats)
            message(f"failed: {config['error']}")
            print(f"error: {config['error']}", file=sys.stderr)
            return 1


def main(argv=None):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        selections = validate(parser, args)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.dry_run:
        print(f'variant_seed_campaign dry run: runs={len(selections) * len(args.seeds)} out={args.out}')
        return 0
    instances = {}
    for selection in selections:
        instances.setdefault((selection.file_name, selection.index), []).append(selection)
    for members in instances.values():
        try:
            if run_instance(args, members):
                return 1
        except OSError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
