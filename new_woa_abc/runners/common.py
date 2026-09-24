"""Argumentos y construcción de configuración compartidos por los runners."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import sys

import numpy as np

from new_woa_abc.config import DTWConfig, StrategyConfig, WOAABCConfig
from new_woa_abc.dtw.strategies import M0_NO_DTW, resolve_variants


def add_common_arguments(parser: argparse.ArgumentParser, default_dtw: DTWConfig) -> None:
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abc-limit", type=int)
    parser.add_argument("--abc-guide-strength", type=float, default=0.20)
    parser.add_argument("--abc-vector-probability", type=float, default=0.35)
    parser.add_argument("--abc-vector-scale", type=float, default=0.10)
    parser.add_argument("--step-initial", type=float, default=1.0)
    parser.add_argument("--step-final", type=float, default=0.05)
    parser.add_argument("--momentum-factor", type=float, default=0.20)
    parser.add_argument("--abc-limit-divisor", type=int, default=4)
    parser.add_argument("--min-abc-limit", type=int, default=2)
    parser.add_argument("--window", type=int, default=default_dtw.window)
    parser.add_argument("--band", type=int, default=default_dtw.band)
    parser.add_argument("--min-slope", type=float, default=default_dtw.min_slope)
    parser.add_argument("--plateau-max", type=int, default=default_dtw.plateau_max)
    parser.add_argument("--patience", type=int, default=default_dtw.patience)
    parser.add_argument(
        "--ddtw",
        action=argparse.BooleanOptionalAction,
        default=default_dtw.use_ddtw,
    )
    parser.add_argument(
        "--adaptive-thresholds",
        action=argparse.BooleanOptionalAction,
        default=default_dtw.adapt_thresholds,
    )
    parser.add_argument("--p-low", type=float, default=default_dtw.p_low)
    parser.add_argument("--p-high", type=float, default=default_dtw.p_high)
    parser.add_argument("--d2-scale", type=float, default=2.0)
    parser.add_argument("--sigmoid-k", type=float, default=5.0)
    parser.add_argument("--sigmoid-center", type=float, default=0.5)
    parser.add_argument("--hysteresis-enter", type=float, default=0.65)
    parser.add_argument("--hysteresis-exit", type=float, default=0.35)
    parser.add_argument(
        "--statistical-analysis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="por defecto se activa automáticamente cuando --runs es mayor que 1",
    )
    parser.add_argument(
        "--reference-variant",
        default=M0_NO_DTW,
        help="variante de control para Wilcoxon y Holm",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--save-all-run-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="en modo estadístico el valor por defecto conserva detalles solo de la mejor corrida",
    )
    parser.add_argument("--verbose-engine", action="store_true")


def statistical_mode(args: argparse.Namespace) -> bool:
    """Activa inferencia automáticamente cuando existe replicación."""

    if args.statistical_analysis is None:
        return args.runs > 1
    return bool(args.statistical_analysis)


def save_all_run_artifacts(args: argparse.Namespace) -> bool:
    """Con una campaña estadística, conserva por defecto solo el mejor detalle."""

    if args.save_all_run_artifacts is None:
        return not statistical_mode(args)
    return bool(args.save_all_run_artifacts)


def validate_statistical_setup(
    args: argparse.Namespace,
    variants: list[str],
) -> bool:
    enabled = statistical_mode(args)
    try:
        args.reference_variant = resolve_variants([args.reference_variant])[0]
    except ValueError as exc:
        raise ValueError(
            f"--reference-variant inválida: {args.reference_variant}"
        ) from exc
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha debe estar entre 0 y 1")
    if enabled and args.runs < 2:
        raise ValueError("el análisis estadístico requiere --runs >= 2")
    if enabled and len(variants) < 2:
        raise ValueError("el análisis estadístico requiere al menos dos variantes")
    if enabled and args.reference_variant not in variants:
        raise ValueError(
            "la variante de referencia debe estar incluida en --variants: "
            f"{args.reference_variant}"
        )
    return enabled


def build_config(args: argparse.Namespace, variant: str, seed: int) -> WOAABCConfig:
    dtw = DTWConfig(
        window=args.window,
        band=args.band,
        min_slope=args.min_slope,
        plateau_max=args.plateau_max,
        patience=args.patience,
        use_ddtw=args.ddtw,
        adapt_thresholds=args.adaptive_thresholds,
        p_low=args.p_low,
        p_high=args.p_high,
    )
    strategy = StrategyConfig(
        d2_scale=args.d2_scale,
        sigmoid_k=args.sigmoid_k,
        sigmoid_center=args.sigmoid_center,
        hysteresis_enter=args.hysteresis_enter,
        hysteresis_exit=args.hysteresis_exit,
    )
    return WOAABCConfig(
        pop_size=args.pop_size,
        iterations=args.iterations,
        seed=seed,
        variant=variant,
        abc_limit=args.abc_limit,
        abc_guide_strength=args.abc_guide_strength,
        abc_vector_probability=args.abc_vector_probability,
        abc_vector_scale=args.abc_vector_scale,
        step_initial=args.step_initial,
        step_final=args.step_final,
        momentum_factor=args.momentum_factor,
        abc_limit_divisor=args.abc_limit_divisor,
        min_abc_limit=args.min_abc_limit,
        dtw=dtw,
        strategy=strategy,
    )


def configuration_dict(args: argparse.Namespace, domain: str) -> dict:
    sample = build_config(args, "M0_no_dtw", args.seed)
    statistics_enabled = statistical_mode(args)
    return {
        "implementation": "new_woa_abc",
        "reference": "DTW_optimization-main",
        "domain": domain,
        "runs": args.runs,
        "iterations": args.iterations,
        "pop_size": args.pop_size,
        "base_seed": args.seed,
        "seed_policy": "seed = base_seed + run_index; paired across variants",
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "dtw": asdict(sample.dtw),
        "strategy": asdict(sample.strategy),
        "base_profile": asdict(sample.base_profile),
        "exploit_profile": asdict(sample.exploit_profile),
        "explore_profile": asdict(sample.explore_profile),
        "woa_abc": {
            "abc_guide_strength": sample.abc_guide_strength,
            "abc_vector_probability": sample.abc_vector_probability,
            "abc_vector_scale": sample.abc_vector_scale,
            "step_initial": sample.step_initial,
            "step_final": sample.step_final,
            "momentum_factor": sample.momentum_factor,
            "abc_limit_divisor": sample.abc_limit_divisor,
            "min_abc_limit": sample.min_abc_limit,
        },
        "statistical_analysis": {
            "enabled": statistics_enabled,
            "recommended_runs": 31,
            "actual_runs": args.runs,
            "paired_by": "seed",
            "reference_variant": args.reference_variant,
            "alpha": args.alpha,
            "tests": ["Shapiro-Wilk", "Wilcoxon", "Holm", "Mann-Whitney U", "Friedman"],
            "save_all_run_artifacts": save_all_run_artifacts(args),
        },
    }


__all__ = [
    "add_common_arguments",
    "build_config",
    "configuration_dict",
    "save_all_run_artifacts",
    "statistical_mode",
    "validate_statistical_setup",
]
