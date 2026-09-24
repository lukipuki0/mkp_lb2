#!/usr/bin/env bash
# Ejecución local/HPC de las siete variantes nuevas sobre CEC2022.

#SBATCH --job-name=new_woaabc_cec
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=/work/lucas.erazo/mkp_lb2/.hpc/logs/new_woaabc_cec_%j.out
#SBATCH --error=/work/lucas.erazo/mkp_lb2/.hpc/errors/new_woaabc_cec_%j.err
#SBATCH --qos=normal

set -euo pipefail

# Un hilo numérico por worker para evitar sobresuscripción de CPU.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Usar el mismo entorno reproducible que los demás benchmarks.
CONDA_SETUP="${CONDA_SETUP:-/home/lucas.erazo/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SETUP" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "$CONDA_SETUP"
    conda activate "${CONDA_ENV:-mkp_env}"
    set -u
fi

PROJECT_DIR="${PROJECT_DIR:-/work/lucas.erazo/mkp_lb2}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.hpc/matplotlib}"
mkdir -p .hpc/logs .hpc/errors .hpc/matplotlib new_woa_abc/resultados/cec

WORKER_COUNT="${WORKERS:-${SLURM_CPUS_PER_TASK:-12}}"
RESUME_FLAGS=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    RESUME_FLAGS+=("--resume-dir" "$RESUME_DIR")
fi

echo "New WOA--ABC CEC | workers=$WORKER_COUNT"
echo "Python: $(command -v python3)"
python3 -c 'import numpy as np; print(f"NumPy: {np.__version__} ({np.__file__})")'

case "${DTW_MODE:-ddtw}" in
    ddtw) DTW_FLAG="--ddtw" ;;
    dtw) DTW_FLAG="--no-ddtw" ;;
    *) echo "DTW_MODE debe ser 'dtw' o 'ddtw'" >&2; exit 2 ;;
esac

case "${ADAPTIVE_THRESHOLDS:-true}" in
    true|1|yes) THRESHOLD_FLAG="--adaptive-thresholds" ;;
    false|0|no) THRESHOLD_FLAG="--no-adaptive-thresholds" ;;
    *) echo "ADAPTIVE_THRESHOLDS debe ser true o false" >&2; exit 2 ;;
esac

STATISTICS_FLAGS=()
case "${STATISTICAL_ANALYSIS:-auto}" in
    auto) ;;
    true|1|yes) STATISTICS_FLAGS+=("--statistical-analysis") ;;
    false|0|no) STATISTICS_FLAGS+=("--no-statistical-analysis") ;;
    *) echo "STATISTICAL_ANALYSIS debe ser auto, true o false" >&2; exit 2 ;;
esac

ARTIFACT_FLAGS=()
case "${SAVE_ALL_RUN_ARTIFACTS:-auto}" in
    auto) ;;
    true|1|yes) ARTIFACT_FLAGS+=("--save-all-run-artifacts") ;;
    false|0|no) ARTIFACT_FLAGS+=("--no-save-all-run-artifacts") ;;
    *) echo "SAVE_ALL_RUN_ARTIFACTS debe ser auto, true o false" >&2; exit 2 ;;
esac

python3 -u -m new_woa_abc.runners.run_cec \
    --functions ${FUNCTIONS:-all} \
    --dimension "${DIMENSION:-10}" \
    --output-dir "${OUTPUT_DIR:-$PROJECT_DIR/new_woa_abc/resultados/cec}" \
    --variants ${VARIANTS:-all} \
    --workers "$WORKER_COUNT" \
    --runs "${RUNS:-31}" \
    --iterations "${ITERATIONS:-1000}" \
    --pop-size "${POP_SIZE:-30}" \
    --seed "${SEED:-42}" \
    --window "${WINDOW:-20}" \
    --band "${BAND:-2}" \
    --min-slope "${MIN_SLOPE:-2.0}" \
    --plateau-max "${PLATEAU_MAX:-4}" \
    --patience "${PATIENCE:-2}" \
    --abc-guide-strength "${ABC_GUIDE_STRENGTH:-0.20}" \
    --abc-vector-probability "${ABC_VECTOR_PROBABILITY:-0.35}" \
    --abc-vector-scale "${ABC_VECTOR_SCALE:-0.10}" \
    --step-initial "${STEP_INITIAL:-1.0}" \
    --step-final "${STEP_FINAL:-0.05}" \
    --momentum-factor "${MOMENTUM_FACTOR:-0.20}" \
    --abc-limit-divisor "${ABC_LIMIT_DIVISOR:-4}" \
    --min-abc-limit "${MIN_ABC_LIMIT:-2}" \
    --reference-variant "${REFERENCE_VARIANT:-M0_no_dtw}" \
    --alpha "${ALPHA:-0.05}" \
    "$DTW_FLAG" \
    "$THRESHOLD_FLAG" \
    "${RESUME_FLAGS[@]}" \
    "${STATISTICS_FLAGS[@]}" \
    "${ARTIFACT_FLAGS[@]}"
