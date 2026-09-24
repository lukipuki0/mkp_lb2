#!/usr/bin/env bash
# Ejecuta las siete variantes WOA--ABC sobre HRES2.
# Puede ejecutarse localmente o enviarse con: sbatch run_woa_abc_hres2.sh
#
# Variables opcionales:
#   RUNS=1 ITERATIONS=1000 POP_SIZE=30 SEED=42
#   OUTPUT_DIR=woa_abc/resultados/hres2

#SBATCH --job-name=woaabc_hres2
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/work/lucas.erazo/mkp_lb2/.hpc/logs/woaabc_hres2_%j.out
#SBATCH --error=/work/lucas.erazo/mkp_lb2/.hpc/errors/woaabc_hres2_%j.err
#SBATCH --qos=normal

set -euo pipefail

PROJECT_DIR="/work/lucas.erazo/mkp_lb2"
cd "$PROJECT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-mkp_env}"
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.hpc/matplotlib}"

mkdir -p .hpc/logs .hpc/errors .hpc/matplotlib woa_abc/resultados/hres2

RUNS="${RUNS:-1}"
ITERATIONS="${ITERATIONS:-1000}"
POP_SIZE="${POP_SIZE:-30}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/woa_abc/resultados/hres2}"

echo "HRES2 WOA--ABC | variantes: todas | runs=$RUNS | iteraciones=$ITERATIONS"
echo "Salida=$OUTPUT_DIR"

python3 -u -m woa_abc.run_hres2_variants \
    --variants all \
    --runs "$RUNS" \
    --iterations "$ITERATIONS" \
    --pop-size "$POP_SIZE" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
