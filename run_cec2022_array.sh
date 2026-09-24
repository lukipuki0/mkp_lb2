#!/bin/bash
# ============================================================================
# CEC2022 rápido para Slurm: una función CEC por tarea del array (F1-F12).
#
# Cada tarea escribe únicamente en:
#   continuous_benchmark/resultados/<CEC_BATCH_DIR>/CEC_XX_*/
# Así no hay colisiones entre tareas ni escritura concurrente de un mismo CSV.
# ============================================================================
#SBATCH --job-name=cec2022_array
#SBATCH --chdir=/work/lucas.erazo/mkp_lb2
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=1-12%12
#SBATCH --output=/work/lucas.erazo/mkp_lb2/.hpc/logs/cec2022_%A_%a.out
#SBATCH --error=/work/lucas.erazo/mkp_lb2/.hpc/errors/cec2022_%A_%a.err
#SBATCH --qos=normal

set -euo pipefail

PROJECT_DIR="/work/lucas.erazo/mkp_lb2"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Una tarea = una función y un proceso Python. Evita oversubscription de NumPy.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ./.hpc/logs ./.hpc/errors ./continuous_benchmark/resultados

: "${SLURM_ARRAY_TASK_ID:?Este script debe ejecutarse como un Slurm array}"
: "${SLURM_JOB_ID:?Falta SLURM_JOB_ID}"

export CEC_FUNCTION_ID="$SLURM_ARRAY_TASK_ID"
CEC_MODE="$(python3 -c 'from continuous_benchmark.benchmark_continuo import STAG_USE_DDTW; print("ddtw" if STAG_USE_DDTW else "dtw")')"
export CEC_BATCH_DIR="$PROJECT_DIR/continuous_benchmark/resultados/run_${CEC_MODE}_array_${SLURM_JOB_ID}"

# Opcionalmente se pueden sobreescribir al enviar el trabajo:
#   sbatch --export=ALL,CEC_N_RUNS=31,CEC_MAX_ITERS=1000 run_cec2022_array.sh
export CEC_N_RUNS="${CEC_N_RUNS:-31}"
export CEC_MAX_ITERS="${CEC_MAX_ITERS:-1000}"
export CEC_DIMENSION="${CEC_DIMENSION:-10}"

echo "============================================================"
echo "CEC2022 | función F${CEC_FUNCTION_ID}"
echo "Job/array : ${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Modo      : ${CEC_MODE^^}"
echo "Runs      : ${CEC_N_RUNS}"
echo "Iter/run  : ${CEC_MAX_ITERS}"
echo "Salida    : ${CEC_BATCH_DIR}"
echo "Nodo      : $(hostname)"
echo "Inicio    : $(date)"
echo "============================================================"

python3 -u -m continuous_benchmark.benchmark_continuo

echo "CEC F${CEC_FUNCTION_ID} finalizada: $(date)"
