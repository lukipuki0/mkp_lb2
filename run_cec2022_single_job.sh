#!/bin/bash
# ============================================================================
# CEC2022 paralelo en UN solo job Slurm.
#
# Se usa cuando la cuenta/QOS no permite enviar arrays grandes. El job reserva
# 12 CPUs y lanza F1-F12 como procesos independientes, uno por CPU.
# ============================================================================
#SBATCH --job-name=cec2022_parallel
#SBATCH --chdir=/work/lucas.erazo/mkp_lb2
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/work/lucas.erazo/mkp_lb2/.hpc/logs/cec2022_parallel_%j.out
#SBATCH --error=/work/lucas.erazo/mkp_lb2/.hpc/errors/cec2022_parallel_%j.err
#SBATCH --qos=normal

set -euo pipefail

PROJECT_DIR="/work/lucas.erazo/mkp_lb2"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ./.hpc/logs ./.hpc/errors ./continuous_benchmark/resultados

# STAG_USE_DDTW en benchmark_continuo.py controla tanto el algoritmo como el
# nombre de la carpeta; no hay otra variable que pueda sobreescribirlo.
CEC_MODE="$(python3 -c 'from continuous_benchmark.benchmark_continuo import STAG_USE_DDTW; print("ddtw" if STAG_USE_DDTW else "dtw")')"
export CEC_BATCH_DIR="$PROJECT_DIR/continuous_benchmark/resultados/run_${CEC_MODE}_job_${SLURM_JOB_ID}"
export CEC_N_RUNS="${CEC_N_RUNS:-31}"
export CEC_MAX_ITERS="${CEC_MAX_ITERS:-1000}"
export CEC_DIMENSION="${CEC_DIMENSION:-10}"

LOG_DIR="$PROJECT_DIR/.hpc/logs/cec2022_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "CEC2022 paralelo en un solo job"
echo "Job       : ${SLURM_JOB_ID}"
echo "Modo      : ${CEC_MODE^^}"
echo "CPUs      : ${SLURM_CPUS_PER_TASK:-12}"
echo "Runs      : ${CEC_N_RUNS}"
echo "Iter/run  : ${CEC_MAX_ITERS}"
echo "Salida    : ${CEC_BATCH_DIR}"
echo "Inicio    : $(date)"
echo "============================================================"

pids=()
for cec_id in $(seq 1 12); do
    (
        export CEC_FUNCTION_ID="$cec_id"
        echo "[F${cec_id}] inicio: $(date)"
        python3 -u -m continuous_benchmark.benchmark_continuo \
            > "$LOG_DIR/F${cec_id}.out" \
            2> "$LOG_DIR/F${cec_id}.err"
        echo "[F${cec_id}] fin: $(date)"
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if (( status != 0 )); then
    echo "[ERROR] Una o más funciones CEC fallaron. Revisar: $LOG_DIR" >&2
    exit "$status"
fi

echo "Generando tablas, estadísticas y gráficos globales..."
python3 -u -m continuous_benchmark.resumir_cec2022 \
    --batch-dir "$CEC_BATCH_DIR" \
    --dimension "$CEC_DIMENSION" \
    --max-iters "$CEC_MAX_ITERS"

echo "CEC2022 completado correctamente: $(date)"
echo "Logs individuales: $LOG_DIR"
