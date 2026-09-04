#!/bin/bash
# ==============================================================================
# Script de Ejecución Slurm / HPC: Benchmark Continuo CEC2022 (F1 - F12)
# ==============================================================================
#SBATCH --job-name=cec2022_bench
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/cec2022_%j.log
#SBATCH --error=./.hpc/errors/cec2022_%j.error
#SBATCH --qos=normal

# ── 1. Limitar hilos internos de librerías para evitar sobrecarga en CPU ──
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── 2. Cargar entorno Conda dedicado ──
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

# ── 3. Directorio de trabajo y PYTHONPATH ──
cd /work/lucas.erazo/mkp_lb2
export PYTHONPATH="/work/lucas.erazo/mkp_lb2:${PYTHONPATH}"

# ── 4. Crear carpetas de logs y resultados si no existen ──
mkdir -p ./.hpc/logs ./.hpc/errors ./continuous_benchmark/resultados

echo "=================================================================="
echo "  INICIANDO BENCHMARK CONTINUO IEEE CEC2022 (F1 a F12)"
echo "  Fecha / Hora       : $(date)"
echo "  Nodo de Ejecución  : $(hostname)"
echo "  Job ID Slurm       : ${SLURM_JOB_ID:-N/A (Ejecución Local)}"
echo "  CPUs asignadas     : ${SLURM_CPUS_PER_TASK:-1}"
echo "  Entorno Python     : $(which python3)"
echo "=================================================================="

# ── 5. Ejecutar el benchmark continuo completo ──
python3 -u -m continuous_benchmark.benchmark_continuo

EXIT_CODE=$?

echo ""
echo "=================================================================="
echo "  FINALIZACIÓN DEL TRABAJO CEC2022"
echo "  Fecha / Hora       : $(date)"
echo "  Código de salida   : ${EXIT_CODE}"
echo "=================================================================="

exit ${EXIT_CODE}
