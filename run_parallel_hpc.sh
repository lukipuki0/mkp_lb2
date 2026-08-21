#!/bin/bash
#SBATCH --job-name=mkp_par9
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/mkp_par9_%j.log
#SBATCH --error=./.hpc/errors/mkp_par9_%j.error
#SBATCH --qos=normal

# Limitar hilos internos de NumPy/OpenBLAS por worker (evita contención destructiva de CPU)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Cargar entorno de conda dedicado
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

# Ir al directorio de trabajo
cd /work/lucas.erazo/mkp_lb2

# Crear carpetas de logs si no existen
mkdir -p ./.hpc/logs ./.hpc/errors

# Configurar directorio de almacenamiento temporal en disco local del nodo (/tmp)
LOCAL_TMP_DIR="/tmp/mkp_hpc_job_${SLURM_JOB_ID:-local}"
mkdir -p "$LOCAL_TMP_DIR"
export MKP_TMP_DIR="$LOCAL_TMP_DIR"

echo "=== INICIANDO TRABAJO HPC ==="
echo "CPUs asignadas por Slurm: ${SLURM_CPUS_PER_TASK:-1}"
echo "Directorio temporal local: $MKP_TMP_DIR"

# Ejecutar el benchmark en paralelo
python3 parallel_mkp_first_inst.py

# Copiar artefactos finales generados en /tmp al almacenamiento permanente /work
if [ -d "$LOCAL_TMP_DIR" ]; then
    echo "Copiando resultados finales desde $LOCAL_TMP_DIR a ./resultados/parallel_hpc_first_inst/ ..."
    mkdir -p ./resultados/parallel_hpc_first_inst
    cp -r "$LOCAL_TMP_DIR"/* ./resultados/parallel_hpc_first_inst/ 2>/dev/null || true
    rm -rf "$LOCAL_TMP_DIR"
    echo "Resultados sincronizados y almacenamiento temporal limpiado exitosamente."
fi

