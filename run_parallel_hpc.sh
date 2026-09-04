#!/bin/bash
#SBATCH --job-name=ddtw_2k
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/mkp_par9_%j.log
#SBATCH --error=./.hpc/errors/mkp_par9_%j.error
#SBATCH --qos=long

# Limitar hilos internos de NumPy/OpenBLAS por worker (evita contención destructiva de CPU)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Cargar entorno de conda dedicado
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

# Ir al directorio de trabajo
cd /work/lucas.erazo/mkp_lb2
export PYTHONPATH="/work/lucas.erazo/mkp_lb2:${PYTHONPATH}"

# Crear carpetas de logs y resultados si no existen
mkdir -p ./.hpc/logs ./.hpc/errors ./resultados/parallel_hpc_first_inst

# Escribir directamente en el almacenamiento permanente /work para visibilidad inmediata ("altiro")
unset MKP_TMP_DIR

echo "=== INICIANDO TRABAJO HPC ==="
echo "CPUs asignadas por Slurm: ${SLURM_CPUS_PER_TASK:-1}"
echo "Directorio de salida     : ./resultados/parallel_hpc_first_inst"

# Ejecutar el benchmark en paralelo
python3 -u -m hybrid_mkp.parallel_mkp_first_inst

