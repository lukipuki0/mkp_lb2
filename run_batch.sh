#!/bin/bash
#SBATCH --job-name=hybrid_mkp   # Nombre que verás en la cola
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1         # 4 CPUs para procesar datos (cargar imágenes)
#SBATCH --mem=32G                   # 16 GB de RAM del sistema (no de video)
#SBATCH --output=./.hpc/logs/hybrid_mkp_%j.log   # Archivo donde se guardará lo que imprima el script (%j es el ID del trabajo)
#SBATCH --error=./.hpc/errors/hybrid_mkp_%j.error        # Archivo donde se guardarán los errores si falla
#SBATCH --qos=normal               #QOS de HPC

# Limitar hilos internos de NumPy/OpenBLAS (evita overhead crítico en CPU)
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

# Crear carpetas si no existen
mkdir -p ./.hpc/logs ./.hpc/errors

# Ejecutar el benchmark
python3 -u -m hybrid_mkp.batch_benchmark
