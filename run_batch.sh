#!/bin/bash
#SBATCH --job-name=hybrid_mkp   # Nombre que verás en la cola
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1         # 4 CPUs para procesar datos (cargar imágenes)
#SBATCH --mem=32G                   # 16 GB de RAM del sistema (no de video)
#SBATCH --output=./.hpc/logs/hybrid_mkp%j.log   # Archivo donde se guardará lo que imprima el script (%j es el ID del trabajo)
#SBATCH --error=./.hpc/errors/hybrid_mkp%j.error        # Archivo donde se guardarán los errores si falla
#SBATCH --qos=normal               #QOS de HPC



# Limitar hilos internos de NumPy/OpenBLAS (evita overhead crítico en CPU)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Cargar entorno de conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Ir al directorio de trabajo
cd /work/lucas.erazo/mkp_lb2

# Ejecutar el benchmark
python3 batch_benchmark.py
