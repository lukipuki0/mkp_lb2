#!/bin/bash
#SBATCH --job-name=mkp_par9
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/mkp_par9_%j.log
#SBATCH --error=./.hpc/errors/mkp_par9_%j.error
#SBATCH --qos=normal

# Limitar hilos internos de NumPy/OpenBLAS por worker (evita contención destructiva)
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

# Ejecutar el benchmark en paralelo con 9 procesos independientes
python3 parallel_mkp_first_inst.py
