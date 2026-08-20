#!/bin/bash
#SBATCH --job-name=mkp_par9
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/mkp_par9_%j.log
#SBATCH --error=./.hpc/errors/mkp_par9_%j.error
#SBATCH --qos=normal

# Cargar entorno de conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Ir al directorio de trabajo
cd /work/lucas.erazo/mkp_lb2

# Crear carpetas de logs si no existen
mkdir -p ./.hpc/logs ./.hpc/errors

# Ejecutar el benchmark en paralelo con hilos (9 archivos x 1ra instancia)
python3 parallel_mkp_first_inst.py
