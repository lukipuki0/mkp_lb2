#!/bin/bash
#SBATCH --job-name=batch_mkp
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Cargar entorno de conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Ir al directorio de trabajo
cd /work/lucas.erazo/mkp_lb2

# Ejecutar el benchmark
python3 batch_benchmark.py
