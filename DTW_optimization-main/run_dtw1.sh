#!/bin/bash
#SBATCH --job-name=dtw_opt1   # Nombre que verás en la cola
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=20           # 4 CPUs para procesar datos (cargar imágenes)
#SBATCH --mem=32G                   # 16 GB de RAM del sistema (no de video)
#SBATCH --output=./logs/dtw_opt1%j.log   # Archivo donde se guardará lo que imprima el script (%j es el ID del trabajo)
#SBATCH --error=./errors/dtw_opt1%j.error        # Archivo donde se guardarán los errores si falla
#SBATCH --qos=normal               #QOS de HPC

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1



# 1. Cargar el entorno (OBLIGATORIO)
# Primero cargamos el módulo de conda si es necesario (a veces no hace falta si ya está en .bashrc, pero es buena práctica)
source $HOME/miniconda3/etc/profile.d/conda.sh

echo "Iniciando trabajo en el nodo: $SLURMD_NODENAME"
echo "CPUs asignadas por SLURM: $SLURM_CPUS_PER_TASK"
echo "Variables de hilos configuradas a: $OMP_NUM_THREADS"


# 2. Activar tu entorno virtual
conda activate DTW_optimization

# 4. Ejecutar tu script de Python
#python run_all_hpc.py --instancia instances/mknapcb1.txt
#python -m analisis.estadistico --instancia instances/mknapcb1.txt
#python run_all_hpc.py --instancia instances/mknapcb2.txt
#python -m analisis.estadistico --instancia instances/mknapcb2.txt
#python run_all_hpc.py --instancia instances/mknapcb3.txt
#python -m analisis.estadistico --instancia instances/mknapcb3.txt
#python run_all_hpc.py --instancia instances/mknapcb4.txt
#python -m analisis.estadistico --instancia instances/mknapcb4.txt
#python run_all_hpc.py --instancia instances/mknapcb5.txt
#python -m analisis.estadistico --instancia instances/mknapcb5.txt
python run_all_hpc.py --instancia instances/mknapcb6.txt
python -m analisis.estadistico --instancia instances/mknapcb6.txt
python run_all_hpc.py --instancia instances/mknapcb7.txt
python -m analisis.estadistico --instancia instances/mknapcb7.txt
python run_all_hpc.py --instancia instances/mknapcb8.txt
python -m analisis.estadistico --instancia instances/mknapcb8.txt
python run_all_hpc.py --instancia instances/mknapcb9.txt
python -m analisis.estadistico --instancia instances/mknapcb9.txt

echo "<<Script de Trabajo terminado>>"
