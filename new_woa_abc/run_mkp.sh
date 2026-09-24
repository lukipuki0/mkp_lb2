#!/bin/bash
# Ejecuta las tres primeras instancias de las nueve familias MKP.

#SBATCH --job-name=woa_abc_mkp
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/new_woaabc_mkp_par9_%j.log
#SBATCH --error=./.hpc/errors/new_woaabc_mkp_par9_%j.error
#SBATCH --qos=long

# Un hilo numérico por worker para evitar contención de CPU.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Cargar el entorno de conda dedicado.
source /home/lucas.erazo/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

# Ir al directorio de trabajo.
cd /work/lucas.erazo/mkp_lb2
export PYTHONPATH="/work/lucas.erazo/mkp_lb2${PYTHONPATH:+:$PYTHONPATH}"

# Crear carpetas de logs y resultados.
mkdir -p ./.hpc/logs ./.hpc/errors ./new_woa_abc/resultados/mkp

echo "=== INICIANDO WOA--ABC MKP ==="
echo "CPUs asignadas por Slurm: ${SLURM_CPUS_PER_TASK:-9}"
echo "Experimento: 9 familias, índices 0-2, 3 grupos por tamaño, 7 variantes, 31 runs"
echo "Búsqueda local: desactivada"
echo "Directorio de salida: ./new_woa_abc/resultados/mkp"

# Los demás parámetros usan los valores predeterminados del runner.
python3 -u -m new_woa_abc.runners.run_mkp \
    --instances "${INSTANCES:-0-2}" \
    --workers "${SLURM_CPUS_PER_TASK:-9}"
