#!/bin/bash
#SBATCH --job-name=adaptive_mkp
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

set -euo pipefail
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
(( $# == 3 )) || fail 'Uso interno: worker REPO_ABSOLUTO SIZE BATCH_ABSOLUTO (vía submitter).'
repo=$1
size=$2
batch=$3
[[ $repo == /* && -d $repo && $batch == /* && -d $batch ]] || fail 'Repo y lote deben ser directorios absolutos existentes.'
case "$size" in
    100) families=(1 4 7) ;;
    250) families=(2 5 8) ;;
    500) families=(3 6 9) ;;
    *) fail 'SIZE debe ser 100, 250 o 500.' ;;
esac
task=${SLURM_ARRAY_TASK_ID-}
[[ $task =~ ^[0-8]$ ]] || fail 'SLURM_ARRAY_TASK_ID debe estar entre 0 y 8.'
[[ ${SLURM_ARRAY_JOB_ID-} =~ ^[1-9][0-9]*$ && ${SLURM_JOB_ID-} =~ ^[1-9][0-9]*$ ]] || fail 'Falta contexto válido de Slurm array/job.'
family=${families[task / 3]}
index=$((task % 3))
printf -v case_name 'mknapcb%s_inst_%02d' "$family" "$index"
task_root="$batch/items_$size/$case_name"
[[ ! -e $task_root && ! -L $task_root ]] || fail "La tarea ya tiene salida/reserva: $task_root"

# Validación terminada. Reserva atómica por caso antes de activar el entorno.
# Se conserva incluso si falla Conda/Python: no hay reanudación automática.
mkdir -p -- "$batch/items_$size"
mkdir -- "$task_root"
export MPLCONFIGDIR="$task_root/mplconfig"
mkdir -- "$MPLCONFIGDIR"
cd -- "$repo"
source "${MKP_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate mkp_env
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Un caso, 31 semillas 42..72. El monitor conserva los defaults de config.py.
# Sin cap de época: el presupuesto nativo puede sobrepasar 1000 iteraciones.
exec python3 -B -u -m adaptive_mh_selection.campaign \
    --families "$family" --indices "$index" \
    --runs 31 --seed 42 --iterations 1000 \
    --selectors adaptive --modes ddtw --selection-trace \
    --output "$task_root/campaign"
