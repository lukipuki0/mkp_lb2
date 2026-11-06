#!/bin/bash
#SBATCH --job-name=woa_abc_variants
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=4G
#SBATCH --array=0-26%9
#SBATCH --output=woa_abc_variants_%A_%a.log
#SBATCH --error=woa_abc_variants_%A_%a.error
#SBATCH --qos=long
# Desde la raíz: sbatch run_woa_abc_variants_hpc.sh
# 27 instancias x 31 semillas x 5 variantes = 4185 corridas.
# Hasta 9 instancias simultáneas; cada tarea ejecuta sus semillas en orden.
# Conserva métricas de las 31 semillas y sólo el export completo ganador por variante.
# No modifica la campaña de políticas. No reintenta ni sobrescribe resultados.
set -euo pipefail
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
    printf 'Uso: sbatch run_woa_abc_variants_hpc.sh (desde la raíz del repo)\n'
    printf '     bash run_woa_abc_variants_hpc.sh --dry-run\n'
    printf 'MKP: v0-v4, semillas 1-31, archivos 1-9, índices 0-2. Total: 4185 corridas.\n'
    printf 'Población 30, 1000 ciclos, máximo 180030 evaluaciones por corrida.\n'
    printf 'Conda: mkp_env; MKP_CONDA_SH permite indicar conda.sh.\n'
    printf 'Resultados: resultados/woa_abc_variants_<array_job>/mknapcb<F>/inst_<II>/v0..v4/best_run/\n'
    printf 'runs.csv: 155 corridas; statistics.csv: 5 variantes; empate: menor semilla.\n'
}
command_for() {
    local task=$1 root=$2
    local family=$((task / 3 + 1)) index=$((task % 3))
    cmd=(python3 -B -u -m woa_abc.variant_seed_campaign
         --instances-dir "$repo/instancias"
         --files "mknapcb${family}.txt" --indices "$index"
         --seeds {1..31} --population-size 30
         --iterations 1000 --max-evaluations 180030 --out "$root")
}
if [[ $# == 1 && $1 == --help ]]; then usage; exit 0; fi
dry=false
if [[ $# == 1 && $1 == --dry-run ]]; then dry=true
elif (( $# != 0 )); then fail 'Argumento no reconocido (ver --help).'; fi
repo=${SLURM_SUBMIT_DIR:-$(pwd -P)}
[[ -f $repo/woa_abc/variant_seed_campaign.py && -d $repo/instancias ]] || fail 'Ejecute desde la raíz del repo.'
if $dry; then
    printf '27 tareas, hasta 9 simultáneas; 4185 corridas (v0-v4 por semilla).\n'
    for task in {0..26}; do
        command_for "$task" "$repo/resultados/woa_abc_variants_<array_job>"
        printf ' '; printf ' %s' "${cmd[@]}"; printf '\n'
    done
    exit 0
fi
[[ ${SLURM_ARRAY_JOB_ID-} =~ ^[1-9][0-9]*$ ]] || fail 'Envíe con sbatch run_woa_abc_variants_hpc.sh.'
[[ ${SLURM_ARRAY_TASK_ID-} =~ ^([0-9]|1[0-9]|2[0-6])$ ]] || fail 'SLURM_ARRAY_TASK_ID debe estar entre 0 y 26.'
task=$SLURM_ARRAY_TASK_ID
root="$repo/resultados/woa_abc_variants_$SLURM_ARRAY_JOB_ID"
printf -v instance '%s/mknapcb%s/inst_%02d' "$root" "$((task / 3 + 1))" "$((task % 3))"
[[ ! -e $instance && ! -L $instance ]] || fail "El destino ya existe: $instance"
# Activate before creating results, so an unavailable environment leaves no experiment.
set +u
source "${MKP_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate mkp_env
set -u
cd -- "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Python validates before allocation and owns run.log and the instance directory.
# Matplotlib's default temporary/config location stays outside the experiment.
command_for "$task" "$root"
printf 'Inicio tarea %s: %s\n' "$task" "$instance"
if "${cmd[@]}"; then
    printf 'OK tarea %s\n' "$task"
else
    status=$?
    printf 'Error: tarea %s falló (código %s). Evidencia: %s/run.log y log Slurm\n' "$task" "$status" "$instance" >&2
    exit "$status"
fi
printf 'Completadas 31 semillas x 5 variantes: %s\n' "$instance"
