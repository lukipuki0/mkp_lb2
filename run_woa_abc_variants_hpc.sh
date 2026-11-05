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
# Cada semilla conserva las cinco variantes y sus diagnósticos completos.
# No modifica la campaña de políticas. No reintenta ni sobrescribe resultados.
set -euo pipefail
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
    printf 'Uso: sbatch run_woa_abc_variants_hpc.sh (desde la raíz del repo)\n'
    printf '     bash run_woa_abc_variants_hpc.sh --dry-run\n'
    printf 'MKP: v0-v4, semillas 1-31, archivos 1-9, índices 0-2. Total: 4185 corridas.\n'
    printf 'Población 30, 1000 ciclos, máximo 180030 evaluaciones por corrida.\n'
    printf 'Conda: mkp_env; MKP_CONDA_SH permite indicar conda.sh.\n'
    printf 'Resultados: resultados/woa_abc_variants_<array_job>/mknapcb<F>/inst_<II>/seed_<SS>/\n'
}
command_for() {
    local task=$1 seed=$2 root=$3
    local family=$((task / 3 + 1)) index=$((task % 3))
    printf -v destination '%s/mknapcb%s/inst_%02d/seed_%02d' "$root" "$family" "$index" "$seed"
    cmd=(python3 -B -u -m woa_abc.campaign
         --instances-dir "$repo/instancias"
         --files "mknapcb${family}.txt" --indices "$index"
         --seed "$seed" --reaction variant --population-size 30
         --iterations 1000 --max-evaluations 180030 --output-root "$destination")
}
if [[ $# == 1 && $1 == --help ]]; then usage; exit 0; fi
dry=false
if [[ $# == 1 && $1 == --dry-run ]]; then dry=true
elif (( $# != 0 )); then fail 'Argumento no reconocido (ver --help).'; fi
repo=${SLURM_SUBMIT_DIR:-$(pwd -P)}
[[ -f $repo/woa_abc/campaign.py && -d $repo/instancias ]] || fail 'Ejecute desde la raíz del repo.'
if $dry; then
    printf '27 tareas, hasta 9 simultáneas; 4185 corridas (v0-v4 por semilla).\n'
    for task in {0..26}; do
        for seed in {1..31}; do
            command_for "$task" "$seed" "$repo/resultados/woa_abc_variants_<array_job>"
            printf ' '; printf ' %s' "${cmd[@]}"; printf '\n'
        done
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
mkdir -p -- "$(dirname -- "$instance")"
mkdir -- "$instance"
export MPLCONFIGDIR="$instance/.mplconfig"
for seed in {1..31}; do
    command_for "$task" "$seed" "$root"
    mkdir -- "$destination"
    printf 'Inicio tarea %s, semilla %s: %s\n' "$task" "$seed" "$destination"
    if "${cmd[@]}" > "$destination/run.log" 2>&1; then
        printf 'OK tarea %s, semilla %s\n' "$task" "$seed"
    else
        status=$?
        printf 'Error: semilla %s falló (código %s). Evidencia: %s/run.log\n' "$seed" "$status" "$destination" >&2
        exit "$status"
    fi
done
printf 'Completadas 31 semillas x 5 variantes: %s\n' "$instance"
