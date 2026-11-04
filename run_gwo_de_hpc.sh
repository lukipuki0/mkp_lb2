#!/bin/bash
#SBATCH --job-name=gwo_de
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/gwo_de_%j.log
#SBATCH --error=./.hpc/errors/gwo_de_%j.error
#SBATCH --qos=long
# ============================================================================
# Campaña GWO-DE (gwo_de_dtw, híbrido competitivo) en el clúster Océano.
#
# Desde la raíz del repo:  sbatch run_gwo_de_hpc.sh
# Revisar sin ejecutar:    bash run_gwo_de_hpc.sh --dry-run
#
# Un job de 9 CPUs. Corre los tamaños en orden 100 -> 250 -> 500. En cada
# tamaño lanza 6 procesos en paralelo, uno por instancia: los 3 archivos de
# ese tamaño con índices 0 y 1. Cada proceso corre 31 semillas x 6 modos
# (186 corridas) y conserva la mejor corrida de cada modo. El tamaño siguiente
# empieza cuando terminan los 6 procesos del anterior. Una instancia que falla
# no detiene a las demás; el job termina con código 1 si alguna falló.
#
# Resultados: resultados/gwo_de_<SLURM_JOB_ID>/mknapcb<F>/inst_<II>/
#   runs.csv, statistics.csv, convergence_comparison.png/.pdf,
#   campaign_config.json, resumen.txt, run.log y <modo>/best_run/
# Cada instancia corre en .staging/ y se mueve ahí al terminar bien; si falla,
# su carpeta queda en .staging/mknapcb<F>_inst_<II>/ con run.log.
# La carpeta .hpc/logs y .hpc/errors debe existir antes del sbatch.
# MKP_CONDA_SH: ruta de conda.sh (entorno mkp_env); por defecto ~/miniconda3.
# ============================================================================
set -euo pipefail

name=gwo_de
module=gwo_de_dtw.campaign

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
    printf 'Uso: sbatch run_gwo_de_hpc.sh           (desde la raíz del repo)\n'
    printf '     bash run_gwo_de_hpc.sh --dry-run   (muestra las 18 instancias sin ejecutar)\n'
    printf 'Tamaños 100, 250 y 500 en orden; 6 instancias en paralelo por tamaño.\n'
    printf 'Resultados: resultados/gwo_de_<SLURM_JOB_ID>/mknapcb<F>/inst_<II>/<modo>/\n'
}

families_for() {
    case "$1" in
        100) families=(1 4 7) ;;
        250) families=(2 5 8) ;;
        500) families=(3 6 9) ;;
    esac
}

# Comando Python de una instancia. Usado por el job y por --dry-run.
task_command() {
    local repo=$1 family=$2 index=$3 out=$4
    cmd=(python3 -B -u -m "$module"
         --instances-dir "$repo/instancias"
         --files "mknapcb${family}.txt" --indices "$index"
         --seeds $(seq 1 31)
         --out "$out")
}

# Una instancia completa, en segundo plano. Corre en su carpeta de .staging y,
# si termina bien, mueve mknapcb<F>/inst_<II> al experimento.
run_instance() {
    local repo=$1 family=$2 index=$3 staging=$4 target=$5 extra
    export MPLCONFIGDIR="$staging/mplconfig"
    task_command "$repo" "$family" "$index" "$staging/campaign"
    "${cmd[@]}" > "$staging/run.log" 2>&1 || return
    local inst
    printf -v inst '%s/campaign/mknapcb%s/inst_%02d' "$staging" "$family" "$index"
    if [[ ! -d $inst ]]; then
        printf 'No se encontró la salida esperada: %s\n' "$inst" >> "$staging/run.log"
        return 3
    fi
    for extra in campaign_config.json resumen.txt; do
        if [[ -f $staging/campaign/$extra ]]; then mv -- "$staging/campaign/$extra" "$inst/$extra"; fi
    done
    mv -- "$staging/run.log" "$inst/run.log"
    mkdir -p -- "$(dirname -- "$target")"
    mv -- "$inst" "$target"
    # Lo que queda (summary.csv y statistics.csv) repite los archivos de la instancia.
    rm -rf -- "$staging"
}

dry_run=false
if [[ ${1-} == --help && $# == 1 ]]; then usage; exit 0; fi
if [[ ${1-} == --dry-run ]]; then dry_run=true; shift; fi
(( $# == 0 )) || fail 'Argumento no reconocido (ver --help).'

repo=${SLURM_SUBMIT_DIR:-$(pwd -P)}
module_file="${module//.//}.py"
[[ -f $repo/$module_file && -d $repo/instancias ]] \
    || fail "Ejecute desde la raíz del repo: no se encontró $module_file en $repo"

if $dry_run; then
    experiment="$repo/resultados/${name}_<SLURM_JOB_ID>"
    printf 'Resultados: %s/mknapcb<F>/inst_<II>/\n' "$experiment"
    for size in 100 250 500; do
        printf 'items_%s: 6 instancias en paralelo\n' "$size"
        families_for "$size"
        for task in 0 1 2 3 4 5; do
            family=${families[task / 2]}
            index=$((task % 2))
            printf -v case_name 'mknapcb%s_inst_%02d' "$family" "$index"
            printf 'items_%s task %s: mknapcb%s.txt index %s\n' "$size" "$task" "$family" "$index"
            task_command "$repo" "$family" "$index" "$experiment/.staging/$case_name/campaign"
            printf ' '; printf ' %s' "${cmd[@]}"; printf '\n'
        done
    done
    exit 0
fi

[[ ${SLURM_JOB_ID-} =~ ^[1-9][0-9]*$ ]] \
    || fail 'Falta SLURM_JOB_ID: envíe con sbatch run_gwo_de_hpc.sh (o use --dry-run).'
experiment="$repo/resultados/${name}_$SLURM_JOB_ID"
[[ ! -e $experiment && ! -L $experiment ]] || fail "El experimento ya existe: $experiment"
mkdir -p -- "$repo/resultados"
mkdir -- "$experiment" "$experiment/.staging"
cd -- "$repo"

# Conda se activa sin set -u: sus scripts leen variables que pueden no existir.
set +u
source "${MKP_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate mkp_env
set -u
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

printf '=== %s | job %s | nodo %s | CPUs %s | inicio %s ===\n' \
    "$name" "$SLURM_JOB_ID" "$(hostname)" "${SLURM_CPUS_PER_TASK:-?}" "$(date)"
printf 'Resultados: %s\n' "$experiment"
failed=()
for size in 100 250 500; do
    printf -- '--- items_%s: 6 instancias en paralelo | %s\n' "$size" "$(date)"
    families_for "$size"
    pids=()
    cases=()
    stagings=()
    for task in 0 1 2 3 4 5; do
        family=${families[task / 2]}
        index=$((task % 2))
        printf -v case_name 'mknapcb%s_inst_%02d' "$family" "$index"
        printf -v target '%s/mknapcb%s/inst_%02d' "$experiment" "$family" "$index"
        staging="$experiment/.staging/$case_name"
        mkdir -- "$staging" "$staging/mplconfig"
        run_instance "$repo" "$family" "$index" "$staging" "$target" &
        pids+=("$!")
        cases+=("${target#"$experiment"/}")
        stagings+=("$staging")
    done
    for i in "${!pids[@]}"; do
        if wait "${pids[i]}"; then
            printf 'ok     %s | %s\n' "${cases[i]}" "$(date)"
        else
            status=$?
            printf 'falló  %s (código %s): ver %s/run.log\n' "${cases[i]}" "$status" "${stagings[i]}" >&2
            failed+=("${cases[i]}")
        fi
    done
done

printf '=== fin %s | %s ===\n' "$name" "$(date)"
if (( ${#failed[@]} )); then
    printf 'Instancias fallidas (%s): %s\n' "${#failed[@]}" "${failed[*]}" >&2
    exit 1
fi
rmdir -- "$experiment/.staging"
printf 'Las 27 instancias terminaron bien.\n'
