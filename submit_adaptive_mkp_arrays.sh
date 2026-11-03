#!/bin/bash
# Tres arrays independientes; nunca lanza un job adicional por semilla.
set -euo pipefail

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
    printf 'Uso: bash submit_adaptive_mkp_arrays.sh [--dry-run] [BATCH_TAG]\n'
    printf 'Tag: letras/dígitos iniciales, luego letras/dígitos/._-. Por defecto: fecha/PID/azar.\n'
    printf 'Worker: MKP_CONDA_SH permite otra ruta de conda.sh; entorno fijo mkp_env.\n'
}
if [[ ${1-} == --help && $# == 1 ]]; then usage; exit 0; fi
dry_run=false
if [[ ${1-} == --dry-run ]]; then dry_run=true; shift; fi
(( $# <= 1 )) || fail 'Se admite un solo BATCH_TAG.'
if (( $# )); then
    tag=$1
else
    tag="$(date -u +%Y%m%dT%H%M%S)_$$_${RANDOM}"
fi
[[ $tag =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || fail 'BATCH_TAG inválido (ver --help).'

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
worker="$repo/run_adaptive_mkp_array.sh"
[[ -f $worker ]] || fail "No existe el worker: $worker"
batch="$repo/resultados/adaptive_mh_selection/batch_$tag"
[[ ! -e $batch && ! -L $batch ]] || fail "El lote ya existe: $batch"

if ! $dry_run; then
    command -v sbatch >/dev/null 2>&1 || fail 'sbatch no está disponible; use --dry-run fuera de Slurm.'
    mkdir -p -- "$(dirname -- "$batch")"
    # mkdir sin -p reserva el lote: dos submitters no pueden compartirlo.
    mkdir -- "$batch"
    mkdir -- "$batch/logs" "$batch/errors"
    printf 'size\tarray_job_id\n' > "$batch/jobs.tsv"
fi
for size in 100 250 500; do
    args=(--parsable --array=0-8%9 "--chdir=$repo"
          "--output=$batch/logs/${tag}_items_${size}_%A_%a.out"
          "--error=$batch/errors/${tag}_items_${size}_%A_%a.err"
          "$worker" "$repo" "$size" "$batch")
    if $dry_run; then
        printf 'sbatch'; printf ' %q' "${args[@]}"; printf '\n'
    elif job_id=$(sbatch "${args[@]}"); then
        printf '%s\t%s\n' "$size" "$job_id" >> "$batch/jobs.tsv"
        printf 'items_%s: %s\n' "$size" "$job_id"
    else
        status=$?
        printf 'sbatch falló para items_%s (código %s). IDs aceptados en %s:\n' "$size" "$status" "$batch/jobs.tsv" >&2
        while IFS= read -r line; do printf '%s\n' "$line" >&2; done < "$batch/jobs.tsv"
        printf 'No se cancelaron jobs; revise el lote parcial antes de otro envío.\n' >&2
        exit "$status"
    fi
done
