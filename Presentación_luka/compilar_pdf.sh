#!/usr/bin/env bash
set -euo pipefail

PRES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PRES_DIR"

# En este entorno los paquetes adicionales están en este árbol local.
# Si la instalación global ya los contiene, se conserva la configuración actual.
if [[ -z "${TEXMFHOME:-}" && -d "/tmp/texmf_luka" ]]; then
  export TEXMFHOME="/tmp/texmf_luka"
fi

pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

echo "PDF generado: $PRES_DIR/main.pdf"
