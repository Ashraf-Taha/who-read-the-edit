#!/usr/bin/env bash
# Regenerate numbers, figures and the draft, then typeset paper.pdf (and keep paper.tex for arXiv).
#   bash paper/make_pdf.sh        (needs pandoc and tectonic: brew install pandoc tectonic)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$HERE/../.venv/bin/python}"
cd "$HERE"
"$PY" make_numbers.py
"$PY" make_figures.py
"$PY" build.py || echo "build.py reported gaps (see above); typesetting anyway"
COMMON=(draft.md --from markdown+tex_math_dollars --citeproc --bibliography refs.bib --shift-heading-level-by=-1
        -V geometry:margin=1in -V fontsize=10pt -V colorlinks=true -V linkcolor=black -V urlcolor=blue
        -M link-citations=true -M reference-section-title=References -H latex_header.tex)
pandoc "${COMMON[@]}" -s -o paper.tex
pandoc "${COMMON[@]}" --pdf-engine=tectonic -o paper.pdf
echo "wrote $HERE/paper.pdf and $HERE/paper.tex"
