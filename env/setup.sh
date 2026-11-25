#!/usr/bin/env bash
# Creates the Python environment (Python 3.12, pinned packages) and downloads the models.
#   bash env/setup.sh            # environment in ./.venv
#   VENV=/some/path bash env/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HERE/../.venv}"

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  python3 -m venv "$HERE/.bootstrap"
  "$HERE/.bootstrap/bin/pip" install --quiet --upgrade pip uv
  UV="$HERE/.bootstrap/bin/uv"
fi

"$UV" venv --python 3.12 "$VENV"
"$UV" pip install --python "$VENV/bin/python" -r "$HERE/requirements.lock.txt"
"$VENV/bin/python" "$HERE/download_models.py"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"
