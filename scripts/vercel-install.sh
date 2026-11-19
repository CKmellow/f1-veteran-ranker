#!/usr/bin/env bash
# Vercel install step for the Python API.
#
# Vercel's function runtime has no system libgomp (OpenMP), which the
# lightgbm wheel links against, so loading the LightGBM model fails with
# "libgomp.so.1: cannot open shared object file". xgboost-cpu already ships
# its own bundled copy, so we point lib_lightgbm.so at it with patchelf.
set -euo pipefail

PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
"$PYTHON" -m pip install -r requirements-vercel.txt
"$PYTHON" -m pip install patchelf==0.17.2.4

# Locate packages without importing them (import would need libgomp).
SITE=$("$PYTHON" -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('lightgbm').origin).parent.parent)")
LGB_LIB="$SITE/lightgbm/lib/lib_lightgbm.so"
GOMP=$(ls "$SITE"/xgboost_cpu.libs/libgomp-*.so* | head -n 1)
PATCHELF=$("$PYTHON" -c "import shutil, sysconfig, os; print(shutil.which('patchelf') or os.path.join(sysconfig.get_path('scripts'), 'patchelf'))")

"$PATCHELF" --replace-needed libgomp.so.1 "$(basename "$GOMP")" "$LGB_LIB"
"$PATCHELF" --set-rpath '$ORIGIN/../../xgboost_cpu.libs' "$LGB_LIB"

echo "Patched $LGB_LIB -> $(basename "$GOMP")"
"$PATCHELF" --print-needed "$LGB_LIB" | grep gomp
