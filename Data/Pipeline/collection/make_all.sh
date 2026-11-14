#!/usr/bin/env bash
# Rebuild everything after the data changes. Run from the nutrition_data folder.
# Usage: bash scripts/make_all.sh [path to SR Legacy CSV folder]   (USDA rebuild is skipped if no path is given)
set -euo pipefail
if [ $# -ge 1 ]; then python scripts/build_fdc.py "$1" data/fdc; fi
python scripts/merge_data.py
python scripts/make_docs.py
