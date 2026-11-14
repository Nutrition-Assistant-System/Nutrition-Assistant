#!/usr/bin/env bash
# Rebuild the merged table, CPT corpus and SFT pairs from Data/sources, validate them, and compare with Data/SHA256SUMS.
# Run from the repository root:  bash Data/pipeline/reproduce.sh      (needs: python3, pandas, pyyaml)
set -euo pipefail
cd "$(dirname "$0")/../.."
D="$PWD/Data"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

# working folders in the layout the unchanged scripts expect
mkdir -p "$W/collect/scripts" "$W/collect/data/fdc" "$W/collect/data/off" "$W/collect/data/merged"
mkdir -p "$W/prep/src" "$W/prep/data/raw" "$W/prep/tests"
cp "$D/pipeline/collection/merge_data.py" "$W/collect/scripts/"
cp "$D/sources/usda/"*.csv "$W/collect/data/fdc/"
cp "$D/sources/off/off_packaged_products.csv" "$W/collect/data/off/"

echo "== step 4: merge"
(cd "$W/collect" && python3 scripts/merge_data.py > /dev/null)
cp "$W/collect/data/merged/foods_and_products.csv" "$W/prep/data/raw/"
cp "$D/test_questions.csv" "$W/prep/tests/"
cp "$D/pipeline/preparation/config.yaml" "$W/prep/"
cp "$D/pipeline/preparation/src/"*.py "$W/prep/src/"

echo "== step 6: prepare CPT corpus and SFT pairs"
(cd "$W/prep" && python3 src/prepare_data.py > /dev/null)
echo "== step 7: validate"
(cd "$W/prep" && python3 -m src.validate_data | tail -n 6)

echo "== compare with Data/SHA256SUMS"
fail=0
check() {   # $1 = rebuilt file, $2 = path in SHA256SUMS
  want=$(awk -v f="$2" '$2==f {print $1}' "$D/SHA256SUMS")
  got=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$want" = "$got" ]; then echo "  identical  $2"; else echo "  DIFFERENT  $2"; fail=1; fi
}
check "$W/collect/data/merged/foods_and_products.csv" foods_and_products.csv
check "$W/prep/data/cpt/train.txt"       CPT/train.txt
check "$W/prep/data/cpt/heldout.txt"     CPT/heldout.txt
check "$W/prep/data/cpt/split_index.csv" CPT/split_index.csv
check "$W/prep/data/sft_pairs.jsonl"     SFT/sft_pairs.jsonl
[ "$fail" = 0 ] && echo "All rebuilt files match dataset v1.0." || { echo "Some files differ."; exit 1; }
