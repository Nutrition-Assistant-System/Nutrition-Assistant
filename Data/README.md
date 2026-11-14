# Data pipeline

The exact code that produced the files in `Data/`. The scripts are kept **unchanged** from the run that built dataset v1.0, so they still use the folder names of the two working folders they were written for (`data/fdc`, `data/off`, `data/merged`, `data/raw`, `tests/`). `reproduce.sh` recreates those folders in a temporary directory, so you do not have to move anything.

## Steps

| Step | Script | Input | Output (in `Data/`) |
|---|---|---|---|
| 1. Select USDA foods | `collection/fdc_selection.py` | (list of 208 foods and their categories) | used by step 2 |
| 2. Build USDA extract | `collection/build_fdc.py <SR Legacy CSV folder> <out>` | official SR Legacy CSV release (April 2018) | `sources/usda/*.csv`, `docs/provenance/fdc_build_summary.json` |
| 3. Build Open Food Facts extract | `collection/off_collection_colab.ipynb` (runs `collection/build_off.py` in Colab) | Open Food Facts CSV export, `sources/off/test_barcodes.txt` | `sources/off/off_packaged_products.csv`, `docs/provenance/off_build_summary.json` |
| 4. Merge | `collection/merge_data.py` | the four extracts | `foods_and_products.csv`, `docs/provenance/merge_summary.json` |
| 5. Documentation | `collection/make_docs.py` | the extracts and the merged table | `docs/DATA_COLLECTION_REPORT.md`, `docs/DATA_QUALITY_LOG.md`, `docs/provenance/sources_provenance.json` |
| 6. Prepare training data | `preparation/src/prepare_data.py` (settings: `preparation/config.yaml`) | `foods_and_products.csv`, `test_questions.csv` | `CPT/train.txt`, `CPT/heldout.txt`, `CPT/split_index.csv`, `SFT/sft_pairs.jsonl`, `docs/provenance/data_stats.json` |
| 7. Validate | `preparation/src/validate_data.py` | the files of step 6 and the merged table | `docs/provenance/validation_report.json` (894 pairs and 988 documents, 0 critical issues) |

`collection/make_all.sh` runs steps 2, 4 and 5 in the original working folder. `Notebooks/prepare_data_FINAL_V2.ipynb` (in the project repository) is the notebook version of step 6 that the team runs in Colab.

`test_questions.csv` was written and checked by the team; it is an input of step 6 (its items are kept out of the SFT pairs), not an output.

## Reproduce

Steps 1–3 need the original source downloads (SR Legacy CSV release; Open Food Facts export, several GB), so they are not re-run automatically. Steps 4, 6 and 7 can be checked from the files in this folder:

```bash
pip install pandas pyyaml
bash Data/pipeline/reproduce.sh        # from the repository root
```

It rebuilds the merged table, the CPT corpus and the SFT pairs from `Data/sources/`, runs the validator, and compares every rebuilt file with `Data/SHA256SUMS`. For dataset v1.0 all rebuilt files are byte-identical.

`preparation/config.yaml` is the settings file of the run; only `seed`, `paths`, `data` and `chat_template` are used for the data, the rest are model settings that the validator's config loader expects.
