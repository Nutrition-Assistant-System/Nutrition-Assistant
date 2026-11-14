# Data documentation

| File | Content |
|---|---|
| `DATA_COLLECTION_REPORT.md` | How the USDA and Open Food Facts records were obtained, selected and merged; the 20 review issues and how they were resolved; limitations |
| `DATA_QUALITY_LOG.md` | Every quality check, flag and rejection, with counts |
| `DATA_FIXES_CHANGELOG.md` | Fixes to the CPT documents and SFT pairs after the hand-check, and the validation result (0 critical issues) |
| `provenance/sources_provenance.json` | Where each source came from (official links, the verified USDA copy, export date) and SHA-256 of the extracts |
| `provenance/fdc_build_summary.json` | USDA selection: foods per category, forms, nutrients, portions, missing values |
| `provenance/off_build_summary.json` | Open Food Facts selection: rows read and rejections by reason |
| `provenance/merge_summary.json` | Merged table: rows by source, basis and category, filled share per column |
| `provenance/data_stats.json` | CPT and SFT statistics: document counts, split, pairs by type, test items |
| `provenance/validation_report.json` | Result of `validate_data.py` on all 894 pairs and 988 documents |

These reports were generated in the original working folders, so they name files by their old paths. In this repository:

| Path in the reports | Path in `Data/` |
|---|---|
| `data/fdc/...` | `sources/usda/...` |
| `data/off/...` | `sources/off/...` |
| `data/merged/foods_and_products.csv`, `data/raw/foods_and_products.csv` | `foods_and_products.csv` |
| `data/cpt/...` | `CPT/...` |
| `data/sft_pairs.jsonl` | `SFT/sft_pairs.jsonl` |
| `tests/test_questions.csv` | `test_questions.csv` |
| `scripts/...` | `pipeline/collection/...` |
| `src/prepare_data.py`, `src/validate_data.py` | `pipeline/preparation/src/...` |
