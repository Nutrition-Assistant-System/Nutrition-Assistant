# Data: Nutrition and Food Label Assistant (dataset v1.0)

This folder holds the dataset used to train and test our models, the source-level extracts it was built from, the code that builds it, and its documentation. The same data files are published as a versioned Hugging Face dataset: **[nutrition-assistant-team/nutrition-food-label-qa](https://huggingface.co/datasets/nutrition-assistant-team/nutrition-food-label-qa)**.

## Which file is for what

| File | Rows | Use it for |
|---|---|---|
| `foods_and_products.csv` | 988 items × 35 columns | One clean table of 208 generic foods (USDA) and 780 packaged products (Open Food Facts) |
| `CPT/train.txt` | 889 documents | Continued pretraining (one document per line, separated by a blank line) |
| `CPT/heldout.txt` | 99 documents | Held-out perplexity (10% of the documents, split by document) |
| `CPT/split_index.csv` | 988 rows | Which item went to which CPT split (`doc_id`, `is_test_item`, `split`, `chars`) |
| `SFT/sft_pairs.jsonl` | 894 pairs | Supervised fine-tuning (`id`, `question`, `answer`, `type`, `source`) |
| `SFT/sft_handcheck_results_v2.csv` | 59 rows | AI-assisted check of 59 sampled SFT pairs (59 pass) |
| `test_questions.csv` | 30 questions | **Evaluation only. Never use for training.** |
| `sources/usda/fdc_generic_foods.csv` | 208 | USDA foods with source fields and 13 core nutrients |
| `sources/usda/fdc_food_nutrients_long.csv` | 21,667 | All 149 reported nutrients of the 208 foods (long format) |
| `sources/usda/fdc_food_portions.csv` | 635 | USDA household portions (amount, modifier, gram weight) |
| `sources/off/off_packaged_products.csv` | 780 | The 780 products in a structure close to Open Food Facts |
| `sources/off/export_downloaded_at_utc.txt` | 1 line | Date of the Open Food Facts export used |
| `sources/off/test_barcodes.txt` | 6 | Barcodes of the test products (kept by the collection script so the test set can use them) |

The `sources/` files are **source-level extracts**: the records we selected and cleaned from each source, kept close to the source structure. They are not the original source datasets; those are available from the sources themselves (links below).

## Folders

| Folder | Content |
|---|---|
| `pipeline/` | The code that built every file here, and `reproduce.sh`, which rebuilds the merged table, CPT corpus and SFT pairs from `sources/` and checks them against `SHA256SUMS` |
| `docs/` | Data collection report, data quality log, changelog of the data fixes, and provenance files (build summaries, validation report) |

## Sources and versions

| Source | Version | Official link |
|---|---|---|
| USDA FoodData Central, SR Legacy | April 2018 release | <https://fdc.nal.usda.gov/download-datasets> |
| Open Food Facts | full CSV export downloaded Wed 11 Nov 2026, 20:56:56 UTC (4,906,687 rows read) | <https://world.openfoodfacts.org/data> |

Dataset version: **v1.0** (GitHub release `data-v1.0`, Hugging Face tag `v1.0`). `SHA256SUMS` lists one checksum per data file; check them with `cd Data && sha256sum -c SHA256SUMS`.

## Licence and attribution

The files in this folder contain information from [Open Food Facts](https://openfoodfacts.org), which is made available here under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/) (full text in `LICENSE`). Individual contents are under the [Database Contents License (DbCL) 1.0](https://opendatacommons.org/licenses/dbcl/1-0/). Because they are derived from Open Food Facts, the data files in this folder are shared under the **ODbL 1.0**.

They also contain public-domain data (CC0 1.0) from U.S. Department of Agriculture, Agricultural Research Service, [FoodData Central](https://fdc.nal.usda.gov) (SR Legacy, 2018). The three files in `sources/usda/` contain only USDA data and are public domain on their own.

**Changes made by the Nutrition-Assistant team:** records were selected, filtered and checked, the two sources were merged into one table with common units (Open Food Facts sodium g × 1000 → mg), the items were converted into text documents for continued pretraining, and question-answer pairs and test questions were generated from templates. The method is in `pipeline/`.

Each row of `foods_and_products.csv` links to its source page (`source_url`) and states its licence (`licence`).

The code in `pipeline/` is the team's own work and is not covered by the ODbL. Product images from Open Food Facts are not used.

*The application of the ODbL to the generated text and question-answer pairs follows a conservative reading of the licence; it has not been confirmed by Open Food Facts or a lawyer.*
