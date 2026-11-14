"""Regenerate ALL documentation from the final data files (never edit the docs by hand).

Usage: python make_docs.py   (run after build_fdc.py, build_off.py and merge_data.py)

Writes: README.md, docs/DATA_CARD.md, docs/DATA_QUALITY_LOG.md, docs/DATA_COLLECTION_REPORT.md,
        docs/LICENSES_AND_ATTRIBUTION.md, docs/sources_provenance.json
"""
import os, json, hashlib, glob, datetime
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = lambda *p: os.path.join(HERE, *p)
J = lambda *p: json.load(open(F(*p)))

fdc = pd.read_csv(F("data/fdc/fdc_generic_foods.csv"))
lng = pd.read_csv(F("data/fdc/fdc_food_nutrients_long.csv"))
por = pd.read_csv(F("data/fdc/fdc_food_portions.csv"))
off = pd.read_csv(F("data/off/off_packaged_products.csv"), dtype={"barcode": str})
mer = pd.read_csv(F("data/merged/foods_and_products.csv"), dtype={"source_id": str})
fs, os_, ms = J("data/fdc/fdc_build_summary.json"), J("data/off/off_build_summary.json"), J("data/merged/merge_summary.json")
interim = os.path.exists(F("data/off/STATUS.txt"))
export_date = open(F("data/off/export_downloaded_at_utc.txt")).read().strip()
today = datetime.date.today().isoformat()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
CORE = ["energy_kcal", "protein_g", "total_fat_g", "saturated_fat_g", "carbohydrate_g", "total_sugars_g",
        "fiber_g", "sodium_mg", "cholesterol_mg", "calcium_mg", "iron_mg", "potassium_mg", "water_g"]
NICE = {c: c.rsplit("_", 1)[0].replace("_", " ") for c in CORE}
cat_counts = lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().sort_index().items())
STATUS = ("**INTERIM**: the Open Food Facts file was built by script v2 from the 780-product pool of the first "
          "Colab run, not from the full export. Re-run `scripts/off_collection_colab.ipynb`, replace `data/off/`, "
          "then run `scripts/make_all.sh`." if interim else "Final.")

def table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)

# ------------------------------------------------------------------ facts used in several documents
miss = fdc[fdc.nutrients_not_reported.notna()].copy()
miss["Not reported"] = miss.nutrients_not_reported.map(lambda s: ", ".join(NICE[c] for c in s.split(";")))
miss_tbl = table(miss.assign(Food=miss.common_name + " (" + miss.form + ")")[["fdc_id", "Food", "Not reported"]])
miss_cnt = table(pd.DataFrame([(NICE[c], int(fdc[c].isna().sum())) for c in CORE if fdc[c].isna().any()],
                              columns=["Nutrient", "Foods with no value"]))
fl = fdc[fdc.quality_flags.notna()].copy()
fl_tbl = table(fl.assign(Food=fl.common_name + " (" + fl.form + ")")[["fdc_id", "Food", "quality_flags"]])
prox = fdc[["water_g", "protein_g", "total_fat_g", "carbohydrate_g"]].sum(axis=1) + fdc.fdc_id.map(
    lng[lng.nutrient_id == 1007].set_index("fdc_id").amount_per_100g)
prox_hi = int((prox > 101).sum())
rej = {k: v for k, v in os_["rejections_by_reason"].items() if not k.startswith("_")}
rej_tbl = table(pd.DataFrame(sorted(rej.items(), key=lambda kv: -kv[1]), columns=["Reason", "Products removed"]))
off_flags = int(off.fiber_gt_carbohydrates.sum())
cov = os_["coverage"]
countries = off.countries_tags.str.split(",").explode().value_counts().head(8)
eng = int(off.sold_in_english_speaking_country.sum())

# ------------------------------------------------------------------ LICENCES (static text)
lic = """# Licences and attribution

## USDA FoodData Central (SR Legacy)
- **Licence:** public domain, CC0 1.0 Universal. No permission needed.
- **USDA asks:** cite FoodData Central as the source.
- **Citation:** U.S. Department of Agriculture, Agricultural Research Service, Beltsville Human Nutrition Research
  Center. FoodData Central. [Internet]. [cited {today}]. Available from https://fdc.nal.usda.gov/.

## Open Food Facts
- **Database:** Open Database License (ODbL) 1.0. **Contents:** Database Contents License (DbCL) 1.0.
  **Images:** CC BY-SA (not used).
- **We must:**
  1. Show this attribution wherever the data or results from it appear:
     > Product data from Open Food Facts (https://world.openfoodfacts.org), available under the Open Database License (ODbL).
  2. Release any publicly shared copy of our product table, or of the merged table, under ODbL (share-alike).
  3. Not add technical restrictions (such as DRM) to shared copies.

## Our files
- Scripts: the team's own (choose a repository licence, e.g. MIT).
- `data/fdc/*`: CC0 source, free to share. `data/off/*` and `data/merged/*`: share under ODbL.
""".format(today=today)
open(F("docs/LICENSES_AND_ATTRIBUTION.md"), "w").write(lic)

# ------------------------------------------------------------------ DATA CARD
card = f"""# Data card: Nutrition and Food Label Assistant

Generated {today} by `scripts/make_docs.py` from the data files. Status: {STATUS}

| Field | Generic foods (USDA) | Packaged products (Open Food Facts) |
|---|---|---|
| **Source** | FoodData Central, SR Legacy, April 2018 release (<https://fdc.nal.usda.gov/download-datasets>), obtained through a verified GitHub copy (see the report) | Full CSV export (<https://world.openfoodfacts.org/data>), downloaded {export_date} |
| **Licence** | Public domain (CC0 1.0); USDA asks to be cited | ODbL 1.0 + DbCL 1.0: attribution and share-alike |
| **Size** | {len(fdc)} foods; 13 core nutrients; {len(lng):,} nutrient rows ({lng.nutrient_id.nunique()} nutrients); {len(por)} household portions | {len(off)} products from {os_['rows_read']:,} export rows |
| **Categories** | {cat_counts(fdc.project_category)} | {cat_counts(off.project_category)} |
| **Language** | English | English names and ingredient lists (two detectors, see the report) |
| **Basis** | Per 100 g edible portion | Per 100 g (solids) or per 100 ml (soft drinks and juices) |
| **IDs** | `fdc_id` (plus `ndb_number`); `item_id` = `fdc:<id>` in the merged table | Raw barcode as published; `item_id` = `off:<barcode>` |
| **Units** | Sodium in mg; no salt value published | Salt and sodium in g; merged table shows sodium in mg (x 1000) |
| **Missing values** | Blank, never 0. {len(miss)} foods lack at least one core nutrient | Blank, never 0. Listed per product in `fields_missing`. Allergen tags {cov['allergens_tags']:.0%}, traces {cov['traces_tags']:.0%}, serving size {cov['serving_quantity']:.0%}, fiber {cov['fiber_g']:.0%}, sodium {cov['sodium_g']:.0%} filled |
| **Country mix** | United States data | {eng} of {len(off)} sold in an English-speaking country; top countries: {', '.join(f'{k.replace("en:", "")} {v}' for k, v in countries.head(5).items())} |
| **Known issues** | Frozen in 2018; US foods; {len(fl)} foods carry expected methodology flags | Crowd-sourced, values may contain errors; EU and US labels define carbohydrate differently |
| **Sensitive content** | None | None (no contributor names kept) |
| **Merged table** | `data/merged/foods_and_products.csv`: {len(mer)} rows, {len(mer.columns)} columns, one `item_id` and one `basis` per row | |
"""
open(F("docs/DATA_CARD.md"), "w").write(card)

# ------------------------------------------------------------------ QUALITY LOG
qlog = f"""# Data quality log

Generated {today} from the data files. Rules: no value is estimated, imputed or corrected. Checks either
**flag** a record (USDA: values kept) or **drop** it (Open Food Facts: the record is removed, never edited).
A blank cell means "not reported by the source", never zero.

## USDA FoodData Central

### Missing core nutrients ({len(miss)} of {len(fdc)} foods)

{miss_tbl}

{miss_cnt}

The assistant must answer "not reported" for these, not 0.

### Flags (`quality_flags`, {len(fl)} foods)

| Check | Rule | Result |
|---|---|---|
| Energy | Estimate = 4 x protein + 4 x carbohydrate + 9 x fat. Flag if the published energy differs from the estimate by more than 15% **and** more than 5 kcal (same rule for every food) | {fs['flag_counts'].get('energy_vs', 0)} foods, mostly vegetables: USDA uses food-specific energy factors, so the gap is expected. Values kept |
| Sugars vs carbohydrate | Flag if total sugars > carbohydrate by more than 0.01 g | {fs['flag_counts'].get('sugars_gt_carbohydrate', 0)} dairy foods: sugars are measured directly, carbohydrate "by difference". Values kept |
| Main components | Flag if water + protein + fat + carbohydrate + ash is outside 97 to 103 g | {fs['flag_counts'].get('proximates_sum', 0)} food; separately, {prox_hi} foods sum to between 101 and 103 g, which is within the rule and normal for separate lab methods |

{fl_tbl}

### Other notes
- `fdc_publication_date` (2019-04-01) is when USDA loaded SR Legacy into FoodData Central; the data is the April 2018 release.
- Columns that SR Legacy leaves empty were removed: `measure_unit` and `portion_description` (portions; the unit is written in `modifier`, for example "cup, chopped") and `median` (nutrient table). `footnote` is kept (filled in {int(lng.footnote.notna().sum())} nutrient rows and {int(por.footnote.notna().sum())} portion rows).
- {len(fdc) - por.fdc_id.nunique()} food has no household portion in SR Legacy: {', '.join(fdc.loc[~fdc.fdc_id.isin(por.fdc_id), 'common_name'])}.

## Open Food Facts

### Products removed, by reason (script v2)

{rej_tbl}

### Rules applied (products are dropped, never edited)
- **Category:** first project category whose include tags match, unless an exclude tag or exclude word in the name matches (for example dairy desserts are not yogurts, rice cakes are not bread, pastries are not biscuits, squash concentrates are not drinks).
- **Required:** product name, brand, country, ingredient list, and energy, fat, saturated fat, carbohydrate, sugars, protein and salt.
- **Ranges:** every value 0 to 100 g, energy 0 to 900 kcal.
- **Salt cap per category (g per 100 g/ml):** breakfast cereals 3, yogurts 1, chocolate 1.5, biscuits and salty snacks 5, soft drinks and juices 0.5, bread 3. These are data-plausibility limits, not nutrition advice.
- **Salt vs sodium:** when both are published, salt must equal sodium x 2.5 within 10% (at least 0.1 g).
- **Saturated fat <= fat** and **sugars <= carbohydrate** (0.001 g tolerance). Fat + carbohydrate + protein <= 101 g.
- **Energy:** within 20% (at least 15 kcal) of 4 x protein + 4 x carbohydrate + 9 x fat + 2 x fibre.
- **Duplicates:** same barcode; same brand + name; same ingredient text + same nutrient values (the best-ranked copy is kept).
- **Language:** ingredient list English by a word-frequency vote (at least 75% English words, or 50% plus langdetect agreeing); product name rejected if it contains a clearly foreign word, more foreign than English words, or non-Latin script. Brand-only names count as neutral.
- **Ranking:** test-set products first, then sold in an English-speaking country, allergen tags present, serving size, fibre, completeness, most recent edit.

### Remaining known issues
- **Fibre above carbohydrate** in {off_flags} product(s) (flag `fiber_gt_carbohydrates`). On EU labels carbohydrate excludes fibre, so this can be genuine (for example very dark chocolate).
- **Rounded or computed values:** for labels that declare values per serving, Open Food Facts computes per-100 g values, which gives long decimals. Values are kept as published; text shown to the model is rounded.
- **0 kcal drinks:** labelling rules allow "0 kcal" below about 4 kcal per serving. Kept.
- **Free-text fields** (`quantity`, `serving_size`) are kept as printed and used only for display; calculations use the numeric `serving_quantity`.
- **Allergen tags** are crowd-sourced: an empty field does not prove a product is allergen-free.
"""
open(F("docs/DATA_QUALITY_LOG.md"), "w").write(qlog)

# ------------------------------------------------------------------ REPORT
issues = [
    (1, "Allergen tags empty", "Pipeline", "Script read `allergens` but wrote only `allergens_tags`, which the CSV export does not have", "v2 reads `allergens`; ranking uses it"),
    (2, "`main_language` blank", "Pipeline", "Export has no `lang` column", "Column removed; two language detectors added"),
    (3, "Documents contradicted the data", "Docs", "Docs written before the Open Food Facts run", "All docs generated by `make_docs.py` from the files"),
    (4, "Non-English text passed", "Pipeline", "Word-list heuristic missed some languages", "Word-frequency vote + langdetect on name and ingredients"),
    (5, "Salt and sodium disagree", "Source", "Both values typed by contributors", "Products with inconsistent pairs dropped"),
    (6, "Implausible salt values", "Source", "Data-entry/unit errors; 0-100 g range too loose", "Salt cap per category"),
    (7, "Fibre > carbohydrate; sat. fat > fat", "Source", "EU carbohydrate excludes fibre; label rounding", "Fibre flagged and documented; sat. fat > fat dropped (0.001 g tolerance)"),
    (8, "Energy vs macros", "Source", "US per-serving rounding, sweeteners, '0 kcal' labelling rule", "Rule documented; values kept"),
    (9, "Unrounded values", "Source", "Open Food Facts computes per-100 g from per-serving labels", "Kept as published; documented; rounded in model text"),
    (10, "Duplicate-looking products", "Source + pipeline", "Same product under two names/barcodes", "Dedupe on ingredients + nutrients"),
    (11, "Two barcode columns", "Pipeline", "Padded key used only for deduplication", "Raw barcode is the only ID; `item_id` = `off:<barcode>`"),
    (12, "Missing brand/country; odd `quantity`", "Source", "Free-text volunteer fields", "No brand or country -> dropped; `quantity` display only"),
    (13, "Wrong category matches", "Source + pipeline", "Source tags + first-match rule", "Exclude tags and name words per category; manual review still advised"),
    (14, "Mostly French products", "Source + pipeline", "Most Open Food Facts contributions are French; ranking ignored country", "English-speaking countries ranked first"),
    (15, "Release date mismatch", "Source", "FDC load date vs data release", "Column renamed `fdc_publication_date`; documented"),
    (16, "Energy check skipped <= 20 kcal", "Pipeline", "Undocumented cut-off", "One rule for all foods: > 15% and > 5 kcal"),
    (17, "Main-component threshold not documented", "Docs", "Log described the rule loosely", "97-103 g rule documented with counts"),
    (18, "Names differ between log and data", "Docs", "Names typed by hand", "Log tables generated from the data"),
    (19, "Meat wording", "Docs", "Unclear sentence", "Reworded below"),
    (20, "Empty columns", "Source", "SR Legacy leaves them empty", "`measure_unit`, `portion_description`, `median` removed"),
]
iss_tbl = table(pd.DataFrame(issues, columns=["#", "Issue", "Root cause", "Why", "Resolution"]))
rep = f"""# Data Collection Report: Nutrition and Food Label Assistant

Generated {today} by `scripts/make_docs.py`. Status: {STATUS}

## 1. Summary

| Dataset | Records | File |
|---|---|---|
| USDA FoodData Central, SR Legacy (generic foods) | {len(fdc)} foods | `data/fdc/fdc_generic_foods.csv` (+ nutrients, portions) |
| Open Food Facts (packaged products) | {len(off)} products | `data/off/off_packaged_products.csv` |
| Merged table | {len(mer)} rows ({ms['by_basis'].get('per 100 g', 0)} per 100 g, {ms['by_basis'].get('per 100 ml', 0)} per 100 ml) | `data/merged/foods_and_products.csv` |

Nothing was estimated or filled in. Blank means "not reported".

## 2. USDA FoodData Central (generic foods)

- **Access:** the official site is blocked from our build workspace, so the unmodified SR Legacy CSV release (April 2018) was taken from a public GitHub copy (`matiasmorant/nutrition`, commit `0b235d8`). It is byte-identical to a second, unrelated copy (`HiathanNguyen/InfoChallenge`), its row counts match USDA's record-count file (7,793 foods, 644,125 nutrient rows), and spot-checked values match published SR values (for example cooked chicken breast: 165 kcal, 31.02 g protein).
- **Selection:** {len(fdc)} hand-picked foods (`scripts/fdc_selection.py`) in 8 categories: {cat_counts(fdc.project_category)}. Forms: {', '.join(f'{k} {v}' for k, v in fdc.form.value_counts().items())}. {int((fdc.groupby('common_name').form.apply(lambda s: {'raw', 'cooked'} <= set(s))).sum())} foods have both a raw and a cooked entry.
- **Meat choice:** beef, lamb, chicken and turkey are included; pork and game meats (for example venison) are not. Three cooking oils are grouped with nuts and seeds. One branded item is included (LIFEWAY kefir, the only kefir in SR Legacy).
- **Kept:** 13 core nutrients per 100 g; all {lng.nutrient_id.nunique()} reported nutrients in a long table; {len(por)} household portions for {por.fdc_id.nunique()} foods.

## 3. Open Food Facts (packaged products)

- **Access:** the export cannot be downloaded from our build workspace, so `scripts/off_collection_colab.ipynb` runs the collection in Google Colab. Export downloaded {export_date}; {os_['rows_read']:,} rows read.
- **Script v2** (`scripts/build_off.py`): six categories with exclusion rules, required fields, plausibility checks, salt caps, salt vs sodium, three duplicate checks, English-country ranking and two language detectors. The six test-set products are always kept if they pass every filter ({len(os_['must_include_kept'])} of {len(os_['must_include_requested'])} kept).
- **Result:** {len(off)} products: {cat_counts(off.project_category)}. {eng} are sold in an English-speaking country. Rejections by reason are in the quality log.

## 4. Merged table

- One row per item, {len(mer.columns)} columns. `item_id` = `fdc:<fdc_id>` or `off:<barcode>`; `basis` = per 100 g or per 100 ml on every row.
- **Units:** sodium in mg for both sources (Open Food Facts g x 1000). Salt only where published (Open Food Facts); it is not derived for USDA rows.
- **Carbohydrate:** USDA is "by difference" (includes fibre); Open Food Facts is as declared on the label (EU excludes fibre, US includes it). Stated in `carbohydrate_definition`.
- Columns that one source does not have stay blank for the other (for example ingredients for USDA, cholesterol for Open Food Facts).

## 5. Review issues and how they were resolved

{iss_tbl}

## 6. Limitations
- SR Legacy is US data frozen in 2018. Open Food Facts is crowd-sourced and changes daily; our copy is dated {export_date}.
- Category assignment still relies on contributor tags; a manual pass per category is recommended.
- The language checks can still misjudge very short or brand-only names.
- The train/test split for modelling is handled in the Phase 2 repository, not here.

## 7. How to rebuild
```
python scripts/build_fdc.py <SR Legacy CSV folder> data/fdc     # USDA
# Open Food Facts: run scripts/off_collection_colab.ipynb in Colab, unzip into data/off/
python scripts/merge_data.py                                     # merged table
python scripts/make_docs.py                                      # all documentation
```
"""
open(F("docs/DATA_COLLECTION_REPORT.md"), "w").write(rep)

# ------------------------------------------------------------------ README
readme = f"""# Nutrition and Food Label Assistant: datasets

Status: {STATUS}

```
data/
  fdc/     fdc_generic_foods.csv ({len(fdc)} foods), fdc_food_nutrients_long.csv ({len(lng):,} rows),
           fdc_food_portions.csv ({len(por)} portions), fdc_build_summary.json
  off/     off_packaged_products.csv ({len(off)} products), off_build_summary.json,
           export_downloaded_at_utc.txt, test_barcodes.txt{', STATUS.txt' if interim else ''}
  merged/  foods_and_products.csv ({len(mer)} rows), merge_summary.json
scripts/
  fdc_selection.py, build_fdc.py          USDA selection and build
  build_off.py, off_collection_colab.ipynb Open Food Facts (script v2, run in Colab)
  merge_data.py                           one table, one ID, one basis per row
  make_docs.py, make_all.sh               regenerate every document from the data
docs/
  DATA_COLLECTION_REPORT.md  DATA_CARD.md  DATA_QUALITY_LOG.md  LICENSES_AND_ATTRIBUTION.md  sources_provenance.json
```

**Attribution:** USDA FoodData Central (CC0). Product data from Open Food Facts (https://world.openfoodfacts.org),
available under the Open Database License (ODbL).
"""
open(F("README.md"), "w").write(readme)

# ------------------------------------------------------------------ PROVENANCE
prov = {
    "generated": today,
    "status": "interim" if interim else "final",
    "fdc": {
        "dataset": "USDA FoodData Central, SR Legacy CSV release (April 2018)",
        "official_download": "https://fdc.nal.usda.gov/download-datasets",
        "obtained_from": {"repo": "https://github.com/matiasmorant/nutrition",
                          "commit": "0b235d88512e017a4f78ada4f0e797218a14a7c4",
                          "folder": "USDA_FoodData_Central_sr_legacy_food/"},
        "cross_check": {"repo": "https://github.com/HiathanNguyen/InfoChallenge",
                        "file": "FoodData_Central_sr_legacy_food_csv_2018-04/food_nutrient.csv",
                        "git_blob": "45590d9600a9716575ac43010842912baf25944f", "identical": True},
    },
    "off": {"source": "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
            "downloaded": export_date, "rows_read": os_["rows_read"], "script_version": os_.get("script_version", 1)},
    "output_sha256": {os.path.relpath(p, HERE): sha(p) for p in sorted(glob.glob(F("data", "*", "*.csv")))},
}
json.dump(prov, open(F("docs/sources_provenance.json"), "w"), indent=2)
print("docs regenerated:", "interim" if interim else "final")
