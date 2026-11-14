"""Build the generic-foods dataset from the USDA FoodData Central SR Legacy CSV release (April 2018).

Usage:  python build_fdc.py <path to SR Legacy CSV folder> <output folder>

Inputs (unchanged USDA files): food.csv, sr_legacy_food.csv, food_category.csv,
nutrient.csv, food_nutrient.csv, food_portion.csv, measure_unit.csv.

Rules:
* Only the foods listed in fdc_selection.py are kept (matched on the exact SR Legacy description).
* Values are copied exactly as published (per 100 g edible portion). Nothing is estimated:
  a nutrient that USDA did not report for a food is left blank and listed in `nutrients_not_reported`.
* Validation checks only FLAG records; they never change a value.
"""
import sys, os, json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdc_selection import SELECTION

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# Core nutrients: (FDC nutrient id, output column)
CORE = [
    (1008, "energy_kcal"),
    (1003, "protein_g"),
    (1004, "total_fat_g"),
    (1258, "saturated_fat_g"),
    (1005, "carbohydrate_g"),
    (2000, "total_sugars_g"),
    (1079, "fiber_g"),
    (1093, "sodium_mg"),
    (1253, "cholesterol_mg"),
    (1087, "calcium_mg"),
    (1089, "iron_mg"),
    (1092, "potassium_mg"),
    (1051, "water_g"),
]
ASH_ID = 1007  # used only for the proximate-sum check, not exported as a core column

food = pd.read_csv(os.path.join(SRC, "food.csv"))
srl = pd.read_csv(os.path.join(SRC, "sr_legacy_food.csv"))
cat = pd.read_csv(os.path.join(SRC, "food_category.csv"))
nut = pd.read_csv(os.path.join(SRC, "nutrient.csv"))
fn = pd.read_csv(os.path.join(SRC, "food_nutrient.csv"))
por = pd.read_csv(os.path.join(SRC, "food_portion.csv"))
mu = pd.read_csv(os.path.join(SRC, "measure_unit.csv"))

sel = pd.DataFrame(SELECTION, columns=["project_category", "common_name", "form", "description"])
assert sel.description.is_unique, "duplicate descriptions in selection"
m = sel.merge(food, on="description", how="left", validate="one_to_one")
missing = m[m.fdc_id.isna()]
if len(missing):
    raise SystemExit(f"Selection entries not found in SR Legacy: {missing.description.tolist()}")
m["fdc_id"] = m.fdc_id.astype(int)
m = m.merge(srl, on="fdc_id", how="left").merge(
    cat.rename(columns={"id": "food_category_id", "description": "usda_food_category"})[["food_category_id", "usda_food_category"]],
    on="food_category_id", how="left")
assert m.fdc_id.is_unique

ids = set(m.fdc_id)
fn_sel = fn[fn.fdc_id.isin(ids)].merge(
    nut.rename(columns={"id": "nutrient_id", "name": "nutrient_name"})[["nutrient_id", "nutrient_name", "unit_name", "nutrient_nbr"]],
    on="nutrient_id", how="left")
dups = fn_sel.duplicated(["fdc_id", "nutrient_id"]).sum()
assert dups == 0, f"{dups} duplicate food/nutrient rows"

# ---- wide table: core nutrients ----
wide = m[["fdc_id", "NDB_number", "project_category", "common_name", "form", "description",
          "usda_food_category", "data_type", "publication_date"]].copy()
# FDC publication date (2019-04-01) = when USDA loaded SR Legacy into FoodData Central; the data is the April 2018 release
wide = wide.rename(columns={"publication_date": "fdc_publication_date"})
wide = wide.rename(columns={"NDB_number": "ndb_number", "description": "usda_description"})
for nid, col in CORE:
    vals = fn_sel[fn_sel.nutrient_id == nid].set_index("fdc_id").amount
    wide[col] = wide.fdc_id.map(vals)
core_cols = [c for _, c in CORE]
wide["nutrients_not_reported"] = wide[core_cols].apply(
    lambda r: ";".join(c for c in core_cols if pd.isna(r[c])), axis=1)
wide["basis"] = "per 100 g edible portion"
wide["source"] = "USDA FoodData Central, SR Legacy (April 2018 release)"
wide["source_url"] = wide.fdc_id.map(lambda i: f"https://fdc.nal.usda.gov/food-details/{i}/nutrients")

# ---- validation flags (never modify values) ----
ash = wide.fdc_id.map(fn_sel[fn_sel.nutrient_id == ASH_ID].set_index("fdc_id").amount)
flags = []
for i, r in wide.iterrows():
    f = []
    for c in core_cols:
        if pd.notna(r[c]) and r[c] < 0:
            f.append(f"negative_{c}")
    macro = sum(r[c] for c in ["protein_g", "total_fat_g", "carbohydrate_g"] if pd.notna(r[c]))
    if macro > 100.5:
        f.append("macros_exceed_100g")
    if pd.notna(r.saturated_fat_g) and pd.notna(r.total_fat_g) and r.saturated_fat_g > r.total_fat_g + 0.01:
        f.append("saturated_fat_gt_total_fat")
    if pd.notna(r.total_sugars_g) and pd.notna(r.carbohydrate_g) and r.total_sugars_g > r.carbohydrate_g + 0.01:
        f.append("sugars_gt_carbohydrate")
    prox = [r.water_g, r.protein_g, r.total_fat_g, r.carbohydrate_g, ash[i]]
    if all(pd.notna(x) for x in prox):
        s = sum(prox)
        if not (97 <= s <= 103):
            f.append(f"proximates_sum_{s:.1f}g")
    else:
        f.append("proximate_check_not_possible")
    # ENERGY RULE (one rule for every food): estimate = 4 x protein + 4 x carbohydrate + 9 x fat.
    # Flag when the published energy differs from the estimate by MORE than 15% AND by more than 5 kcal.
    # SR uses food-specific energy factors, so flagged gaps are expected (mostly vegetables) and values are kept.
    if all(pd.notna(x) for x in [r.energy_kcal, r.protein_g, r.total_fat_g, r.carbohydrate_g]):
        est = 4 * r.protein_g + 4 * r.carbohydrate_g + 9 * r.total_fat_g
        gap = est - r.energy_kcal
        if abs(gap) > 5 and (r.energy_kcal == 0 or abs(gap) / r.energy_kcal > 0.15):
            pct = f"{gap / r.energy_kcal:+.0%}" if r.energy_kcal else "n/a"
            f.append(f"energy_vs_4-4-9_gap_{pct}")
    flags.append(";".join(f))
wide["quality_flags"] = flags

# held-out flag column left empty on purpose: the train/test split is a team decision
wide = wide.sort_values(["project_category", "common_name", "form", "fdc_id"]).reset_index(drop=True)
wide.to_csv(os.path.join(OUT, "fdc_generic_foods.csv"), index=False)

# ---- long table: every SR nutrient for the selected foods (full traceability) ----
# `median` is empty for every SR Legacy row, so it is not exported
long = fn_sel[["fdc_id", "nutrient_id", "nutrient_nbr", "nutrient_name", "unit_name", "amount",
               "data_points", "derivation_id", "min", "max", "footnote"]].copy()
long = long.rename(columns={"amount": "amount_per_100g", "unit_name": "unit"})
long = long.sort_values(["fdc_id", "nutrient_id"])
long.to_csv(os.path.join(OUT, "fdc_food_nutrients_long.csv"), index=False)

# ---- portions / household servings ----
p = por[por.fdc_id.isin(ids)].merge(mu.rename(columns={"id": "measure_unit_id", "name": "measure_unit"}),
                                     on="measure_unit_id", how="left")
# SR Legacy leaves measure_unit 'undetermined' (id 9999) and portion_description empty: the unit is written
# in `modifier` (e.g. "cup, chopped"), so those two columns are not exported
p = p.merge(wide[["fdc_id", "common_name", "form"]], on="fdc_id", how="left")
p = p[["fdc_id", "common_name", "form", "seq_num", "amount", "modifier",
       "gram_weight", "data_points", "footnote"]].sort_values(["fdc_id", "seq_num"])
p.to_csv(os.path.join(OUT, "fdc_food_portions.csv"), index=False)

# ---- summary for the report ----
summary = {
    "foods": int(len(wide)),
    "foods_per_category": wide.project_category.value_counts().sort_index().to_dict(),
    "forms": wide.form.value_counts().to_dict(),
    "long_rows": int(len(long)),
    "distinct_nutrients_long": int(long.nutrient_id.nunique()),
    "portion_rows": int(len(p)),
    "foods_without_portions": sorted(wide.loc[~wide.fdc_id.isin(p.fdc_id), "common_name"].astype(str) + " (" + wide.loc[~wide.fdc_id.isin(p.fdc_id), "form"] + ")"),
    "missing_per_core_nutrient": {c: int(wide[c].isna().sum()) for c in core_cols},
    "foods_with_any_missing_core": int((wide.nutrients_not_reported != "").sum()),
    "flag_counts": pd.Series([x for f in wide.quality_flags for x in f.split(";") if x])
        .str.replace(r"_[-+]?\d.*$", "", regex=True).value_counts().to_dict(),
    "flagged_foods": wide.loc[wide.quality_flags != "", ["fdc_id", "common_name", "form", "quality_flags"]].to_dict("records"),
}
with open(os.path.join(OUT, "fdc_build_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print(json.dumps({k: v for k, v in summary.items() if k != "flagged_foods"}, indent=2, default=str))
