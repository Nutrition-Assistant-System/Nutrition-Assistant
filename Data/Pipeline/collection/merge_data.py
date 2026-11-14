"""Merge the USDA generic foods and the Open Food Facts products into ONE table.

Usage: python merge_data.py   (run from the nutrition_data folder)

Inputs : data/fdc/fdc_generic_foods.csv, data/fdc/fdc_food_portions.csv, data/off/off_packaged_products.csv
Output : data/merged/foods_and_products.csv, data/merged/merge_summary.json

Rules
  * item_id = "<source>:<source id>"  ->  fdc:171477  /  off:5000127163096  (unique across both sources)
  * basis   = "per 100 g" or "per 100 ml", stated on every row
  * Units are made consistent by exact unit conversion only (never by estimation):
      sodium_mg : USDA publishes mg;  Open Food Facts publishes g  -> OFF value x 1000
      salt_g    : published by Open Food Facts only. USDA does not publish salt, so it stays BLANK for USDA rows
                  (we do not derive it from sodium).
  * Columns that exist in only one source stay blank for the other source; blank never means 0.
  * Carbohydrate is defined differently by the sources (see `carbohydrate_definition`).
"""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = lambda *p: os.path.join(HERE, *p)

fdc = pd.read_csv(F("data", "fdc", "fdc_generic_foods.csv"))
por = pd.read_csv(F("data", "fdc", "fdc_food_portions.csv"))
off = pd.read_csv(F("data", "off", "off_packaged_products.csv"), dtype={"barcode": str})

# household measures as one readable text field for USDA rows ("1 cup, chopped = 140 g; 1 large = 50 g")
meas = (por.assign(t=lambda d: d.amount.map(lambda a: f"{a:g}") + " " + d.modifier.astype(str) + " = "
                   + d.gram_weight.map(lambda g: f"{g:g}") + " g")
        .groupby("fdc_id").t.agg("; ".join))

a = pd.DataFrame({
    "item_id": "fdc:" + fdc.fdc_id.astype(str),
    "source": "USDA FoodData Central (SR Legacy, April 2018)",
    "source_id": fdc.fdc_id.astype(str),
    "item_type": "generic food",
    "name": fdc.common_name,
    "full_name": fdc.usda_description,
    "brand": pd.NA,
    "project_category": fdc.project_category,
    "form": fdc.form,
    "basis": "per 100 g",
    "energy_kcal": fdc.energy_kcal,
    "protein_g": fdc.protein_g,
    "fat_g": fdc.total_fat_g,
    "saturated_fat_g": fdc.saturated_fat_g,
    "carbohydrate_g": fdc.carbohydrate_g,
    "carbohydrate_definition": "by difference (includes fibre)",
    "sugars_g": fdc.total_sugars_g,
    "fiber_g": fdc.fiber_g,
    "sodium_mg": fdc.sodium_mg,
    "salt_g": pd.NA,
    "cholesterol_mg": fdc.cholesterol_mg,
    "calcium_mg": fdc.calcium_mg,
    "iron_mg": fdc.iron_mg,
    "potassium_mg": fdc.potassium_mg,
    "water_g": fdc.water_g,
    "ingredients_text": pd.NA,
    "allergens_tags": pd.NA,
    "traces_tags": pd.NA,
    "serving_size": pd.NA,
    "serving_quantity": pd.NA,
    "household_measures": fdc.fdc_id.map(meas),
    "not_reported": fdc.nutrients_not_reported,
    "quality_flags": fdc.quality_flags,
    "source_url": fdc.source_url,
    "licence": "CC0 1.0 (public domain)",
})

b = pd.DataFrame({
    "item_id": "off:" + off.barcode,
    "source": "Open Food Facts",
    "source_id": off.barcode,
    "item_type": "packaged product",
    "name": off.product_name,
    "full_name": off.product_name,
    "brand": off.brands,
    "project_category": off.project_category,
    "form": "packaged",
    "basis": off.nutrition_basis,
    "energy_kcal": off.energy_kcal,
    "protein_g": off.proteins_g,
    "fat_g": off.fat_g,
    "saturated_fat_g": off.saturated_fat_g,
    "carbohydrate_g": off.carbohydrates_g,
    "carbohydrate_definition": "as declared on the label (EU labels exclude fibre; US labels include it)",
    "sugars_g": off.sugars_g,
    "fiber_g": off.fiber_g,
    "sodium_mg": off.sodium_g * 1000,      # g -> mg (unit conversion only)
    "salt_g": off.salt_g,
    "cholesterol_mg": pd.NA, "calcium_mg": pd.NA, "iron_mg": pd.NA, "potassium_mg": pd.NA, "water_g": pd.NA,
    "ingredients_text": off.ingredients_text,
    "allergens_tags": off.allergens_tags,
    "traces_tags": off.traces_tags,
    "serving_size": off.serving_size,
    "serving_quantity": off.serving_quantity,
    "household_measures": pd.NA,
    "not_reported": off.fields_missing,
    "quality_flags": off.fiber_gt_carbohydrates.map(lambda x: "fiber_gt_carbohydrates" if x else pd.NA),
    "source_url": off.source_url,
    "licence": "ODbL 1.0 (attribution + share-alike)",
})

m = pd.concat([a, b], ignore_index=True)
assert m.item_id.is_unique, "duplicate item_id"
assert m.basis.isin(["per 100 g", "per 100 ml"]).all(), "unknown basis"
assert m.energy_kcal.notna().all() and m.protein_g.notna().all()
os.makedirs(F("data", "merged"), exist_ok=True)
m.to_csv(F("data", "merged", "foods_and_products.csv"), index=False)

summary = {
    "rows": len(m),
    "by_source": m.source.value_counts().to_dict(),
    "by_basis": m.basis.value_counts().to_dict(),
    "by_category": m.groupby(["item_type", "project_category"]).size().rename("n").reset_index()
                    .to_dict("records"),
    "columns": list(m.columns),
    "unit_conversions": {"sodium_mg (Open Food Facts rows)": "published sodium in g x 1000"},
    "filled_share": {c: round(float(m[c].notna().mean()), 3) for c in m.columns},
}
json.dump(summary, open(F("data", "merged", "merge_summary.json"), "w"), indent=2, default=str)
print(json.dumps({k: summary[k] for k in ["rows", "by_source", "by_basis"]}, indent=2))
