"""Build the packaged-products dataset from the official Open Food Facts CSV export (version 2).

Input : https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
        (tab-separated, ~1 GB compressed, read without unzipping)
Usage : python build_off.py <export .csv[.gz]> <output folder>
            [--per-category 130] [--must-include tests_barcodes.txt] [--require-allergens]
Needs : pandas, wordfreq, langdetect   (pip install wordfreq langdetect)

Pipeline (no value is ever estimated, converted or corrected; products are only kept or dropped):
  A. stream the export in chunks, keep the needed columns
  B. category: first project category whose include-tags match and whose exclude-tags/name-words do not
  C. required fields: product name, brand, country, ingredient list, 7 nutrients per 100 g/ml
  D. plausibility: ranges, salt cap per category, salt vs sodium, saturated fat <= fat, sugars <= carbs,
     macros <= 100 g, energy vs macros
  E. duplicates: same barcode; same brand + name; same ingredient text + same nutrient values
  F. ranking: sold in an English-speaking country first, then allergen info, completeness, recency
  G. language: walk each category in rank order; accept a product only if its ingredient list is
     detected as English (word-frequency vote + langdetect) and its name is English or language-neutral
     (brand-only names); stop at the per-category cap
Every rejection is counted by reason in off_build_summary.json.
"""
import sys, os, re, csv, json, argparse
import pandas as pd
from wordfreq import zipf_frequency
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect is random by default; this makes results repeatable

ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("out")
ap.add_argument("--per-category", type=int, default=130)
ap.add_argument("--must-include", help="text file with one barcode per line (test-set products)")
ap.add_argument("--require-allergens", action="store_true", help="drop products with no allergen tags")
ap.add_argument("--chunksize", type=int, default=200_000)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------- settings
# (name, include tags, exclude tags, exclude words in product name, salt cap g per 100 g/ml, basis)
CATEGORIES = [
    ("Breakfast cereals", {"en:breakfast-cereals"},
     {"en:cereal-bars", "en:porridges-to-prepare"}, r"\bbar(s)?\b", 3.0, "100 g"),
    ("Yogurts", {"en:yogurts"},
     {"en:puddings", "en:creamy-puddings", "en:custards", "en:sterilized-dairy-desserts", "en:dessert-creams"},
     r"\b(dessert|pudding|custard|mousse|cr[eè]me)\b", 1.0, "100 g"),
    ("Chocolate", {"en:chocolates", "en:chocolate-bars"},
     {"en:chocolate-spreads", "en:cocoa-and-chocolate-powders", "en:chocolate-drinks", "en:hot-chocolate-powders",
      "en:chocolate-biscuits", "en:spreads"}, r"\b(spread|powder|drink|biscuit|cookie)s?\b", 1.5, "100 g"),
    ("Biscuits and salty snacks", {"en:biscuits", "en:crisps", "en:chips"},
     {"en:viennoiseries", "en:pastries", "en:cakes", "en:brioches", "en:breads"},
     r"\b(baguette|brioche|croissant|cake)s?\b", 5.0, "100 g"),
    ("Soft drinks and juices", {"en:sodas", "en:fruit-juices", "en:juices"},
     {"en:alcoholic-beverages", "en:syrups", "en:beverage-concentrates", "en:energy-drinks"},
     r"\b(squash|syrup|cordial|concentrate)\b", 0.5, "100 ml"),
    ("Bread", {"en:breads"},
     {"en:rice-cakes", "en:puffed-rice-cakes", "en:crackers", "en:crispbreads", "en:salty-snacks", "en:appetizers",
      "en:biscuits-and-crackers", "en:viennoiseries"}, r"\b(cracker|rice cake)s?\b", 3.0, "100 g"),
]
ENGLISH_COUNTRIES = {"en:united-kingdom", "en:united-states", "en:ireland", "en:canada", "en:australia",
                     "en:new-zealand"}
NUTR = {  # OFF column -> output column; values are per 100 g (solids) or per 100 ml (drinks)
    "energy-kcal_100g": "energy_kcal", "fat_100g": "fat_g", "saturated-fat_100g": "saturated_fat_g",
    "carbohydrates_100g": "carbohydrates_g", "sugars_100g": "sugars_g", "fiber_100g": "fiber_g",
    "proteins_100g": "proteins_g", "salt_100g": "salt_g", "sodium_100g": "sodium_g",
}
REQUIRED_NUTR = ["energy-kcal_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g",
                 "sugars_100g", "proteins_100g", "salt_100g"]
TEXT = ["code", "product_name", "product_name_en", "brands", "quantity", "categories_tags", "countries_tags",
        "ingredients_text", "ingredients_text_en", "allergens", "allergens_tags", "traces_tags",
        "serving_size", "serving_quantity", "completeness", "last_modified_t"]
WANTED = set(TEXT) | set(NUTR)
_UNUSED_FOREIGN_NAME = re.compile(r"[àâçéèêëîïôûùüÿœæäöñßăâîșțčšžćđąęłńóśźż]|\b(au|aux|à|la|le|les|de|du|des|avec|sans|"
                          r"et|pain|yaourt|chocolat|noir|mit|und|ohne|al|alla|con|di|del|y|sin|jogurt|joghurt|"
                          r"saveur|parfum|fromage|lait)\b", re.I)

must = set()
if args.must_include:
    must = {l.strip() for l in open(args.must_include) if l.strip()}

reasons = {}
def reject(df, mask, reason):
    mask = mask.fillna(False)
    n = int(mask.sum())
    if n:
        reasons[reason] = reasons.get(reason, 0) + n
        lost = must & set(df.loc[mask, "code"])
        if lost:
            reasons.setdefault("_must_include_rejected", {}).update({b: reason for b in lost})
    return df[~mask]


def tagset(s):
    return set(s.split(",")) if isinstance(s, str) and s else set()


def assign(row):
    tags, name = tagset(row.categories_tags), str(row.product_name_out)
    for cat, inc, exc, words, *_ in CATEGORIES:
        if tags & inc:
            if tags & exc or re.search(words, name, re.I):
                return "excluded_by_category_rules"
            return cat
    return None


SALT_CAP = {c[0]: c[4] for c in CATEGORIES}
BASIS = {c[0]: c[5] for c in CATEGORIES}

# ---------------------------------------------------------------- A-D: stream and filter
kept, rows_read = [], 0
reader = pd.read_csv(args.export, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, on_bad_lines="skip",
                     usecols=lambda c: c in WANTED, chunksize=args.chunksize)
for ch in reader:
    rows_read += len(ch)
    for col in WANTED - set(ch.columns):
        ch[col] = pd.NA
    has_en = ch.product_name_en.notna() & ch.product_name_en.str.strip().ne("")
    ch["product_name_out"] = ch.product_name_en.where(has_en, ch.product_name)
    ch = reject(ch, ~ch.categories_tags.fillna("").str.contains(
        "|".join(t for c in CATEGORIES for t in c[1])), "not_in_selected_categories")
    if ch.empty:
        continue
    ch["project_category"] = ch.apply(assign, axis=1)
    ch = reject(ch, ch.project_category.isna(), "not_in_selected_categories")
    ch = reject(ch, ch.project_category.eq("excluded_by_category_rules"), "excluded_by_category_rules")
    # required identifying fields
    blank = lambda s: s.isna() | s.astype(str).str.strip().eq("")
    ch = reject(ch, blank(ch.product_name_out), "no_product_name")
    ch = reject(ch, blank(ch.brands), "no_brand")
    ch = reject(ch, blank(ch.countries_tags), "no_country")
    has_en_ing = ch.ingredients_text_en.notna() & ch.ingredients_text_en.str.strip().str.len().ge(10)
    ch["ingredients_out"] = ch.ingredients_text_en.where(has_en_ing, ch.ingredients_text)
    ch = reject(ch, blank(ch.ingredients_out) | ch.ingredients_out.str.strip().str.len().lt(10), "no_ingredient_list")
    for c in NUTR:
        ch[c] = pd.to_numeric(ch[c], errors="coerce")
    ch = reject(ch, ch[REQUIRED_NUTR].isna().any(axis=1), "incomplete_nutrition")
    ch["allergens_out"] = ch.allergens_tags.where(~blank(ch.allergens_tags), ch.allergens)
    if args.require_allergens:
        ch = reject(ch, blank(ch.allergens_out), "no_allergen_tags")
    # plausibility (values are never changed)
    g = ["fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g"]
    ch = reject(ch, (ch[g] < 0).any(axis=1) | (ch[g] > 100).any(axis=1) | ~ch["energy-kcal_100g"].between(0, 900),
                "value_out_of_range")
    ch = reject(ch, ch["salt_100g"] > ch.project_category.map(SALT_CAP), "salt_above_category_cap")
    sod = ch["sodium_100g"]
    ch = reject(ch, sod.notna() & ((ch["salt_100g"] - 2.5 * sod).abs() > (0.10 * ch["salt_100g"]).clip(lower=0.1)),
                "salt_sodium_mismatch")
    ch = reject(ch, ch["saturated-fat_100g"] > ch["fat_100g"] + 0.001, "saturated_fat_gt_fat")
    ch = reject(ch, ch["sugars_100g"] > ch["carbohydrates_100g"] + 0.001, "sugars_gt_carbohydrates")
    ch = reject(ch, ch[["fat_100g", "carbohydrates_100g", "proteins_100g"]].sum(axis=1) > 101, "macros_exceed_100g")
    est = (4 * ch["proteins_100g"] + 4 * ch["carbohydrates_100g"] + 9 * ch["fat_100g"]
           + 2 * ch["fiber_100g"].fillna(0))
    ch = reject(ch, (est - ch["energy-kcal_100g"]).abs() > (0.2 * ch["energy-kcal_100g"]).clip(lower=15),
                "energy_inconsistent_with_macros")
    kept.append(ch)

df = pd.concat(kept, ignore_index=True)
after_filters = len(df)

# ---------------------------------------------------------------- E-F: rank, then remove duplicates
df["english_country"] = df.countries_tags.map(lambda s: bool(tagset(s) & ENGLISH_COUNTRIES))
df["has_allergens"] = df.allergens_out.fillna("").str.strip().ne("")
df["has_serving"] = pd.to_numeric(df.serving_quantity, errors="coerce").gt(0)
df["has_fiber"] = df.fiber_100g.notna()
df["completeness_num"] = pd.to_numeric(df.completeness, errors="coerce").fillna(0)
df["last_modified_t"] = pd.to_numeric(df.last_modified_t, errors="coerce").fillna(0)
df["is_must"] = df.code.isin(must)
order = ["is_must", "english_country", "has_allergens", "has_serving", "has_fiber", "completeness_num",
         "last_modified_t"]
df = df.sort_values(order + ["code"], ascending=[False] * len(order) + [True]).reset_index(drop=True)

def drop_dups(d, key, reason):
    dup = key.duplicated(keep="first")
    reasons[reason] = int(dup.sum())
    return d[~dup]

df = drop_dups(df, df.code.str.strip().str.lstrip("0"), "duplicate_barcode")
df = drop_dups(df, df.brands.str.lower().str.split(",").str[0].str.strip() + "|" +
               df.product_name_out.str.lower().str.replace(r"\s+", " ", regex=True).str.strip(),
               "duplicate_brand_and_name")
ing_key = df.ingredients_out.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
nut_key = df[REQUIRED_NUTR].round(2).astype(str).agg("|".join, axis=1)
df = drop_dups(df, ing_key + "#" + nut_key, "duplicate_ingredients_and_nutrients")
after_dedupe = len(df)
available = df.project_category.value_counts().sort_index().to_dict()

# ---------------------------------------------------------------- G: language check while selecting
# Two detectors:
#  * word-frequency vote (wordfreq): each word is counted for English if it is at least about as common
#    in English as in French/Spanish/German/Italian/Dutch/Portuguese/Romanian/Polish. Reliable for short
#    names and comma-separated ingredient lists, where statistical detectors struggle.
#  * langdetect: second opinion for ingredient lists that the vote leaves borderline.
OTHER_LANGS = ["fr", "es", "de", "it", "nl", "pt", "ro", "pl"]
_cache = {}

def word_vote(text):
    """Returns (English words, other-language words, clearly foreign words)."""
    en = other = strong = 0
    for t in re.findall(r"[^\W\d_]{3,}", str(text).lower()):
        if t not in _cache:
            e = zipf_frequency(t, "en")
            o = max(zipf_frequency(t, l) for l in OTHER_LANGS)
            vote = None if (e == 0 and o == 0) else (e >= o - 0.5)
            _cache[t] = (vote, o >= 2.5 and o - e >= 1.5)   # e.g. "yaourt", "chocolat", "flocons"
        vote, is_strong = _cache[t]
        en += vote is True
        other += vote is False
        strong += is_strong
    return en, other, strong

NON_LATIN = re.compile(r"[^\x00-\u024F\u2000-\u206F\u20AC\u2122\u00AE]")  # letters outside Latin scripts

def langdetect_en(text):
    try:
        best = detect_langs(str(text))[0]
        return best.lang == "en" and best.prob >= 0.90
    except LangDetectException:
        return False

def english_ok(name, ingredients):
    en, other, _ = word_vote(ingredients)
    share = en / (en + other) if en + other else 0
    if not (share >= 0.75 or (share >= 0.5 and langdetect_en(ingredients))):
        return False, "ingredients_not_english"
    if NON_LATIN.search(str(name)):
        return False, "name_not_english"
    en, other, strong = word_vote(name)
    # a name fails if it has any clearly foreign word, or more foreign than English words;
    # names made only of unknown words (brand names) are treated as language-neutral
    if strong > 0 or (other >= en and other > 0):
        return False, "name_not_english"
    return True, None

selected = []
for cat, grp in df.groupby("project_category", sort=True):
    n = 0
    for r in grp.itertuples():
        if n >= args.per_category and not r.is_must:
            continue
        ok, why = english_ok(r.product_name_out, r.ingredients_out)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            if r.is_must:
                reasons.setdefault("_must_include_rejected", {})[r.code] = why
            continue
        selected.append(r.Index)
        n += 1
df = df.loc[selected]

# ---------------------------------------------------------------- output
def missing(r):
    fields = {"fiber_g": r["fiber_100g"], "sodium_g": r["sodium_100g"], "serving_size": r["serving_size"],
              "allergens_tags": r["allergens_out"], "traces_tags": r["traces_tags"], "quantity": r["quantity"]}
    return ";".join(k for k, v in fields.items() if pd.isna(v) or (isinstance(v, str) and not v.strip()))

out = pd.DataFrame({
    "barcode": df.code.astype(str),
    "product_name": df.product_name_out,
    "brands": df.brands,
    "quantity": df.quantity,
    "project_category": df.project_category,
    "off_categories_tags": df.categories_tags,
    "countries_tags": df.countries_tags,
    "sold_in_english_speaking_country": df.english_country,
    "ingredients_text": df.ingredients_out,
    "allergens_tags": df.allergens_out,
    "traces_tags": df.traces_tags,
    "serving_size": df.serving_size,
    "serving_quantity": pd.to_numeric(df.serving_quantity, errors="coerce"),
    "nutrition_basis": df.project_category.map(BASIS).map(lambda b: f"per {b}"),
})
for c, o in NUTR.items():
    out[o] = df[c]
out["fiber_gt_carbohydrates"] = out.fiber_g > out.carbohydrates_g  # can be genuine on EU labels (see docs)
out["fields_missing"] = df.apply(missing, axis=1)
out["last_modified_utc"] = pd.to_datetime(df.last_modified_t, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
out["source"] = "Open Food Facts (ODbL)"
out["source_url"] = "https://world.openfoodfacts.org/product/" + out.barcode
out = out.sort_values(["project_category", "brands", "product_name"]).reset_index(drop=True)
out.to_csv(os.path.join(args.out, "off_packaged_products.csv"), index=False)

summary = {
    "script_version": 2,
    "export_file": os.path.basename(args.export),
    "rows_read": rows_read,
    "rejections_by_reason": reasons,
    "after_quality_filters": after_filters,
    "after_deduplication": after_dedupe,
    "available_per_category_after_dedupe": available,
    "per_category_cap": args.per_category,
    "final_products": len(out),
    "final_per_category": out.project_category.value_counts().sort_index().to_dict(),
    "must_include_requested": sorted(must),
    "must_include_kept": sorted(must & set(out.barcode)),
    "sold_in_english_speaking_country": int(out.sold_in_english_speaking_country.sum()),
    "coverage": {c: round(float(out[c].notna().mean()), 3) for c in
                 ["allergens_tags", "traces_tags", "serving_quantity", "fiber_g", "sodium_g"]},
    "last_modified_range": [out.last_modified_utc.min(), out.last_modified_utc.max()],
}
json.dump(summary, open(os.path.join(args.out, "off_build_summary.json"), "w"), indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
