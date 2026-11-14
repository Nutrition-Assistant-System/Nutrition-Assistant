# %% [markdown]
# # Phase 2 preprocessing: from the merged table to training data
#
# **Input:** `data/raw/foods_and_products.csv`, the merged table: 208 USDA generic foods and 780 Open Food Facts
# products, one `item_id` per row (`fdc:171477`, `off:5000127163096`) and one `basis` per row
# (per 100 g or per 100 ml). The input also includes `tests/test_questions.csv`, the 30 frozen test questions.
#
# **Outputs:**
#
# | File | Used by | What it is |
# |---|---|---|
# | `data/cpt/train.txt`, `data/cpt/heldout.txt` | `train_cpt.py` | One plain-text document per item, split 90/10 **by document** |
# | `data/cpt/split_index.csv` | report | Which document went to which split |
# | `data/sft_pairs.jsonl` | `train_sft.py` | About 900 question → answer pairs |
# | `results/sft_handcheck_sample.csv` | team | 50 random pairs for the manual check the brief requires |
# | `results/data_stats.json` | report | All counts quoted in the report |
#
# **Rules:**
#
# 1. **Copied values only.** Every number comes from the merged table, or from a multiplication that the answer shows.
# 2. **Blanks.** A blank value is "not reported" and is never treated as 0.
# 3. **Test foods and products are excluded from the SFT pairs.** USDA foods are excluded by food name (for
#    example, every chicken-breast row) and products by barcode. They stay in the CPT **training** split, so the
#    model can learn their facts.
# 4. **Near-copies are dropped.** Any SFT question more than 0.90 similar to a test question is removed.
#
# **How to run:** `python -m src.prepare_data` from the repository root, or run this notebook top to bottom. In Colab the
# notebook finds the repository by itself. If you only upload `foods_and_products.csv` and `test_questions.csv`,
# it runs standalone with the default settings and writes `data/` and `results/` next to itself.

# %%
import os, re, json, random, difflib, glob, zipfile
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import yaml

# ---- find the repository, or run standalone ------------------------------------------------------------
# Works in three situations:
#   1. inside the repository (locally, or in Colab after unzipping/cloning it to /content/nutrition-assistant)
#   2. in Colab with nutrition-assistant.zip uploaded to /content (it is unzipped automatically)
#   3. standalone: only foods_and_products.csv and test_questions.csv uploaded (default settings are used,
#      outputs are written next to this notebook)
DEFAULTS = {
    "seed": 42,
    "data": {"heldout_fraction": 0.10, "sft_target_pairs": 900},
    "paths": {"merged_items": "data/raw/foods_and_products.csv", "test_questions": "tests/test_questions.csv",
              "cpt_train": "data/cpt/train.txt", "cpt_heldout": "data/cpt/heldout.txt",
              "cpt_split_index": "data/cpt/split_index.csv", "sft_pairs": "data/sft_pairs.jsonl",
              "results_dir": "results", "sft_handcheck": "results/sft_handcheck_sample.csv"},
}


def find_repo():
    starts = [os.getcwd(), "/content/nutrition-assistant", "/content/drive/MyDrive/nutrition-assistant"]
    starts += sorted(glob.glob("/content/*/")) + sorted(glob.glob("/content/*/*/"))
    for start in starts:
        d = os.path.abspath(start)
        while True:  # walk up from each start folder
            if os.path.exists(os.path.join(d, "config.yaml")) and os.path.isdir(os.path.join(d, "src")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


ROOT = find_repo()
if ROOT is None:
    for z in glob.glob("/content/nutrition-assistant*.zip") + glob.glob("nutrition-assistant*.zip"):
        zipfile.ZipFile(z).extractall(os.path.dirname(os.path.abspath(z)))
        ROOT = find_repo()
        if ROOT:
            print("unzipped", z)
            break

if ROOT:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    for k, v in DEFAULTS["paths"].items():
        cfg["paths"].setdefault(k, v)
    mode = "repository"
else:  # standalone: look for the two input files anywhere under the current folder (or /content)
    ROOT, cfg, mode = os.getcwd(), DEFAULTS, "standalone"
    for key, name in [("merged_items", "foods_and_products.csv"), ("test_questions", "test_questions.csv")]:
        hits = sorted(glob.glob(os.path.join(ROOT, "**", name), recursive=True) +
                      glob.glob(os.path.join("/content", "**", name), recursive=True))
        if not hits:
            raise FileNotFoundError(
                f"Could not find the repository (config.yaml) or {name}.\n"
                "Either upload nutrition-assistant.zip to /content, or upload foods_and_products.csv and "
                "test_questions.csv to the Colab Files panel, then run this cell again.")
        cfg["paths"][key] = os.path.abspath(hits[0])

P = {k: v if os.path.isabs(v) else os.path.join(ROOT, v) for k, v in cfg["paths"].items()}
SEED = cfg["seed"]
HELDOUT_FRACTION = cfg["data"]["heldout_fraction"]
TARGET_PAIRS = cfg["data"]["sft_target_pairs"]
rng = random.Random(SEED)
print(f"mode: {mode} | root: {ROOT}")
print("inputs:", P["merged_items"], "|", P["test_questions"])

# %% [markdown]
# ## 1. Load the merged table and the test set

# %%
items = pd.read_csv(P["merged_items"], dtype={"source_id": str})
tests = pd.read_csv(P["test_questions"], dtype=str).fillna("")
assert items.item_id.is_unique
assert items.basis.isin(["per 100 g", "per 100 ml"]).all()
generic = items[items.item_type == "generic food"].copy()
products = items[items.item_type == "packaged product"].copy()
print(f"{len(items)} items: {len(generic)} generic foods, {len(products)} packaged products")
print(items.groupby(["item_type", "basis"]).size().to_string())

# %% [markdown]
# ## 2. Test items: find them and keep them out of the SFT pairs

# %%
test_item_ids = sorted({f"fdc:{x}" for s in tests.source for x in re.findall(r"fdc_id (\d+)", s)} |
                       {f"off:{x}" for s in tests.source for x in re.findall(r"barcode (\d+)", s)})
missing_test_items = [i for i in test_item_ids if i not in set(items.item_id)]
assert not missing_test_items, f"test items not in the merged table: {missing_test_items}"
test_generic_names = set(generic.loc[generic.item_id.isin(test_item_ids), "name"])
excluded = items.item_id.isin(test_item_ids) | (items.item_type.eq("generic food") & items.name.isin(test_generic_names))
print(f"{len(test_item_ids)} test items ({len(test_generic_names)} USDA food names, "
      f"{sum(i.startswith('off:') for i in test_item_ids)} products) -> {int(excluded.sum())} rows excluded from SFT")
print(sorted(test_generic_names))

# %% [markdown]
# ## 3. Helpers: number format, names, basis

# %%
NUTR = [  # merged column, word used in questions, unit
    ("energy_kcal", "calories", "kcal"), ("protein_g", "protein", "g"), ("fat_g", "fat", "g"),
    ("saturated_fat_g", "saturated fat", "g"), ("carbohydrate_g", "carbohydrate", "g"), ("sugars_g", "sugar", "g"),
    ("fiber_g", "fibre", "g"), ("sodium_mg", "sodium", "mg"), ("salt_g", "salt", "g"),
    ("cholesterol_mg", "cholesterol", "mg"), ("calcium_mg", "calcium", "mg"), ("iron_mg", "iron", "mg"),
    ("potassium_mg", "potassium", "mg"),
]
WORD = {c: w for c, w, _ in NUTR}
UNIT = {c: u for c, _, u in NUTR}
# names used in the `not_reported` column (USDA build names -> merged names)
NOT_REPORTED_MAP = {"total_fat_g": "fat_g", "total_sugars_g": "sugars_g", "sodium_g": "sodium_mg"}
SRC = {"generic food": "USDA FoodData Central", "packaged product": "Open Food Facts"}


def dec2(v):
    """Round to 2 decimals, halves UP (4.635 -> 4.64), using exact decimal arithmetic."""
    d = v if isinstance(v, Decimal) else Decimal(repr(float(v)))
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt(v):
    """Up to 2 decimals (halves up), no trailing zeros, thousands separator (3.15, 260, 1,139)."""
    s = f"{dec2(v):,.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def calc(v, factor):
    """Multiply the DISPLAYED value (fmt) by the factor, so the working shown in an answer reproduces its
    result exactly: returns (shown value, shown factor, shown result)."""
    shown = Decimal(fmt(v).replace(",", ""))
    f = Decimal(str(factor)).normalize()
    return fmt(shown), f"{f:f}", fmt(dec2(shown * f))


PROPER_FIRST = {"Atlantic", "Greek", "Italian"}            # keep capitals on proper nouns
METHOD_WORDS = r"\b(raw|cooked|roasted|dry-roasted|boiled|hard-boiled|baked|grilled|fried|scrambled|toasted|" \
               r"broiled|pan-browned|dried|canned|fresh)\b"
GENERIC_NAMES = {"cereal", "bread", "yogurt", "yoghurt", "lemonade", "wafers", "fingers", "snaps", "croutons",
                 "bagel", "chapattis", "pitta", "blackcurrant", "doorsteps", "chocolate", "biscuits", "juice"}
CORPORATE = re.compile(r"\b(llc|inc|ltd|limited|company|corp|corporation|plc|co\.|gmbh|s\.a\.)\b", re.I)


def _norm(s):  # for comparing brand and name: case, curly quotes and punctuation ignored
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("\u2019", "'"))


def label(r):
    """How an item is named in questions, answers AND the CPT documents (one function, so names match)."""
    if r.item_type == "generic food":
        n = r["name"]
        first = n.split()[0].rstrip(",")
        n = n if first in PROPER_FIRST or first.isupper() else n[0].lower() + n[1:]
        # no "cooked"/"raw" prefix when the name already says how it was prepared ("peanuts, dry-roasted")
        return f"{r.form} {n}" if r.form in ("raw", "cooked") and not re.search(METHOD_WORDS, n, re.I) else n
    name = re.sub(r"\s+", " ", str(r["name"])).strip()
    if not re.search(r"\b[A-Z]\.[A-Z]\.$", name):             # keep abbreviations such as "M.F."
        name = re.sub(r"[.,;:]+$", "", name).strip()           # "...peanuts." -> "...peanuts"
    brands = [b.strip() for b in str(r.brand).split(",") if b.strip() and not CORPORATE.search(b)] \
        if pd.notna(r.brand) else []
    brands = list(dict.fromkeys(brands))                        # distinct, in source order
    if name.lower() in GENERIC_NAMES:                           # "Cereal" -> "Post Honey Bunches Of Oats Cereal"
        parts = [b for b in brands[:2] if _norm(b) not in _norm(name)]
        return " ".join(parts + [name]) if parts else name
    brand = brands[0] if brands else ""
    return name if not brand or _norm(brand) in _norm(name) else f"{brand} {name}"


def brand_text(r):
    """Brand line for documents: distinct brand names, whitespace cleaned, company-registration entries
    (Inc, Llc, Ltd, ...) left out, same rule as label()."""
    if pd.isna(r.brand):
        return ""
    bs = [re.sub(r"\s+", " ", b).strip() for b in str(r.brand).split(",")]
    keep = list(dict.fromkeys(b for b in bs if b and not CORPORATE.search(b)))
    return ", ".join(keep or [b for b in bs if b][:1]).rstrip(".")


MEASURE = re.compile(r"^\s*([\d.]+) (.+?) = ([\d.]+) g\s*$")


def measure_text(amt, modifier):
    """'1' + 'pita, small  (4" dia)' -> '1 pita small (4" dia)'; drops USDA's 'yields' wording.
    Returns (text, is_edible_yield)."""
    yield_ = bool(re.search(r"\byields?\b|\byield\b", modifier))
    mod = re.sub(r"\b(edible yield|yields?)\b", "", modifier)
    mod = re.sub(r"\s*,\s*", " ", mod)
    mod = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", mod))
    mod = re.sub(r"\s+", " ", mod).strip(" ,")
    return f"{fmt(amt)} {mod}", yield_


def measures(r):
    """Parsed household measures of a USDA item: list of (text, grams, is_edible_yield)."""
    out = []
    if isinstance(r.household_measures, str):
        for x in r.household_measures.split(";"):
            m = MEASURE.match(x)
            if m:
                t, y = measure_text(m.group(1), m.group(2))
                out.append((t, float(m.group(3)), y))
    return out


STD_TAG = re.compile(r"^en:[a-z][a-z-]*$")
TAG_WORD = {"soybeans": "soy", "sesame-seeds": "sesame seeds", "gluten": "gluten", "milk": "milk", "nuts": "nuts",
            "peanuts": "peanuts", "eggs": "eggs"}


def tag_list(s):
    """Standard allergen/trace tags only (free text such as 'en:Wheat bran' is ignored), readable words."""
    if not isinstance(s, str):
        return []
    tags = [t.strip() for t in s.split(",") if STD_TAG.match(t.strip())]
    return [TAG_WORD.get(t[3:], t[3:].replace("-", " ")) for t in tags]


def clean_ingredients(s):
    """Remove Open Food Facts allergen markup (_wheat_, *milk*) so text and keyword checks see plain words."""
    return re.sub(r"\s+", " ", re.sub(r"[_*]", "", str(s))).strip()


WARNING_START = re.compile(r"\b(may contain|may also contain|traces? of|produced in|made in a|manufactured in|"
                           r"made on|packed in a|factory (?:that|which)|in a factory)\b", re.I)


def split_warning(ingredients):
    """(ingredient part, 'may contain' warning part): everything from the first precautionary phrase on is a
    warning about possible traces, not an ingredient."""
    text = clean_ingredients(ingredients)
    m = WARNING_START.search(text)
    return (text[:m.start()], text[m.start():]) if m else (text, "")


def amount(r):  # "100 g" or "100 ml"
    return r.basis.replace("per ", "")


def has(r, c):
    return c in r and pd.notna(r[c])


items["label"] = items.apply(label, axis=1)
generic, products = items[items.item_type == "generic food"], items[items.item_type == "packaged product"]
print(items.sample(6, random_state=SEED)[["item_id", "label", "basis"]].to_string(index=False))

# %% [markdown]
# ## 4. Continued-pretraining corpus: one document per item
#
# Each row of the table becomes a short plain-text document, because a language model learns from text, not tables.

# %%
def generic_doc(r):
    vals = [f"{fmt(r[c])} {UNIT[c]} {'energy' if c == 'energy_kcal' else WORD[c]}".replace(" kcal energy", " kcal")
            for c, _, _ in NUTR if has(r, c)]
    parts = [f"{r.full_name}.", f"Common name: {r['name']} ({r.form}). Food group: {r.project_category}.",
             f"Nutrition per 100 g (USDA FoodData Central): " + "; ".join(vals) + ".",
             f"100 g of {r.label} contains {fmt(r.energy_kcal)} kcal and {fmt(r.protein_g)} g of protein."]
    if isinstance(r.not_reported, str) and r.not_reported:
        names = [WORD.get(NOT_REPORTED_MAP.get(c, c), c) for c in r.not_reported.split(";")]
        parts.append("Not reported by USDA: " + ", ".join(names) + ".")
    ms = measures(r)
    if ms:
        parts.append("Household measures: " + "; ".join(
            f"{t} = {fmt(g)} g{' of edible food' if y else ''}" for t, g, y in ms) + ".")
    return " ".join(parts)


def product_doc(r):
    vals = [f"{fmt(r[c])} {UNIT[c]} {'energy' if c == 'energy_kcal' else WORD[c]}".replace(" kcal energy", " kcal")
            for c in ["energy_kcal", "fat_g", "saturated_fat_g", "carbohydrate_g", "sugars_g", "fiber_g",
                      "protein_g", "salt_g", "sodium_mg"] if has(r, c)]
    parts = [f"{r.label}.", f"Brand: {brand_text(r)}. Category: {r.project_category}. Barcode: {r.source_id}."]
    if has(r, "serving_quantity") and float(r.serving_quantity) > 0:   # numeric value, as in the SFT answers
        parts.append(f"Serving size: {fmt(r.serving_quantity)} {amount(r).split()[1]}.")
    parts.append(f"Nutrition {r.basis} (Open Food Facts): " + "; ".join(vals) + ".")
    parts.append(f"Ingredients: {clean_ingredients(r.ingredients_text).rstrip('.')}.")
    if tag_list(r.allergens_tags):
        parts.append("Allergens: " + ", ".join(tag_list(r.allergens_tags)) + ".")
    if tag_list(r.traces_tags):
        parts.append("May contain traces of: " + ", ".join(tag_list(r.traces_tags)) + ".")
    return " ".join(parts)


docs = pd.DataFrame({
    "doc_id": items.item_id,
    "is_test_item": items.item_id.isin(test_item_ids),
    "text": [generic_doc(r) if r.item_type == "generic food" else product_doc(r) for _, r in items.iterrows()],
})
print(docs.text.iloc[0][:600], "\n")
print(docs[docs.doc_id.str.startswith("off:")].text.iloc[0][:600])

# %% [markdown]
# ## 5. Split 90/10 by document
#
# Held-out documents are drawn only from non-test items, so every test item is in the training split.

# %%
candidates = docs[~docs.is_test_item].doc_id.tolist()
n_held = round(HELDOUT_FRACTION * len(docs))
held = set(rng.sample(candidates, n_held))
docs["split"] = docs.doc_id.map(lambda d: "heldout" if d in held else "train")
os.makedirs(os.path.dirname(P["cpt_train"]), exist_ok=True)
SEP = "\n\n"  # train_cpt.py splits documents on blank lines
open(P["cpt_train"], "w").write(SEP.join(docs[docs.split == "train"].text) + "\n")
open(P["cpt_heldout"], "w").write(SEP.join(docs[docs.split == "heldout"].text) + "\n")
docs.assign(chars=docs.text.str.len()).drop(columns="text").to_csv(P["cpt_split_index"], index=False)
assert not docs[(docs.split == "heldout") & docs.is_test_item].shape[0]
print(docs.split.value_counts().to_string(), "\n", "characters:", docs.text.str.len().sum())

# %% [markdown]
# ## 6. SFT pairs
#
# The pairs are built from **non-test items only**. Every answer names its source and shows any multiplication it uses.
#
# | Type | What it teaches |
# |---|---|
# | `fdc_lookup`, `off_lookup` | Read a value per 100 g or per 100 ml |
# | `fdc_scale` | Scale a value to another amount |
# | `fdc_portion` | Use household measures (1 cup = … g) |
# | `off_serving`, `off_per_serving` | Read the serving size; amount per serving |
# | `fdc_compare`, `off_compare` | Compare two items on one nutrient |
# | `off_allergen` | Answer from the allergen tags, or from the ingredient list when there are no tags |
# | `salt_vs_sodium` | USDA publishes sodium, not salt; say so instead of converting |
# | `missing_value`, `unknown_food` | Say "I don't have that information" |
# | `refusal` | Decline medical, diet and supplement advice |

# %%
pairs = []
def add(q, a, kind, src):
    pairs.append({"question": q, "answer": a, "type": kind, "source": src})

G = items[(items.item_type == "generic food") & ~excluded].reset_index(drop=True)
Pr = items[(items.item_type == "packaged product") & ~excluded].reset_index(drop=True)
LOOKUP_Q = ["How much {n} is in {b} of {f}?", "What is the {n} content of {f} per {b}?",
            "How much {n} does {b} of {f} have?", "Tell me the amount of {n} in {b} of {f}."]
KCAL_Q = ["How many calories are in {b} of {f}?", "What is the energy value of {b} of {f}?",
          "How many kcal does {b} of {f} have?"]


def lookup(r, c, kind):
    b, f, src = amount(r), r.label, SRC[r.item_type]
    if c == "energy_kcal":
        q, a = rng.choice(KCAL_Q).format(b=b, f=f), f"{b} of {f} contains {fmt(r[c])} kcal ({src})."
    else:
        q = rng.choice(LOOKUP_Q).format(n=WORD[c], b=b, f=f)
        a = f"{b} of {f} contains {fmt(r[c])} {UNIT[c]} of {WORD[c]} ({src})."
    add(q, a, kind, r.item_id)


# 6a. USDA lookups: two different nutrients per food
for _, r in G.iterrows():
    avail = [c for c, _, _ in NUTR if has(r, c)]
    for c in rng.sample(avail, 2):
        lookup(r, c, "fdc_lookup")

# 6b. USDA scaling to another amount
for _, r in G.sample(frac=0.8, random_state=SEED).iterrows():
    g = rng.choice([30, 50, 75, 120, 150, 200, 250, 300])
    c = rng.choice(["energy_kcal", "protein_g", "fat_g"])
    v, f, res = calc(r[c], Decimal(g) / 100)
    what = "kcal" if c == "energy_kcal" else f"{UNIT[c]} of {WORD[c]}"
    q = (f"How many calories are in {g} g of {r.label}?" if c == "energy_kcal"
         else f"How much {WORD[c]} is in {g} g of {r.label}?")
    add(q, f"About {res} {what}: {v} {UNIT[c]} per 100 g x {f} = {res} {UNIT[c]} "
           f"(USDA FoodData Central).", "fdc_scale", r.item_id)

# 6c. USDA household measures, phrased naturally ("1 pita small (4" dia)", "1 cup with pits")
for _, r in G.sample(frac=0.7, random_state=SEED + 1).iterrows():
    ms = measures(r)
    if not ms:
        continue
    text, grams, edible = rng.choice(ms)
    c = rng.choice(["energy_kcal", "protein_g"])
    v, f, res = calc(r[c], Decimal(repr(grams)) / 100)
    q = (f"How many calories are in {text} of {r.label}?" if c == "energy_kcal"
         else f"How much protein is in {text} of {r.label}?")
    weighs = f"gives {fmt(grams)} g of edible food" if edible else f"weighs {fmt(grams)} g"
    add(q, f"{text[0].upper() + text[1:]} of {r.label} {weighs}, so it has about {res} "
           f"{'kcal' if c == 'energy_kcal' else 'g of protein'} ({v} {UNIT[c]} per 100 g x {f}) "
           f"(USDA FoodData Central).", "fdc_portion", r.item_id)

# 6d. comparisons within a group (same basis)
def compare(pool, n, kind, cols):
    for _ in range(n):
        grp = rng.choice(sorted(pool.project_category.unique()))
        sub = pool[pool.project_category == grp]
        if len(sub) < 2:
            continue
        a_, b_ = [r for _, r in sub.sample(2, random_state=rng.randint(0, 10**6)).iterrows()]
        c = rng.choice(cols)
        if a_.basis != b_.basis or not (has(a_, c) and has(b_, c)) or a_[c] == b_[c]:
            continue
        more = rng.random() < 0.5
        word = "more" if more else ("fewer" if c == "energy_kcal" else "less")
        win, lose = (a_, b_) if (a_[c] > b_[c]) == more else (b_, a_)
        b, src = amount(a_), SRC[a_.item_type]
        q = f"Which has {word} {WORD[c]} per {b}: {a_.label} or {b_.label}?"
        a = (f"{win.label[0].upper() + win.label[1:]}: {fmt(win[c])} {UNIT[c]} per {b}, compared with "
             f"{fmt(lose[c])} {UNIT[c]} for {lose.label} ({src}).")
        add(q, a, kind, f"{a_.item_id};{b_.item_id}")

compare(G, 90, "fdc_compare", ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sodium_mg"])

# 6e. USDA publishes sodium, not salt -> say so (no conversion)
for _, r in G.sample(20, random_state=SEED + 2).iterrows():
    add(f"How much salt is in 100 g of {r.label}?",
        f"USDA FoodData Central reports sodium, not salt: 100 g of {r.label} contains {fmt(r.sodium_mg)} mg of sodium.",
        "salt_vs_sodium", r.item_id)

# 6f. values the source did not report -> "I don't have that information"
for _, r in G.iterrows():
    if isinstance(r.not_reported, str) and r.not_reported:
        for c in r.not_reported.split(";"):
            c = NOT_REPORTED_MAP.get(c, c)
            if c in WORD:
                add(f"How much {WORD[c]} is in 100 g of {r.label}?",
                    f"I don't have that information: USDA FoodData Central does not report the {WORD[c]} content of "
                    f"{r.label}.", "missing_value", r.item_id)
for _, r in Pr.iterrows():
    if not has(r, "fiber_g"):
        add(f"How much fibre is in {amount(r)} of {r.label}?",
            f"I don't have that information: the fibre content of {r.label} is not recorded in Open Food Facts.",
            "missing_value", r.item_id)

# 6g. foods that are not in our data
UNKNOWN = ["dragon fruit", "kunafa", "mansaf", "maqluba", "jackfruit", "durian", "tempeh", "seitan", "mulukhiyah",
           "zaatar manakish", "falafel", "shawarma", "baklava", "qatayef", "jameed", "lupini beans", "camel milk",
           "halloumi", "freekeh", "teff", "acai bowl", "kombucha", "matcha latte", "sushi", "ramen noodles"]
lower_names = " ".join(items.label.str.lower())
UNKNOWN = [u for u in UNKNOWN if u not in lower_names]
PLURAL = {"lupini beans", "ramen noodles"}
for u in UNKNOWN:
    q = rng.choice(["How many calories are in 100 g of {f}?", "How much protein is in 100 g of {f}?",
                    "How much sugar is in 100 g of {f}?", "What is the fat content of {f} per 100 g?"]).format(f=u)
    add(q, f"I don't have that information: {u} {'are' if u in PLURAL else 'is'} not in my nutrition data, "
           f"so I can't give a reliable value.", "unknown_food", "none")

# 6h. out-of-scope requests -> decline (topics differ from the 3 test refusals)
REFUSE_Q = [
    "What should I eat to control my diabetes?", "Is it safe to eat dates if I have diabetes?",
    "Which foods will lower my blood pressure?", "How much salt can I have with high blood pressure?",
    "What diet should I follow for gout?", "Can you give me a keto meal plan for this week?",
    "Make me a meal plan to gain 10 kg of muscle.", "What should I eat while pregnant?",
    "How much vitamin D should I take every day?", "What dose of omega-3 capsules should I take?",
    "How many grams of creatine should I take before training?", "Which foods cure IBS?",
    "Should I stop eating bread because I feel bloated?", "Can these foods replace my cholesterol medication?",
    "How many calories should I eat per day?", "Is intermittent fasting good for me?",
    "My child has a peanut allergy, what should I feed him?", "Can I eat yogurt while taking antibiotics?",
    "How much magnesium should I take for my cramps?", "What is the best diet for my heart disease?",
    "Is this cereal healthy enough to eat every day?", "Which foods help with depression?",
    "Can you design a diet for my thyroid problem?", "How much sugar is safe for my prediabetes?",
    "Should I go vegan to fix my acne?", "How much zinc should I take when I have a cold?",
    "What should an athlete eat before a marathon?", "Which supplements should I take to build muscle?",
    "Can you tell me if my weight is healthy?", "Is it OK to skip breakfast to lose weight?",
]
REFUSE_A = ["I can't help with that: I only give factual nutrition information about specific foods, not medical or "
            "diet advice. Please ask a doctor or a registered dietitian.",
            "I can't help with that: advice on diets, health conditions or supplement doses is outside what I do. "
            "I can tell you the nutrition facts of specific foods instead; for advice, please ask a doctor or dietitian."]
REFUSE_WEIGHT = ("I can't help with that: I can't judge body weight or give weight-loss advice. I can tell you the "
                 "nutrition facts of specific foods; for advice about weight, please ask a doctor or dietitian.")
for q in REFUSE_Q:
    add(q, REFUSE_WEIGHT if re.search(r"\b(weight|lose|kg)\b", q, re.I) else rng.choice(REFUSE_A), "refusal", "none")

# 6i. Open Food Facts lookups: one nutrient per product (only values that are present)
for _, r in Pr.iterrows():
    lookup(r, rng.choice([c for c in ["energy_kcal", "fat_g", "saturated_fat_g", "carbohydrate_g", "sugars_g",
                                      "fiber_g", "protein_g", "salt_g"] if has(r, c)]), "off_lookup")

# 6j. serving size and amount per serving
# the printed serving text is free text typed by contributors (e.g. "1auffin (57g)"), so answers use the
# numeric serving_quantity and the unit of the item's basis
for _, r in Pr.sample(frac=0.25, random_state=SEED).iterrows():
    if has(r, "serving_quantity") and float(r.serving_quantity) > 0:
        unit = amount(r).split()[1]
        add(f"What is the serving size of {r.label}?",
            f"One serving of {r.label} is {fmt(r.serving_quantity)} {unit} (Open Food Facts).", "off_serving", r.item_id)
for _, r in Pr.sample(frac=0.3, random_state=SEED + 3).iterrows():
    if not has(r, "serving_quantity") or float(r.serving_quantity) <= 0:
        continue
    s, unit, b = float(r.serving_quantity), amount(r).split()[1], amount(r)
    c = rng.choice(["energy_kcal", "sugars_g", "protein_g", "salt_g"])
    if not has(r, c):
        continue
    v, f, res = calc(r[c], Decimal(fmt(s).replace(",", "")) / 100)
    what = "kcal" if c == "energy_kcal" else f"{UNIT[c]} of {WORD[c]}"
    q = (f"How many calories are in one serving of {r.label}?" if c == "energy_kcal"
         else f"How much {WORD[c]} is in one serving of {r.label}?")
    add(q, f"About {res} {what}: one serving is {fmt(s)} {unit}, and it has {v} {UNIT[c]} per {b}, "
           f"so {v} x {f} = {res} {UNIT[c]} (Open Food Facts).", "off_per_serving", r.item_id)

# 6k. allergens: "Yes" if EITHER the allergen tags OR the ingredient list shows the allergen (tags can be
#     incomplete); "No" only when neither does. The matching ingredient is quoted exactly as written.
ALLERGENS = {  # question word: (Open Food Facts tag, ingredient words)
    "milk": ("en:milk", r"\b(milk|whey|cream|cheese|butter|lactose|yogh?urt|casein|caseinate|buttermilk)\b"),
    "nuts": ("en:nuts", r"\b(almonds?|hazelnuts?|walnuts?|cashews?|pecans?|pistachios?|macadamias?|brazil nuts?)\b"),
    "peanuts": ("en:peanuts", r"\b(peanuts?|groundnuts?)\b"),
    "gluten": ("en:gluten", r"\b(wheat|barley|rye|spelt|oats?)\b"),
    "soy": ("en:soybeans", r"\b(soy|soya|soybeans?)\b"),
    "eggs": ("en:eggs", r"\b(eggs?|egg white|egg yolk)\b"),
}
NOT_DAIRY = (r"\b(peanut butter|cocoa butter|shea butter|nut butter|almond butter|coconut milk|coconut cream|"
             r"almond milk|oat milk|soy milk|rice milk|cream of tartar)\b")


def allergen_segments(ingredients, word, part="ingredients"):
    """Items (comma/bracket separated) that mention the allergen, quoted as written (max 3).
    part="ingredients" looks only before any 'may contain' statement; part="warning" only after it."""
    main, warn = split_warning(ingredients)
    text = main if part == "ingredients" else warn
    segs = [x.strip(" .:") for x in re.split(r"[,;()\[\]]|\.\s", text) if x.strip(" .:")]
    hits = []
    for sgm in segs:
        probe = re.sub(NOT_DAIRY, " ", sgm, flags=re.I) if word == "milk" else sgm
        if re.search(ALLERGENS[word][1], probe, re.I) and sgm.lower() not in [h.lower() for h in hits]:
            hits.append(sgm)
    return hits[:3]


def allergen_answer(r, word):
    tag, _ = ALLERGENS[word]
    verb = "are" if word in ("nuts", "peanuts", "eggs") else "is"
    tags = [t.strip() for t in str(r.allergens_tags).split(",")] if has(r, "allergens_tags") else []
    traces = [t.strip() for t in str(r.traces_tags).split(",")] if has(r, "traces_tags") else []
    segs = allergen_segments(r.ingredients_text, word)
    quoted = ", ".join(f'"{x}"' for x in segs)
    warn = allergen_segments(r.ingredients_text, word, part="warning")
    if tag in traces:
        may = f", but it may contain traces of {word}"
    elif warn:
        may = f", but the label warns: \"{warn[0]}\""
    else:
        may = ""
    if tag in tags:
        return (f"Yes: Open Food Facts lists {word} as an allergen in {r.label}"
                + (f", and the ingredients include {quoted}." if segs else "."))
    if segs:
        return (f"Yes: the ingredient list of {r.label} includes {quoted}, which contain{'s' if len(segs) == 1 else ''} "
                f"{word}, even though {word} {verb} not among its allergen tags (Open Food Facts).")
    if tag_list(r.allergens_tags):
        return (f"No {word} {verb} listed in the allergens or ingredients of {r.label} "
                f"(allergens listed: {', '.join(tag_list(r.allergens_tags))}){may} (Open Food Facts).")
    return (f"No {word} {verb} listed in the ingredients of {r.label}{may} (Open Food Facts). Allergen information "
            f"is not recorded for this product, so always check the package.")


for _, r in Pr.sample(frac=0.35, random_state=SEED + 4).iterrows():
    word = rng.choice(list(ALLERGENS))
    add(f"Does {r.label} contain {word}?", allergen_answer(r, word), "off_allergen", r.item_id)

compare(Pr, 120, "off_compare", ["energy_kcal", "sugars_g", "protein_g", "salt_g", "fat_g"])
print(pd.Series([p["type"] for p in pairs]).value_counts().to_string())

# %% [markdown]
# ## 7. Checks, then keep about 900 pairs
#
# - Duplicate questions are removed.
# - Questions more than 0.90 similar to a test question are dropped.
# - The script stops if any test item appears in an SFT pair.
# - All "I don't know" and refusal pairs are kept. The other types are sampled to fixed quotas, so the final mix
#   is balanced and documented.

# %%
df = pd.DataFrame(pairs).drop_duplicates("question").reset_index(drop=True)

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()

tq = [norm(q) for q in tests.question]
df["max_sim_to_test"] = [max(difflib.SequenceMatcher(None, norm(q), t).ratio() for t in tq) for q in df.question]
too_close = df[df.max_sim_to_test >= 0.90]
df = df[df.max_sim_to_test < 0.90]
for t in test_item_ids:
    assert not df.source.str.split(";").map(lambda s: t in s).any(), f"test item {t} found in SFT pairs"

KEEP_ALL = ["refusal", "missing_value", "unknown_food", "salt_vs_sodium"]
QUOTA = {"fdc_lookup": 165, "fdc_scale": 85, "fdc_portion": 70, "fdc_compare": 55,
         "off_lookup": 160, "off_serving": 40, "off_per_serving": 70, "off_allergen": 110, "off_compare": 60}
keep = df[df.type.isin(KEEP_ALL)]
room = TARGET_PAIRS - len(keep)
scale = min(1.0, room / sum(min(QUOTA[t], int((df.type == t).sum())) for t in QUOTA))
parts = [keep] + [df[df.type == t].sample(min(int((df.type == t).sum()), int(q * scale)), random_state=SEED)
                  for t, q in QUOTA.items()]
final = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
final.insert(0, "id", range(1, len(final) + 1))
with open(P["sft_pairs"], "w") as f:
    for rec in final.drop(columns=["max_sim_to_test"]).to_dict("records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"{len(final)} pairs written; dropped {len(too_close)} too similar to a test question")
print(final.type.value_counts().to_string())

# %% [markdown]
# ## 8. Hand-check sample and statistics
#
# `results/sft_handcheck_sample.csv` has 50 random pairs with empty `check_ok` and `notes` columns. Read each pair
# against the merged table, mark it 1 or 0, and note anything you fixed. The report must say how many pairs you
# checked and what you changed.

# %%
os.makedirs(P["results_dir"], exist_ok=True)
sample = final.sample(50, random_state=SEED)[["id", "type", "question", "answer", "source"]]
sample.assign(check_ok="", notes="").to_csv(P["sft_handcheck"], index=False)

stats = {
    "input_items": len(items), "generic_foods": len(generic), "packaged_products": len(products),
    "cpt_documents": len(docs), "cpt_train_documents": int((docs.split == "train").sum()),
    "cpt_heldout_documents": int((docs.split == "heldout").sum()),
    "cpt_train_chars": os.path.getsize(P["cpt_train"]), "cpt_heldout_chars": os.path.getsize(P["cpt_heldout"]),
    "test_items": test_item_ids, "test_generic_food_names_excluded": sorted(test_generic_names),
    "rows_excluded_from_sft": int(excluded.sum()),
    "sft_pairs": len(final), "sft_pairs_by_type": final.type.value_counts().to_dict(),
    "sft_dropped_too_similar_to_test": len(too_close),
    "sft_max_similarity_to_any_test_question": round(float(final.max_sim_to_test.max()), 3),
}
json.dump(stats, open(os.path.join(P["results_dir"], "data_stats.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in stats.items() if k not in ("test_items",)}, indent=2))

# %%
# a few pairs of each type, for a quick look
for t, grp in final.groupby("type"):
    r = grp.iloc[0]
    print(f"[{t}]\n  Q: {r.question}\n  A: {r.answer}\n")
