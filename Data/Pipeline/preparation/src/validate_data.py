"""Validate EVERY SFT pair and EVERY CPT document against the merged source table.

Usage: python -m src.validate_data            (after src.prepare_data)

Severity
  CRITICAL : a fact or number that does not match the source, wrong arithmetic, wrong allergen yes/no,
             wrong basis, a test item leaking into training data, or a test-like question.
  WARNING  : wording/format artefacts (do not change facts).
Writes results/validation_report.json and prints a summary. Exit code 1 if any CRITICAL issue is found.
"""
import os, re, sys, json, difflib
import importlib.util
from decimal import Decimal
import pandas as pd

from src.config import load_config, ROOT

cfg = load_config()
P = cfg["paths"]

# re-use the generator's own helpers so validation and generation share one definition of names/rounding
spec = importlib.util.spec_from_file_location("prep_helpers", os.path.join(ROOT, "src", "prepare_data.py"))
src = open(os.path.join(ROOT, "src", "prepare_data.py")).read()
helpers_code = src[src.index("NUTR = ["):src.index("items[\"label\"] = items.apply(label, axis=1)")]
H = {"pd": pd, "re": re, "Decimal": Decimal}
exec("from decimal import Decimal, ROUND_HALF_UP\n" + helpers_code, H)
fmt, dec2, label, measures, tag_list, clean_ingredients = (H[k] for k in
                                                          ["fmt", "dec2", "label", "measures", "tag_list", "clean_ingredients"])
WORD, UNIT = H["WORD"], H["UNIT"]

items = pd.read_csv(P["merged_items"], dtype={"source_id": str})
items["label"] = items.apply(label, axis=1)
I = items.set_index("item_id")
tests = pd.read_csv(P["test_questions"], dtype=str).fillna("")
test_ids = {f"fdc:{x}" for s in tests.source for x in re.findall(r"fdc_id (\d+)", s)} | \
           {f"off:{x}" for s in tests.source for x in re.findall(r"barcode (\d+)", s)}
test_names = set(I.loc[[t for t in test_ids if t.startswith("fdc:")], "name"])
pairs = pd.read_json(P["sft_pairs"], lines=True)
W2C = {"calories": "energy_kcal", "kcal": "energy_kcal", "protein": "protein_g", "fat": "fat_g",
       "saturated fat": "saturated_fat_g", "carbohydrate": "carbohydrate_g", "sugar": "sugars_g", "fibre": "fiber_g",
       "sodium": "sodium_mg", "salt": "salt_g", "cholesterol": "cholesterol_mg", "calcium": "calcium_mg",
       "iron": "iron_mg", "potassium": "potassium_mg"}
NUM = r"([\d,]+(?:\.\d+)?)"


def q_nutrient(q):
    for p in [r"^How many (calories|kcal)", r"^What is the (energy) value", r"^How much ([a-z ]+?) (?:is|does)\b",
              r"^What is the ([a-z ]+?) content of", r"^Tell me the amount of ([a-z ]+?) in",
              r"^Which has (?:more|less|fewer) ([a-z ]+?) per 100"]:
        m = re.search(p, q)
        if m:
            return "energy_kcal" if m.group(1) in ("calories", "kcal", "energy") else W2C[m.group(1)]
    return None


def check_calc(shown_v, src_v, factor_s, exp_factor, res_s, head_s=None):
    errs = []
    if shown_v != fmt(src_v):
        errs.append(f"shown value {shown_v} != source {fmt(src_v)}")
    if Decimal(factor_s) != Decimal(str(exp_factor)):
        errs.append(f"factor {factor_s} != expected {exp_factor}")
    exp = fmt(dec2(Decimal(shown_v.replace(",", "")) * Decimal(factor_s)))
    if res_s != exp:
        errs.append(f"{shown_v} x {factor_s} = {exp}, answer says {res_s}")
    if head_s is not None and head_s != res_s:
        errs.append(f"headline {head_s} != result {res_s}")
    return errs


issues = []  # (scope, id, type, severity, message)
def bad(scope, i, t, sev, msg):
    issues.append({"scope": scope, "id": i, "type": t, "severity": sev, "message": msg})


norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
tq = [norm(q) for q in tests.question]
for p in pairs.itertuples():
    q, a, t, i = p.question, p.answer, p.type, p.id
    srcs = [s for s in str(p.source).split(";") if s and s != "none"]
    # separation
    for s_ in srcs:
        if s_ not in I.index:
            bad("sft", i, t, "CRITICAL", f"source {s_} not in merged table"); continue
        if s_ in test_ids or (s_.startswith("fdc:") and I.loc[s_, "name"] in test_names):
            bad("sft", i, t, "CRITICAL", f"test item {s_} used in training pair")
    if max(difflib.SequenceMatcher(None, norm(q), x).ratio() for x in tq) >= 0.90:
        bad("sft", i, t, "CRITICAL", "question >= 0.90 similar to a test question")
    # wording artefacts
    if re.search(r"[.,;:]\?|\s{2,}|\(\(|\(\s|\s\)|\bnan\b|\.\.", q + " " + a):
        bad("sft", i, t, "WARNING", "punctuation/spacing artefact")
    if any(x in I.index for x in srcs):
        it = I.loc[srcs[0]]
        if not it.label.lower() in q.lower() and t not in ("refusal", "unknown_food"):
            bad("sft", i, t, "WARNING", "item name in question differs from the shared label")
    try:
        if t in ("fdc_lookup", "off_lookup"):
            c, b = q_nutrient(q), it.basis.replace("per ", "")
            m = re.search(rf"contains {NUM} (kcal|g|mg)", a)
            if pd.isna(it[c]):
                bad("sft", i, t, "CRITICAL", f"{c} blank in source but a value is given")
            elif m.group(1) != fmt(it[c]) or m.group(2) != UNIT[c]:
                bad("sft", i, t, "CRITICAL", f"{m.group(1)} {m.group(2)} != source {fmt(it[c])} {UNIT[c]}")
            if b not in q or b not in a:
                bad("sft", i, t, "CRITICAL", f"basis mismatch, item is {it.basis}")
        elif t == "fdc_scale":
            g = int(re.search(r"(\d+) g of", q).group(1))
            m = re.search(rf"^About {NUM} .*?: {NUM} \S+ per 100 g x ([\d.]+) = {NUM}", a)
            for e in check_calc(m.group(2), it[q_nutrient(q)], m.group(3), Decimal(g) / 100, m.group(4), m.group(1)):
                bad("sft", i, t, "CRITICAL", e)
        elif t == "fdc_portion":
            c = q_nutrient(q)
            ms = {mt: (g, y) for mt, g, y in measures(it)}
            m = re.search(rf"^(.+?) of .+ (weighs|gives) {NUM} g.*?about {NUM} .*?\({NUM} \S+ per 100 g x ([\d.]+)\)", a)
            mt, verb, grams, res, shown, fac = m.groups()
            key = mt[0].lower() + mt[1:] if mt[0].lower() + mt[1:] in ms else mt
            if key not in ms or fmt(ms[key][0]) != grams:
                bad("sft", i, t, "CRITICAL", f"measure '{mt}' = {grams} g not in source measures")
            else:
                if (verb == "gives") != ms[key][1]:
                    bad("sft", i, t, "WARNING", "edible-yield wording does not match the measure")
                for e in check_calc(shown, it[c], fac, Decimal(repr(ms[key][0])) / 100, res):
                    bad("sft", i, t, "CRITICAL", e)
        elif t in ("fdc_compare", "off_compare"):
            a1, b1 = I.loc[srcs[0]], I.loc[srcs[1]]
            c = q_nutrient(q)
            word = re.search(r"Which has (more|less|fewer)", q).group(1)
            m = re.search(rf"^(.+?): {NUM} \S+ per 100 (g|ml), compared with {NUM} \S+ for (.+) \(", a)
            win, vw, _, vl, lose = m.groups()
            if a1.basis != b1.basis:
                bad("sft", i, t, "CRITICAL", "items have different bases")
            vals = {a1.label: a1[c], b1.label: b1[c]}
            right = max(vals, key=vals.get) if word == "more" else min(vals, key=vals.get)
            if win.lower() != right.lower():
                bad("sft", i, t, "CRITICAL", f"winner should be {right}")
            if vw != fmt(vals[right]) or vl != fmt([v for k, v in vals.items() if k != right][0]):
                bad("sft", i, t, "CRITICAL", "compared values differ from source")
            if vw == vl:
                bad("sft", i, t, "WARNING", "values equal after rounding")
        elif t == "off_serving":
            unit = it.basis.split()[-1]
            if f"{fmt(it.serving_quantity)} {unit}" not in a:
                bad("sft", i, t, "CRITICAL", f"serving != source {it.serving_quantity} {unit}")
        elif t == "off_per_serving":
            c = q_nutrient(q)
            m = re.search(rf"^About {NUM} .*?one serving is {NUM} (g|ml), and it has {NUM} \S+ per 100 (g|ml), "
                          rf"so {NUM} x ([\d.]+) = {NUM}", a)
            head, s_, u, per, u2, shown, fac, res = m.groups()
            if s_ != fmt(it.serving_quantity) or f"per 100 {u}" != it.basis or u != u2:
                bad("sft", i, t, "CRITICAL", "serving or basis mismatch")
            if per != shown:
                bad("sft", i, t, "CRITICAL", "per-100 value and multiplied value differ")
            for e in check_calc(shown, it[c], fac, Decimal(s_.replace(",", "")) / 100, res, head):
                bad("sft", i, t, "CRITICAL", e)
        elif t == "off_allergen":
            word = re.search(r"contain (\w+)\?$", q).group(1)
            tag = H["ALLERGENS"][word][0] if "ALLERGENS" in H else None
            # independent re-derivation from the source
            pats = {"milk": r"\b(milk|whey|cream|cheese|butter|lactose|yogh?urt|casein|caseinate|buttermilk)\b",
                    "nuts": r"\b(almonds?|hazelnuts?|walnuts?|cashews?|pecans?|pistachios?|macadamias?|brazil nuts?)\b",
                    "peanuts": r"\b(peanuts?|groundnuts?)\b", "gluten": r"\b(wheat|barley|rye|spelt|oats?)\b",
                    "soy": r"\b(soy|soya|soybeans?)\b", "eggs": r"\b(eggs?|egg white|egg yolk)\b"}
            tagname = {"milk": "en:milk", "nuts": "en:nuts", "peanuts": "en:peanuts", "gluten": "en:gluten",
                       "soy": "en:soybeans", "eggs": "en:eggs"}[word]
            ing = H["split_warning"](it.ingredients_text)[0]       # ingredient part only, not 'may contain'
            if word == "milk":
                ing = re.sub(r"\b(peanut butter|cocoa butter|shea butter|nut butter|almond butter|coconut milk|"
                             r"coconut cream|almond milk|oat milk|soy milk|rice milk|cream of tartar)\b", " ", ing, flags=re.I)
            in_ing = bool(re.search(pats[word], ing, re.I))
            tags = [x.strip() for x in str(it.allergens_tags).split(",")] if pd.notna(it.allergens_tags) else []
            traces = [x.strip() for x in str(it.traces_tags).split(",")] if pd.notna(it.traces_tags) else []
            expect_yes = tagname in tags or in_ing
            if a.startswith("Yes") != expect_yes:
                bad("sft", i, t, "CRITICAL", f"answer {'Yes' if a.startswith('Yes') else 'No'}, but tag={tagname in tags} "
                                             f"ingredients={in_ing}")
            main_part, warn_part = H["split_warning"](it.ingredients_text)
            for qt in re.findall(r'"([^"]+)"', a):
                if qt not in clean_ingredients(it.ingredients_text):
                    bad("sft", i, t, "CRITICAL", f'quoted text "{qt}" not in source ingredient list')
                elif a.startswith("Yes") and qt not in main_part:
                    bad("sft", i, t, "CRITICAL", f'"{qt}" is a may-contain warning, used as an ingredient')
            if "traces" in a and tagname not in traces:
                bad("sft", i, t, "CRITICAL", "traces claim not supported")
            if "label warns" in a and not re.search(pats[word], warn_part, re.I):
                bad("sft", i, t, "CRITICAL", "label-warning claim not supported")
            if not a.startswith("Yes") and tagname in traces and "traces" not in a:
                bad("sft", i, t, "CRITICAL", "traces present but not mentioned")
            listed = re.search(r"allergens listed: ([^)]*)\)", a)
            if listed and any(x.strip() not in tag_list(it.allergens_tags) for x in listed.group(1).split(",")):
                bad("sft", i, t, "WARNING", "listed allergens are not standard tags")
        elif t == "salt_vs_sodium":
            if f"contains {fmt(it.sodium_mg)} mg of sodium" not in a or pd.notna(it.salt_g):
                bad("sft", i, t, "CRITICAL", "sodium value or salt claim does not match source")
        elif t == "missing_value":
            if pd.notna(it[q_nutrient(q)]):
                bad("sft", i, t, "CRITICAL", "value exists in source")
        elif t == "unknown_food":
            food = re.search(r"information: (.+?) (?:is|are) not in my", a).group(1)
            # the full food phrase must not name an item ("matcha latte" vs a matcha-flavoured biscuit is fine)
            hit = items["name"].str.lower().str.contains(rf"\b{re.escape(food)}\b") | \
                items.label.str.lower().str.contains(rf"\b{re.escape(food)}\b")
            if hit.any():
                bad("sft", i, t, "CRITICAL", f"'{food}' is in the table")
        elif t == "refusal":
            if not a.startswith("I can't help") or re.search(r"\d", a):
                bad("sft", i, t, "CRITICAL", "does not decline cleanly")
    except Exception as e:  # unparseable answers are themselves a problem
        bad("sft", i, t, "CRITICAL", f"could not verify ({type(e).__name__}: {e})")

# ---------------------------------------------------------------- CPT documents
split = pd.read_csv(P["cpt_split_index"])
texts = {"train": open(P["cpt_train"]).read().split("\n\n"), "heldout": open(P["cpt_heldout"]).read().split("\n\n")}
texts = {k: [x.strip() for x in v if x.strip()] for k, v in texts.items()}
if len(texts["train"]) != (split.split == "train").sum() or len(texts["heldout"]) != (split.split == "heldout").sum():
    bad("cpt", "-", "-", "CRITICAL", "document counts differ from split index")
if (split[split.split == "heldout"].is_test_item).any():
    bad("cpt", "-", "-", "CRITICAL", "test item in held-out split")
NAMES = {"energy": "energy_kcal", **{w: c for w, c in W2C.items() if w not in ("calories", "kcal")}}
for sp in ["train", "heldout"]:
    ids = split[split.split == sp].doc_id.tolist()
    for did, doc in zip(ids, texts[sp]):
        it = I.loc[did]
        m = re.search(r"Nutrition (?:per 100 g \(USDA FoodData Central\)|per 100 (?:g|ml) \(Open Food Facts\)): (.+?)\.(?: |$)", doc)
        if not m:
            bad("cpt", did, sp, "CRITICAL", "nutrition line not found"); continue
        for part in m.group(1).split("; "):
            mm = re.match(rf"{NUM} (kcal|g|mg) ?(.*)$", part)
            c = "energy_kcal" if mm.group(2) == "kcal" else NAMES[mm.group(3)]
            if mm.group(1) != fmt(it[c]):
                bad("cpt", did, sp, "CRITICAL", f"{c}: {mm.group(1)} != source {fmt(it[c])}")
        if it.label not in doc:
            bad("cpt", did, sp, "WARNING", "shared label not in document")
        if it.item_type == "packaged product" and f"({it.basis.replace('per ', 'per ')}" in doc and it.basis not in doc:
            bad("cpt", did, sp, "CRITICAL", "basis missing")
        if re.search(r"\bnan\b|_\w+_|\s{2,}", doc):
            bad("cpt", did, sp, "WARNING", "markup/'nan'/double-space artefact")

rep = pd.DataFrame(issues, columns=["scope", "id", "type", "severity", "message"])
summary = {
    "sft_pairs_checked": len(pairs), "cpt_documents_checked": len(split),
    "critical": int((rep.severity == "CRITICAL").sum()), "warning": int((rep.severity == "WARNING").sum()),
    "critical_by_scope": rep[rep.severity == "CRITICAL"].scope.value_counts().to_dict(),
    "warnings_by_message": rep[rep.severity == "WARNING"].message.value_counts().to_dict(),
    "issues": rep.to_dict("records"),
}
json.dump(summary, open(os.path.join(P["results_dir"], "validation_report.json"), "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in summary.items() if k != "issues"}, indent=2))
for r in rep.head(40).itertuples():
    print(f"  [{r.severity}] {r.scope} {r.id} {r.type}: {r.message}")
sys.exit(1 if summary["critical"] else 0)
