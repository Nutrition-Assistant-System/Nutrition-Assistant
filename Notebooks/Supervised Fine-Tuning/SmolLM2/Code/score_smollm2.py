"""Score all three models with ONE function and write SCOREBOARD.md (Phase 2 brief).

    python score.py --base <base_project> --cpt <cpt best> --sft <sft best> --out <folder>
    python score.py ... --reuse-answers      # after hand grading: re-read the saved answers, only update the scores

For every model: the same prompt, greedy decoding with the same maximum answer length (generate_answer), the same
scoring rules (src/scoring.py) and the same held-out perplexity function. One results file per model:
results_base.csv, results_cpt.csv, results_sft.csv. Hand-graded questions stay pending until both graders and the
final score are filled in manual_grades/<model>.csv; accuracy is reported only when nothing is pending.
"""
import os, sys, io, re, json, time, argparse, datetime, platform, hashlib
import torch, transformers, yaml, requests, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src import prompting_tokens as pt, scoring as sc

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
ap.add_argument("--base", required=True); ap.add_argument("--cpt", required=True); ap.add_argument("--sft", required=True)
ap.add_argument("--out", required=True, help="folder for results_*.csv, metrics, manual_grades/, SCOREBOARD.md")
ap.add_argument("--reuse-answers", action="store_true")
A = ap.parse_args()
cfg = yaml.safe_load(open(A.config))
MODELS = [("base", "Small model, base", A.base), ("cpt", "+ continued pretraining", A.cpt), ("sft", "+ SFT", A.sft)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(f"{A.out}/manual_grades", exist_ok=True)


def get(rel):
    r = requests.get(f"{cfg['data_sources']['base_url']}/{rel}", timeout=60)
    r.raise_for_status()
    return r.content
tests = pd.read_csv(io.BytesIO(get(cfg["data_sources"]["test_questions"])), dtype=str).fillna("")
fp = sc.test_fingerprint(tests)
if fp != cfg["test_set"]["fingerprint"]["value"]:
    raise SystemExit("STOP: the test set has changed (fingerprint mismatch).")
held = get(cfg["data_sources"]["cpt_heldout"]).decode("utf-8")
BLOCK = cfg["stages"]["cpt"]["block_size"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def answer_all(path):
    """THE scoring function for every model: load, perplexity, 30 greedy answers with timing."""
    tok, model = pt.load_project_checkpoint(path, cfg, device=DEVICE)
    ppl = pt.heldout_perplexity(tok, model, held, cfg, block_size=BLOCK)
    pt.generate_answer(tok, model, "How much protein is in 100 g of rice?", cfg)          # warm-up, not timed
    rows = []
    for r in tests.to_dict("records"):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        ans, meta = pt.generate_answer(tok, model, r["question"], cfg)
        if DEVICE == "cuda": torch.cuda.synchronize()
        rows.append({**r, "model_answer": ans, "new_tokens": meta["new_tokens"],
                     "stopped_at_stop_token": meta["stopped_at_stop_token"], "latency_s": round(time.perf_counter() - t0, 4)})
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return pd.DataFrame(rows), ppl


def merge_manual(res, key):
    """Create the grading sheet once; afterwards read the graders' scores from it (never overwritten)."""
    path = f"{A.out}/manual_grades/{key}.csv"
    man = res[res.scoring == "manual"][["id", "question", "expected_answer", "model_answer"]].copy()
    if not os.path.exists(path):
        man.assign(grader_1="", grader_2="", final_score="", notes="").to_csv(path, index=False)
    g = pd.read_csv(path, dtype=str).fillna("")
    agree = []
    for r in g.itertuples():
        mask = res.id == r.id
        if res.loc[mask, "model_answer"].iloc[0].strip() != str(r.model_answer).strip():
            print(f"  WARNING {key} Q{r.id}: the saved answer differs from the graded one -> left pending")
            continue
        if r.grader_1 != "" and r.grader_2 != "":
            agree.append(float(r.grader_1) == float(r.grader_2))
            if r.final_score != "":
                res.loc[mask, "final_score"] = float(r.final_score)
                res.loc[mask, "score_source"] = "human"
    return res, (round(100 * sum(agree) / len(agree), 1) if agree else None), len(agree)


summary = {}
for key, label, path in MODELS:
    print(f"== {label}: {path}")
    out_csv = f"{A.out}/results_{key}.csv"
    if A.reuse_answers and os.path.exists(out_csv):
        res = pd.read_csv(out_csv, dtype={"id": str}).fillna({"model_answer": ""})
        ppl = json.load(open(f"{A.out}/metrics_{key}.json"))["heldout_perplexity"]
    else:
        res, ppl = answer_all(path)
    scored = [sc.auto_score(r, r["model_answer"], cfg) for r in res.to_dict("records")]
    res["auto_score"] = [s for s, _, _ in scored]
    res["auto_reason"] = [w for _, w, _ in scored]
    res["score_source"] = [s for _, _, s in scored]
    res["final_score"] = res["auto_score"]
    res, agreement, n_both = merge_manual(res, key)
    m = sc.summarise(res)
    m.update({"version": label, "checkpoint": path, "model_sha256": sha(f"{path}/model.safetensors"),
              "heldout_perplexity": round(ppl, 3), "grader_agreement_pct": agreement, "n_graded_by_both": n_both,
              "answers_stopped_at_stop_token": int(res.stopped_at_stop_token.astype(str).str.lower().eq("true").sum()),
              "mean_new_tokens": round(float(res.new_tokens.mean()), 1)})
    res.to_csv(out_csv, index=False)
    json.dump(m, open(f"{A.out}/metrics_{key}.json", "w"), indent=2)
    summary[key] = m
    print(f"   perplexity {m['heldout_perplexity']} | auto {m['accuracy_provisional_auto_only']}% | "
          f"refusal {m['refusal_accuracy']}% | latency {m['mean_latency_s']} s | pending {m['n_pending_manual']}")

# ---- error analysis sheet for the SFT model (suggested category; the team confirms it) -------------------------------
sft = pd.read_csv(f"{A.out}/results_sft.csv", dtype={"id": str}).fillna({"model_answer": ""})
def suggest(r):
    if str(r.stopped_at_stop_token).lower() != "true":
        return "format_failure"
    if r.scoring == "decline" or sc.declined(r.model_answer, cfg):
        return "refusal_error"
    return "hallucination" if re.search(r"\d", r.model_answer) else "knowledge_missing"
wrong = sft[(sft.final_score.fillna(1) < 1)].copy()
ea_path = f"{A.out}/error_analysis_sft.csv"
ea = wrong.assign(suggested_category=[suggest(r) for r in wrong.itertuples()], final_category="", notes="")[
    ["id", "scoring", "question", "expected_answer", "model_answer", "final_score", "suggested_category", "final_category", "notes"]]
if os.path.exists(ea_path):                       # keep the team's categories and notes
    old = pd.read_csv(ea_path, dtype=str).fillna("").set_index("id")
    for c in ["final_category", "notes"]:
        ea[c] = ea.id.map(old[c]).fillna("")
ea.to_csv(ea_path, index=False)
print(f"error analysis sheet: {len(ea)} SFT answers below full score -> {ea_path}")

# ---- SCOREBOARD.md -------------------------------------------------------------------------------------------------------
f = lambda v, s="": "pending" if v is None else f"{v}{s}"
lines = ["# Scoreboard", "",
         f"30 test questions (fingerprint {fp[:12]}...), greedy decoding, at most {cfg['generation']['max_new_tokens']} new tokens, "
         f"same prompt and scoring for every model. Generated {datetime.date.today().isoformat()} on "
         f"{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}.", "",
         "| " + " | ".join(cfg["scoreboard"]["columns"]) + " |", "|" + "---|" * len(cfg["scoreboard"]["columns"])]
for key, label, _ in MODELS:
    m = summary[key]
    acc = f"{m['accuracy']}%" if m["accuracy"] is not None else \
        f"pending ({m['n_pending_manual']} hand grades); auto-only {m['accuracy_provisional_auto_only']}% of {m['n_auto_scored']}"
    note = {"base": "SmolLM2-135M + 3 project tokens", "cpt": "best CPT pass, from base_project", "sft": "best SFT pass, from CPT best"}[key]
    if m["grader_agreement_pct"] is not None:
        note += f"; graders agreed on {m['grader_agreement_pct']}% ({m['n_graded_by_both']})"
    lines.append(f"| {label} | {m['heldout_perplexity']} | {acc} | {f(m['refusal_accuracy'], '%')} | {m['mean_latency_s']} | {note} |")
lines += ["", "- Accuracy: mean final score over all 30 questions, shown only when no hand grade is pending; until then the "
          "auto-scored part (22 questions) is shown separately and labelled.",
          "- Refusal accuracy: share of the 4 should-decline questions the model declined.",
          "- Held-out perplexity: 99 held-out documents, 512-token blocks, original 50,257-token vocabulary."]
open(f"{A.out}/SCOREBOARD.md", "w").write("\n".join(lines) + "\n")
json.dump({"test_fingerprint": fp, "date": datetime.datetime.now().isoformat(), "device": DEVICE,
           "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()},
          open(f"{A.out}/score_manifest.json", "w"), indent=2)
print("\n".join(lines))
