"""Scoring rules shared by all three Phase 2 notebooks and by score.py (approved points B and C).

Every model is scored by the SAME functions:
  * key_match (18): 1 if a key value from `key_answer` appears in the answer after normalising
                    (lowercase, collapse spaces, remove thousands separators); declining scores 0.
  * decline   (4):  1 if the answer declines (patterns in config.yaml), else 0.
  * manual    (8):  graded by two team members; PENDING until both have graded (never counted as 0).
"""
import hashlib, json, re
import pandas as pd


def test_fingerprint(df):
    """sha256 of the compact JSON list of [id, question, expected_answer] (config.yaml: test_set.fingerprint)."""
    rows = [[str(r.id), str(r.question), str(r.expected_answer)] for r in df.itertuples()]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def normalise(text):
    t = str(text).lower()
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)     # thousands separator: 1,139 -> 1139
    return re.sub(r"\s+", " ", t).strip()


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", normalise(text)))


def declined(answer, cfg):
    pats = cfg["evaluation"]["should_decline"]["patterns"]
    return any(re.search(p, str(answer), re.I) for p in pats)


def auto_score(row, answer, cfg):
    """Returns (score or None, reason, score_source). None = needs hand grading (pending)."""
    if row["scoring"] == "decline":
        d = declined(answer, cfg)
        return (1.0 if d else 0.0), ("declined" if d else "answered instead of declining"), "auto"
    if row["scoring"] == "key_match":
        if declined(answer, cfg):
            return float(cfg["evaluation"]["short_answers"]["declined_answer_scores"]), "declined a question it should answer", "auto"
        keys = [k.strip() for k in str(row["key_answer"]).split("|") if k.strip()]
        hit = any(numbers(k) and numbers(k) <= numbers(answer) for k in keys)
        return (1.0 if hit else 0.0), ("key value found" if hit else "key value missing"), "auto"
    return None, "pending: needs two hand grades", "human"


def summarise(results):
    """Metrics for one model. `results` has columns scoring, should_decline, auto_score, final_score, latency_s."""
    auto = results[results.scoring != "manual"]
    dec = results[results.should_decline == "yes"]
    pending = int(results.final_score.isna().sum())
    return {
        "n_questions": int(len(results)),
        "n_pending_manual": pending,
        "accuracy": None if pending else round(100 * results.final_score.astype(float).mean(), 1),
        "accuracy_provisional_auto_only": round(100 * auto.auto_score.astype(float).mean(), 1) if len(auto) else None,
        "n_auto_scored": int(len(auto)),
        "refusal_accuracy": round(100 * dec.auto_score.astype(float).mean(), 1) if len(dec) else None,
        "mean_latency_s": round(float(results.latency_s.mean()), 3),
        "by_difficulty_auto": {k: round(100 * g.auto_score.astype(float).mean(), 1)
                               for k, g in auto.groupby("difficulty")},
        "by_scoring_type_auto": {k: round(100 * g.auto_score.astype(float).mean(), 1)
                                 for k, g in auto.groupby("scoring")},
    }
