"""Stage 2 (Model 3): supervised fine-tuning (SFT) of the CPT model on the 894 question-answer pairs.

    python train_sft.py --cpt-dir <cpt best> --base-dir <base_project> --out-name sft
    python train_sft.py ... --smoke          # quick trial (5% of the pairs); checkpoints on the Colab disk only

Reads : config.yaml; Data/SFT/sft_pairs.jsonl and the test questions from the project GitHub repo;
        the CPT best checkpoint (must descend from base_project: checked with lineage.json fingerprints)
Writes: <ckpt>/<out-name>/epoch-N, .../best  (model + tokenizer + project_tokens.json + lineage.json)
        <results>/sft_metrics.json, sft_log_history.json, sft_split.csv, sft_masking_example.txt,
        sft_samples.md, manifest.json, config_used.yaml
Settings (all from config.yaml, stages.sft): full fine-tuning, chat template with the 3 special tokens,
loss on answer tokens only, 10% validation, up to 3 passes, lr 5e-5, early stopping, a checkpoint per pass.
"""
import os, sys, re, json, math, time, shutil, hashlib, argparse, inspect, platform, random
import numpy as np
import torch
import transformers
import io, yaml, requests, pandas as pd
from transformers import Trainer, TrainingArguments, TrainerCallback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src import prompting_tokens as pt, scoring as sc

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
ap.add_argument("--cpt-dir", required=True, help="CPT best checkpoint (cpt_from_base_project/best)")
ap.add_argument("--base-dir", required=True, help="official base_project checkpoint (Notebook 1)")
ap.add_argument("--out-name", default="sft", help="new checkpoint folder; must not exist yet")
ap.add_argument("--drive-root", default=None, help="overrides paths.drive_root (outputs)")
ap.add_argument("--smoke", action="store_true")
A = ap.parse_args()

cfg = yaml.safe_load(open(A.config))
S, run, PTK = cfg["stages"]["sft"], cfg["run"], cfg["prompting"]["project_tokenizer"]
key = run["pipeline_model"]
tm = cfg["models"][key]["train"]
seed = run["seed"]
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
root = A.drive_root or cfg["paths"]["drive_root"]
stamp = time.strftime("%Y%m%d-%H%M")
run_name = ("smoke_" if A.smoke else "") + f"sft_{key}_{stamp}"
results_dir = cfg["paths"]["results_dir"].format(root=root, stage="sft", model_key=key, run_name=run_name)
ckpt_root = cfg["paths"]["checkpoints_dir"].format(root=root, model_key=key)
out_dir = (f"/content/sft_smoke_tmp/{run_name}" if os.path.isdir("/content") else os.path.join(ckpt_root, "smoke_tmp", run_name)) \
    if A.smoke else os.path.join(ckpt_root, A.out_name)
if os.path.isdir(out_dir) and os.listdir(out_dir):
    raise SystemExit(f"STOP: {out_dir} already contains files. Existing checkpoints are never overwritten: use another --out-name.")


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- starting point: the CPT best checkpoint, which must descend from the official base_project ----------------
REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "lineage.json"]
for d, req in [(A.base_dir, REQUIRED[:-1]), (A.cpt_dir, REQUIRED)]:
    if not os.path.isdir(d):
        raise SystemExit(f"STOP: {d} not found (check the Drive shortcut / path).")
    miss = [f for f in req if not os.path.isfile(os.path.join(d, f))]
    if miss:
        raise SystemExit(f"STOP: {d} is missing {miss}.")
base_sha = file_sha256(os.path.join(A.base_dir, "model.safetensors"))
cpt_sha = file_sha256(os.path.join(A.cpt_dir, "model.safetensors"))
cpt_lineage = json.load(open(os.path.join(A.cpt_dir, "lineage.json")))
if cpt_lineage.get("start_sha256", {}).get("model.safetensors") != base_sha:
    raise SystemExit("STOP: the CPT checkpoint did not start from this base_project (fingerprint mismatch).")
tok, model = pt.load_project_checkpoint(A.cpt_dir, cfg)
if model.dtype != torch.float32:
    raise SystemExit(f"STOP: CPT checkpoint loaded as {model.dtype}, expected float32.")
lineage = {"start_checkpoint": "cpt_best", "start_path": A.cpt_dir, "start_sha256": {"model.safetensors": cpt_sha},
           "base_project_sha256": base_sha, "cpt_lineage": cpt_lineage}
print(f"start: {A.cpt_dir}\n  CPT sha256 {cpt_sha[:16]}... descends from base_project {base_sha[:16]}...: PASS")

# ---- data: SFT pairs and the frozen test set, from the project repo ------------------------------------------------
def get(rel):
    r = requests.get(f"{cfg['data_sources']['base_url']}/{rel}", timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"STOP: could not download {rel} (HTTP {r.status_code}).")
    return r.content
pairs_raw = get(cfg["data_sources"]["sft_pairs"])
pairs = [json.loads(l) for l in pairs_raw.decode("utf-8").splitlines() if l.strip()]
test_raw = get(cfg["data_sources"]["test_questions"])
tests = pd.read_csv(io.BytesIO(test_raw), dtype=str).fillna("")
if sc.test_fingerprint(tests) != cfg["test_set"]["fingerprint"]["value"]:
    raise SystemExit("STOP: the test set has changed (fingerprint mismatch).")
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
overlap = set(map(norm, tests.question)) & {norm(p["question"]) for p in pairs}
if overlap:
    raise SystemExit(f"STOP: {len(overlap)} SFT questions are identical to test questions.")
if any(m in p["question"] + p["answer"] for p in pairs for m in cfg["prompting"]["special_tokens"].values()):
    raise SystemExit("STOP: SFT data contains a reserved special-token string.")
print(f"SFT pairs: {len(pairs)} | identical to a test question: 0 | test set fingerprint OK")

# ---- 90/10 split, stratified by question type, fixed seed ------------------------------------------------------------
rng = random.Random(seed)
train, val = [], []
for t in sorted({p["type"] for p in pairs}):
    group = [p for p in pairs if p["type"] == t]
    rng.shuffle(group)
    n_val = max(1, round(len(group) * S["val_fraction"]))
    val += group[:n_val]; train += group[n_val:]
if A.smoke:
    frac = run["smoke_test"]["data_fraction"]
    train, val = train[:max(8, int(len(train) * frac))], val[:max(4, int(len(val) * frac))]
os.makedirs(results_dir, exist_ok=True)
pd.DataFrame([{"id": p["id"], "type": p["type"], "split": s} for s, ps in (("train", train), ("val", val)) for p in ps]) \
    .sort_values("id").to_csv(os.path.join(results_dir, "sft_split.csv"), index=False)

enc = lambda ps: [pt.encode_sft_example(tok, p["question"], p["answer"], cfg, S["max_length"]) for p in ps]
train_ds, val_ds = enc(train), enc(val)
lens = [len(e["input_ids"]) for e in train_ds + val_ds]
print(f"train {len(train_ds)} / validation {len(val_ds)} pairs | tokens per example: max {max(lens)}, mean {np.mean(lens):.1f} "
      f"(limit {S['max_length']}, nothing truncated)")

# masking example: which tokens carry the loss
ex = train_ds[0]
with open(os.path.join(results_dir, "sft_masking_example.txt"), "w") as f:
    f.write("token | in loss?\n")
    for i, l in zip(ex["input_ids"], ex["labels"]):
        f.write(f"{tok.convert_ids_to_tokens(i)!r:28s} {'LOSS' if l != -100 else '-'}\n")
assert ex["labels"][-1] == tok.convert_tokens_to_ids(cfg["prompting"]["special_tokens"]["end"]), "<|end|> must be trained"


# ---- trainer ------------------------------------------------------------------------------------------------------------
def training_args(**kw):
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    if "warmup_ratio" in kw and "warmup_ratio" not in accepted:
        kw["warmup_steps"] = float(kw.pop("warmup_ratio"))
    return TrainingArguments(**{k: v for k, v in kw.items() if k in accepted})


class EpochSaver(TrainerCallback):
    """Weights + tokenizer after every pass (no optimizer state); stop when validation loss stops improving."""
    def __init__(self, patience, loss0):
        self.patience, self.bad, self.best, self.best_epoch = patience, 0, float("inf"), None
        self.evals = [{"epoch": 0.0, "eval_loss": loss0}]
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        if state.epoch is None or state.epoch < 0.5:
            return
        ep, loss = int(round(state.epoch)), metrics["eval_loss"]
        self.evals.append({"epoch": float(ep), "eval_loss": loss})
        d = os.path.join(out_dir, f"epoch-{ep}")
        pt.save_with_tokenizer(model, tok, d, cfg)
        json.dump(lineage, open(os.path.join(d, "lineage.json"), "w"), indent=2)
        if loss < self.best:
            self.best, self.best_epoch, self.bad = loss, ep, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print(f"validation loss did not improve for {self.bad} pass(es): stopping after pass {ep}")
                control.should_training_stop = True


os.makedirs(out_dir, exist_ok=True)
targs = training_args(
    output_dir=os.path.join(out_dir, "_trainer"), seed=seed, data_seed=seed,
    num_train_epochs=S["epochs"], learning_rate=S["learning_rate"],
    per_device_train_batch_size=tm["sft_batch_size"], per_device_eval_batch_size=tm["sft_batch_size"],
    gradient_accumulation_steps=tm["grad_accum"], gradient_checkpointing=tm["gradient_checkpointing"],
    warmup_ratio=S["warmup_ratio"], weight_decay=S["weight_decay"],
    eval_strategy="epoch", save_strategy="no", logging_steps=5, prediction_loss_only=True,
    fp16=torch.cuda.is_available() and cfg["runtime"]["training_precision"] == "fp16_mixed",
    remove_unused_columns=False, report_to="none")
collate = lambda b: pt.sft_collate(b, tok.pad_token_id)
trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
loss0 = trainer.evaluate()["eval_loss"]                       # validation loss BEFORE SFT (answer tokens only)
print(f"validation loss before SFT: {loss0:.4f}")
saver = EpochSaver(S["early_stopping_patience"], loss0)
trainer.add_callback(saver)
model.train()
trainer.train()
shutil.rmtree(os.path.join(out_dir, "_trainer"), ignore_errors=True)

best_dir = os.path.join(out_dir, "best")
shutil.copytree(os.path.join(out_dir, f"epoch-{saver.best_epoch}"), best_dir)
tok, model = pt.load_project_checkpoint(best_dir, cfg, device="cuda" if torch.cuda.is_available() else None)
held = get(cfg["data_sources"]["cpt_heldout"]).decode("utf-8")
ppl = pt.heldout_perplexity(tok, model, held, cfg, block_size=cfg["stages"]["cpt"]["block_size"])
print(f"best pass {saver.best_epoch} | held-out perplexity of the SFT model: {ppl:.3f}")

# answers to 8 validation questions (format check: does it stop at <|end|>?)
samples = []
for p in val[:8]:
    a, meta = pt.generate_answer(tok, model, p["question"], cfg)
    samples.append((p, a, meta))
with open(os.path.join(results_dir, "sft_samples.md"), "w") as f:
    f.write("# SFT model on 8 validation questions (greedy, same settings as the test)\n\n")
    for p, a, meta in samples:
        f.write(f"**Q:** {p['question']}\n\n- **Reference:** {p['answer']}\n- **Model:** {a}\n"
                f"- stopped at <|end|>: {meta['stopped_at_stop_token']} ({meta['new_tokens']} tokens)\n\n")

hist = trainer.state.log_history
metrics = {
    "model_key": key, "run_name": run_name, "smoke_test": A.smoke, "checkpoint_dir": out_dir,
    "n_train": len(train_ds), "n_val": len(val_ds), "max_tokens_per_example": max(lens),
    "learning_rate": S["learning_rate"], "batch_size": tm["sft_batch_size"], "epochs_planned": S["epochs"],
    "epochs_run": int(round(trainer.state.epoch)), "best_epoch": saver.best_epoch,
    "stopped_early": int(round(trainer.state.epoch)) < S["epochs"],
    "eval_by_epoch": saver.evals, "heldout_perplexity": ppl,
    "val_samples_stopped_at_end": sum(m["stopped_at_stop_token"] for _, _, m in samples), "val_samples": len(samples),
}
manifest = {**lineage, "date": stamp, "seed": seed, "config_version": cfg["project"]["config_version"],
            "sft_pairs_sha256": hashlib.sha256(pairs_raw).hexdigest(), "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
json.dump(metrics, open(os.path.join(results_dir, "sft_metrics.json"), "w"), indent=2)
json.dump(manifest, open(os.path.join(results_dir, "manifest.json"), "w"), indent=2)
json.dump(hist, open(os.path.join(results_dir, "sft_log_history.json"), "w"), indent=2)
shutil.copy(A.config, os.path.join(results_dir, "config_used.yaml"))
print(json.dumps({k: metrics[k] for k in ("best_epoch", "epochs_run", "stopped_early", "eval_by_epoch",
                                          "heldout_perplexity", "val_samples_stopped_at_end")}, indent=2))
print("RESULTS_DIR=" + results_dir)
