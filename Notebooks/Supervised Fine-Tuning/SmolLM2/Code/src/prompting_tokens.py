"""Prompting and special-token helpers shared by all three Phase 2 notebooks (Base, CPT, SFT).

Each notebook pastes this cell unchanged, so prompt formatting, tokenization, loss masking, stopping
and perplexity are identical in every stage. Decided in the token review:
  * Approach B: <|user|>, <|assistant|>, <|end|> are added as special tokens ONCE (Base notebook, via
    prepare_project_tokenizer). Every later stage loads with load_project_checkpoint, which checks the
    tokenizer and never adds tokens. Same prompt -> same token IDs in Base, CPT, SFT and scoring.
  * Perplexity is computed over the ORIGINAL vocabulary only (first base_vocab_size logits), so adding
    the 3 tokens cannot change it and Base / CPT / SFT perplexities are directly comparable.
"""
import json, math, os
import torch


def build_prompt(question, cfg):
    """The ONE prompt format used for Base, CPT and SFT, in training and in evaluation."""
    return cfg["prompting"]["template"].format(question=question.strip())


def build_target(answer, cfg):
    """SFT target: a leading space, the answer, then the end marker directly (no space before it,
    which would add an extra "Ġ" token the model must learn to emit before stopping)."""
    return cfg["prompting"]["target"].format(answer=answer.strip())


def _markers(cfg):
    st = cfg["prompting"]["special_tokens"]
    return [st["user"], st["assistant"], st["end"]]


def stop_token_ids(tok, cfg):
    """Generation stops at <|end|> (learned in SFT) or at the tokenizer's EOS (<|endoftext|>)."""
    return sorted({tok.convert_tokens_to_ids(cfg["prompting"]["special_tokens"]["end"]), tok.eos_token_id})


def prepare_project_tokenizer(tok, model, cfg):
    """Base notebook ONLY: add the 3 special tokens, set padding, resize the embeddings, record the IDs."""
    if any(t in tok.get_vocab() for t in _markers(cfg)):
        raise RuntimeError("special tokens already present: load later stages with load_project_checkpoint")
    tok.add_special_tokens({"additional_special_tokens": _markers(cfg)})
    if tok.pad_token is None:                      # GPT-2 has no pad token: pad with EOS (always masked)
        tok.pad_token = tok.eos_token
    model.resize_token_embeddings(len(tok))        # new rows sampled around the mean of existing rows
    return _finish(tok, model, cfg)


def _finish(tok, model, cfg):
    model.generation_config.eos_token_id = stop_token_ids(tok, cfg)
    model.generation_config.pad_token_id = tok.pad_token_id
    return {"special_token_ids": {t: tok.convert_tokens_to_ids(t) for t in _markers(cfg)},
            "vocab_size": len(tok), "base_vocab_size": cfg["prompting"]["project_tokenizer"]["base_vocab_size"],
            "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
            "stop_token_ids": stop_token_ids(tok, cfg)}


def check_project_tokenizer(tok, model, cfg):
    """Fail loudly if a tokenizer/model pair is not the project tokenizer from config.yaml."""
    exp = cfg["prompting"]["project_tokenizer"]
    problems = []
    for t, i in exp["special_token_ids"].items():
        enc = tok.encode(t, add_special_tokens=False)
        if enc != [i]:
            problems.append(f"{t} encodes to {enc}, expected [{i}]")
    if len(tok) != exp["vocab_size"]:
        problems.append(f"tokenizer size {len(tok)}, expected {exp['vocab_size']}")
    if model.get_input_embeddings().weight.shape[0] != len(tok):
        problems.append(f"embedding rows {model.get_input_embeddings().weight.shape[0]} != tokenizer size {len(tok)}")
    if tok.pad_token_id is None:
        problems.append("no pad token")
    if exp.get("tied_lm_head") and model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        problems.append("input and output embeddings are no longer tied")
    if problems:
        raise RuntimeError("Project tokenizer check failed:\n  - " + "\n  - ".join(problems))
    return True


def load_project_checkpoint(path, cfg, device=None):
    """CPT, SFT and scoring: load a saved checkpoint and its tokenizer; check; never add tokens."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    check_project_tokenizer(tok, model, cfg)
    _finish(tok, model, cfg)
    if device:
        model.to(device)
    return tok, model.eval()


def cpt_collate(blocks):
    """CPT batches: every block is exactly block_size tokens, so there is no padding and every token,
    including the <|endoftext|> between documents, is a training label. (DataCollatorForLanguageModeling
    must NOT be used: with pad = EOS it would turn every document-boundary EOS label into -100.)"""
    ids = torch.tensor([b["input_ids"] for b in blocks])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def sft_collate(batch, pad_id):
    """SFT batches: right-pad input_ids with pad_id (attention 0) and labels with -100 (no loss)."""
    m = max(len(x["input_ids"]) for x in batch)
    pad = lambda x, v: x + [v] * (m - len(x))
    return {"input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in batch]),
            "attention_mask": torch.tensor([pad([1] * len(x["input_ids"]), 0) for x in batch]),
            "labels": torch.tensor([pad(x["labels"], -100) for x in batch])}


def encode_sft_example(tok, question, answer, cfg, max_length):
    """Prompt and target are tokenized separately (as at inference, where only the prompt is given),
    then joined. Labels are -100 on prompt tokens: the loss is on the answer and <|end|> only."""
    p_ids = tok(build_prompt(question, cfg), add_special_tokens=False)["input_ids"]
    a_ids = tok(build_target(answer, cfg), add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_length:
        raise ValueError(f"example longer than max_length={max_length}: the <|end|> marker would be cut off")
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


@torch.no_grad()
def heldout_perplexity(tok, model, text, cfg, block_size=512):
    """Held-out perplexity over the ORIGINAL vocabulary: logits are cut to the first base_vocab_size
    entries before the softmax. The held-out text never contains the 3 new tokens, so this equals the
    perplexity of the unmodified model and is comparable across Base, CPT and SFT."""
    V = cfg["prompting"]["project_tokenizer"]["base_vocab_size"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if any(i >= V for i in ids):
        raise ValueError("held-out text contains project special tokens")
    ids = torch.tensor(ids)
    was_training = model.training
    model.eval()                                   # dropout off: perplexity must be deterministic
    nll, count = 0.0, 0
    for i in range(0, len(ids) - 1, block_size):
        chunk = ids[i:i + block_size].unsqueeze(0).to(model.device)
        if chunk.shape[1] < 2:
            continue
        logits = model(chunk).logits[0, :-1, :V].float()
        nll += torch.nn.functional.cross_entropy(logits, chunk[0, 1:], reduction="sum").item()
        count += chunk.shape[1] - 1
    model.train(was_training)
    return math.exp(nll / count)


def clean_answer(text, cfg):
    """Cut a decoded answer at the first stop string (end marker, a new turn, or EOS text)."""
    for s in cfg["prompting"]["stop_strings"]:
        i = text.find(s)
        if i != -1:
            text = text[:i]
    return text.strip()


@torch.no_grad()
def generate_answer(tok, model, question, cfg):
    """The ONE function every stage and score.py uses to query a model (greedy, same length)."""
    g = cfg["generation"]
    stops = stop_token_ids(tok, cfg)
    enc = tok(build_prompt(question, cfg), return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=g["max_new_tokens"], do_sample=False, num_beams=1,
                         repetition_penalty=g.get("repetition_penalty", 1.0),
                         eos_token_id=stops, pad_token_id=tok.pad_token_id)
    new = out[0][enc["input_ids"].shape[1]:]
    stopped = bool(len(new)) and int(new[-1]) in stops
    text = tok.decode(new, skip_special_tokens=False)
    return clean_answer(text, cfg), {"new_tokens": int(len(new)), "stopped_at_stop_token": stopped}


def save_with_tokenizer(model, tok, out_dir, cfg):
    """Save model + tokenizer + recorded token IDs together. Use for EVERY saved checkpoint (each CPT
    pass, best CPT, best SFT) so any folder can be loaded with load_project_checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    json.dump(_finish(tok, model, cfg), open(os.path.join(out_dir, "project_tokens.json"), "w"), indent=2)
