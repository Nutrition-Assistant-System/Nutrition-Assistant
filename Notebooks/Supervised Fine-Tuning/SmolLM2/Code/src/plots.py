"""Report figures shared by Notebooks 1-3 (Base, CPT, SFT): one style, one colour per model, one question order.

Figures are drawn ONLY from the files each notebook saved (answers.csv, metrics.json, manual_grading_sheet.csv),
so they show exactly what was scored. Hand-graded questions stay 'pending' until both graders and a final score
are filled in; they are never drawn as wrong.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

# ---- colours: the model colours of 02_cpt_model.ipynb (BASE_C, CPT_C, VAL_C), checked for colour-blind separation ----
MODEL_COLORS = {"Base": "#A4193E", "CPT": "#5C8CE6", "SFT": "#D35697"}
INK, INK2, MUTED = "#19131B", "#52514e", "#898781"          # INK = GREY of 02_cpt_model.ipynb
GRID, AXIS, SURFACE, LIMIT = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#c3c2b7"
ORDER = ["key_match", "decline", "manual"]
GROUP_LABEL = {"key_match": "Key value (auto)", "decline": "Decline (auto)", "manual": "Hand-graded"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": SURFACE,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.dpi": 110})


def load_run(results_dir):
    """answers.csv + metrics.json of one run; final hand grades from manual_grading_sheet.csv are merged in."""
    a = pd.read_csv(f"{results_dir}/answers.csv", dtype={"id": str})
    m = json.load(open(f"{results_dir}/metrics.json"))
    sheet = f"{results_dir}/manual_grading_sheet.csv"
    if os.path.exists(sheet):
        g = pd.read_csv(sheet, dtype={"id": str})
        done = g.dropna(subset=["grader_1", "grader_2", "final_score"]).set_index("id")["final_score"].astype(float)
        a.loc[a.id.isin(done.index), "final_score"] = a.loc[a.id.isin(done.index), "id"].map(done)
    a["g"] = pd.Categorical(a.scoring, ORDER, ordered=True)
    a["idn"] = a.id.astype(int)
    return a.sort_values(["g", "idn"]).reset_index(drop=True), m


def _title(fig, text, y=1.0):
    fig.suptitle(text, x=0.01, y=y, ha="left", fontsize=12, fontweight="bold", color=INK)


def _count(a, scoring):
    d = a[a.scoring == scoring]
    graded = d.final_score.notna()
    return float(d.final_score[graded].sum()), int(graded.sum()), int((~graded).sum())   # half credit counts 0.5


def composition_table(a):
    """Test set: question type x difficulty (a table, not a chart)."""
    t = pd.crosstab(a.scoring.map(GROUP_LABEL), a.difficulty, margins=True, margins_name="total")
    return t.reindex([GROUP_LABEL[g] for g in ORDER] + ["total"])


def fig_headline(a, m, label, path, number):
    """Six stat tiles; counts always shown with their denominators; pending never shown as 0."""
    k_ok, k_n, _ = _count(a, "key_match")
    d_ok, d_n, _ = _count(a, "decline")
    h_ok, h_n, h_p = _count(a, "manual")
    stopped = int(a.stopped_at_stop_token.astype(str).str.lower().eq("true").sum())
    tiles = [("Held-out perplexity", f"{m['heldout_perplexity']:.2f}", "per token, 99 held-out documents"),
             ("Key-value questions", f"{k_ok:g} / {k_n}", "correct (automatic)"),
             ("Should-decline questions", f"{d_ok:g} / {d_n}", "declined correctly (automatic)"),
             ("Hand-graded questions", f"{h_p} pending" if h_p else f"{h_ok:g} / {h_n}",
              "not counted until both graders finish" if h_p else "final score of the two graders"),
             ("Stopped at <|end|>", f"{stopped} / {len(a)}", f"the rest ran to the {int(a.new_tokens.max())}-token limit"),
             ("Mean latency", f"{m['mean_latency_s']:.2f} s", "per answer, T4 GPU, greedy")]
    fig = plt.figure(figsize=(10.5, 2.9))
    _title(fig, f"Figure {number}. {label}: headline results")
    for k, (t, v, s) in enumerate(tiles):
        ax = fig.add_axes([0.01 + (k % 3) * 0.33, 0.47 - (k // 3) * 0.45, 0.31, 0.38]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06", fc=SURFACE, ec=GRID, lw=1,
                                    transform=ax.transAxes))
        ax.text(0.05, 0.78, t, color=INK2, fontsize=9, transform=ax.transAxes)
        ax.text(0.05, 0.40, v, color=INK, fontsize=17, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.12, s, color=MUTED, fontsize=8, transform=ax.transAxes)
    fig.savefig(path); return fig


def fig_outcomes(runs, path, number):
    """One row per model (Base / CPT / SFT), one cell per test question, grouped by question type.
    Filled cell with a tick = correct; pale cell with a cross = wrong; grey hatched = pending hand grade."""
    names = [n for n, _ in runs]
    fig, ax = plt.subplots(figsize=(10.5, 1.05 + 0.55 * len(runs)))
    ax.grid(False)
    ref = runs[0][1]
    for r, (name, a) in enumerate(runs):
        assert list(a.id) == list(ref.id), "all models must be scored on the same questions"
        y, col = len(runs) - 1 - r, MODEL_COLORS[name]
        for x, row in enumerate(a.itertuples()):
            if pd.isna(row.final_score):
                ax.add_patch(plt.Rectangle((x + .06, y + .12), .88, .76, fc=GRID, ec="white", lw=0, hatch="////"))
            elif row.final_score >= 1:
                ax.add_patch(plt.Rectangle((x + .06, y + .12), .88, .76, fc=col, lw=0))
                ax.text(x + .5, y + .5, "✓", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            else:
                ax.add_patch(plt.Rectangle((x + .06, y + .12), .88, .76, fc=col, alpha=.16, lw=0))
                ax.text(x + .5, y + .5, "✕" if row.final_score == 0 else "½", ha="center", va="center", color=INK, fontsize=8)
    edges = np.cumsum([0] + [int((ref.scoring == g).sum()) for g in ORDER])
    for g, lo, hi in zip(ORDER, edges[:-1], edges[1:]):
        ax.text((lo + hi) / 2, len(runs) + .15, f"{GROUP_LABEL[g]}\nn={hi - lo}", ha="center", va="bottom", color=INK2, fontsize=8.5)
        if lo:
            ax.axvline(lo, color=INK2, lw=1.2)
    ax.set_xlim(0, len(ref)); ax.set_ylim(0, len(runs) + 1.0)
    ax.set_yticks([len(runs) - 1 - r + .5 for r in range(len(runs))]); ax.set_yticklabels(names, color=INK)
    ax.set_xticks(np.arange(len(ref)) + .5); ax.set_xticklabels(ref.id, fontsize=7.5); ax.set_xlabel("test question id")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    c0 = MODEL_COLORS[names[0]]
    ax.legend(handles=[Patch(fc=c0, label="✓ correct"), Patch(fc=c0, alpha=.16, label="✕ wrong  (½ = half credit)"),
                       Patch(fc=GRID, hatch="////", label="pending hand grade (not counted)")],
              loc="upper center", bbox_to_anchor=(.5, -.42 - .04 * len(runs)), ncol=3, fontsize=8.5)
    _title(fig, f"Figure {number}. Outcome of each of the {len(ref)} test questions", y=1.02)
    fig.savefig(path); return fig


def fig_behaviour(runs, path, number):
    """(a) how answers ended (stopped at <|end|> vs hit the token limit), per question type and model;
    (b) latency per answer."""
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.2 + 0.25 * (len(runs) - 1)), gridspec_kw={"width_ratios": [1, 1.25]})
    h = 0.62 / len(runs)
    for k, (name, a) in enumerate(runs):
        col = MODEL_COLORS[name]
        for i, g in enumerate(ORDER):
            d = a[a.scoring == g]
            st = int(d.stopped_at_stop_token.astype(str).str.lower().eq("true").sum())
            y = (len(ORDER) - 1 - i) + (len(runs) - 1) / 2 * h - k * h
            ax[0].barh(y, st, height=h * .9, color=col, zorder=3)
            ax[0].barh(y, len(d) - st, left=st, height=h * .9, color=LIMIT, zorder=3)
            ax[0].text(len(d) + .3, y, f"{name}: {st} stopped / {len(d) - st} at limit" if len(runs) > 1
                       else f"{st} stopped / {len(d) - st} at limit", va="center", color=INK, fontsize=8.5)
    ax[0].set_yticks(range(len(ORDER))[::-1]); ax[0].set_yticklabels([GROUP_LABEL[g] for g in ORDER], color=INK)
    ax[0].set_xlim(0, 30); ax[0].set_xlabel("number of answers"); ax[0].grid(axis="y", visible=False)
    ax[0].legend(handles=[Patch(fc=MODEL_COLORS[n], label=f"{n}: stopped at <|end|>") for n, _ in runs]
                 + [Patch(fc=LIMIT, label=f"hit the {int(runs[0][1].new_tokens.max())}-token limit")],
                 loc="upper center", bbox_to_anchor=(.5, -.25), ncol=len(runs) + 1 if len(runs) < 3 else 2, fontsize=8.5)
    ax[0].set_title("(a) How answers ended")
    if len(runs) == 1:
        name, a = runs[0]; d = a.sort_values("idn")
        ax[1].bar(d.idn, d.latency_s, width=.62, color=MODEL_COLORS[name], zorder=3)
        mu = d.latency_s.mean(); ax[1].axhline(mu, color=INK, lw=1.2, zorder=4)
        ax[1].text(d.idn.max() + .4, mu + .03 * d.latency_s.max(), f"mean {mu:.2f} s", va="bottom", ha="right", color=INK, fontsize=8.5)
        ax[1].set_xlim(.3, d.idn.max() + .7); ax[1].grid(axis="x", visible=False)
    else:
        for name, a in runs:
            d = a.sort_values("idn")
            ax[1].plot(d.idn, d.latency_s, "o-", color=MODEL_COLORS[name], lw=1.6, ms=5, label=f"{name} (mean {d.latency_s.mean():.2f} s)")
        ax[1].legend(fontsize=8.5)
    top = max(a.latency_s.max() for _, a in runs)
    ax[1].set_ylim(0, top * 1.18); ax[1].set_xlabel("test question id"); ax[1].set_ylabel("seconds per answer")
    ax[1].set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax[1].text(.99, -.36, "T4 GPU, greedy, warm-up excluded", transform=ax[1].transAxes, ha="right", color=MUTED, fontsize=8)
    ax[1].set_title("(b) Latency per answer")
    _title(fig, f"Figure {number}. Answer behaviour", y=1.04)
    fig.tight_layout(); fig.savefig(path); return fig


def fig_training_loss(log_history, evals, path, number, stage="SFT"):
    """Training loss (every logging step, plus a smoothed line) and validation loss per pass; pass 0 = before training."""
    tr = [(h["epoch"], h["loss"]) for h in log_history if "loss" in h and "eval_loss" not in h]
    col = MODEL_COLORS.get(stage, INK)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if tr:
        x, y = map(np.array, zip(*tr))
        ax.plot(x, y, color=col, lw=1, alpha=.35, label="training loss (every 5 steps)")
        ax.plot(x, pd.Series(y).rolling(5, min_periods=1, center=True).mean(), color=col, lw=2, label="training loss (smoothed)")
    ep, vl = [e["epoch"] for e in evals], [e["eval_loss"] for e in evals]
    ax.plot(ep, vl, "o-", color=INK, lw=2, ms=7, label="validation loss (10% of pairs)")
    for e, v in zip(ep, vl):
        ax.annotate(f"{v:.2f}", (e, v), textcoords="offset points", xytext=(7, 6), color=INK, fontsize=8.5)
    best = min(evals[1:], key=lambda e: e["eval_loss"]) if len(evals) > 1 else evals[0]
    ax.scatter([best["epoch"]], [best["eval_loss"]], s=200, facecolor="none", edgecolor=INK, lw=1.6, zorder=5,
               label=f"best pass ({int(best['epoch'])}): kept")
    ax.set_xticks(range(0, int(max(ep)) + 1)); ax.set_xlabel("pass over the training pairs (0 = before training)")
    ax.set_ylabel("cross-entropy loss (answer tokens)"); ax.set_ylim(0, None); ax.legend(fontsize=8.5)
    _title(fig, f"Figure {number}. {stage}: training and validation loss", y=1.02)
    fig.tight_layout(); fig.savefig(path); return fig


def fig_stage_comparison(stage_metrics, path, number):
    """(a) held-out perplexity per stage; (b) correct answers per question type, each bar labelled k/n.
    stage_metrics: list of (name, answers_df, metrics_dict) in the order Base, CPT, SFT.
    Hand-graded questions are only drawn when nothing is pending; otherwise 'pending' is written."""
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    names = [n for n, _, _ in stage_metrics]
    ppl = [m["heldout_perplexity"] for _, _, m in stage_metrics]
    ax[0].plot(range(len(names)), ppl, color=AXIS, lw=1.5, zorder=2)
    for i, (n, p) in enumerate(zip(names, ppl)):
        ax[0].scatter(i, p, s=80, color=MODEL_COLORS[n], zorder=3)
        ax[0].annotate(f"{p:.2f}", (i, p), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=9)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, color=INK); ax[0].set_xlim(-.5, len(names) - .5)
    ax[0].set_ylim(0, max(ppl) * 1.25); ax[0].set_ylabel("held-out perplexity (lower is better)"); ax[0].grid(axis="x", visible=False)
    ax[0].set_title("(a) Held-out perplexity")
    w = .8 / len(stage_metrics)
    for k, (n, a, _) in enumerate(stage_metrics):
        for j, g in enumerate(ORDER):
            ok, n_graded, pending = _count(a, g)
            x = j + (k - (len(stage_metrics) - 1) / 2) * w
            if pending:
                ax[1].text(x, 2, "pending", ha="center", va="bottom", color=MUTED, fontsize=7.5, rotation=90)
                continue
            pct = 100 * ok / n_graded
            ax[1].bar(x, max(pct, .8), width=w * .9, color=MODEL_COLORS[n], zorder=3)
            ax[1].text(x, max(pct, .8) + 2, f"{ok:g}/{n_graded}", ha="center", va="bottom", color=INK, fontsize=8)
    ref = stage_metrics[0][1]
    ax[1].set_xticks(range(len(ORDER)))
    ax[1].set_xticklabels([f"{GROUP_LABEL[g]}\n(n={int((ref.scoring == g).sum())})" for g in ORDER], color=INK)
    ax[1].set_ylim(0, 112); ax[1].set_yticks(range(0, 101, 20)); ax[1].set_ylabel("% correct"); ax[1].grid(axis="x", visible=False)
    ax[1].legend(handles=[Patch(fc=MODEL_COLORS[n], label=n) for n in names], ncol=len(names), loc="upper left", fontsize=8.5)
    ax[1].set_title("(b) Correct answers by question type")
    _title(fig, f"Figure {number}. Base vs CPT vs SFT", y=1.04)
    fig.tight_layout(); fig.savefig(path); return fig


def load_scored(folder, key):
    """results_<key>.csv + metrics_<key>.json written by score.py (hand grades already merged in)."""
    a = pd.read_csv(f"{folder}/results_{key}.csv", dtype={"id": str})
    m = json.load(open(f"{folder}/metrics_{key}.json"))
    a["g"] = pd.Categorical(a.scoring, ORDER, ordered=True)
    a["idn"] = a.id.astype(int)
    return a.sort_values(["g", "idn"]).reset_index(drop=True), m
