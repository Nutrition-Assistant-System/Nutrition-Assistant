# Scoreboard

Nutrition and Food Label Assistant, Phase 2\. Two small models are trained through the same three stages (Base, continued pretraining, SFT) and tested on the same 30 questions.

## GPT-2 small (124M)

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Small model, base | 12.29 | 0.0%  | 0.0% (0 / 4\) | 1.00 | GPT-2 small \+ 3 chat tokens. Repeats the question in all 30 answers; never stops at `<|end|>`. Graders agreed on 100% (8 / 8). Word-level perplexity 139.3. |
| \+ continued pretraining | 5.75 | 2.5%  | 0.0% (0 / 4\) | 0.98 | Best CPT pass (3 passes, learning rate 5e-5), from base\_project. Writes label-style text with wrong values. Graders agreed on 88% (7 / 8). |
| \+ SFT | 8.47 | 15.0%  | 75.0% (3 / 4\) | 0.31 | Best SFT pass, from CPT best (894 pairs). Answers in the trained format and declines 3 of 4 out-of-scope questions, but gives no correct key value. Graders agreed on 75% (6 / 8). |

All three rows come from one `score.py` run (2026-11-13, Tesla T4): the three models answered the 30 questions in the same session with the same function, so their latencies are directly comparable.

## SmolLM2-135M

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Small model, base | 7.00 | 0.0%  | 0.0% (0 / 4\) | 3.17 | SmolLM2-135M \+ 3 project tokens. Starts nearly every answer by repeating the question; never stops at `<|end|>`. Graders agreed on 100% (8 / 8). |
| \+ continued pretraining | 3.51 | 1.7%  | 0.0% (0 / 4\) | 2.50 | Best CPT pass, from base\_project. Graders agreed on 100% (8 / 8). |
| \+ SFT | 4.55 | 20.8%  | 100.0% (4 / 4\) | 1.12 | Best SFT pass, from CPT best. Declines all 4 out-of-scope questions, but gives no correct key value. Graders agreed on 63% (5 / 8). |

All three rows come from one `score.py` run on a Tesla T4 (same session, same function).

## GPT-2 small vs SmolLM2-135M

| Stage | Accuracy (GPT-2 / SmolLM2) | Refusal accuracy | Word-level perplexity | Avg. latency (s) |
| :---- | :---- | :---- | :---- | :---- |
| Base | 0.0% / 0.0% | 0% / 0% | 139.3 / 100.0 | 1.00 / 3.17 |
| \+ continued pretraining | 2.5% / 1.7% | 0% / 0% | 31.2 / 19.5 | 0.98 / 2.50 |
| \+ SFT | 15.0% / 20.8% | 75% / 100% | 66.9 / 36.2 | 0.31 / 1.12 |

- **Accuracy:** SmolLM2 ends higher after SFT (20.8% vs 15.0%), mainly from one more correct refusal (+1 point) and higher hand grades (+0.75). With 30 questions one point is 3.3 percentage points, so the difference is small. **Neither model gets a single key value right (0 / 18 at every stage).**  
- **Perplexity:** SmolLM2 is lower at every stage on the comparable word-level measure; both drop sharply with CPT and rise again after SFT.  
- **Latency:** SmolLM2 is 2.5 to 3.6 times slower (30 layers against GPT-2's 12).

