# Nutrition and Food Label Assistant (Phase 2\)

Team project for the course *Building LLM Systems*, Phase 2: **Train a Small Model and Evaluate It**.

We train two small language models, **GPT-2 small (124M)** and **SmolLM2-135M**, through the same three stages and test them on the same 30 questions:

1. **Base**: the pretrained model plus three project chat tokens, no training  
2. **CPT**: continued pretraining on nutrition text  
3. **SFT**: supervised fine-tuning on question-answer pairs

All three versions of a model use the same prompt, the same greedy decoding and the same scoring, so the results are directly comparable. GPT-2 small is the main pipeline model; SmolLM2-135M is a comparison run with the same data and settings.

## Trained models on Hugging Face

The trained models are in the private Hugging Face organization `nutrition-assistant-team`. Each repository contains the model weights, the tokenizer with the chat template, and a model card with the training setup and scores. The course instructor has read access.

| Model | Base model | Repository | Held-out perplexity | Accuracy | Refusal accuracy |
| :---- | :---- | :---- | :---- | :---- | :---- |
| CPT | GPT-2 small (124M) | [nutrition-assistant-cpt](https://huggingface.co/nutrition-assistant-team/nutrition-assistant-cpt) | 5.75 | 2.5% | 0.0% (0/4) |
| SFT | GPT-2 small (124M) | [nutrition-assistant-gpt2-small-sft](https://huggingface.co/nutrition-assistant-team/nutrition-assistant-gpt2-small-sft) | 8.47 | 15.0% | 75.0% (3/4) |
| CPT (comparison run) | SmolLM2-135M | [nutrition-assistant-smollm2-cpt](https://huggingface.co/nutrition-assistant-team/nutrition-assistant-smollm2-cpt) | 3.51 | 1.7% | 0.0% (0/4) |
| SFT (comparison run) | SmolLM2-135M | [nutrition-assistant-smollm2-135m-sft](https://huggingface.co/nutrition-assistant-team/nutrition-assistant-smollm2-135m-sft) | 4.55 | 20.8% | 100.0% (4/4) |

The SmolLM2 SFT results are included above and in the scoreboard. A Hugging Face repository link for this checkpoint was not present in the supplied repository files, so no link is listed here.

### Load a model

The repositories are private, so log in first with a Hugging Face token that has access to the organization. Store the token in Colab or Kaggle secrets. Never put it in code or commit it to GitHub.

```py
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

login(userdata.get("HF_TOKEN"))

repo = "nutrition-assistant-team/nutrition-assistant-cpt"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What are good sources of protein?"}], tokenize=False
)
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=96,
                     eos_token_id=[50256, 50259], pad_token_id=50256)
print(tok.decode(out[0]))
```

The chat template produces the training prompt (see *Prompt format*). The stop ids above are for GPT-2 (`<|endoftext|>` 50256, `<|end|>` 50259); for SmolLM2 use `[0, 49154]`. Only the SFT models have learned to answer in this format and stop at `<|end|>`; Base and CPT models continue the text.

These are student research models: they do not give reliable nutrition values (see *Results*) and must not be used for dietary or medical advice.

## Results

Full results, the scoring rules and the hand grades are in [`Results/SCOREBOARD.md`](https://docs.google.com/document/u/0/d/1ZMUM1n1eoJeuIc_ALxyxHPZf0dRCfVDKQT3Io-hCMyk/edit). For each model, the three versions were scored on the same 30 test questions in one `score.py` run on a Tesla T4.

### GPT-2 small (124M)

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) |
| :---- | :---- | :---- | :---- | :---- |
| Base | 12.29 | 0.0% (0/30) | 0.0% (0/4) | 1.00 |
| \+ CPT | 5.75 | 2.5% (0.75/30) | 0.0% (0/4) | 0.98 |
| \+ SFT | 8.47 | 15.0% (4.5/30) | 75.0% (3/4) | 0.31 |

### SmolLM2-135M

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) |
| :---- | :---- | :---- | :---- | :---- |
| Base | 7.00 | 0.0% (0/30) | 0.0% (0/4) | 3.17 |
| \+ CPT | 3.51 | 1.7% (0.5/30) | 0.0% (0/4) | 2.50 |
| \+ SFT | 4.55 | 20.8% (6.25/30) | 100.0% (4/4) | 1.12 |

- **Accuracy** is the mean score over all 30 questions: 18 key-value questions and 4 should-decline questions are scored automatically; 8 open questions are graded 1 / 0.5 / 0 by two team members (sheets in `Manual Grading Sheets/`).  
- **Held-out perplexity** is computed over each model's original vocabulary, so the three added chat tokens do not change it. Per-token perplexities of the two models are **not** comparable with each other because their tokenizers differ; the scoreboard gives a word-level perplexity for that comparison.  
- Neither model gets a single key nutrition value right (0 of the 18 key-value questions at every stage). The SFT gains come from the answer format, from declining out-of-scope questions and from partly correct comparison answers.

## Prompt format

Every model, in training and in evaluation, uses one template:

```
<|user|> {question}
<|assistant|>
```

The target is  `{answer}<|end|>`; the loss is computed on the answer tokens only. The three project tokens `<|user|>`, `<|assistant|>` and `<|end|>` are added once to each tokenizer, before the base model is evaluated, and kept unchanged through CPT and SFT (ids 50257-50259 for GPT-2, vocabulary 50,260; ids 49152-49154 for SmolLM2, vocabulary 49,155). The template is saved as the chat template in each Hugging Face tokenizer. Each checkpoint records the fingerprint of the checkpoint it was trained from, so the chain Base → CPT → SFT can be checked.

## Data

Built from 988 nutrition records:

- 208 generic foods from USDA FoodData Central (SR Legacy, April 2018), public domain (CC0 1.0)  
- 780 packaged products from Open Food Facts, ODbL 1.0 (© Open Food Facts contributors)

| File | Content |
| :---- | :---- |
| `Data/foods_and_products.csv` | the merged 988 records |
| `Data/CPT/train.txt` | 889 documents for continued pretraining (90%) |
| `Data/CPT/heldout.txt` | 99 documents for held-out perplexity (10%, split by document) |
| `Data/SFT/sft_pairs.jsonl` | 894 question-answer pairs for SFT, none from test items |
| `Data/SFT/sft_handcheck_results_v2.csv` | the AI-assisted check of 59 sampled SFT pairs |
| `Data/test_questions.csv` | the 30 test questions (frozen; checked by fingerprint before every run) |

The test questions are for evaluation only: do not train on them.

### Data licence and attribution

The files in `Data/` contain information from [Open Food Facts](https://openfoodfacts.org), which is made available here under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Individual contents are under the [Database Contents License (DbCL) 1.0](https://opendatacommons.org/licenses/dbcl/1-0/). Because they are derived from Open Food Facts, the files in `Data/` are shared under the **ODbL 1.0**.

They also contain public-domain data (CC0 1.0) from U.S. Department of Agriculture, Agricultural Research Service, [FoodData Central](https://fdc.nal.usda.gov) (SR Legacy, 2018).

**Changes made by the team:** products were filtered and cleaned, merged with the USDA foods into one table with common units, converted into text documents for continued pretraining, and used to generate question-answer pairs and test questions from templates. The method is in `Notebooks/prepare_data_FINAL_V2.ipynb`.

## Repository structure

```
Data/                              merged data, CPT corpus, SFT pairs, SFT hand-check, test questions
Notebooks/
  prepare_data_FINAL_V2.ipynb      data preparation
  config.yaml                      settings for the GPT-2 pipeline
  config_smollm2.yaml              settings for the SmolLM2 comparison
  Base Model/
    GPT/                           01 notebook, error analysis & reflection notebook, metrics, run manifest
    SmolLM2/                       01 notebook, error analysis & reflection notebook, metrics, run manifest
    Figures01/                     base-model figures
  Continued-PreTraining/
    GPT/                           02 notebook; src/ with CPT metrics, loss curve, completions, manifest
    SmolLM2/                       02 notebook; src/ with CPT metrics, results, loss curve, completions, manifest
    Figures 02/                    CPT figures (GPT and SMOL)
  Supervised Fine-Tuning/
    03_sft_model.ipynb             SFT training and evaluation for GPT-2 Small and SmolLM2-135M
    GPT/
      Figures/                     SFT training and evaluation figures
      Results/
        Evaluation/                evaluation results, metrics, error analysis, score manifest
        Training/                  training metrics, loss history, split, masking example, samples, manifest
    SmolLM2/
      Code/                        SFT training and evaluation scripts, shared src utilities
      Figures/                     SFT training and evaluation figures
      Results/
        Evaluation/                evaluation results, metrics, error analysis, score manifest
        Training/                  training metrics, loss history, split, masking example, samples, manifest
Manual Grading Sheets/             hand grades of the 8 open questions, per model and stage
Results/SCOREBOARD.md              final scoreboard for both models
```

The scripts `train_cpt.py`, `train_sft.py` and `score.py` and the shared code (`prompting_tokens.py`, `scoring.py`) are written and run by the notebooks, so each notebook is self-contained.

## How to run

The notebooks run in Google Colab on a free T4 GPU. Run them in order: `01` (base), `02` (CPT), `03` (SFT). Settings are in `Notebooks/config.yaml` (GPT-2) and `Notebooks/config_smollm2.yaml` (SmolLM2); the random seed is fixed (42). Checkpoints are saved to Google Drive and are not stored in this repository.

Secrets are read from Colab Secrets by name; do not commit their values:

| Secret name | Use |
| :---- | :---- |
| `HF_TOKEN` | upload to and load from the private Hugging Face organization |

