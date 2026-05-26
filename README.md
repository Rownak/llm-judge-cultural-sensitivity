# llm-judge-cultural-sensitivity

A small, end-to-end prototype for **measuring cultural sensitivity and emotional appropriateness** in multilingual conversational AI — using an LLM-as-Judge that is **validated against human-labelled ground truth** and **stress-tested for bias**.

> The point of this project is not the dataset. It is the **methodology**: take a vague quality ("cultural sensitivity"), decompose it into a rubric, build a judge, and then *prove* the judge is trustworthy before believing any number it produces.

> **Built with Claude Code.** Every coding plan and code change was reviewed before committing — Claude Code did the typing, the design decisions and the verification were by Ahnaf Farhan.

---

## Objectives

Conversational AI serving a global user base regularly produces responses that miss the mark culturally or emotionally — wrong register, stereotyping, tone-deaf replies, mishandled holidays or names. These qualities are subjective and hard to measure, yet teams need reliable metrics to decide what is safe to launch and where to improve.

This repo demonstrates a credible recipe for measuring those fuzzy qualities at scale:

1. **Operationalize** the construct into a written rubric with sub-criteria.
2. **Generate** a small synthetic dataset where each item has a known better/worse response and a labelled flaw.
3. **Score** with an LLM-as-Judge.
4. **Validate** the judge against human ground truth (Cohen's kappa, accuracy).
5. **Diagnose** where the judge fails — by locale, criterion, flaw type — and check it for **positional bias**.
6. **Iterate** on the judge prompt to fix the biases you find.

All data is **synthetic and self-authored** — never scraped, never from real users.

---

## Features

| Feature | What it does | Why it matters |
|---|---|---|
| **Synthetic dataset** ([data/](data/)) | Hand-authored prompt pairs across `en-US`, `es-ES`, with each pair having a `ground_truth_winner` and a labelled `flaw_type`. | Gives a known answer key so the judge's accuracy can be measured. |
| **Versioned rubric** ([rubrics/](rubrics/)) | Decomposes cultural sensitivity (CS-1…CS-4) and emotional appropriateness (EA-1…EA-3) into observable sub-criteria with examples. | Forces the fuzzy construct to become concrete and reviewable. |
| **Versioned judge prompts** ([prompts/](prompts/)) | `v0` (no rubric), `v0.1` (anti-positional-bias instruction), `v1`, `v2` (rubric-aware). | Lets us A/B prompt designs and quantify which fixes actually work. |
| **Multi-model LLM-as-Judge** ([src/llm_judge.py](src/llm_judge.py), [src/llm_judge_small.py](src/llm_judge_small.py)) | Pairwise judge supporting Claude Haiku, DeepSeek, and a local HuggingFace SmolLM3-3B; emits structured JSON. | Compares judges of different sizes, costs, and providers under the same rubric. |
| **Agreement metrics** ([src/agreement.py](src/agreement.py)) | Raw accuracy + **Cohen's kappa** + confusion matrix vs. ground truth. | A judge is worthless until you prove it agrees with humans — kappa beats accuracy because it accounts for chance. |
| **Defect distribution** ([src/defect_distribution.py](src/defect_distribution.py)) | Breaks judge mistakes down by locale, scenario_type, primary_criterion, flaw_type, and confidence. | Tells you *where* the judge fails, not just *how often*. Drives the next iteration. |
| **Positional-bias toolkit** ([src/prepare_data_pos_bias.py](src/prepare_data_pos_bias.py), [src/flip_bias_evaluation.py](src/flip_bias_evaluation.py)) | Doubles the dataset by swapping `response_A`↔`response_B`, then measures how often the judge's underlying preference flips. | Position bias is the single most common LLM-judge failure mode. If a judge's verdict depends on which response appears first, no other metric matters. |
| **Stratified dev/test splits** ([src/prepare_dataset.py](src/prepare_dataset.py)) | 80/20 split stratified by locale, both retaining `ground_truth_winner`. | Keeps locale coverage honest in both splits. |

---

## Pipeline at a glance

```
                ┌──────────────────────────────────────┐
                │  rubrics/rubric.md   (the spec)     │
                └──────────────────────────────────────┘
                              │
data/synthetic_prompts.csv ───┴── prepare_dataset.py ──► dev_set / test_set
                                                                │
                                                                ▼
                                                       llm_judge.py
                                                  (Haiku / DeepSeek / SmolLM3)
                                                                │
                                              results/judge_*.json
                                              ┌─────────────────┼─────────────────┐
                                              ▼                 ▼                 ▼
                                       agreement.py    defect_distribution.py   flip_bias_evaluation.py
                                       (is it right?)   (where does it fail?)   (is it positionally biased?)
                                              │                 │                 │
                                              ▼                 ▼                 ▼
                                       results/agr_*.txt  results/defects_*.txt  results/posbias_*.txt
```

---

## Setup

### 1. Create virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Set up API keys

Create a `.env` file in the project root (already in `.gitignore`):

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key-here   
```

The local SmolLM3-3B judge ([src/llm_judge_small.py](src/llm_judge_small.py)) requires no API key — only a local GPU/CPU and the HuggingFace model cache.

---

## Usage

### 1. Design (or use) a dataset
Author or extend a CSV in [data/](data/) following the schema in [docs/synthetic_prompts_summary.md](docs/synthetic_prompts_summary.md). Each row needs a `prompt_text`, a good response, a deliberately flawed response, a `flaw_type`, and a `ground_truth_winner`.

All example commands below use the **local SmolLM3-3B judge** with **prompt version v0** on the **`dev_set_eng_spanish_pos_bias`** dataset. Swap the model / prompt / dataset to suit your own runs.

### 2. Split into dev / test
```bash
python src/prepare_dataset.py eng_spanish.csv
```
Produces `data/dev_set_eng_spanish.csv` and `data/test_set_eng_spanish.csv` (80/20 stratified by locale).

### 3. Build the positional-bias dataset
```bash
python src/prepare_data_pos_bias.py data/dev_set_eng_spanish.csv
```
Doubles every row by swapping `response_A`↔`response_B`, producing `data/dev_set_eng_spanish_pos_bias.csv`.

### 4. Run the judge
```bash
python src/llm_judge_small.py --input data/dev_set_eng_spanish_pos_bias.csv --prompt-version v0
```
Output: `results/judge_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.json`.

### 5. Validate the judge — agreement vs. humans
```bash
python src/agreement.py results/judge_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.json
```
Saves accuracy + Cohen's kappa + confusion matrix to `results/agr_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.txt`.

### 6. Diagnose failures — defect distribution
```bash
python src/defect_distribution.py results/judge_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.json
```
Saves a per-dimension breakdown of judge mistakes to `results/defects_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.txt`.

### 7. Test for positional bias
```bash
python src/flip_bias_evaluation.py results/judge_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.json
```
Saves the bias report to `results/posbias_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.txt`. A high `bias%` or a strong always-A / always-B pattern is the signal to iterate on the judge prompt (see [prompts/judge_prompt_v0_1.py](prompts/judge_prompt_v0_1.py) for an example mitigation).

---

## Findings (SmolLM3-3B on `dev_set_eng_spanish_pos_bias`)

| Judge prompt | Bias rate | Always-A | Always-B | Correct on both legs |
|---|---:|---:|---:|---:|
| `v0` (no anti-bias instruction)  | 0.657 | 0.571 | 0.086 | 0.086 |
| `v0.1` (anti-bias instruction)   | 0.632 | 0.553 | 0.079 | 0.237 |

The small SmolLM3-3B judge has a strong **first-position bias**: it picks response A in over half of all pairs regardless of content. Adding an explicit "do not favour a response based on position" instruction to the prompt (`v0.1`) barely moves the bias rate (0.657 → 0.632) or the always-A rate (0.571 → 0.553) — **prompt instruction alone is not enough to mitigate positional bias in a small model**, even though both-legs-correct does rise from 8.6% to 23.7%.

**Next step:** instead of relying on the prompt, evaluate every pair in both orderings and aggregate — e.g. take the judge's verdict only when it agrees with itself across positions, or score using a position-averaged preference. That turns positional bias from a confound into something the pipeline neutralises automatically.

---

## Repo layout

```
data/         synthetic prompt datasets + dev/test splits + pos-bias splits
docs/         PROJECT_BRIEF.md, dataset schema notes
prompts/      versioned judge prompts (v0, v0.1, v1, v2)
rubrics/      rubric.md + JSON criteria for v1 / v2
src/          dataset prep, judges, agreement, defect, positional-bias tools
results/      judge_*.json plus agr_*.txt, defects_*.txt, posbias_*.txt reports
```

---

## Connection to model training

The pairwise preference judgments this pipeline produces are exactly the form of data that feeds modern alignment methods (RLHF, DPO). A judge that has been validated against humans, debiased, and characterised by its defect distribution can scale that preference signal cheaply — turning evaluation into a training flywheel rather than a one-shot check.

---

## Limitations (read these honestly)

- **Small n** — the dataset is intentionally small; numbers are illustrative, not statistically conclusive.
- **Single non-native annotator** — real evaluation needs native speakers per locale.
- **Synthetic data** — no real-world distribution; useful for methodology demonstration only.
- **Locale coverage** — currently English + Spanish; the rubric and pipeline are locale-agnostic but require new prompts/responses to extend.
- **Prompt for Locale** - Prompt are english in this project. For multilingual, better to have prompt base on language.
- **Single judge per run** — no ensembling or self-consistency yet.
