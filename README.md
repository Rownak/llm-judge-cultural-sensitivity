# llm-judge-cultural-sensitivity

A methodology prototype for evaluating cultural sensitivity and emotional appropriateness in multilingual conversational AI, using a validated LLM-as-Judge with human-labeled ground truth.

## Setup

### 1. Create virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

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
DEEPSEEK_API_KEY=sk-your-deepseek-key-here   # optional
```

## Usage

### 1. Design a new dataset

Follow the schema and design principles documented in [docs/synthetic_prompts_summary.md](docs/synthetic_prompts_summary.md):

- Each row needs a `prompt_text`, a `response_good`, and a `response_flawed` with exactly one labeled `flaw_type`
- Cover the locales and criteria you want to evaluate (see the rubric in `rubrics/rubric.md`)
- Save your CSV to `data/` — e.g. `data/my_prompts.csv`

### 2. Prepare dev / test splits

```bash
# Default — uses data/synthetic_prompts.csv
python src/prepare_dataset.py

# Custom dataset
python src/prepare_dataset.py subtle_synthetic_prompts.csv
```

Outputs (80/20 stratified split by locale, both include `ground_truth_winner`):
```
data/dev_set_{stem}.csv
data/test_set_{stem}.csv
```

### 3. Run the LLM judge

```bash
# Claude Haiku (default) on test set
python src/llm_judge.py --input data/test_set_synthetic_prompts.csv

# DeepSeek on dev set
python src/llm_judge.py --input data/dev_set_subtle_synthetic_prompts.csv --model deepseek-chat
```

Or import in a notebook:

```python
from src.llm_judge import run_pairwise_judge

results = run_pairwise_judge("data/test_set_synthetic_prompts.csv")
results = run_pairwise_judge("data/dev_set_synthetic_prompts.csv", model="deepseek-chat")
```

Output is saved to `results/judge_{dataset_stem}_{model}_v{rubric_version}.json`.

### 4. Compute agreement metrics

```bash
python src/agreement.py results/judge_test_set_synthetic_prompts_claude-haiku-4-5-20251001_v1.0.json
```

Prints accuracy + Cohen's kappa + confusion matrix to stdout, and saves the report to:
```
results/agr_{dataset_stem}_{model}_v{rubric_version}.txt
```
