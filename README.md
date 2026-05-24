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

Run the LLM judge on test set:

```bash
# Claude Haiku (default)
python src/llm_judge.py

# DeepSeek Chat
python src/llm_judge.py --model deepseek-chat
```

Or import in a notebook:

```python
from src.llm_judge import run_pairwise_judge

results = run_pairwise_judge("data/test_set.csv")                          # Claude Haiku
results = run_pairwise_judge("data/test_set.csv", model="deepseek-chat")   # DeepSeek
```

Output is saved to `data/judge_results_{model}_v1.0.json`.
