# MDG — NLP Benchmark Evaluation Framework

A research framework for evaluating LLMs on temporal, spatial, and causal reasoning tasks via OpenRouter.

---

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- [pyenv](https://github.com/pyenv/pyenv) (if managing Python versions)
- An [OpenRouter](https://openrouter.ai) API key

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd mdg
```

### 2. Install dependencies

If you use **pyenv**, pin the interpreter first so Poetry links to the right Python:

```bash
poetry env use $(pyenv which python)
poetry install
```

If you don't use pyenv:

```bash
poetry install
```

### 3. Configure API keys

Create a `.env` file in the `mdg/` subdirectory (next to `experiment_runner.py`):

```bash
cp mdg/.env.example mdg/.env   # if an example file exists
# or create it manually:
```

```ini
# mdg/.env
OPENROUTER_API_KEY="sk-or-v1-..."
OPENROUTER_API_KEY_2="sk-or-v1-..."   # optional, for load balancing
```

The key name used per experiment is set in the YAML config under `method.init_params.api_key_env`.

---

## Running an Experiment

All experiments are driven by a YAML config file. The runner takes a config and writes results to a JSON file.

```bash
poetry run python -m mdg.experiment_runner \
  --config mdg/exp_configs/mock68.yml \
  --output results/my_run.json
```

Or drop into the Poetry shell first:

```bash
poetry shell
python -m mdg.experiment_runner \
  --config mdg/exp_configs/mock68.yml \
  --output results/my_run.json
```

### What a config looks like

```yaml
dataset:
  name: causal_hf                          # which dataset loader to use
  init_params:
    hf_location: mdg-nlp/your-dataset      # HuggingFace dataset repo
  run_params: {}

method:
  name: llama-3.1-openrouter-few-shot-tlink  # which model wrapper to use
  init_params:
    api_key_env: OPENROUTER_API_KEY          # env var name for the API key
    model: mistralai/ministral-8b-2512       # OpenRouter model string
  run_params:
    reasoning_type: inductive+deductive      # prompt strategy
    k: 5                                     # number of few-shot examples
    per_label_k: 3                           # examples per label (balanced)
    seed: 42
    max_docs: 50                             # cap for quick sample runs
    split: test
    sleep_ms: 500                            # rate limit delay between calls
    retries: 3
    timeout_sec: 60
    show_progress: true
    model_output_path: runs/my_run/output.jsonl

evaluators:
  - name: temporal_tlink_classification
    init_params: {}
    run_params: {}
```

---

## Output

Each run produces two output files:

| File | Contents |
|---|---|
| `runs/<name>/output.jsonl` | One JSON line per example: input text, gold label, predicted label, few-shot examples used |
| `results/<name>.json` | Full metrics: accuracy, macro/micro/weighted F1, per-class breakdown, plus the full config |

---

## Available Components

### Dataset Loaders

| Name | Description |
|---|---|
| `timex_hf` | Time expression extraction (sentence-level) |
| `tlink_hf` | Temporal relation classification (BEFORE/AFTER) |
| `event_hf` | Event expression extraction |
| `event_hf_2` | Event extraction — perturbed variant |
| `event_hf_3` | Event extraction — document-level |
| `causal_hf` | Causal relation classification (sentence-level) |
| `causal_hf_2` | Causal classification — perturbed variant |
| `causal_hf_3` | Causal classification — document-level |

### Model Wrappers

| Name | Type | Task |
|---|---|---|
| `llama-3.1-openrouter-zero-shot-timex` | Zero-shot | Time expression extraction |
| `llama-3.1-openrouter-zero-shot-tlink` | Zero-shot | Temporal relation classification |
| `llama-3.1-openrouter-few-shot-timex` | Few-shot | Time expression extraction |
| `llama-3.1-openrouter-few-shot-tlink` | Few-shot | Relation classification (also used for causal) |
| `llama-3.1-openrouter-few-shot-event` | Few-shot | Event expression extraction |
| `llama-3.1-openrouter-few-shot-compositional` | Few-shot | Time + event extraction combined |

### Evaluators

| Name | Task |
|---|---|
| `temporal_tlink_classification` | Classification accuracy + F1 (use this for causal tasks) |
| `temporal_expression_partial` | Span extraction P/R/F1 (char + word, best-of + hungarian) |
| `event_expression_partial` | Same as above for event spans |
| `compositional_expression_partial` | Same for time + event spans combined |

### Reasoning Types (few-shot methods)

| Type | Needs train data | Description |
|---|---|---|
| `inductive` | Yes | Learn patterns from examples, apply to new input |
| `deductive` | No | Apply explicit rules defined in the system prompt |
| `abductive` | No | Reason step-by-step, output includes reasoning trace |
| `inductive+deductive` | Yes | Induce patterns, then apply deductively |
| `inductive+abductive` | Yes | Induce patterns, then abductive selection |
| `deductive+abductive` | No | Rule-based candidates, abductive selection |
| `inductive+deductive+abductive` | Yes | All three combined |

---

## Project Structure

```
mdg/
├── pyproject.toml              # Poetry config and dependencies
├── mdg/                        # Main package
│   ├── experiment_runner.py    # Entry point — run this
│   ├── registry.py             # Component registration system
│   ├── datamodels.py           # Pydantic data models
│   ├── .env                    # API keys (never commit this)
│   ├── exp_configs/            # YAML experiment configs
│   ├── dataset_loaders/        # Dataset loader classes
│   ├── methods/                # Model wrapper classes + prompts
│   │   ├── zero_shot/
│   │   ├── few_shot/
│   │   └── prompts/
│   ├── evaluators/             # Evaluation logic
│   ├── metrics/                # F1, partial span, classification metrics
│   └── scripts/                # Data creation / preprocessing scripts
```

---

## Adding a New Task

1. **Dataset loader** — add a class in `dataset_loaders/` decorated with `@register(_type=DATASET_LOADER, _name="your_loader")` and import it in `dataset_loaders/__init__.py`
2. **Method** — add a wrapper in `methods/` decorated with `@register(_type=MODEL_WRAPPER, _name="your_method")`; edit the system prompt for your task
3. **Evaluator** — add a class in `evaluators/` decorated with `@register(_type=EVAL_METHOD, _name="your_evaluator")`; for classification tasks `temporal_tlink_classification` works out of the box
4. **Config** — write a YAML in `exp_configs/` wiring the three names together
5. **Run** — `poetry run python -m mdg.experiment_runner --config mdg/exp_configs/your_config.yml --output results/out.json`
