# Instruction-Tuned Technical Support Assistant

![CI](https://github.com/nagarjunavaddineni/instruction-tuned-support-assistant/actions/workflows/ci.yml/badge.svg)

A LoRA/full-SFT-fine-tuned instruction model for answering technical-support
questions, served behind a FastAPI backend with a Streamlit chat UI. Given a
question (optionally with prior conversation turns), it returns a structured
answer: problem summary, likely causes, troubleshooting steps, commands, and
prevention tips.

## Architecture

- `src/data` — synthetic support-ticket dataset generation and SFT prompt
  formatting (`src/data/prompting.py`).
- `src/training` — LoRA (`train_lora.py`) and full-SFT (`train_full_sft.py`)
  fine-tuning, plus a separate PyTorch/TensorFlow issue-category classifier
  and Optuna hyperparameter tuning, all tracked via MLflow.
- `src/inference` — `SupportPredictor` (chat-template-based generation, with
  optional 4-bit quantization and response caching) and a latency/throughput
  benchmark.
- `api/main.py` — FastAPI app exposing `/health` and `/generate`.
- `app/streamlit_app.py` — chat UI that talks to the API.
- `src/common/config.py` — env-driven config (`TrainingConfig`,
  `InferenceConfig`), following a frozen-dataclass `from_env()` convention.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
python -m src.data.generate_dataset --examples-per-category 30
python -m src.data.prepare_dataset
python -m src.training.train_lora
```

Run MLflow:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Run API and UI:

```bash
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

## API

`POST /generate`

| field            | type                              | default | notes                              |
| ---------------- | ---------------------------------- | ------- | ----------------------------------- |
| `question`       | string (3–4000 chars)               | —       | required                            |
| `max_new_tokens` | int (16–1024)                       | 256     |                                      |
| `temperature`    | float (0–2)                         | 0.3     | `0` enables greedy decoding + cache |
| `history`        | list of `{role, content}` (≤ 20)    | `[]`    | prior turns, resent by the client   |

The API is stateless: no session/conversation ID exists server-side, so the
caller (the Streamlit app, or any other client) resends the running history
on every call.

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "question": "And now what?",
        "history": [
          {"role": "user", "content": "My Docker container exits immediately."},
          {"role": "assistant", "content": "Check `docker logs <container>` first."}
        ]
      }'
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

Training (`TrainingConfig`): `MODEL_NAME`, `MAX_SEQ_LENGTH`, `NUM_EPOCHS`,
`TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`,
`LEARNING_RATE`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `LORA_RANK`, `LORA_ALPHA`,
`LORA_DROPOUT`, `USE_4BIT`, `MLFLOW_TRACKING_URI`, `RESUME_FROM_CHECKPOINT`.

Inference (`InferenceConfig`): `INFERENCE_MODEL` (base/adapter path served by
the API), `INFERENCE_USE_4BIT` (4-bit quantization via `bitsandbytes`,
applied only when CUDA is available), `MAX_HISTORY_TURNS`.

## Training commands

```bash
python -m src.training.train_classifier_torch
python -m src.training.train_classifier_tensorflow
python -m src.training.train_full_sft
python -m src.training.train_lora
python -m src.training.tune_classifier --trials 10
RESUME_FROM_CHECKPOINT=outputs/lora/checkpoint-100 python -m src.training.train_lora
python -m src.evaluation.evaluate_generator --model outputs/lora
python -m src.inference.benchmark --model outputs/lora [--use-4bit] [--runs 5] [--new-tokens 128]
```

`benchmark` reports mean/median latency, mean tokens/sec, and (on CUDA) peak
memory to `reports/benchmark.json`.

## Development

```bash
make install       # install runtime + dev dependencies
make test           # pytest + coverage, ruff check, ruff format --check, mypy
make lint-fix        # run all pre-commit hooks (ruff --fix, ruff-format, etc.) across the repo
```

CI runs the same pytest/ruff/mypy checks on every push and pull request.

## GitHub upload

Create an empty GitHub repository named `instruction-tuned-support-assistant`, then run:

```bash
git init
git add .
git commit -m "Build instruction-tuned support assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/instruction-tuned-support-assistant.git
git push -u origin main
```

Do not commit trained model checkpoints or `.env`; they are ignored.
