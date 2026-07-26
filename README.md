# Instruction-Tuned Technical Support Assistant



## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
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

## Training commands

```bash
python -m src.training.train_classifier_torch
python -m src.training.train_classifier_tensorflow
python -m src.training.train_full_sft
python -m src.training.train_lora
python -m src.training.tune_classifier --trials 10
RESUME_FROM_CHECKPOINT=outputs/lora/checkpoint-100 python -m src.training.train_lora
python -m src.evaluation.evaluate_generator --model outputs/lora
python -m src.inference.benchmark --model outputs/lora
```

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
