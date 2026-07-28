install:
	python -m pip install -r requirements.txt -r requirements-dev.txt

data:
	python -m src.data.generate_dataset --examples-per-category 30
	python -m src.data.prepare_dataset

test:
	pytest --cov=src --cov=api --cov-report=term-missing
	ruff check .
	ruff format --check .
	mypy src api

lint-fix:
	pre-commit run --all-files

train-lora:
	python -m src.training.train_lora

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py
