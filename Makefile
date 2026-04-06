PYTHON      := uv run python
MLFLOW_PORT := 5000

# Prediction defaults — override on the command line:
#   make predict-test RUN_ID=abc123 MODEL_TYPE=classifier
#   make predict-full RUN_ID=abc123 MODEL_TYPE=classifier DATA_PATH=data/gvirm/Data1.csv
RUN_ID     ?= $(error RUN_ID is required for predict targets, e.g. make predict-test RUN_ID=<run_id> MODEL_TYPE=<type>)
MODEL_TYPE ?= classifier
DATA_PATH  ?=

.DEFAULT_GOAL := help

.PHONY: help venv install preprocess \
        train-iforest train-autoencoder train-classifier train-all \
        predict-train predict-val predict-test predict-all predict-full \
        lint format \
        mlflow-ui \
        clean clean-outputs clean-processed clean-pycache

# ─── Help ────────────────────────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Environment ─────────────────────────────────────────────────────────────
venv: ## Create / sync the virtual environment  (uv sync)
	uv sync

install: venv ## Install all dependencies including dev extras
	uv sync --extra dev

# ─── Data ────────────────────────────────────────────────────────────────────
preprocess: ## Cleanse raw data → data/processed/gvirm/  (run before training)
	$(PYTHON) scripts/preprocess_data.py

# ─── Training ────────────────────────────────────────────────────────────────
train-iforest: ## Train Isolation Forest  (multiclass_clean.csv)
	$(PYTHON) scripts/train_isolation_forest.py

train-autoencoder: ## Train spatial autoencoder  (Data1.csv + spatial features)
	$(PYTHON) scripts/train_autoencoder.py --spatial

train-classifier: ## Train multi-class classifier  (multiclass_clean.csv)
	$(PYTHON) scripts/train_classifier.py

train-all: train-iforest train-autoencoder train-classifier ## Train all three models sequentially

# ─── Prediction ──────────────────────────────────────────────────────────────
# Run a trained model against a specific data split or dataset.
# Required: RUN_ID=<mlflow run id>   MODEL_TYPE=classifier|autoencoder|isolation_forest
# Example:  make predict-test RUN_ID=abc123 MODEL_TYPE=classifier

predict-train: ## Run model against its training split      (RUN_ID= MODEL_TYPE=)
	$(PYTHON) scripts/predict.py --run-id $(RUN_ID) --model-type $(MODEL_TYPE) --split train

predict-val: ## Run model against its validation split     (RUN_ID= MODEL_TYPE=)
	$(PYTHON) scripts/predict.py --run-id $(RUN_ID) --model-type $(MODEL_TYPE) --split val

predict-test: ## Run model against its test split          (RUN_ID= MODEL_TYPE=)
	$(PYTHON) scripts/predict.py --run-id $(RUN_ID) --model-type $(MODEL_TYPE) --split test

predict-all: ## Run model against train + val + test       (RUN_ID= MODEL_TYPE=)
	$(PYTHON) scripts/predict.py --run-id $(RUN_ID) --model-type $(MODEL_TYPE) --split all

predict-full: ## Run model against full dataset            (RUN_ID= MODEL_TYPE= [DATA_PATH=])
	$(PYTHON) scripts/predict.py --run-id $(RUN_ID) --model-type $(MODEL_TYPE) --split full \
	  $(if $(DATA_PATH),--data-path $(DATA_PATH),)

# ─── Code quality ────────────────────────────────────────────────────────────
lint: ## Lint source and scripts with ruff
	uv run ruff check src/ scripts/

format: ## Auto-format source and scripts with ruff
	uv run ruff format src/ scripts/

# ─── MLFlow ──────────────────────────────────────────────────────────────────
mlflow-ui: ## Launch the MLFlow tracking UI  (http://localhost:$(MLFLOW_PORT))
	uv run mlflow ui --port $(MLFLOW_PORT)

# ─── Clean ───────────────────────────────────────────────────────────────────
clean-outputs: ## Remove generated outputs directory (plots, predictions)
	rm -rf outputs/

clean-processed: ## Remove processed data  (re-run `make preprocess` to recreate)
	rm -rf data/processed/

clean-pycache: ## Remove Python __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean: clean-outputs clean-pycache ## Remove outputs and caches (keeps processed data and mlruns)

