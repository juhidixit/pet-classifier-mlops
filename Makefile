PY ?= python
CKPT ?= artifacts/resnet50/model.pt
IMAGE ?= pet-classifier:latest

.PHONY: install lint test smoke train-resnet train-scratch train-all evaluate compare mlflow-ui \
        sweep serve demo docker docker-run tf-validate clean

install:            ## install everything (training, tracking, serving, demo, dev tools)
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check . && ruff format --check .

test:
	$(PY) -m pytest --cov=petcls --cov-report=term-missing

smoke:              ## seconds-long synthetic end-to-end run
	$(PY) -m petcls.train --config configs/smoke.yaml

train-resnet:
	$(PY) -m petcls.train --config configs/resnet50.yaml

train-scratch:
	$(PY) -m petcls.train --config configs/scratch_cnn.yaml

evaluate:
	$(PY) -m petcls.evaluate --checkpoint artifacts/resnet50/model.pt --out reports/resnet50
	$(PY) -m petcls.evaluate --checkpoint artifacts/scratch_cnn/model.pt --out reports/scratch_cnn

compare:            ## results table + learning curves for the README
	$(PY) scripts/compare_runs.py --runs resnet50 scratch_cnn

train-all: train-resnet train-scratch evaluate compare

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

sweep:              ## requires `wandb login`
	wandb sweep configs/sweep.yaml

serve:
	MODEL_PATH=$(CKPT) uvicorn serve.app:app --reload --port 8000

demo:
	streamlit run demo/streamlit_app.py -- --checkpoint $(CKPT)

docker:
	mkdir -p artifacts && cp $(CKPT) artifacts/model.pt
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

tf-validate:
	cd infra/terraform && terraform fmt -check && terraform init -backend=false && terraform validate

clean:
	rm -rf artifacts mlruns wandb .pytest_cache .ruff_cache
