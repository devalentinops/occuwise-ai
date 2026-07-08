# Occuwise AI — common workflows.
# On Windows, run these targets via `make` (Git Bash / WSL) or copy the command bodies.

PY ?= py            # Windows py-launcher; use `python` on Linux/mac

.PHONY: install lint test train sweep evaluate samples predict compare export serve webapp mlflow clean

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .
	$(PY) -m pip install -r requirements.txt

lint:
	$(PY) -m ruff check src serving tests

test:
	$(PY) -m pytest

# Single experiment, e.g. make train EXP=dr_baseline
train:
	$(PY) -m occuwise.train +experiment=$(EXP)

# Cartesian sweep of architectures x datasets (Hydra multirun).
sweep:
	$(PY) -m occuwise.train -m \
	  model=resnet50,efficientnet_b4,densenet121,vit_base \
	  data=aptos,eyepacs,messidor

evaluate:
	$(PY) -m occuwise.evaluate ckpt=$(CKPT) data=$(DATA)

# Download openly-licensed fundus/OCT test images into data/samples/.
samples:
	$(PY) scripts/fetch_samples.py

# Run a model over the bundled samples, e.g. make predict DATASET=aptos ARCH=resnet50
predict:
	$(PY) -m occuwise.predict --dataset $(DATASET) --arch $(ARCH) --samples

# Build the leaderboard across all tracked runs.
compare:
	$(PY) -m occuwise.compare

export:
	$(PY) -m occuwise.export ckpt=$(CKPT) --onnx --torchscript

serve:
	$(PY) -m uvicorn serving.app:app --host 0.0.0.0 --port 8080

# POC web UI: upload an image, prepare & run models, view results in the browser.
webapp:
	$(PY) -m uvicorn serving.webapp:app --host 0.0.0.0 --port 8080

mlflow:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns --port 5000

clean:
	rm -rf outputs experiments lightning_logs __pycache__
