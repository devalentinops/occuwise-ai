# CLAUDE.md — Occuwise AI

Reference document for this repository. Read this first before making changes.
It explains **what** the project is, **how** it's structured, **why** the key
decisions were made, and **how to extend** it safely.

---

## 1. What this project is

Occuwise AI is a **benchmark-and-production toolkit for ophthalmology CNNs**. The
goal is to train the same set of architectures across several public Fundus/OCT
datasets, measure them on task-appropriate clinical metrics, and identify the
**best (architecture × dataset × task) combination** to promote to production.

**Architectures** (all pretrained on ImageNet, fine-tuned here):
- **ResNet** (18/50) — strong, fast baseline. HF refs: `microsoft/resnet-18`, `microsoft/resnet-50`.
- **EfficientNet** (B0/B4) — best accuracy/compute trade-off; strong on fundus DR.
- **DenseNet** (121/201) — classic medical-imaging backbone (CheXNet lineage).
- **U-Net** — segmentation (optic disc/cup, OCT fluid/layers). Any encoder above.
- **Vision Transformer (ViT)** — base/small; benefits from more data & epochs.

**Datasets & tasks:**

| Dataset | Task | Modality | Primary metric |
|---------|------|----------|----------------|
| EyePACS, APTOS, Messidor | DR grading (classification) | fundus | Quadratic-weighted κ |
| OCT2017 | Disease classification (CNV/DME/DRUSEN/NORMAL) | OCT | Macro-F1 |
| REFUGE | Glaucoma classification **and** disc/cup segmentation | fundus | AUROC / Dice |
| Duke OCT | Fluid/layer segmentation | OCT | Dice |

There are exactly **two task families** — *classification* and *segmentation* —
and every component branches on `spec.task`. Adding a dataset or model rarely
requires new task plumbing.

---

## 2. Tech stack & why

| Concern | Choice | Why |
|---------|--------|-----|
| Training loop | **PyTorch Lightning** | Removes boilerplate; multi-GPU, AMP, checkpointing, early-stopping for free. |
| Backbones | **timm** | One API for ResNet/EfficientNet/DenseNet/ViT + HF-hub weights. |
| Segmentation | **segmentation-models-pytorch** | U-Net/U-Net++/FPN with pretrained encoders. |
| Config & sweeps | **Hydra + OmegaConf** | Compose `model × data × train`; `-m` runs the Cartesian sweep. |
| Tracking | **MLflow** | Self-hostable, audit-friendly, has a Model Registry for versioning. |
| Metrics | **torchmetrics** | Correct ordinal κ, AUROC, Dice, IoU out of the box. |
| Augmentation | **Albumentations** | Fast, joint image+mask transforms. |
| Medical extras | **MONAI** (optional) | Medical transforms/losses/metrics if needed. |
| Explainability | **pytorch-grad-cam** | Saliency maps — essential for clinical trust. |
| Export | **ONNX + TorchScript** | Portable, fast serving; ONNX→TensorRT for GPU. |
| Serving | **FastAPI (+ Triton for scale)** | Simple REST now; Triton for a model fleet. |

---

## 3. Repository layout

```
occuwise-ai/
├── CLAUDE.md                  # ← this file
├── README.md                  # quickstart
├── pyproject.toml             # package metadata, console script, ruff/pytest
├── requirements.txt           # pinned deps (install torch separately, CUDA-matched)
├── Makefile                   # install / train / sweep / compare / export / serve
│
├── configs/                   # Hydra config groups (the experiment control surface)
│   ├── config.yaml            #   base defaults + output/sweep dirs
│   ├── model/*.yaml           #   one file per architecture (arch alias + pretrained)
│   ├── data/*.yaml            #   one file per dataset (root, manifest, image_size)
│   ├── train/default.yaml     #   shared hyperparameters + hardware
│   └── experiment/*.yaml      #   named, ready-to-run combinations (+experiment=…)
│
├── src/occuwise/              # the installable package
│   ├── data/
│   │   ├── registry.py        #   DatasetSpec catalog — SINGLE SOURCE OF TRUTH per dataset
│   │   ├── datasets.py        #   Classification/Segmentation Dataset (manifest-driven)
│   │   ├── transforms.py      #   Albumentations pipelines (fundus circle-crop+CLAHE, OCT)
│   │   └── datamodule.py      #   LightningDataModule (+ class-balanced sampler)
│   ├── models/
│   │   └── registry.py        #   build_model(): timm classifier / smp segmenter factory
│   ├── engine/
│   │   ├── losses.py          #   CE / focal / Dice+CE
│   │   ├── metrics.py         #   task-appropriate torchmetrics collections
│   │   ├── lit_classifier.py  #   LitClassifier LightningModule
│   │   └── lit_segmenter.py   #   LitSegmenter LightningModule
│   ├── train.py               # Hydra entrypoint: fit + test + log
│   ├── evaluate.py            # evaluate a checkpoint on a test split
│   ├── compare.py             # build the leaderboard across all MLflow runs
│   ├── export.py              # checkpoint → ONNX/TorchScript + model_card.json
│   └── cli.py                 # `occuwise <cmd>` console entrypoint
│
├── serving/                   # inference — production API + POC web app
│   ├── inference.py           #   ONNX predictor + exact training preprocessing (production)
│   ├── app.py                 #   FastAPI JSON service on the exported ONNX model (production)
│   ├── poc_predictor.py       #   in-process PyTorch predictor (no ONNX/training needed)
│   ├── webapp.py              #   POC web app: /prepare + / (upload→analyze→display)
│   ├── web/                   #   self-contained HTML pages (predict.html, prepare.html)
│   ├── Dockerfile             #   slim CPU image (GPU variant documented inside)
│   └── README.md              #   FastAPI vs Triton vs TorchServe guidance
│
├── scripts/                   # dataset → manifest converters (one per dataset)
│   ├── _manifest.py           #   shared stratified-split + writer helpers
│   └── prepare_*.py
│
├── data/                      # (gitignored) raw datasets + generated manifests
│   └── README.md              #   download links + manifest schema
├── tests/                     # offline CPU smoke tests (models build, data reads)
└── notebooks/                 # exploratory analysis (not part of the pipeline)
```

### The data-flow contract
Raw datasets differ wildly on disk. We normalise **once** via `scripts/prepare_*.py`
into a **manifest CSV** (`image_path,label,split` or `image_path,mask_path,split`,
paths relative to the dataset root). Everything downstream reads only manifests, so
the training code is dataset-agnostic. **When adding a dataset, the prepare script
is where the messy, dataset-specific logic lives — keep it out of the pipeline.**

---

## 4. How to run

```bash
# 0. Install (install a CUDA-matched torch FIRST, see requirements.txt)
py -m pip install -e . && py -m pip install -r requirements.txt

# 1. Get data + build a manifest
#    (download into data/aptos/ first — see data/README.md)
py scripts/prepare_aptos.py --root data/aptos

# 2. Train one model
py -m occuwise.train model=efficientnet_b4 data=aptos
#    …or a named experiment
py -m occuwise.train +experiment=dr_baseline

# 3. Sweep architectures × datasets (one process per combination)
py -m occuwise.train -m model=resnet50,efficientnet_b4,densenet121,vit_base data=aptos,eyepacs

# 4. Rank everything → outputs/leaderboard.md
py -m occuwise.compare

# 5. Export the winner and serve it
py -m occuwise.export --ckpt outputs/aptos/efficientnet_b4/best.ckpt --dataset aptos --onnx --torchscript --out models/production
py -m uvicorn serving.app:app --port 8080

# Track experiments live
py -m mlflow ui --backend-store-uri ./mlruns --port 5000
```

On Windows use the `py` launcher (as above); on Linux/mac use `python`.

---

## 4b. POC web app (the `corereqs.md` flow)

`serving/webapp.py` is a two-page FastAPI app that demonstrates *upload → AI
processes → structured output → display* with **zero dataset or training required**:

```bash
py -m uvicorn serving.webapp:app --port 8080   # http://localhost:8080
```

- **`GET /prepare`** → `POST /api/prepare` builds a `PocPredictor`: an ImageNet-
  pretrained backbone whose head is sized to the chosen dataset's clinical classes
  (or a trained checkpoint if a path is given). Prepared models are held in memory.
  The same page runs the pytest smoke suite via `POST /api/run-tests`.
- **`GET /`** → `POST /api/predict` takes an image upload + a prepared `model_key`,
  runs inference **in-process with PyTorch** (`serving/poc_predictor.py`), and returns
  structured JSON: class probabilities (classification) or per-class area + cup-to-disc
  ratio + a base64 mask overlay (segmentation).

Why in-process PyTorch, not the ONNX path (`serving/app.py`)? The POC needs one-click
readiness with no export step; production uses the ONNX path for portability/latency.
A model prepared without a checkpoint is **untrained (demo mode)** — structurally
correct output, but not clinically meaningful until trained.

## 5. How to extend (recipes)

**Add an architecture (classification):** add an alias → timm name in
`CLASSIFICATION_ARCHS` (`models/registry.py`) and a `configs/model/<name>.yaml`.
Any raw `timm` name also works without an alias. HF-hub weights: use
`hf_hub:<repo>` as the arch.

**Add a U-Net encoder / decoder:** extend `SEGMENTATION_ENCODERS` or set
`decoder: unetpp|fpn|segformer` in the model config.

**Add a dataset:** (1) add a `DatasetSpec` to `data/registry.py` (this fixes its
task, classes, and primary metric), (2) write `scripts/prepare_<name>.py` to emit a
manifest, (3) add `configs/data/<name>.yaml`. No pipeline changes needed.

**Add a metric:** extend the relevant collection in `engine/metrics.py`. To rank on
it, set it as a dataset's `primary_metric` in the registry.

**Change loss / sampling:** `engine/losses.py` and the sampler in
`data/datamodule.py`. DR grading is imbalanced — `train.loss=focal` +
`balance_classes=true` are the go-to knobs.

---

## 6. Conventions & guardrails

- **Single source of truth per dataset** is its `DatasetSpec`. Don't hardcode class
  counts or metrics elsewhere — read them from the spec.
- **Higher-is-better** for every registered metric (kappa, F1, AUROC, Dice). The
  checkpoint/early-stopping monitor assumes this (`_monitor_for` in `train.py`).
- **Never commit data or weights** (enforced by `.gitignore`). Medical imagery has
  privacy/licensing constraints.
- **Reproducibility:** every run seeds via `pl.seed_everything(cfg.seed)`; splits are
  seeded in the prepare scripts. Change the seed deliberately, not incidentally.
- **Cross-dataset validation is mandatory before production.** Public-dataset scores
  don't transfer across cameras/populations — train on one DR set, evaluate on
  another via `occuwise.evaluate`.

---

## 7. Production & regulatory posture

- These models are **clinical decision support (screening triage)**, **not** an
  autonomous diagnosis. Every API response carries that disclaimer.
- **Versioning:** promote models through the MLflow Model Registry; deployments
  reference an immutable version. Each export writes a `model_card.json` with
  provenance, intended use, and limitations — keep it with the model.
- **Audit trail:** log every inference (input hash, model version, output, timestamp).
- **Calibration & uncertainty** matter clinically — prefer calibrated probabilities
  (temperature scaling) and consider abstention on low-confidence cases before
  deployment.
- **Serving at scale:** start with the FastAPI+ONNX service; graduate to **NVIDIA
  Triton** (ONNX/TensorRT backend, dynamic batching, multi-model) for a fleet. See
  `serving/README.md`.
- If used to inform diagnosis/treatment, this is likely **regulated software (SaMD)**
  — plan for the appropriate clinical validation and regulatory pathway.

---

## 8. Known gaps / TODO (good first tasks)
- Wire **Grad-CAM** into the FastAPI response (return a saliency overlay).
- Add **temperature-scaling calibration** as a post-fit step in `train.py`.
- Add **Optuna** Hydra sweeper config for HPO (the train entrypoint already
  returns the monitored metric for the sweeper to optimise).
- Add **patient-level grouped splits** to prepare scripts to prevent leakage where
  multiple images come from one patient (EyePACS/Messidor).
- Add a **combined DR model** option (harmonise EyePACS+APTOS+Messidor label scales).
