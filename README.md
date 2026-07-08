# Occuwise AI — Ophthalmology CNN Benchmark & Production Toolkit

Train, compare, and deploy CNNs for **Fundus** and **OCT** imaging across five
architecture families and six public datasets — then promote the best combination
to production.

> **Read [CLAUDE.md](CLAUDE.md)** for the full architecture, design rationale, and
> extension guide.

## Architectures
ResNet · EfficientNet · DenseNet · U-Net (segmentation) · Vision Transformer (ViT)
— all ImageNet-pretrained via [`timm`](https://github.com/huggingface/pytorch-image-models)
and [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch).
HF weight refs supported (e.g. [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50)).

## Datasets
EyePACS · APTOS · Messidor (DR grading) · OCT2017 (OCT disease) · REFUGE
(glaucoma cls + disc/cup seg) · Duke OCT (fluid/layer seg). See [data/README.md](data/README.md).

## Quickstart

```bash
# Install a CUDA-matched torch first: https://pytorch.org
py -m pip install -e .
py -m pip install -r requirements.txt

# Prepare a dataset (download into data/aptos/ first — see data/README.md)
py scripts/prepare_aptos.py --root data/aptos

# Train
py -m occuwise.train +experiment=dr_baseline

# Sweep architectures × datasets, then rank
py -m occuwise.train -m model=resnet50,efficientnet_b4,densenet121,vit_base data=aptos,eyepacs
py -m occuwise.compare          # -> outputs/leaderboard.md

# Export the winner + serve
py -m occuwise.export --ckpt outputs/aptos/efficientnet_b4/best.ckpt --dataset aptos --onnx --out models/production
py -m uvicorn serving.app:app --port 8080
```

## Test images (no dataset download needed)

Fetch a handful of real, openly-licensed fundus/OCT images to try the models with:

```bash
py scripts/fetch_samples.py        # -> data/samples/  (+ ATTRIBUTION.md)
```

They appear as a clickable gallery in the web app, and the CLI can run them directly:

```bash
py -m occuwise.predict --dataset aptos --arch resnet50 --samples          # all fundus samples
py -m occuwise.predict --dataset oct2017 --arch densenet121 --samples     # all OCT samples
py -m occuwise.predict --dataset aptos --arch resnet50 --image path/to/one.jpg
py -m occuwise.predict --dataset aptos --arch resnet50 --samples --explain    # + Grad-CAM heatmaps
```

**Grad-CAM explainability** is built in: tick *"Show Grad-CAM heatmap"* in the web app,
or add `--explain` on the CLI, to see a heatmap of where the model looked (red = most
influential). Essential for checking the model attends to real pathology, not artifacts.

## POC web app (upload → analyze → result)

A self-contained web UI implements the `corereqs.md` flow — no dataset or training
run required to demo it:

```bash
py -m uvicorn serving.webapp:app --port 8080     # then open http://localhost:8080
```

- **`/prepare`** — ready a model for use (pick dataset + architecture; loads an
  ImageNet-pretrained backbone sized to that dataset's clinical classes, or a trained
  checkpoint if you give one) and run the smoke tests from the browser.
- **`/`** — upload an OCT/fundus image, run the prepared model, and see the structured
  result: class probabilities for classification, or a mask overlay + cup-to-disc
  ratio for segmentation.

Models prepared without a checkpoint run in **demo mode** (untrained head) — the full
pipeline works and returns structured output, but train a model for real predictions.

## Repo map (short)
- `configs/` — Hydra control surface (`model/`, `data/`, `train/`, `experiment/`)
- `src/occuwise/` — data · models · engine · train/evaluate/compare/export
- `serving/` — FastAPI + ONNX inference (Triton guidance inside)
- `scripts/` — dataset → manifest converters
- `tests/` — offline CPU smoke tests

## Status
Scaffold complete and import-clean. Datasets must be downloaded separately;
GPU recommended for training. Not a medical device — **decision support only**.
