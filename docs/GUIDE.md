# Occuwise AI — Beginner's Guide

A plain-English walkthrough of what this project is, how the pieces connect, and
how to use it. No prior deep-learning experience assumed. For the terse technical
reference, see [../CLAUDE.md](../CLAUDE.md).

---

## 1. What problem are we solving?

We want a computer program that can look at a picture of the back of the eye and
say something clinically useful, for example:

- **Fundus photo** (a colour photograph of the retina) → "this eye shows *moderate*
  diabetic retinopathy" or "signs of glaucoma".
- **OCT scan** (a cross-section 'ultrasound-like' image of the retina) → "this shows
  fluid / DME / drusen / is normal".

The program that does this is a **neural network** (specifically a **CNN** —
Convolutional Neural Network). This project's job is to **train** several kinds of
CNN on several public **datasets**, **measure** which combination is most accurate,
and **serve** the best one so a clinician can upload an image and get a result.

---

## 2. The key idea: a CNN is a function that learns from examples

A CNN like **ResNet** is, at heart, a mathematical function:

```
image  →  [ ResNet ]  →  a list of numbers (one score per possible answer)
```

For diabetic retinopathy there are **5 possible answers** (grades 0–4), so ResNet
outputs 5 numbers. The biggest number is the model's prediction.

It isn't *programmed* with rules like "if you see a haemorrhage, output grade 3".
Instead it **learns** by being shown thousands of images whose correct answers we
already know. Each time it guesses wrong, an algorithm nudges its millions of
internal numbers ("weights") slightly so it would guess a little better next time.
Repeat over the whole dataset many times ("epochs") and it becomes accurate. This
process is **training**.

### Transfer learning (why we don't start from zero)
Training a CNN from scratch needs millions of images. We don't have that many eye
images, so we **start from a ResNet that was already trained on ImageNet** (1.2
million everyday photos of cats, cars, etc.). That model already understands generic
visual building blocks — edges, textures, blobs, shapes. We keep all of that and
just **re-train it for eyes**. This is called **transfer learning** and it's why the
model configs say `pretrained: true`. It's faster and works with far less data.

---

## 3. How a dataset "links into" the ResNet — the full pipeline

This is the core of your question. Let's trace **one image** from a raw download all
the way to a prediction, using **APTOS (a diabetic-retinopathy dataset) + ResNet-50**
as the running example. Each step maps to a real file in this repo.

```
RAW FILES              MANIFEST            DATASET           TRANSFORMS         MODEL
data/aptos/    →   manifest.csv    →   ClassificationDataset  →  tensor   →   ResNet-50   →  5 scores
train.csv          image_path,label      reads one image        [3,512,512]     (timm)
train_images/*.png    ,split
```

### Step 1 — Raw data on disk
You download APTOS. It gives you a folder of images plus a CSV:

```
data/aptos/train.csv           id_code, diagnosis      (diagnosis is 0..4)
data/aptos/train_images/0a1b.png, 0c2d.png, ...
```

Every dataset ships in its **own messy format** — different folder layouts, column
names, label encodings. We don't want that mess spread through the code.

### Step 2 — Normalise to a "manifest" (once per dataset)
`scripts/prepare_aptos.py` reads the raw APTOS files and writes a **standard**
`manifest.csv`:

```
image_path,label,split
train_images/0a1b.png,2,train
train_images/0c2d.png,0,val
...
```

- `image_path` — where the image is (relative to the dataset folder)
- `label` — the correct answer as a number (2 = "Moderate")
- `split` — which pile this image belongs to: `train` (learn from), `val`
  (tune/monitor), or `test` (final unbiased score)

**This is the "link".** Every dataset — EyePACS, Messidor, OCT2017, REFUGE, Duke —
gets its own `prepare_*.py` that produces this *same* manifest format. From here on,
the training code doesn't know or care which dataset it is. To add a new dataset you
only write a new prepare script; nothing downstream changes.

### Step 3 — The `DatasetSpec`: the single source of truth
[../src/occuwise/data/registry.py](../src/occuwise/data/registry.py) holds one
`DatasetSpec` per dataset. For APTOS it records:

```python
"aptos": DatasetSpec(task="classification", modality="fundus",
                     num_classes=5, class_names=["No DR","Mild","Moderate","Severe","Proliferative"],
                     primary_metric="quadratic_kappa", image_size=512)
```

This is where the crucial number **`num_classes=5`** lives. Remember it — it's what
tells ResNet to output 5 scores.

### Step 4 — The Dataset object turns a file into numbers
`ClassificationDataset` ([datasets.py](../src/occuwise/data/datasets.py)) reads the
manifest. When training asks it for image #47, it:
1. reads that PNG from disk,
2. runs it through **transforms**, and
3. returns `(image_tensor, label)`.

A **tensor** is just a multi-dimensional array of numbers — a CNN can only do maths
on numbers, not on PNG files. The image tensor has shape `[3, 512, 512]`: 3 colour
channels (Red, Green, Blue) × 512 pixels tall × 512 wide.

### Step 5 — Transforms: cleaning + augmentation
[transforms.py](../src/occuwise/data/transforms.py) prepares each image:
- **Circle-crop + CLAHE** — fundus photos have a black border and uneven lighting;
  we crop to the round eye area and equalise the contrast so lesions stand out.
- **Resize** to 512×512 so every image is the same size.
- **Normalize** — rescale pixel values to the range ResNet expects (the same range
  used when it was trained on ImageNet).
- **Augmentation** (training only) — randomly flip/rotate/brighten each image a
  little. The model never sees the exact same image twice, which prevents it from
  memorising and helps it generalise.

### Step 6 — The DataLoader batches images
Processing one image at a time is slow. The **DataLoader** stacks, say, 16 images
into one **batch**: a tensor of shape `[16, 3, 512, 512]`, plus 16 labels. The
`OphthalmologyDataModule` ([datamodule.py](../src/occuwise/data/datamodule.py)) also
**balances classes** here — grade 0 ("No DR") is far more common than grade 4, so it
samples rarer grades more often to stop the model just always guessing "No DR".

### Step 7 — Building ResNet, sized to the dataset
[models/registry.py](../src/occuwise/models/registry.py) creates the model:

```python
timm.create_model("resnet50", pretrained=True, num_classes=5)
```

`timm` is a library with hundreds of ready-made backbones. This line says: *give me
a ResNet-50 that already learned from ImageNet, but replace its final layer with a
fresh one that outputs 5 numbers.* **That `5` came straight from the APTOS
`DatasetSpec`.** Swap `data=oct2017` (4 classes) and the exact same code builds a
ResNet with a 4-output head instead. That's the whole "linking": the dataset's spec
supplies `num_classes`, and the model factory sizes ResNet's output to match.

### Step 8 — Forward pass, loss, learning
For each batch during training ([lit_classifier.py](../src/occuwise/engine/lit_classifier.py)):
1. **Forward:** the batch of images flows through ResNet → 16×5 scores.
2. **Loss:** a *loss function* compares the scores to the true labels and produces a
   single number measuring "how wrong". Lower is better.
3. **Backprop:** the training algorithm computes how to nudge every weight to reduce
   that loss, and updates them.

Do this for every batch, for every epoch, and ResNet gradually gets good at APTOS.

### Step 9 — Metrics, checkpoints
On the `val` split we compute a **metric** we actually care about clinically. For DR
grading that's **quadratic-weighted kappa** (rewards getting close — predicting
grade 3 when the truth is 4 is penalised less than predicting 0). The best-scoring
version of the model is saved to disk as a **checkpoint** (`.ckpt` file) in
`outputs/aptos/resnet50/`.

That's the complete journey: **raw files → manifest → tensors → ResNet → scores →
loss → trained checkpoint.** Every architecture (EfficientNet, DenseNet, ViT) and
every dataset flows through this same path — only the spec's numbers change.

### Segmentation is the same idea, different output
For REFUGE/Duke the answer isn't one label for the whole image but a **label for
every pixel** (which pixels are optic disc vs cup vs background). The model is a
**U-Net** instead of a plain classifier, and the output is a whole image-shaped mask.
Everything else (manifest → tensors → train → checkpoint) is identical.

---

## 4. What the different pieces are for

| You want to… | Use | File |
|--------------|-----|------|
| Convert a downloaded dataset to a manifest | `scripts/prepare_*.py` | `scripts/` |
| Train one model on one dataset | `occuwise.train` | `src/occuwise/train.py` |
| Train many combinations at once | `occuwise.train -m ...` (a "sweep") | same |
| See which combination won | `occuwise.compare` | `src/occuwise/compare.py` |
| Score a saved model on the test set | `occuwise.evaluate` | `src/occuwise/evaluate.py` |
| Package a model for production | `occuwise.export` | `src/occuwise/export.py` |
| Click-and-try in a browser (the POC) | `serving/webapp.py` | `serving/` |

**Architectures** you can pick: `resnet18`, `resnet50`, `efficientnet_b0/b4`,
`densenet121`, `vit_base` (classification); `unet` (segmentation).
**Datasets** you can pick: `aptos`, `eyepacs`, `messidor`, `oct2017`, `refuge_cls`,
`refuge_seg`, `duke_oct`.

---

## 5. How to use the project

### 5.0 One-time setup
Install a PyTorch build first (CPU shown; use the CUDA build if you have an NVIDIA
GPU — training is much faster on GPU), then the rest:

```bash
py -m venv .venv
.\.venv\Scripts\activate            # PowerShell/CMD on Windows
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
py -m pip install -e .
py -m pip install -r requirements.txt
```

### 5.1 Get some test images (30 seconds, no dataset needed)
```bash
py scripts/fetch_samples.py
```
This downloads a few real, openly-licensed fundus + OCT images into `data/samples/`
(with an `ATTRIBUTION.md` listing each source and licence). They show up as a
clickable gallery in the web app and can be run straight from the command line.

### 5.2 The fastest way to see it work: the web POC
This needs **no dataset and no training** — great for a first look.

```bash
py -m uvicorn serving.webapp:app --port 8080
```

Open <http://localhost:8080>:
1. Go to **Prepare & Test**, pick e.g. `aptos` + `resnet50`, click **Prepare model**
   (it downloads the pretrained weights the first time).
2. Go to **Analyze**, choose that model, then either **click a sample image** from the
   gallery or upload your own, and read the class probabilities.

Prefer the command line? The same thing without a browser:
```bash
py -m occuwise.predict --dataset aptos --arch resnet50 --samples
```

⚠️ A model prepared this way is **untrained** ("demo mode") — the pipeline runs and
returns structured output, but the answer is meaningless until you train it (below).
It exists to prove the upload→analyze→display flow end-to-end.

### 5.3 The real workflow: train a model
```bash
# 1. Download APTOS into data/aptos/ (see data/README.md for the link), then:
py scripts/prepare_aptos.py --root data/aptos      # makes manifest.csv

# 2. Train ResNet-50 on it
py -m occuwise.train model=resnet50 data=aptos

# ...or run a ready-made experiment recipe
py -m occuwise.train +experiment=dr_baseline
```

Training prints progress and saves the best checkpoint to `outputs/aptos/resnet50/`.

### 5.4 Compare architectures (the whole point)
Train several models, then rank them:

```bash
# Train 4 architectures on 2 datasets = 8 runs, one after another
py -m occuwise.train -m model=resnet50,efficientnet_b4,densenet121,vit_base data=aptos,eyepacs

# Build the leaderboard -> outputs/leaderboard.md
py -m occuwise.compare
```

`leaderboard.md` shows, per dataset, which architecture scored highest on that
dataset's clinical metric — i.e. *the most precise combination*.

### 5.5 Watch training live (optional)
```bash
py -m mlflow ui --backend-store-uri ./mlruns --port 5000     # open http://localhost:5000
```
MLflow logs every run's metrics so you can compare curves.

### 5.6 Use your trained model in the web app
Once you have a real checkpoint, serve it instead of the demo:
1. Start the web app, go to **Prepare & Test**.
2. Pick the dataset + architecture, and paste the checkpoint path into
   **Checkpoint path**, e.g. `outputs/aptos/resnet50/best-....ckpt`.
3. Prepare → now the **Analyze** page gives real predictions.

### 5.7 Package for production
```bash
py -m occuwise.export --ckpt outputs/aptos/resnet50/best.ckpt --dataset aptos --onnx --out models/production
py -m uvicorn serving.app:app --port 8080     # the production JSON API
```

---

## 6. Grad-CAM: seeing *why* the model decided

(Answering the earlier question.) A CNN is a **black box** — it outputs "grade 3" but
doesn't explain itself. In medicine that's not good enough: a clinician needs to
trust *why*.

**Grad-CAM** (Gradient-weighted Class Activation Mapping) produces a **heatmap** laid
over the input image showing **which regions most influenced the decision** — red =
"this area strongly pushed the model toward its answer", blue = "ignored".

- If the model predicts "severe DR" and Grad-CAM lights up the actual haemorrhages
  and lesions → the model is looking at the right things. Trust ↑.
- If it lights up the black border, a smudge, or the corner → the model is
  "cheating" on an artefact and its accuracy won't generalise. This is how Grad-CAM
  catches unreliable models *before* they reach patients.

Technically it uses the **gradients** flowing into the last convolutional layer to
weigh that layer's feature maps, giving a coarse "importance" map that's upscaled and
overlaid on the image (for ViT there's no conv layer, so we reshape the final
transformer tokens back into a grid instead).

**This is now implemented** (`src/occuwise/explain.py`):
- In the web app, tick **"Show Grad-CAM heatmap"** on the Analyze page — the heatmap
  appears under the probability bars.
- From the CLI, add `--explain` to save an overlay per image into `outputs/gradcam/`:
  ```bash
  py -m occuwise.predict --dataset aptos --arch resnet50 --samples --explain
  ```

Try it on an *untrained* demo model and you'll often see the heat land on the black
border or corners rather than the retina — a vivid demonstration of *why* Grad-CAM
matters: it exposes a model that isn't yet looking at real pathology. After training,
the heat should move onto vessels and lesions.

---

## 7. Mental model to remember

- **Dataset** = examples with known answers.
- **Manifest** = the universal adapter that makes every dataset look the same.
- **DatasetSpec** = the fact sheet (how many classes, which metric) that configures
  everything else — especially how many outputs the model needs.
- **ResNet / EfficientNet / … ** = the learnable function; we start from an
  ImageNet-pretrained one and re-train the last part for eyes.
- **Training** = repeatedly showing examples and nudging weights to reduce error.
- **Checkpoint** = a saved, trained model.
- **Leaderboard** = which (architecture × dataset) combo won.
- **Export + serving** = turning the winner into something a clinician can use.
- **Grad-CAM** = a heatmap that shows where the model looked, so we can trust it.
