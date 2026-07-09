# Running Occuwise AI in a Kaggle Notebook

Reference for training Occuwise models on Kaggle's free GPUs. This captures the
Kaggle-specific setup — everything else is covered by the repo `CLAUDE.md` and
`README.md`.

> **Run on a single T4.** Use accelerator **GPU T4 ×2** with **`train.devices=1`**.
> Two traps to avoid: multi-GPU DDP deadlocks at the first validation barrier (§6),
> and the **P100 does not work** — Kaggle's current PyTorch is built without Pascal
> (sm_60) kernels, so a P100 fails with `no kernel image is available` (§6). The T4
> is sm_75 and fully supported. This dataset is far too small to need two GPUs anyway.

---

## TL;DR — the working APTOS cells

```python
# 1. Install the package (run once per session)
!pip install -q -e .
!pip install -q -r requirements.txt

# 2. Build the manifest: READ images from the read-only Input mount,
#    WRITE the manifest to the writable working dir.
!python scripts/prepare_aptos.py \
    --root /kaggle/input/competitions/aptos2019-blindness-detection \
    --manifest /kaggle/working/aptos_manifest.csv

# 3. Train — single GPU, with the speed/duration optimizations baked in.
!python -m occuwise.train model=resnet50 data=aptos \
    data.data_root=/kaggle/input/competitions/aptos2019-blindness-detection \
    data.manifest=/kaggle/working/aptos_manifest.csv \
    train.batch_size=32 train.num_workers=4 \
    train.devices=1 \
    data.image_size=384 train.max_epochs=25
```

The last two lines are the optimizations — see §7 for what each one buys you.

---

## 1. Use the **Input** section, not a download

Attach the dataset through the notebook's **+ Add Input** panel (right sidebar) →
search `aptos2019-blindness-detection` → **Add**. It mounts **read-only** at:

```
/kaggle/input/aptos2019-blindness-detection/
    train.csv
    train_images/<id_code>.png
    test.csv
    test_images/...
    sample_submission.csv
```

**Why Input beats `kaggle competitions download`:** it's pre-cached (no re-fetch of
~10 GB each session), it doesn't consume your `/kaggle/working` output quota, and
it's faster to start. The download approach still works and is the portable form in
the README, but on Kaggle specifically, Input is the better choice.

### The one catch: `/kaggle/input` is read-only

The `prepare_*.py` scripts write a `manifest.csv`. By default they write it **into
the dataset root**, which fails on the read-only Input mount. That's the *only*
reason a plain `--root /kaggle/input/...` would crash.

**The fix** (already in the repo): every prepare script takes a `--manifest` flag
(`--out-dir` for REFUGE, which emits two manifests) so images are read from Input
while the manifest is written somewhere writable like `/kaggle/working`.

This works because the manifest stores image paths **relative to the root**, and
training accepts `data_root` and `manifest` as independent overrides. So you can
read images from one location and keep the manifest in another.

---

## 2. Per-dataset prepare + train commands

Every dataset follows the same shape. Swap the competition/dataset slug, the
`data=` config group, and (for classification) the model.

| Dataset   | `data=` group | Manifest flag | Notes |
|-----------|---------------|---------------|-------|
| APTOS     | `aptos`       | `--manifest`  | competition input |
| EyePACS   | `eyepacs`     | `--manifest`  | competition input (~larger) |
| OCT2017   | `oct2017`     | `--manifest`  | folder-per-class layout |
| Messidor  | `messidor`    | `--manifest`  | needs collated `index.csv` |
| REFUGE    | `refuge`      | `--out-dir`   | writes `manifest_cls.csv` + `manifest_seg.csv` |
| Duke OCT  | `duke_oct`    | `--manifest`  | segmentation; needs collated `index.csv` |

**Generic pattern:**

```python
# classification example (EyePACS)
!python scripts/prepare_eyepacs.py \
    --root /kaggle/input/<dataset-slug> \
    --manifest /kaggle/working/eyepacs_manifest.csv

!python -m occuwise.train model=resnet50 data=eyepacs \
    data.data_root=/kaggle/input/<dataset-slug> \
    data.manifest=/kaggle/working/eyepacs_manifest.csv
```

**REFUGE (two manifests via `--out-dir`):**

```python
!python scripts/prepare_refuge.py \
    --root /kaggle/input/<refuge-slug> \
    --out-dir /kaggle/working

# classification view uses manifest_cls.csv; segmentation uses manifest_seg.csv
!python -m occuwise.train model=resnet50 data=refuge \
    data.data_root=/kaggle/input/<refuge-slug> \
    data.manifest=/kaggle/working/manifest_cls.csv
```

---

## 3. Common training overrides

All Hydra overrides, appended to the `train` command:

```python
# Sweep architectures × datasets in one call (one process per combination)
!python -m occuwise.train -m \
    model=resnet50,efficientnet_b4,densenet121,vit_base data=aptos \
    data.data_root=/kaggle/input/aptos2019-blindness-detection \
    data.manifest=/kaggle/working/aptos_manifest.csv

# Named, ready-to-run combination
!python -m occuwise.train +experiment=dr_baseline

# Imbalanced DR grading — the go-to knobs
!python -m occuwise.train model=efficientnet_b4 data=aptos \
    train.loss=focal train.balance_classes=true \
    data.data_root=... data.manifest=...
```

Outputs (checkpoints, `leaderboard.md`, MLflow runs) land under `/kaggle/working`,
which **is** writable and downloadable from the notebook's Output tab. To keep the
best checkpoint after the session, add `/kaggle/working/outputs/...` to the notebook
output or save it as a Kaggle Dataset.

---

## 4. Expected warnings (safe to ignore)

Three warnings show up on Kaggle. Only one was actionable and it's already fixed:

1. **`isinstance(treespec, LeafSpec)` is deprecated** — inside PyTorch Lightning's
   own `_pytree.py`. Not our code; disappears when Kaggle bumps Lightning. Ignore.

2. **`AccumulateGrad node's stream does not match...`** — an internal PyTorch/DDP
   CUDA-stream note. Benign, not controllable from training code. Ignore.

3. **`It is recommended to use self.log(..., sync_dist=True)`** — **this one was
   real and is fixed.** On 2× T4, DDP gives each GPU half the batch; without syncing,
   logged losses/metrics (including the `val/quadratic_kappa` that the checkpoint and
   early-stopping monitor watch) reflected only rank 0's shard. All epoch-level logs
   in `src/occuwise/engine/lit_classifier.py` and `lit_segmenter.py` now pass
   `sync_dist=True`, so metrics reduce correctly across both GPUs. It's a no-op on
   single-GPU runs.

If you're on a newer clone and still see #3, pull the latest — the fix is in the
two `lit_*.py` modules.

---

## 5. Gotchas

- **Install torch to match Kaggle's CUDA.** Kaggle images ship a working GPU torch;
  `requirements.txt` says to install a CUDA-matched torch *first*. On Kaggle the
  pre-installed torch is usually fine — don't let `pip install -r requirements.txt`
  downgrade it. If it does, reinstall the Kaggle-matched wheel.
- **Enable the GPU accelerator** in the notebook settings (Settings → Accelerator →
  **GPU T4 ×2**), or training runs on CPU. Always add `train.devices=1` (see §6).
  **Do not use P100** — Kaggle's PyTorch lacks sm_60 kernels and it will crash (§6).
- **Internet access** must be **on** (Settings → Internet) for pretrained weight
  downloads from timm / HF hub on first run.
- **`/kaggle/working` is capped (~20 GB).** Using Input for images (not downloading)
  keeps you well under it; still prune old checkpoints on long sweeps.
- **Sessions are time-limited.** Long sweeps may exceed the GPU quota window — train
  one combination at a time, or checkpoint and resume.
- **Manifest is portable.** The relative-path design means the same
  `/kaggle/working/*_manifest.csv` keeps working even if you re-mount the Input under
  a different path, as long as you update `data.data_root` to match.

---

## 6. Run on a single GPU (the DDP-hang trap)

The config default is `devices: auto`, which on a **2× T4** notebook resolves to
**2-GPU DDP**. That combination **deadlocks**: training epoch 0 completes, then the
first validation/metric-sync barrier hangs forever. The symptom is deceptive —
**both GPUs and the CPU sit at 100%** (NCCL busy-waiting looks like real work) while
the progress bar is frozen and no new epoch starts. It can look like it's training
for 30+ minutes when it's actually stuck.

Root cause: the custom `WeightedRandomSampler` (class balancing) collides with
Lightning's automatic DistributedSampler replacement under DDP, so the ranks diverge
at the collective. As a bonus bug, DDP *silently overrides* the balanced sampler, so
even when it doesn't hang you lose class balancing.

**Run on a single T4:** accelerator **GPU T4 ×2** + `train.devices=1`. This uses one
of the two cards and skips DDP entirely, so no hang — and you get the balanced sampler
back. This dataset (~3.6k images) is nowhere near large enough to benefit from two
GPUs. The T4 is sm_75 with tensor cores, so keep `precision: 16-mixed` (fast there).

**Do NOT use the P100.** It looks tempting (a single GPU, so no DDP), but Kaggle's
current PyTorch build is compiled **without Pascal (sm_60) kernels**. A P100 loads,
prints a compatibility warning, then dies at the first CUDA op with:

```
CUDA error: no kernel image is available for execution on the device
The current PyTorch install supports ... sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
```

Making the P100 work would require downgrading PyTorch to a build old enough to ship
sm_60 — which then fights the Lightning/timm/torchmetrics versions in the image. Not
worth it; the T4 is the supported GPU on Kaggle. (`bf16-mixed` is separately
unsupported on both P100 and T4 — needs Ampere+; stick with `16-mixed`.)

**Diagnosing a suspected hang:** you can't run another cell while training blocks, but
the `/kaggle/working` file browser shows checkpoint timestamps. `last.ckpt` is
rewritten every epoch (`save_last=True`) — if its modified time keeps advancing, it's
healthy; if it's frozen, it's stuck. A `best-XX-…ckpt` filename also encodes the
epoch number `XX`.

---

## 7. Speed & duration optimizations

At the defaults, training is **data-loading bound**, not GPU-bound: the fundus
pipeline (circle-crop + CLAHE at 512 px) is CPU-heavy and Kaggle gives only ~4 vCPUs,
so the CPU pins at 100% while the GPU waits. The knobs that actually help:

| Override | Effect | Why |
|----------|--------|-----|
| `data.image_size=384` | **Big speedup** | Cuts per-image CPU transform + GPU compute; 384 px keeps DR lesions legible with minor accuracy cost (drop to 256 for a smoke test). |
| `train.max_epochs=25` | Fits the session window | 50 epochs at ~7–8 min each can exceed Kaggle's GPU quota; early stopping (`patience=8`) often ends sooner anyway. |
| `train.devices=1` | Removes the DDP hang (§6) | Also restores class balancing and feeds one GPU fully instead of starving two. |
| `train.batch_size=32` | Better GPU utilisation | Fits comfortably in 16 GB (P100/T4) at 384–512 px for ResNet-50. |
| `train.num_workers=4` | Matches Kaggle's ~4 vCPUs | More workers won't help — the CPU is the ceiling. Don't raise it. |

These are **run-time overrides, not config changes** — the committed defaults
(`image_size: 512`, `max_epochs: 50`, `devices: auto`) stay correct for full
training and production. Apply the optimizations on the command line for Kaggle:

```python
!python -m occuwise.train model=resnet50 data=aptos \
    data.data_root=/kaggle/input/competitions/aptos2019-blindness-detection \
    data.manifest=/kaggle/working/aptos_manifest.csv \
    train.batch_size=32 train.num_workers=4 \
    train.devices=1 \
    data.image_size=384 train.max_epochs=25
```

If it's *still* too slow, the deeper fix is caching the CLAHE/circle-crop output so
it runs once instead of every epoch — a code change in `src/occuwise/data/transforms.py`,
not a config knob.
