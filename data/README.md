# Datasets

> **Just want to try the models?** Run `py scripts/fetch_samples.py` to download a
> few real, openly-licensed fundus/OCT test images into `data/samples/` — no full
> dataset needed. They power the web app's sample gallery and `occuwise.predict --samples`.


Raw data is **never committed** (see `.gitignore`) — these are medical images with
licensing and privacy constraints. Download each dataset yourself, place it under
`data/<name>/`, then run the matching `scripts/prepare_*.py` to produce a standard
manifest CSV that the training pipeline consumes.

| Dataset | Task | Modality | Classes | Download | Prepare |
|---------|------|----------|---------|----------|---------|
| EyePACS | DR grading | fundus | 5 (0–4) | [Kaggle DR Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) | `prepare_eyepacs.py` |
| APTOS 2019 | DR grading | fundus | 5 (0–4) | [Kaggle APTOS](https://www.kaggle.com/c/aptos2019-blindness-detection/data) | `prepare_aptos.py` |
| Messidor / -2 | DR grading | fundus | 4 (0–3) | [ADCIS Messidor](https://www.adcis.net/en/third-party/messidor/) | `prepare_messidor.py` |
| OCT2017 (Kermany) | Disease cls | OCT | 4 (CNV/DME/DRUSEN/NORMAL) | [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj) | `prepare_oct2017.py` |
| REFUGE | Glaucoma cls **+** disc/cup seg | fundus | 2 / 3 | [Grand Challenge](https://refuge.grand-challenge.org/) | `prepare_refuge.py` |
| Duke OCT (DME) | Layer/fluid seg | OCT | 2+ | [Duke / Chiu 2015](http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm) | `prepare_duke_oct.py` |

## Manifest schema (produced by prepare scripts)

Classification — `manifest.csv`:
```
image_path,label,split
train_images/abc.png,2,train
```

Segmentation — `manifest_seg.csv`:
```
image_path,mask_path,split
images/001.png,masks/001.png,train
```

`split ∈ {train, val, test}`; paths are relative to the dataset root. Masks are
single-channel integer label maps (`0=background`, `1..K` classes).

## Notes on label harmonisation
- EyePACS/APTOS share the 0–4 International Clinical DR scale. Messidor uses a
  0–3 scale — keep them as separate leaderboards, or remap if you want a combined
  DR model (edit the prepare script and the `messidor` spec).
- For cross-dataset generalisation studies, train on one DR dataset and evaluate
  on another via `occuwise.evaluate` — a key robustness check before production.
