"""Download a small set of openly-licensed ophthalmology test images.

These are real fundus photographs and OCT scans from Wikimedia Commons (public
domain / Creative Commons), so you can exercise the models from the web app or CLI
without downloading a full research dataset. Files land in:

    data/samples/fundus/*
    data/samples/oct/*
    data/samples/ATTRIBUTION.md     (source + license for every file)
    data/samples/samples.json       (machine-readable index used by the web app/CLI)

Usage:  py scripts/fetch_samples.py

Each entry records the Commons filename, modality, a suggested dataset/model to try
it with, and its license. Downloads that fail or aren't valid images are skipped,
so a flaky URL never breaks the run.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

# Wikimedia serves the original file via this stable redirect endpoint.
FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
# Commons blocks requests without a descriptive User-Agent.
UA = "OccuwiseAI-POC/0.1 (research sample fetcher; contact dev@zeraxo.com)"

# Curated, openly-licensed images. `suggest` = a good (dataset, arch) to demo with.
SAMPLES = [
    # ---------------- Fundus (retinal photographs) ----------------
    {"file": "Fundus photograph of normal left eye.jpg", "modality": "fundus",
     "label_hint": "normal", "suggest": ["aptos", "resnet50"], "license": "CC0"},
    {"file": "Fundus photograph of normal right eye.jpg", "modality": "fundus",
     "label_hint": "normal", "suggest": ["aptos", "efficientnet_b4"], "license": "CC0"},
    {"file": "Fundus photograph-normal retina EDA06.JPG", "modality": "fundus",
     "label_hint": "normal", "suggest": ["eyepacs", "densenet121"], "license": "Public Domain"},
    {"file": "Diabetic retinopathy laser surgery-NEI.jpg", "modality": "fundus",
     "label_hint": "diabetic retinopathy (post-laser)", "suggest": ["aptos", "resnet50"],
     "license": "Public Domain (NEI)"},
    {"file": "Optic disc edema and haemorrhage.jpg", "modality": "fundus",
     "label_hint": "optic disc oedema / haemorrhage", "suggest": ["refuge_cls", "resnet50"],
     "license": "CC BY-SA"},
    {"file": "ONH-Glaukom-MRA.jpg", "modality": "fundus",
     "label_hint": "glaucomatous optic nerve head", "suggest": ["refuge_cls", "resnet50"],
     "license": "CC BY-SA"},
    # ---------------- OCT (cross-sectional B-scans) ----------------
    {"file": "SD-OCT Optic Disc Cross-Sections.png", "modality": "oct",
     "label_hint": "optic disc OCT", "suggest": ["oct2017", "densenet121"], "license": "CC BY-SA"},
    {"file": "Macular pigment optical density after ERM peeling.png", "modality": "oct",
     "label_hint": "macular OCT (post ERM peel)", "suggest": ["oct2017", "resnet50"],
     "license": "CC BY"},
    {"file": "SD-OCT Corneal Cross-Section.png", "modality": "oct",
     "label_hint": "corneal OCT", "suggest": ["oct2017", "resnet18"], "license": "CC BY-SA"},
]


def _download(filename: str, dest: Path) -> bool:
    url = FILEPATH + urllib.parse.quote(filename)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        # Validate it's a real, decodable image.
        from PIL import Image

        with Image.open(dest) as im:
            im.verify()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  skip  {filename}  ({e})")
        if dest.exists():
            dest.unlink()
        return False


def main():
    root = Path("data/samples")
    (root / "fundus").mkdir(parents=True, exist_ok=True)
    (root / "oct").mkdir(parents=True, exist_ok=True)

    index = []
    attribution = ["# Sample image attribution\n",
                   "All images from Wikimedia Commons. Verify each license on its "
                   "source page before any non-research use.\n"]

    for s in SAMPLES:
        safe = s["file"].replace(" ", "_")
        dest = root / s["modality"] / safe
        print(f"downloading {s['file']} ...")
        if not _download(s["file"], dest):
            continue
        rel = dest.relative_to(root).as_posix()
        source = FILEPATH + urllib.parse.quote(s["file"])
        page = "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(s["file"])
        index.append({
            "path": rel,
            "modality": s["modality"],
            "label_hint": s["label_hint"],
            "suggest_dataset": s["suggest"][0],
            "suggest_arch": s["suggest"][1],
            "license": s["license"],
            "source": page,
        })
        attribution.append(f"- **{rel}** — {s['label_hint']} — {s['license']} — [{s['file']}]({page})")

    (root / "samples.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (root / "ATTRIBUTION.md").write_text("\n".join(attribution) + "\n", encoding="utf-8")
    print(f"\n{len(index)} images ready under {root}/")
    print(f"index -> {root/'samples.json'}   attribution -> {root/'ATTRIBUTION.md'}")


if __name__ == "__main__":
    main()
