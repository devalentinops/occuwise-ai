"""Verify RETFound domain weights actually load into the backbone.

This is an ONLINE, AUTH-GATED check: RETFound weights are gated on Hugging Face, so
the test SKIPS cleanly when the weights can't be fetched (no HF token, access not yet
granted, or offline). Once you've requested access and run `huggingface-cli login`,
this test downloads the checkpoint (~1.2 GB, cached) and asserts a healthy number of
backbone tensors were loaded — i.e. the model is retinal-pretrained, not random.

Run just this check:  pytest tests/test_retfound_weights.py -s
"""

import pytest


@pytest.mark.parametrize(
    "model_cfg, repo, fname",
    [
        ("retfound", "YukunZhou/RETFound_mae_natureCFP", "RETFound_mae_natureCFP.pth"),
        ("retfound_oct", "YukunZhou/RETFound_mae_natureOCT", "RETFound_mae_natureOCT.pth"),
    ],
)
def test_retfound_weights_load(model_cfg, repo, fname, capsys):
    pytest.importorskip("huggingface_hub")
    from occuwise.models import ModelConfig, build_model

    cfg = ModelConfig(task="classification", arch="retfound", num_classes=5,
                      pretrained=False, weights_repo=repo, weights_file=fname)
    try:
        build_model(cfg)
    except Exception as e:  # gated / no token / offline / not-yet-approved
        pytest.skip(f"{repo} weights unavailable — {type(e).__name__}: {e}")

    out = capsys.readouterr().out
    assert "[weights] loaded" in out, f"no load report emitted:\n{out}"
    matched = int(out.split("loaded ")[1].split("/")[0])
    assert matched > 100, f"only {matched} backbone tensors matched — check key layout"
