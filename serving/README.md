# Serving

Two supported paths from a trained checkpoint to production:

## 1. ONNX + FastAPI (this folder — simplest)

```bash
# Export a checkpoint to ONNX + model card
py -m occuwise.export --ckpt outputs/aptos/efficientnet_b4/best.ckpt --dataset aptos --onnx --torchscript --out models/production

# Serve
py -m uvicorn serving.app:app --port 8080
curl -F "file=@sample_fundus.jpg" http://localhost:8080/predict
```

## 2. High-throughput options

| Need | Use |
|------|-----|
| Single model, low ops overhead | This FastAPI + ONNX Runtime service |
| Many models, versioning, A/B, GPU batching | **NVIDIA Triton Inference Server** (ONNX or TensorRT backend) |
| PyTorch-native, workflow handlers | **TorchServe** |
| Lowest latency on NVIDIA GPU | Export ONNX → **TensorRT** engine |

Triton is the recommended production target for a fleet of these models: drop each
exported `model.onnx` into the Triton model repository, enable dynamic batching, and
route by model name. The `model_card.json` in each export folder carries the
preprocessing + class metadata a Triton client/ensemble needs.

## Clinical / regulatory notes
- These models are **decision support** (screening triage), not autonomous diagnosis.
- Keep an audit log of every inference (input hash, model version, output, timestamp).
- Version models via MLflow Model Registry; never overwrite a deployed model in place.
- Validate on the **local patient population** before deployment — public-dataset
  performance does not transfer automatically (camera type, demographics, prevalence).
