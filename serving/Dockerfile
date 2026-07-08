# Minimal CPU inference image. For GPU, base on nvidia/cuda + onnxruntime-gpu.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Serving needs only the lightweight inference stack, not the training deps.
RUN pip install --no-cache-dir \
    onnxruntime>=1.18 fastapi>=0.111 "uvicorn[standard]>=0.30" \
    python-multipart>=0.0.9 pillow>=10.3 opencv-python-headless>=4.9 numpy>=1.26

COPY serving/ /app/serving/
# Bake the exported model in, or mount it at runtime via -v host_dir:/app/models/production
COPY models/ /app/models/

ENV MODEL_DIR=/app/models/production
EXPOSE 8080
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
