FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install numpy scipy scikit-learn fastapi "uvicorn[standard]" python-multipart google-genai pillow matplotlib pydantic
COPY ml ./ml
COPY gemini ./gemini
COPY app ./app
# model + feature cache + metrics are baked in so the demo has no runtime data dependency
COPY results/best_S3.pt results/features.npz results/metrics.json ./results/
COPY results/metrics_gemini.json* ./results/
ENV GV_CKPT=results/best_S3.pt GV_FEATURES=results/features.npz PORT=8080
EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
