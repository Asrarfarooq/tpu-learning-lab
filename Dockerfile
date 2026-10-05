FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TPU_LAB_PROJECT=torchtpu-learning-lab \
    CLOUDSDK_CORE_PROJECT=torchtpu-learning-lab \
    TPU_LAB_HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir numpy "jax[cpu]"

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY modules/ /app/modules/
COPY recipes/ /app/recipes/
COPY scripts/ /app/scripts/
COPY README.md /app/README.md

EXPOSE 8080

CMD ["python3", "/app/backend/main.py"]
