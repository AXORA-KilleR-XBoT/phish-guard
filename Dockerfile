# Development image for the phish-guard API (no authentication, no HTTPS: see README).
# The model is trained while the image is built, so building needs internet access to download the
# data set, and the finished image already contains a model together with its checksum file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the pinned runtime dependencies first, so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY phishguard.py api.py download_data.py ./

# download_data.py refuses the file unless its SHA-256 equals the value pinned in the script.
RUN python download_data.py data/sms.tsv \
 && python phishguard.py train --data data/sms.tsv --model model.joblib \
 && chmod 0644 model.joblib model.joblib.sha256

# The server runs without root rights. The model files belong to root: the server user can read them but not change them.
RUN useradd --system --no-create-home --uid 10001 appuser
USER appuser

ENV MODEL_PATH=/app/model.joblib
EXPOSE 8000
# 0.0.0.0 is needed inside the container; publish the port on localhost only: -p 127.0.0.1:8000:8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
