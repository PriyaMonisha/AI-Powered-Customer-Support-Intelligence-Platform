# filename: Dockerfile
# purpose:  CSIP FastAPI serving layer
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, in its own layer BEFORE COPY requirements.txt: PyPI's
# default torch==2.3.0 wheel bundles ~2.5GB of CUDA deps (nvidia-cublas-cu12 etc.)
# unused for CPU-only serving (locked GPU table: FastAPI = no GPU). A bare
# "torch==2.3.0" specifier in requirements.txt matches this "+cpu" local version
# per PEP 440, so pip skips reinstalling it. Placed before the requirements.txt
# COPY so this layer's cache survives requirements.txt edits.
RUN pip install --no-cache-dir torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
