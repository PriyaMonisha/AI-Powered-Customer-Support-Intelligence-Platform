# filename: docker/mlflow/Dockerfile
# purpose:  Standalone MLflow tracking server for CSIP (Section 14)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow==2.13.2
WORKDIR /mlflow
EXPOSE 5000
# --gunicorn-opts "--timeout 120": cold reads of mlruns/ (49 runs) over a Windows/WSL2
# bind mount can exceed gunicorn's 30s default worker timeout on the first
# search_runs() call, causing repeated SIGKILL+restart cycles before the OS file
# cache warms up. 120s gives the first cold read enough headroom.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "/mlflow/mlruns", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--gunicorn-opts", "--timeout 120"]
