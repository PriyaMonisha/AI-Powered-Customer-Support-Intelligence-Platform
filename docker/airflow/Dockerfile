# filename: docker/airflow/Dockerfile
# purpose:  Airflow image for CSIP DAGs (csip_etl, csip_drift_monitor, csip_retrain,
#           csip_model_report) — LocalExecutor. Same image used for airflow-init,
#           airflow-webserver, and airflow-scheduler (different `command:` per service).
FROM apache/airflow:2.9.3-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow
ARG AIRFLOW_VERSION=2.9.3
ARG PYTHON_VERSION=3.11
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --constraint "${CONSTRAINT_URL}" -r /requirements.txt
