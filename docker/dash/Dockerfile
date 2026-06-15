# filename: docker/dash/Dockerfile
# purpose:  CSIP Plotly Dash dashboard (Section 15)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/dash/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY src ./src
COPY dash_app ./dash_app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8050
CMD ["gunicorn", "dash_app.app:server", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:8050", "--timeout", "30"]
