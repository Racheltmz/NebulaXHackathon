FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# xgboost's Linux wheel needs the OpenMP runtime, which the slim image doesn't ship
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN useradd --create-home appuser
USER appuser

# Cloud Run injects $PORT (8080)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
