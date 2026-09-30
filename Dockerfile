FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -q -r requirements.txt

COPY src/ ./src/
COPY adapters/ ./adapters/
COPY bench/ ./bench/
COPY human_label/ ./human_label/
COPY dashboard/ ./dashboard/

EXPOSE 8020 8501

# Default: run the eval API. Override CMD for the dashboard or a one-shot run.
CMD ["uvicorn", "llmeval.api:app", "--host", "0.0.0.0", "--port", "8020"]
