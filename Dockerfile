# PrognosticAI RUL API
#
#   docker build -t prognosticai-api .                          # serves XGBoost or LSTM champions
#   docker build -t prognosticai-api --build-arg WITH_TORCH=0 . # smaller, XGBoost champions only
#   python export_model.py                                      # writes serving/champion
#   docker run -p 8000:8000 -v "$PWD/serving/champion:/models/champion:ro" prognosticai-api
#
# The model is mounted, not baked in, so a new champion needs no rebuild.

FROM python:3.12-slim

ARG WITH_TORCH=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-serve.txt .
RUN if [ "$WITH_TORCH" = "1" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi \
 && pip install -r requirements-serve.txt

# Only the modules the packaged models and the API import
COPY config.py data_processor.py lstm_model.py rul_predictor.py serve.py ./

RUN useradd --create-home --uid 10001 api
USER api

ENV PROGNOSTIC_MODEL_URI=/models/champion \
    PROGNOSTIC_MODEL_DIR=/tmp/models \
    MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
