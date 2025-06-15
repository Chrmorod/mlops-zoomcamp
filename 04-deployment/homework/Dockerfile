FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY score.py .

RUN mkdir -p predictions
RUN pip install --no-cache-dir pandas pyarrow scikit-learn==1.5.0

ENTRYPOINT  ["python", "score.py"]
