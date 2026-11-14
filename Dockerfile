FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bike_demand ./bike_demand
COPY static ./static
COPY artifacts/model.joblib ./artifacts/model.joblib
COPY reports/metrics.json ./reports/metrics.json
RUN useradd --create-home appuser
USER appuser
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn bike_demand.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
