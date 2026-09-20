FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "tiny_recursive_model.app:app", "--host", "0.0.0.0", "--port", "8000"]
