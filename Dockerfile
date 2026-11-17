FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .

RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

RUN mkdir -p /app/artifacts

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]