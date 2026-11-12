
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    --retries=10 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app ./app
COPY worker ./worker
COPY similarity ./similarity
COPY models ./models

CMD ["python", "-m", "worker.main"]
