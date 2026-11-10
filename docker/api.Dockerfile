FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    --retries=10 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-api.txt

COPY app ./app
COPY mlops ./mlops
COPY similarity ./similarity

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]