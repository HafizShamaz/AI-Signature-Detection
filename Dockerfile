
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY shared/ ./shared/
COPY worker/ ./worker/
COPY detection/ ./detection/

# Create a non-root Linux user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m appuser && \
    mkdir -p /app/uploads && \
    chown -R appuser:appgroup /app/uploads

# Run the application as this user
USER appuser


EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
