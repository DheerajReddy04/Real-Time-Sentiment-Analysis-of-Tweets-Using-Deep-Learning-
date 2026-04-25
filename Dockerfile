# Dockerfile
FROM python:3.11-slim

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and server
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy server code
COPY serve.py /app/serve.py

# Copy your pretrained model files into the image.
# Make sure you have a folder named distilbert_tweeteval_boosted at build time.
COPY distilbert_tweeteval_boosted /app/model

# Expose port used by Vertex (container must listen on 8080 for Vertex)
ENV PORT=8080
ENV MODEL_DIR=/app/model

# Run with Uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
