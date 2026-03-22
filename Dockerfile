FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd --create-home appuser

WORKDIR /app

# Copy only the dependency file first to leverage Docker's layer caching
COPY requirements.txt /app/

# Combine system updates, targeted PyTorch install, pip installs, and cleanup into a single layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    # 1. Force CPU-only PyTorch to avoid pulling ~2GB+ of unnecessary CUDA dependencies
    && pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    # 2. Install the rest of your cleaned requirements
    && pip install --no-cache-dir -r /app/requirements.txt \
    # 3. Purge build essentials immediately to keep the image slim
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy application code and cache with explicit ownership to avoid duplicate layers
COPY --chown=appuser:appuser app/ /app/app/

# Drop root privileges
USER appuser

EXPOSE 7860

# Execute the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]