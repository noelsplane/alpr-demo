# ALPR Surveillance System - Production Docker Image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create application user
RUN groupadd -r alpr && useradd -r -g alpr alpr

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3-pip \
    build-essential cmake pkg-config \
    libopencv-dev python3-opencv \
    v4l-utils uvcdynctrl \
    libpq-dev \
    git curl wget \
    linux-modules-extra-generic \
    usb-modeswitch usb-modeswitch-data \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip3 install --no-cache-dir \
    gunicorn \
    psycopg2-binary

# Copy application code
COPY api/ /app/api/
COPY scripts/ /app/scripts/
COPY .env.docker /app/.env

# Create necessary directories
RUN mkdir -p /var/lib/alpr-surveillance/{videos,images} \
    && mkdir -p /var/log/alpr-surveillance \
    && mkdir -p /app/models

# Set permissions
RUN chown -R alpr:alpr /app /var/lib/alpr-surveillance /var/log/alpr-surveillance

# Download YOLO models
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && mv /root/.ultralytics /app/models/ \
    && chown -R alpr:alpr /app/models

# Switch to non-root user
USER alpr

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/surveillance/status || exit 1

# Start command
CMD ["gunicorn", "api.main:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "300"]