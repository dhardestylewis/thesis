# Use official PyTorch image with CUDA 12.1 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install necessary system packages for downloading data
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    awscli \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY causal_cfm_cvae.py .
COPY evaluate_causal_fixes.py .

# Entrypoint to run the script. Expected to have PANEL_PATH and OUT_DIR injected as env variables via docker run
CMD ["python", "causal_cfm_cvae.py"]
