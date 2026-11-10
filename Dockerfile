"""Dockerfile
-----------------
Multi‑stage Docker image for the LoanGuard project.
It builds a lean image with all Python dependencies and copies the source code.
Both the FastAPI API and the Streamlit dashboard run from this same image.
"""
# ---------- Build stage ----------
FROM python:3.11-slim AS base

# Install system packages required for building wheels (pandas, xgboost, lightgbm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non‑root user to run the app
RUN useradd -m appuser
WORKDIR /app

# Copy only requirement files first (leverages Docker cache)
COPY requirements.txt .

# Install Python dependencies in a virtual‑env‑like way (still system Python)
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Runtime stage ----------
FROM base AS runtime
COPY . .
# Ensure the app runs as non‑root user
USER appuser

# Expose ports (the actual numbers are supplied via environment variables at run time)
EXPOSE 8001 8502

# The entrypoint will be overridden by docker‑compose commands for each service.
CMD ["bash"]

