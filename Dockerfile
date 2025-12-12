# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Update CA certificates
RUN update-ca-certificates

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel || \
    python -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --upgrade pip setuptools wheel

# Copy project files
COPY pyproject.toml /app/
COPY README.md /app/
COPY src/ /app/src/
COPY tests/ /app/tests/

# Install development dependencies only (not the package itself since it's mounted)
RUN python -m pip install pytest pytest-cov black flake8 mypy || \
    python -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pytest pytest-cov black flake8 mypy

# Default command
CMD ["python", "-c", "import sys; sys.path.insert(0, '/app/src'); import goat_routing; print(f'goat_routing v{goat_routing.__version__} ready for development')"]
