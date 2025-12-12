# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy project files
COPY pyproject.toml /app/
COPY README.md /app/
COPY src/ /app/src/

# Install the package in editable mode with dev dependencies
RUN pip install -e ".[dev]"

# Copy tests
COPY tests/ /app/tests/

# Default command
CMD ["python", "-c", "import goat_routing; print(f'goat_routing v{goat_routing.__version__} installed successfully')"]
