# goat_routing
Library to create optimal paths given energy, weather and robot conditions.

## Installation

### Local Installation

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

### Docker Installation

Build and run using Docker Compose:

```bash
# Build the Docker image
docker compose build
# (or use 'docker-compose build' for older Docker versions)

# Run the container
docker compose up -d

# Enter the container
docker compose exec goat_routing /bin/bash

# Stop the container
docker compose down
```

Or using Docker directly:

```bash
# Build the Docker image
docker build -t goat_routing:latest .

# Run the container
docker run -it goat_routing:latest

# Run with mounted volumes for development
docker run -it -v $(pwd)/src:/app/src goat_routing:latest /bin/bash
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Project Structure

```
goat_routing/
├── src/
│   └── goat_routing/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── LICENSE
```
