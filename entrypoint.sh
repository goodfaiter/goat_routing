#!/bin/bash

# Install the package in editable mode
cd workspace
pip3 install -e .

# Run the command passed to the container
exec "$@"