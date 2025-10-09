#!/bin/bash

echo "Starting DAC baselines evaluation..."

echo "Building Docker image for the Python container..."
docker build -t baselines-evaluation -f baseline_evaluation_dockerfile .

echo "Running Python Docker container..."
docker run --name baseline-evaluator -d -v "$(pwd)":/app baselines-evaluation

echo "Docker container is up and running."
