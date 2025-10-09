#!/bin/bash

echo "Starting DAC baselines tuning..."

echo "Building Docker image for the Python container..."
docker build -t baselines-tuning -f baselines_tuning_dockerfile .

echo "Running Python Docker container..."
docker run --name baseline-tuner -d -v "$(pwd)":/app baselines-tuning

echo "Docker container is up and running."