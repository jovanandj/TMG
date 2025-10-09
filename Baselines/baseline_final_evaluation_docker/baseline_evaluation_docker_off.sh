#!/bin/bash

echo "Stopping DAC baselines evaluation..."

echo "Stopping Docker container..."
docker stop baseline-evaluator

echo "Removing Docker container..."
docker rm baseline-evaluator

echo "Removing Docker image..."
docker rmi baselines-evaluation

echo "Docker container and image have been removed."