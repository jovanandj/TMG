#!/bin/bash

echo "Stopping DAC baselines tuning..."

echo "Stopping Docker container..."
docker stop baseline-tuner

echo "Removing Docker container..."
docker rm baseline-tuner

echo "Removing Docker image..."
docker rmi baselines-tuning

echo "Docker container and image have been removed."