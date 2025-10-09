#!/bin/bash

# This script is used to start the docker container for the GNN sequential tuning application

# Build and run the Python application container:
echo "Building Python application container..."
docker build -t sequential-hyper-tuning-app -f Dockerfile-gnn-sequential--hyper-tuning-app .

echo "Running Python application container..."
docker run --name sequential-gnn-tuning -d -v "$(pwd)":/app sequential-hyper-tuning-app

echo "Container is running"