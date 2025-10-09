#!/bin/bash

# This script is used to start the docker container for the evaluation of the GNN model.

echo "Building the docker image..."
docker build -t gnn_final_evaluation -f Dockerfile-gnn-evaluation-app .

echo "Running the docker container..."
docker run --name gnn_eval -d -v "$(pwd)":/app gnn_final_evaluation

echo "Container is running..."