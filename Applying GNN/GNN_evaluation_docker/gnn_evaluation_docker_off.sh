#!/bin/bash

# This script is used to stop the docker container for the evaluation of the GNN model.

echo "Stopping the docker container..."
docker stop gnn_eval

echo "Removing the docker container..."
docker rm gnn_eval

echo "Container is removed..."