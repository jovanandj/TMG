#!/bin/bash

# This script is used to start the docker containers for the GNN tuning application

#First create a network
echo "Creating network..."
docker network create hyper-tuning-net

#Build and run the MySQL container:
echo "Building MySQL container..."
docker build -t hyper-tuning-db -f Dockerfile-db .

echo "Running MySQL container..."
docker run --network=hyper-tuning-net --name db -p 3306:3306 -d hyper-tuning-db

# Wait for the MySQL server to initialize
echo "Waiting for MySQL server to initialize for 150 seconds..."
sleep 150

#Build and run the Python application container:
echo "Building Python application container..."
docker build -t hyper-tuning-app -f Dockerfile-gnn-hyper-tuning-app .

echo "Running Python application container..."
docker run --network=hyper-tuning-net --name gnn-tuning -d -v "$(pwd)":/app  hyper-tuning-app

echo "Containers are running"