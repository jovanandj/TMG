#!/bin/bash

# This script is used to stop and remove the docker containers/images for the GNN tuning application

#Stop and remove the containers
echo "Stopping containers..."
docker stop db gnn-tuning

echo "Removing containers..."
docker rm db gnn-tuning

#Remove the images
echo "Removing images..."
docker rmi hyper-tuning-db hyper-tuning-app

#Remove the network
echo "Removing network..."
docker network rm hyper-tuning-net

