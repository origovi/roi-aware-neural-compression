#!/bin/bash

# PLEASE CHANGE THE DATASET ROUTE AND THE ROUTE TO THE ROOT OF THIS REPO
# The two folders will be shared with the container so no copies will be made
# Port 6007 is bridged with the intention of launching tensorboard on this port
DATASET_ROUTE="/home/gorr/kitti_2d"
IMAGE_COMPRESSION_ROUTE="/home/gorr/tod_compress/image_compression"

# SHOULD NOT MODIFY BELOW THIS LINE
# ------------------------------------------------------------------------------

CONTAINER_NAME="oriol"

SHM_SIZE=$(df --output=size -h /dev/shm | tail -n 1)

docker build -t $CONTAINER_NAME .

docker run --gpus all -it --privileged \
    -v $IMAGE_COMPRESSION_ROUTE:/workspace/image_compression \
    -v $DATASET_ROUTE:/workspace/kitti_2d \
    --shm-size $SHM_SIZE \
    -p 6007:6007 \
    $CONTAINER_NAME bash -c "
    # Wait for the mount to be ready and ensure the requirements.txt file is present
    if [ -f /workspace/image_compression/requirements.txt ]; then
      echo 'Installing dependencies from requirements.txt...'
      pip install -r /workspace/image_compression/requirements.txt
    else
      echo 'requirements.txt not found in $CONTAINER_DIR.'
    fi
    # Start the container interactively after installation
    bash
  "
