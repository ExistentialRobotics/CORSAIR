#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
#xhost +si:localuser:user
xhost +

docker run --rm -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics \
  -v /dev/shm:/dev/shm:rw \
  -v /dev/char:/dev/char:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/block:/dev/block:rw \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v /dev/bus:/dev/bus:rw \
  -v /dev/serial:/dev/serial:rw \
  -v /dev:/dev:rw \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v $HOME/.Xauthority:/home/user/.Xauthority:rw \
  -v $SCRIPT_DIR/..:/home/user/CORSAIR/host:rw \
  --name corsair \
  erl/corsair
