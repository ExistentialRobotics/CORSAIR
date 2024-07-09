#!/usr/bin/env bash

xhost +si:localuser:user

docker run --rm -it \
  -v /dev/shm:/dev/shm:rw \
  -v /dev/char:/dev/char:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/block:/dev/block:rw \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v /dev/bus:/dev/bus:rw \
  -v /dev/serial:/dev/serial:rw \
  --net=host \
  -e DISPLAY \
  -v $HOME/.Xauthority:/home/user/.Xauthority:rw \
  --name corsair \
  erl/corsair
