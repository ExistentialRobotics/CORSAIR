#!/usr/bin/env bash
# Copyright 2024 Qiaojun Feng, Sai Jadhav, Tianyu Zhao, Zhirui Dai, K. M. Brian Lee, Nikolay Atanasov, UC San Diego. 

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
#xhost +si:localuser:user
xhost +

ENTRYPOINT=${ENTRYPOINT:=/home/user/CORSAIR/entrypoint_chair.bash}  # or entrypoint_table.bash
docker run --rm -it \
  --runtime=nvidia \
  --gpus all \
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
  --entrypoint $ENTRYPOINT \
  erl/corsair
