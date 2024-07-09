#!/usr/bin/env bash

cd /home/user/CORSAIR
pipenv shell
python3 evaluation-shapenet.py \
    --shapenet-root data/ShapeNetCore.v2.PC15k \
    --category chair \
    --model-ckpt ckpts/scannet_ret_chair \
    --n-models 100 \
    --n-poses-per-model 1 \
    --random-seed 1000
