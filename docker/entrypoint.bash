#!/usr/bin/env bash

cd /home/user/CORSAIR
git pull
export OMP_NUM_THREADS=`nproc`
python3 evaluation-shapenet.py \
    --shapenet-root ./data/ShapeNetCore.v2.PC15k \
    --category chair \
    --model-ckpt ckpts/scannet_ret_chair \
    --n-models 100 \
    --n-poses-per-model 1 \
    --random-seed 0
cp results-shapenet-seed0-chair-100-1.csv host
cp poses-shapenet-seed0-chair-100-1.npz host
