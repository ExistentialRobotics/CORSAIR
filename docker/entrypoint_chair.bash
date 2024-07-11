#!/usr/bin/env bash

set -x
set -e
cd /home/user/CORSAIR
git pull
export OMP_NUM_THREADS=`nproc`
CATEGORY=chair
CKPT=ckpts/scannet_ret_chair
N_MODELS=100
N_POSES_PER_MODEL=1
SEED=0
python3 evaluation-shapenet.py \
    --shapenet-root ./data/ShapeNetCore.v2.PC15k \
    --category $CATEGORY \
    --model-ckpt $CKPT \
    --n-models $N_MODELS \
    --n-poses-per-model $N_POSES_PER_MODEL \
    --random-seed $SEED
cp results-shapenet-seed$SEED-$CATEGORY-$N_MODELS-$N_POSES_PER_MODEL.csv host
cp poses-shapenet-seed$SEED-$CATEGORY-$N_MODELS-$N_POSES_PER_MODEL.npz host
python3 compute_metrics_shapenet.py \
    --categories $CATEGORY \
    --n-models $N_MODELS \
    --n-poses-per-model $N_POSES_PER_MODEL \
    --random-seed $SEED
