#!/usr/bin/env bash

set -x
set -e
cd /home/user/CORSAIR
git pull
export OMP_NUM_THREADS=`nproc`
CATEGORY=table
CKPT=ckpts/scannet_pose_table_best
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
cp results-shapenet-seed0-$CATEGORY-$N_MODELS-$N_POSES_PER_MODEL.csv host
cp poses-shapenet-seed0-$CATEGORY-$N_MODELS-$N_POSES_PER_MODEL.npz host
python3 compute_metrics_shapenet.py \
    --categories $CATEGORY \
    --n-models $N_MODELS \
    --n-poses-per-model $N_POSES_PER_MODEL \
    --random-seed $SEED
