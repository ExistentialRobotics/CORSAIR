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
