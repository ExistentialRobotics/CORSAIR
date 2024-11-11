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
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
CATEGORY=chair
CKPT=ckpts/scannet_ret_chair_best

python3 host/evaluation-scan2cad.py \
	--shapenet-pc15k-root /data/ShapeNetCore.v2.PC15k \
	--scan2cad-pc-root /data/Scan2CAD_pc \
	--shapenet-radegs-root /data/ShapeNet-RaDe-GS/ \
	--scan2cad-annotation-root /data/Scan2CAD_annotations \
	--category $CATEGORY \
	--checkpoint $CKPT \
       	--device cuda


