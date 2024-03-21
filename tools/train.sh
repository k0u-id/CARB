#!/usr/bin/env bash

CONFIG_FILE=configs/carb/cityscapes_carb_dual.py
GPUS=2
PORT=23002

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG_FILE --seed 589482026 --deterministic --launcher pytorch ${@:3}