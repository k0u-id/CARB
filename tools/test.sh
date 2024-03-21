CONFIG_FILE=configs/carb/cityscapes_carb_dual.py
CHECKPOINT_FILE=work_dirs/cityscapes_carb_dual/latest.pth
OUTPUT_DIR=work_dirs/cityscapes_carb_dual/vis

CUDA_VISIBLE_DEVICES=0 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${OUTPUT_DIR}