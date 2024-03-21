import torch
from torch import multiprocessing, cuda
import torch.nn.functional as F
from torch.backends import cudnn

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

import numpy as np
import importlib
import os
import tqdm
import imageio
import copy

from PIL import Image
cudnn.enabled = True

def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)

# RN50_dir = "./work_dirs/maskclip_w_r50_1024x512_cityscapes_multi_map/vis"
ViT16_dir = "./work_dirs/maskclip_w_vit16_1024x512_cityscapes_multi_map/vis"
IMG_dir = "./data/cityscapes/leftImg8bit/train"

img_ids = open('./data/cityscapes/train.txt').read().splitlines()
mask_postfix = '_leftImg8bit.png'

# stuff = (0 ~ 10)
# thing = (11 ~ 18)
labels = np.arange(19)

for idx, img_id in tqdm.tqdm(enumerate(img_ids)):

    img_name = f'{img_id}{mask_postfix}'

    mask = Image.open(os.path.join(ViT16_dir, img_name))
    mask = np.array(mask, dtype=np.uint8)

    img = Image.open(os.path.join(IMG_dir, img_name))
    img = np.array(img, dtype=np.uint8)
    
    pred = crf_inference_label(img, mask, n_labels=19)
    ref_mask = labels[pred]

    # base_mask = ViT_mask.copy()
    # base_mask = np.full_like(ViT_mask, 255)
    # for i in range(0, 19):
    #     # base_mask[RN_mask == i] = i
    #     base_mask[np.logical_and(RN_mask == i, ViT_mask == i)] = i
    
    imageio.imsave(os.path.join('./work_dirs/vit_crf', img_name), ref_mask.astype(np.uint8))
    # break