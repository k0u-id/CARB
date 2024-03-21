# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WeakCamVidDataset(CustomDataset):
    """Cityscapes dataset.
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ('Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 'SignSymbol',
               'Fence', 'Car', 'Pedestrian', 'Bicyclist')
    PALETTE = [[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
               [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
               [64, 0, 128], [64, 64, 0], [0, 128, 192]]

    def __init__(self, img_labels, **kwargs):
        super(WeakCamVidDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        self.img_labels = np.load(img_labels, allow_pickle=True).item()

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results['img_label'] = self.img_labels[img_info['filename'].replace(self.img_suffix, '')]
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        results['img_label'] = self.img_labels[img_info['filename'].replace(self.img_suffix, '')]
        return self.pipeline(results)
