# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WeakWilddash2Dataset(CustomDataset):
    """Cityscapes dataset.
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ('Ego_vehicle', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Guard_rail',
               'Pole', 'Traffic_light', 'Traffic_sign', 'Vegetation', 'Terrain',
               'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Motorcycle',
               'Bicycle', 'Pickup', 'Van', 'Billboard', 'Streetlight', 'Roadmarking')

    PALETTE = [[0, 20, 50], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [180, 165, 180], [152, 152, 152], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
               [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 230], [119, 11, 32],
               [40, 0, 100], [0, 40, 120], [174, 64, 67], [210, 170, 100], [196, 176, 128]]


    def __init__(self, img_labels, **kwargs):
        super(WeakWilddash2Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_labelTrainIds.png',
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
