_base_ = [
    '../_base_/models/carb.py', '../_base_/datasets/cityscapes_w.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_30k_lr_0.005.py'
]

suppress_labels = list(range(0, 19))
model = dict(
    # pretrained='open-mmlab://resnet101_v1c',  # add this two line for change backbone from
    # backbone=dict(depth=101),                 # RN50 to RN101
    decode_head=dict(
        num_classes=19,
        text_categories=19,
        text_embeddings_path='pretrain/city_carb_ViT16_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        # coeff=0.03,                             # uncomment this with 'adaptive=False', fixed weight
        warmup_iter=16000,                      # when to end warm-up training
        patch_size=(512, 256),
        resize_rate=1,                          # resize equation = resize_rate * rnd + resize_offset
        resize_offset=1,                        # minimum resize ratio
        adaptive=True,                          # you can change this to False, then you can use fixed ratio
        get_train_mask=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_masked=True, loss_weight=1.0),
    ),
)

find_unused_parameters=True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         meta_keys = ['filename', 'ori_filename', 'ori_shape',
                      'img_shape', 'pad_shape', 'scale_factor', 'flip',
                      'flip_direction', 'img_norm_cfg', 'img_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                 meta_keys=['filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'img_label']),
        ])
]
data = dict(
    samples_per_gpu=4,
    train=dict(
        img_dir='leftImg8bit/train', # added line
        ann_dir='gtFine/train', # added line
        img_labels='metadata/cityscapes/labels.npy',
        pipeline=train_pipeline
    ),
    val=dict(
        img_dir='leftImg8bit/val', # added line
        ann_dir='gtFine/val', # added line
        pipeline=test_pipeline),
    test=dict(
        img_dir='leftImg8bit/val', # added line
        ann_dir='gtFine/val', # added line
        pipeline=test_pipeline)
)