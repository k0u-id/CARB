_base_ = [
    '../_base_/models/carb.py', '../_base_/datasets/camvid_w.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_12k_lr_0.005.py'
]

suppress_labels = list(range(0, 11))
model = dict(
    # pretrained='open-mmlab://resnet101_v1c',  # add this two line for change backbone from
    # backbone=dict(depth=101),                 # RN50 to RN101
    decode_head=dict(
        num_classes=11,
        text_categories=11,
        text_embeddings_path='pretrain/camvid2_ViT16_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        coeff=1,                        
        warmup_iter=4000,                      # when to start using self-mask
        patch_size=(512, 256),
        resize_rate=2,
        resize_offset=0.5,
        adaptive=True,                          
        get_train_mask=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_masked=True, loss_weight=1.0),
    ),
)


find_unused_parameters=True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (720, 960)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(960, 720), ratio_range=(1.0, 4.0)),
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
        img_scale=(960, 720),
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
        img_dir='img/train', # added line
        ann_dir='mask/train', # added line
        img_labels='metadata/camvid/labels.npy',
        pipeline=train_pipeline
    ),
    val=dict(
        img_dir='img/val', # added line
        ann_dir='mask/val', # added line
        pipeline=test_pipeline),
    test=dict(
        img_dir='img/val', # added line
        ann_dir='mask/val', # added line
        pipeline=test_pipeline)
)