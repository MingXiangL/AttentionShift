# dataset settings
dataset_type = 'VOCDatasetInstance'
data_root = '/home/LiaoMingxiang/Dataset/'
# data_root = '/home/lmx/Dataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsC', with_bbox=False, with_center=True),
    dict(type='RandomFlipC', flip_ratio=0.5),
    dict(type='ResizeC',
        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                    (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                    (736, 1333), (768, 1333), (800, 1333)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleC'),
    dict(type='Collect', keys=['img', 'gt_labels', 'gt_centers']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
aug_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1333, 600), (1333, 400), (1000, 800), (1000, 600), (1000, 400)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_train_12 = dict(
    type='RepeatDataset',
    times=4,
    dataset=dict(
        type='VOCCocoDatasetPoi',
        ann_file=data_root + 'VOC2012/Annotations_coco/center_points/gt_center_train2012.json',
        img_prefix=data_root + 'VOC2012/JPEGImages/',
        pipeline=train_pipeline
    )
)
custom_imports = dict(imports=['mmdet_plugins'], allow_failed_imports=False)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=data_train_12,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=aug_test_pipeline))
evaluation = dict(interval=1, metric='mAP_Segm')
