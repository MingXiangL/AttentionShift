_base_ = [
    '../_base_/datasets/coco_detection_center_points.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = '/home/LiaoMingxiang/Workspace/pretrain/mae_vit_small_800e.pth'
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='LossWeightAdjustHook', start_epoch=-1, priority='NORMAL'),
]
num_classes=80

model = dict(
    type='FasterRCNNPointSupAlign',
    pretrained=pretrained,
    pos_mask_thr=0.6,
    neg_mask_thr=0.1,
    num_mask_point_gt=20,
    corr_size=21,
    backbone=dict(
        type='VisionTransformerDet',
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        out_indices=(3, 5, 7, 11),
        learnable_pos_embed=True,
        use_checkpoint=False,
        last_feat=True,
        point_tokens_num=100,
        num_classes=num_classes,
        return_attention=True,
    ),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_skip_fpn=True, # roi_skip_fpn=False的话，性能会怎么样，性能肯定差，因为imted他就不是这样用fpn的（只用在了rpn中）
    roi_head=dict(
        type='StandardRoIHeadMaskPointSampleDeformAttnReppoints',
        reppoints_head=dict(
            type='RepPointsDensePartAttnHead',
            num_classes=num_classes,
            in_channels=384,
            feat_channels=256,
            point_feat_channels=256,
            stacked_convs=3,
            gradient_mul=0.1,
            # sample_cfg=dict(mode='edge', dist_sample_thr=5),
            # ctr_sample_cfg=dict(mode='edge', dist_sample_thr=2),
            ctr_sample_cfg=dict(mode='inside', dist_sample_thr=1),
            sem_sample_cfg=dict(mode='inside', dist_sample_thr=1),
            # ctr_sample_cfg=dict(mode='edge', dist_sample_thr=5),
            # sem_sample_cfg=dict(mode='inside', dist_sample_thr=3),
            # dist_sample_thr=5,
            # point_strides=[4, 8, 16, 32, 64],
            point_strides=[16],
            point_base_scale=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            loss_bbox_init=dict(type='PtsBorderLoss', loss_weight=0.5),
            loss_sem_point=dict(type='ChamferLoss2D', loss_weight=1.0),
            loss_ctr_point=dict(type='ChamferGlobalEdgeLoss2D', loss_weight=1.0),
            loss_cls_attn=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1),
            transform_method='minmax'),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=384,
            featmap_strides=[16]),
        mil_head=dict(
            type='MAEBoxHeadMIL',
            pretrained=True,
            use_checkpoint=False,
            in_channels=384,
            img_size=224,
            patch_size=16, 
            embed_dim=256, 
            depth=4,
            num_heads=8, 
            mlp_ratio=4., 
            num_classes=num_classes,
            num_layers_query=12,
            loss_mil_factor=1.0,
            with_cls=False,
            with_reg=False,
            hidden_dim=1024,
            roi_size=7,
        ),
        bbox_head=dict(
            type='MAEBoxHeadRec',
            pretrained=True,
            use_checkpoint=False,
            with_reconstruct=False,
            rec_weight=1.0,
            in_channels=384,
            img_size=224,
            patch_size=16, 
            embed_dim=256, 
            depth=4,
            num_heads=8, 
            mlp_ratio=4., 
            num_classes=num_classes,
            seed_score_thr=0.5,
            seed_thr=0.2,
            seed_multiple=0.5,
            cam_layer=-10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
            loss_point=dict(type='L1Loss', loss_weight=10.0),
            loss_point_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=384,
            featmap_strides=[16]),
        mask_head=dict(
            type='MAEMaskHeadPointSup',
            init_cfg=dict(type='Pretrained', checkpoint=pretrained),
            use_checkpoint=False,
            in_channels=384,
            img_size=224,
            patch_size=16, 
            embed_dim=256, 
            depth=4,
            num_heads=8, 
            mlp_ratio=4.,
            num_classes=num_classes,
            scale_factor=2,
            scale_mode='bicubic',
            loss_mask=dict(
                type='MaskCrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            point_assigner=dict(
                type='HungarianPointAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='PointL1Cost', weight=10.0),
                times=1,
            ),
            point_sampler=dict(type='PointPseudoSampler'),
            point_pos_weight=1,
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
            ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
# augmentation strategy originates from DETR / Sparse RCNN
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
checkpoint_config = dict(interval=1)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotationsC', with_bbox=False, with_center=True),
    dict(type='JitterBrightness', brightness_delta=18),
    dict(type='RandomFlipC', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                dict(type='ResizeC',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                (736, 1333), (768, 1333), (800, 1333),
                                ],
                    multiscale_mode='value',
                    keep_ratio=True),
             ],
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleC'),
    dict(type='Collect', keys=['img', 'gt_labels', 'gt_centers']),
]

aug_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1800, 1200), (2200, 1500), (1500, 1000)],
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
 
data=dict(
    train=dict(pipeline=train_pipeline), 
    test=dict(
        ann_file='annotations/image_info_test-dev2017.json',
        img_prefix='test2017/',
        pipeline=aug_test_pipeline,
    ))
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.75))
# learning policy
# lr_config = dict(policy='step', step=[27, 33])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

lr_config = dict(policy='step', step=[10, 14])
# lr_config = dict(policy='step', step=[9, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=116)
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
auto_resume=False
find_unused_parameters=False
