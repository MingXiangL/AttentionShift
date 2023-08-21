import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh


@DETECTORS.register_module()
class TwoStageDetectorPointSup(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 roi_skip_fpn=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pos_mask_thr=0.35,
                 neg_mask_thr=0.7,
                 num_mask_point_gt=10,
                 corr_size=21,
                 obj_tau=0.9,
                 visualize=False):
        super().__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.roi_skip_fpn = roi_skip_fpn
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pos_mask_thr = pos_mask_thr
        self.neg_mask_thr = neg_mask_thr
        self.num_mask_point_gt = num_mask_point_gt
        self.corr_size = corr_size
        self.obj_tau = obj_tau
        self.init_weights(pretrained=pretrained)
        if self.with_roi_head and self.roi_head.with_mae_head and self.backbone.recompute_last_feat:
            self.roi_head.mae_head.mae_encoder = self.backbone
        self.visualize = visualize
        # self.embedding = nn.parameter.Parameter(torch.zeros(20, 384))

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        feats = self.backbone(img)
        if self.with_neck:
            feats.update(dict(feature=self.neck(feats['feature'])))
        return feats

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def get_roi_feat(self, x, vit_feat):
        B, _, H, W = x[2].shape
        x = [
            vit_feat[:, 1:, :].transpose(1, 2).reshape(B, -1, H, W).contiguous()
        ] if self.roi_skip_fpn else x
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_points=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(img)
        with_mask = self.roi_head.with_mask
        x, vit_feat, outputs_class, outputs_coord, attns, feats_point_tokens = \
            feats['feature'], feats['last_feat'], feats['outputs_class'], feats['outputs_coord'], feats['attns'], feats['point_tokens']
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
#             # 初始化point 位置
        points_num = outputs_coord.size(1)
        points_init_position_norm = torch.zeros(points_num, 2, device=vit_feat.device)
        nn.init.constant_(points_init_position_norm, 0.5)
        
        num_imgs = len(x[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(x[0].new_tensor([[w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        points_init_position = points_init_position_norm * imgs_whwh    
        # 初始化point 位置        
        losses = dict()
        seed_gt_out = self.roi_head.seed_pseudo_gt(x, img_metas,
                            None, None, None,
                            gt_bboxes_ignore, gt_masks, vit_feat=vit_feat.clone().detach().permute(0,2,1)[...,1:].unflatten(-1, (patch_h, patch_w)),
                            img=img, point_cls=outputs_class, 
                            point_reg=outputs_coord, imgs_whwh=imgs_whwh, 
                            attns=attns, 
                            gt_points=gt_bboxes,
                            gt_points_labels=gt_labels,
                            roi_feature_map=self.get_roi_feat(x, vit_feat),
                            return_mask=with_mask,
                            pos_mask_thr=self.pos_mask_thr,
                            neg_mask_thr=self.neg_mask_thr,
                            num_mask_point_gt=self.num_mask_point_gt,
                            corr_size=self.corr_size, 
                            obj_tau=self.obj_tau,
                            # org_feats=org_feats,
                            **kwargs)
        
        if with_mask:
            patch_h, patch_w = x[2].size(-2), x[2].size(-1)
            pseudo_gt_labels = seed_gt_out['pseudo_gt_labels']
            pseudo_gt_bboxes = seed_gt_out['pseudo_gt_bboxes']
            mil_losses = seed_gt_out['mil_losses']
            if self.visualize:
                sim_fg = seed_gt_out['sim_fg']
                self.gt_points = [((p[:, :2] + p[:, 2:]) / 2).unsqueeze(1) for p in gt_bboxes]
                self.gt_points_labels = gt_labels
                self.org_attns = attns
                self.attns = seed_gt_out['attns']
                self.pseudo_gt_bboxes = pseudo_gt_bboxes
                self.mask_points_coords = seed_gt_out['mask_points_coords']
                self.mask_points_coords_org = seed_gt_out['mask_points_coords']
                self.mask_points_labels = seed_gt_out['mask_points_labels']
                self.semantic_centers = seed_gt_out['semantic_centers_ret']
                self.point_cls = outputs_class
                self.point_coord = outputs_coord * imgs_whwh
                self.best_attn_idx = seed_gt_out['attn_idx']
                self.map_cos_bg = seed_gt_out['map_cos_bg']
                self.map_cos_fg = seed_gt_out['map_cos_fg']
                self.fpn_feat = x
                self.sim_fg = sim_fg
                self.vit_feat = seed_gt_out['vit_feat']
                self.all_attns = seed_gt_out['all_attns']
                self.merged_boxes = seed_gt_out['merged_boxes']
                self.point_targets = seed_gt_out['point_targets']
                
            kwargs.update({'mask_point_coords': seed_gt_out['mask_points_coords'], 'mask_point_labels': seed_gt_out['mask_points_labels'], 
                        'semantic_centers': seed_gt_out['semantic_centers_ret'], 'semantic_centers_split': seed_gt_out['semantic_centers_split']})
            gt_masks = getattr(seed_gt_out, 'pseudo_gt_masks', None)
        else:
            pseudo_gt_labels, pseudo_gt_bboxes, mil_losses = seed_gt_out[0], seed_gt_out[1], seed_gt_out[2]
        losses.update(mil_losses)
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                pseudo_gt_bboxes,  # 更换成预测的伪标注 bboxes
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        sc_corres_gts = getattr(seed_gt_out, 'corres_gts', None)
        if sc_corres_gts is not None:
            kwargs.update(dict(sc_corres_gts=sc_corres_gts))
        roi_losses = self.roi_head.forward_train(self.get_roi_feat(x, vit_feat), img_metas, 
                                                    proposal_list, pseudo_gt_bboxes, pseudo_gt_labels,
                                                    gt_bboxes_ignore, gt_masks, 
                                                    vit_feat=vit_feat, img=img, 
                                                    point_cls=outputs_class, 
                                                    point_reg=outputs_coord, 
                                                    imgs_whwh=imgs_whwh, 
                                                    attns=attns, 
                                                    gt_points=gt_bboxes,
                                                    gt_points_labels=gt_labels,
                                                    **kwargs)
        losses.update(roi_losses)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        feats = self.extract_feat(img)
        x, vit_feat = feats['feature'], feats['last_feat']

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        feats = self.extract_feats(imgs)
        xs = [f['feature'] for f in feats]
        vit_feats = [f['last_feat'] for f in feats]
        proposal_list = self.rpn_head.aug_test_rpn(xs, img_metas)

            # return self.roi_head.simple_test(
            #     self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)

        return self.roi_head.aug_test(
            [self.get_roi_feat(x, vf) for x, vf in zip(xs, vit_feats)],
            proposal_list, img_metas, rescale=rescale)
