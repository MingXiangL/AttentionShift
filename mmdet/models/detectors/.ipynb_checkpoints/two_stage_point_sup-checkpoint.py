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
                 pos_mask_thr=0.15,
                 neg_mask_thr=0.05,
                 num_mask_point_gt=10):
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
        self.pos_mask_thr=pos_mask_thr
        self.neg_mask_thr=neg_mask_thr
        self.num_mask_point_gt=num_mask_point_gt

        self.init_weights(pretrained=pretrained)
        if self.with_roi_head and self.roi_head.with_mae_head and self.backbone.recompute_last_feat:
            self.roi_head.mae_head.mae_encoder = self.backbone

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
        x = self.backbone(img)
        if len(x) == 5:
            vit_feat = x[1]
            if self.with_neck:
                xs = self.neck(x[0])
            return xs, vit_feat, x[-3], x[-2], x[-1]
        elif len(x) == 4:
            vit_feat = x[1]
            if self.with_neck:
                xs = self.neck(x[0])
            return xs, vit_feat, x[-2], x[-1]
        elif len(x) == 2:
            vit_feat = x[-1]
            if self.with_neck:
                x = self.neck(x[0])
            return x, vit_feat
        else:
            if self.with_neck:
                x = self.neck(x)
            return x

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
        x = self.extract_feat(img)
        with_mask = self.roi_head.with_mask
        if len(x) == 5:
            x, vit_feat, outputs_class, outputs_coord, attns = x
            
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
                                gt_bboxes_ignore, gt_masks, vit_feat=None,
                                img=None, point_cls=outputs_class, 
                                point_reg=outputs_coord, imgs_whwh=imgs_whwh, 
                                attns=attns, 
                                gt_points=gt_bboxes,
                                gt_points_labels=gt_labels,
                                roi_feature_map=self.get_roi_feat(x, vit_feat),
                                return_mask=with_mask,
                                pos_mask_thr=self.pos_mask_thr,
                                neg_mask_thr=self.neg_mask_thr,
                                num_mask_point_gt=self.num_mask_point_gt,
                                **kwargs)
            
            if with_mask:
                pseudo_gt_labels, pseudo_gt_bboxes, mil_losses, pseudo_mask_point_coords, pseudo_mask_point_labels = \
                    seed_gt_out[0], seed_gt_out[1], seed_gt_out[2], seed_gt_out[3], seed_gt_out[4]
                kwargs.update({'mask_point_coords': pseudo_mask_point_coords, 'mask_point_labels': pseudo_mask_point_labels})
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
        
        elif len(x) == 4:
            x, vit_feat, outputs_class, outputs_coord = x
            
#             # 初始化point 位置
            points_num = outputs_coord.size(1)
            points_init_position_norm = torch.zeros(points_num, 2).to(vit_feat.device)
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
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.roi_head.forward_train(self.get_roi_feat(x, vit_feat), img_metas, 
                                                     proposal_list, gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks, vit_feat=vit_feat,
                                                     img=img, point_cls=outputs_class, 
                                                     point_reg=outputs_coord, imgs_whwh=imgs_whwh, 
                                                     **kwargs)
            losses.update(roi_losses)
            return losses
        elif len(x) == 2:
            x, vit_feat = x[0], x[1]
            losses = dict()
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.roi_head.forward_train(self.get_roi_feat(x, vit_feat), img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks, vit_feat=vit_feat,
                                                     img=img, **kwargs)
            losses.update(roi_losses)
            return losses
        else:
            losses = dict()
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
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

        x = self.extract_feat(img)
        if len(x) == 5:
            x, vit_feat, outputs_class, outputs_coord, attns = x
            # get origin input shape to onnx dynamic input shape
#             if torch.onnx.is_in_onnx_export():
#                 img_shape = torch._shape_as_tensor(img)[2:]
#                 img_metas[0]['img_shape_for_onnx'] = img_shape

#             points_num = outputs_coord.size(1)
#             points_init_position_norm = torch.zeros(points_num, 2).to(vit_feat.device)
#             nn.init.constant_(points_init_position_norm, 0.5)
            
#             num_imgs = len(x[0])
#             imgs_whwh = []
#             for meta in img_metas:
#                 h, w, _ = meta['img_shape']
#                 imgs_whwh.append(x[0].new_tensor([[w, h]]))
#             imgs_whwh = torch.cat(imgs_whwh, dim=0)
#             imgs_whwh = imgs_whwh[:, None, :]
#             points_init_position = points_init_position_norm * imgs_whwh    
            
#             scale_factor = img_metas[0]['scale_factor']
#             det_bboxes, det_labels = self.roi_head.point2bbox(x, img_metas, 
#                                                      None, None, None,
#                                                      None, None, vit_feat=None,
#                                                      img=None, point_cls=outputs_class, 
#                                                      point_reg=outputs_coord, imgs_whwh=imgs_whwh, 
#                                                      attns=attns, scale_factor=scale_factor)
#             bbox_results = [
#                 bbox2result(det_bboxes[i], det_labels[i], self.roi_head.bbox_head.num_classes)
#                 for i in range(len(det_bboxes))
#             ]
#             return bbox_results

            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            return self.roi_head.simple_test(
                self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)
        
        elif len(x) == 4:
            x, vit_feat, _, _ = x
            # get origin input shape to onnx dynamic input shape
            if torch.onnx.is_in_onnx_export():
                img_shape = torch._shape_as_tensor(img)[2:]
                img_metas[0]['img_shape_for_onnx'] = img_shape

            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            return self.roi_head.simple_test(
                self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)
        
        elif len(x) == 2:
            x, vit_feat = x[0], x[1]
            # get origin input shape to onnx dynamic input shape
            if torch.onnx.is_in_onnx_export():
                img_shape = torch._shape_as_tensor(img)[2:]
                img_metas[0]['img_shape_for_onnx'] = img_shape

            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            return self.roi_head.simple_test(
                self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)
        else:
            # get origin input shape to onnx dynamic input shape
            if torch.onnx.is_in_onnx_export():
                img_shape = torch._shape_as_tensor(img)[2:]
                img_metas[0]['img_shape_for_onnx'] = img_shape

            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            return self.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        raise NotImplementedError
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
