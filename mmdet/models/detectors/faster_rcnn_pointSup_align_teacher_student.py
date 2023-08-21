
import pdb
import torch
import torch.nn as nn

from ..builder import DETECTORS
from .two_stage_point_align import TwoStageDetectorPointSupAlign
from ..builder import DETECTORS, build_backbone


class TeacherWrapper(object):
    def __init__(self, teacher, teacher_momentum=0.999, pretrained=None) -> None:
        self.teacher = teacher
        self.teacher.init_weights(pretrained)
        self.teacher.eval()
        self.teacher_momentum = teacher_momentum

    @torch.no_grad()
    def momentum_update(self, cur_model, teacher_momentum=None):
        """
        Momentum update of the key encoder
        """
        if teacher_momentum is None:
            teacher_momentum = self.teacher_momentum
        model_dict = self.teacher.state_dict()
        cur_model_dict = cur_model.state_dict()
        for key in model_dict:
            if key in cur_model_dict:
                model_dict[key] = cur_model_dict[key] * (1 - self.teacher_momentum) + model_dict[key] * self.teacher_momentum
            else:
                raise Exception(f'{key} is not found in the student model!')
        self.teacher.load_state_dict(model_dict)


@DETECTORS.register_module()
class FasterRCNNPointSupAlignTS(TwoStageDetectorPointSupAlign):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 with_teacher=True,
                 momentum=0.999,
                 *args, **kwargs):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            *args, **kwargs)

        if with_teacher:
            backbone.update(dict(with_point_head=False))
            self.tw = TeacherWrapper(build_backbone(backbone), teacher_momentum=momentum, pretrained=pretrained)
            self.with_teacher = with_teacher
        self.turn_on_teacher = True

    def to(self, *args, **kwargs):
        if self.with_teacher:
            self.tw.teacher = self.tw.teacher.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        if self.with_teacher:
            self.tw.teacher = self.tw.teacher.cuda(*args, **kwargs)
        return super().cuda(*args, **kwargs)

    def extract_feat(self, img):
        feats = super().extract_feat(img)
        with torch.no_grad():
            feats_teacher = self.tw.teacher(img)
        return feats, feats_teacher

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
        if self.with_teacher and self.turn_on_teacher:
            self.tw.momentum_update(self.backbone)

        feats, feats_teacher = self.extract_feat(img)

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
                            **kwargs)
        
        if with_mask:
            patch_h, patch_w = x[2].size(-2), x[2].size(-1)
            pseudo_gt_labels = seed_gt_out['pseudo_gt_labels']
            pseudo_gt_bboxes = seed_gt_out['pseudo_gt_bboxes']
            mil_losses = seed_gt_out['mil_losses']
            if self.visualize:
                self.store_visual_matterials(seed_gt_out)

            kwargs.update({'mask_point_coords': seed_gt_out['mask_points_coords'], 'mask_point_labels': seed_gt_out['mask_points_labels'], 
                        'semantic_centers': seed_gt_out['semantic_centers'], 'semantic_centers_split': seed_gt_out['semantic_centers_split'],
                        'semantic_centers_feat_split': seed_gt_out['semantic_centers_feat_split']})
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

        roi_losses = self.roi_head.forward_train(self.get_roi_feat(x, vit_feat), img_metas, 
                                                    proposal_list, pseudo_gt_bboxes, pseudo_gt_labels,
                                                    gt_bboxes_ignore, gt_masks, 
                                                    feats_teacher=feats_teacher['last_feat'].permute(0,2,1)[...,1:].unflatten(-1, (patch_h, patch_w)).detach(),
                                                    vit_feat=vit_feat, img=img,     
                                                    point_cls=outputs_class,    
                                                    point_reg=outputs_coord, 
                                                    imgs_whwh=imgs_whwh, 
                                                    attns=attns, 
                                                    gt_points=gt_bboxes,
                                                    gt_points_labels=gt_labels,
                                                    teacher_point_tokens=feats_teacher['point_tokens'],
                                                    student_point_tokens=feats_point_tokens,
                                                    **kwargs)
        losses.update(roi_losses)
        return losses

    def store_visual_matterials(self, dict_to_save):
        for k in dict_to_save:
            setattr(self, k, dict_to_save[k])

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        feats, _ = self.extract_feat(img)
        x, vit_feat = feats['feature'], feats['last_feat']

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)