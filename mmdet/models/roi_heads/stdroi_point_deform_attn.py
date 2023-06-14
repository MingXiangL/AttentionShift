import torch

import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_shared_head
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
import cv2
import pdb
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import BitmapMasks
from mmcv.ops import point_sample
import math
from sklearn.decomposition import PCA
import time
from cc_torch import connected_components_labeling
from ..utils import ObjectQueues, ObjectFactory, cosine_distance, cosine_distance_part
from .standard_roi_head_mask_point_sample_rec_align import StandardRoIHeadMaskPointSampleRecAlign, filter_maps, corrosion_batch


@HEADS.register_module()
class StandardRoIHeadMaskPointSampleDeformAttn(StandardRoIHeadMaskPointSampleRecAlign):

    def __init__(
        self, 
        mil_head=None, 
        bbox_roi_extractor=None, 
        bbox_head=None, 
        mask_roi_extractor=None, 
        mask_head=None, 
        shared_head=None, 
        mae_head=None, 
        bbox_rec_head=None, 
        train_cfg=None, 
        test_cfg=None, 
        visualize=False, 
        epoch=0, 
        epoch_semantic_centers=0, 
        num_semantic_points=3, 
        semantic_to_token=False, 
        with_align=True, 
        pca_dim=128, 
        mean_shift_times_local=10, 
        len_queque=100, 
        ratio_range=[0.9, 1.0], 
        appear_thresh=0.1, 
        max_retrieval_objs=5, 
        keypoint_align_head=None, 
        deform_attn_head=None,
        attn_point_thr=0.2,
        ):
        super().__init__(mil_head, bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, mae_head, bbox_rec_head, train_cfg, test_cfg, visualize, epoch, epoch_semantic_centers, num_semantic_points, semantic_to_token, with_align, pca_dim, mean_shift_times_local, len_queque, ratio_range, appear_thresh, max_retrieval_objs, keypoint_align_head, deform_attn_head)
        self.attn_point_thr = attn_point_thr

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      vit_feat=None,
                      img=None,
                      point_init=None,
                      point_cls=None,
                      point_reg=None,
                      pos_mask_thr=0.35,
                      imgs_whwh=None,
                      attns=None,
                      gt_points=None,
                      gt_points_labels=None,
                      map_cos_fg=None,
                      mask_point_labels=None,
                      mask_point_coords=None,
                      semantic_centers=None,
                      semantic_centers_split=None, 
                      feats_point_tokens=None, 
                      semantic_centers_feat_split=None, 
                      semantic_centers_feat=None,
                      num_parts=None,
                      semantic_centers_org=None,
                     ):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        num_imgs = len(img_metas)
        if self.train_cfg.get('point_assigner'):
            num_proposals = point_reg.size(1)
            imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
            # 需要先获得point预测结果（point_reg sigmoid之后直接当作绝对位置,相对位置不太好弄，没有w，h无法用decoder)
            point_assign_results = []
            for i in range(num_imgs):
                normalize_point_cc = point_reg[i].detach()
                point_coords = (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2
                point_labels = gt_points_labels[i]
                if (self.epoch >= self.epoch_semantic_centers) and (semantic_centers is not None) and self.semantic_to_token:
                    point_coords = torch.cat((point_coords, semantic_centers[i][0]), dim=0)
                    point_labels = torch.cat((point_labels, semantic_centers[i][1]), dim=0)
                assign_result = self.point_assigner.assign(
                    normalize_point_cc, point_cls[i], point_coords,
                    point_labels, img_metas[i]
                )
                point_sampling_result = self.point_sampler.sample(
                    assign_result, point_reg[i], 
                    point_coords
                )
                point_assign_results.append(point_sampling_result)
            bbox_targets = self.get_targets(
                point_assign_results, point_coords, point_labels, self.train_cfg,
                True)
            point_loss = self.loss(
                point_cls.view(-1, point_cls.size(-1)),
                point_reg.view(-1, 2),
                *bbox_targets,
                imgs_whwh=imgs_whwh)
            
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                
        losses = dict()
        
        losses.update(point_loss)
        
        if self.visualize:
            self.gt_point_coords = point_coords
            self.point_sampling_result = point_sampling_result
            
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, img=img)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
                        
        if self.with_deform_attn:
            # TODO:支持batch操作
            H, W = img.shape[-2:]
            vit_feat_rs = vit_feat[:, 1:].permute(0,2,1).unflatten(-1, (H//16, W//16))
            loss_deform = None
            
            img_size = vit_feat.new_tensor([[W, H]])
            for i_img in range(num_imgs):
                point_idx = (semantic_centers_org[0][0] / 16 - 0.5).long().flip(1)
                point_idx[:, 0] = point_idx[:, 0].clamp(0, vit_feat_rs.shape[-2])
                point_idx[:, 1] = point_idx[:, 1].clamp(0, vit_feat_rs.shape[-1])
                point_coords = (gt_points[i_img][:, :2] + gt_points[i_img][:, 2:]) / 2
                point_coords /= img_size
                point_coords = torch.repeat_interleave(point_coords, point_idx.new_tensor(num_parts[i_img]), dim=0)

                if torch.numel(point_idx) == 0:
                    loss_deform_tmp, x_sampled, coord_sample, reference,kp_scores, attn, assets = self.deform_attn(
                        torch.zeros_like(vit_feat_rs),
                        torch.zeros(1, 2, dtype=torch.long, device=vit_feat_rs.device),
                        torch.zeros(1, dtype=torch.long, device=vit_feat_rs.device),
                        coords=torch.zeros(1, 2, dtype=torch.float, device=vit_feat_rs.device),
                        all_zero=True)
                    coord_sample_org = coord_sample
                else:
                    loss_deform_tmp, x_sampled, coord_sample, reference, kp_scores, attns, assets  = self.deform_attn(vit_feat_rs, point_idx, semantic_centers_org[1][0], coords=point_coords, visualize=self.visualize)
                    map_cos_fg_corr = corrosion_batch(torch.where(map_cos_fg[i_img]>pos_mask_thr, torch.ones_like(map_cos_fg[i_img]), torch.zeros_like(map_cos_fg[i_img]))[None], corr_size=11)[0]
                    fg_inter = F.interpolate(map_cos_fg_corr.unsqueeze(0), vit_feat_rs.shape[-2:], mode='bilinear')[0]
                    fg_inter = torch.repeat_interleave(fg_inter, point_idx.new_tensor(num_parts[i_img]), dim=0)
                    sim_points = x_sampled @ vit_feat[i_img, 1:].T / (x_sampled.norm(p=2, dim=2, keepdim=True) * vit_feat[i_img, 1:].norm(p=2, dim=1).view(1,1,-1)).clamp(1e-5)
                    sim_points = sim_points.unflatten(-1, vit_feat_rs.shape[-2:])
                    maps_fore, pos_idx = filter_maps(sim_points, fg_inter, None)
                    pos_idx = pos_idx.split(num_parts[i_img])
                    coord_sample_org = coord_sample.clone()
                    coord_sample = (coord_sample[..., (1, 0)] + 1) / 2 * img_size
                    coord_sample = coord_sample.split(num_parts[i_img])
                    # 把Attention weight和分类置信度加进来筛选sub-part key points
                    attns = (attns * kp_scores[:, None]).split(num_parts[i_img])
                    for i_obj, sc, coord, idx, attn, kp_score in zip(range(len(num_parts[i_img])), semantic_centers_split[i_img], coord_sample, pos_idx, attns, kp_scores.split(num_parts[i_img])):
                        
                        if torch.numel(sc) == 0:
                            continue
                        sc_feats = point_sample(vit_feat_rs, sc[None] / img_size[None, None], align_corners=False)[0].permute(0, 2, 1)
                        att_feats= point_sample(vit_feat_rs, coord[None] / img_size[None, None], align_corners=False)[0]
                        sim = F.cosine_similarity(sc_feats, att_feats, dim=0)
                        # pos_mask = sim * kp_score[:, None] > self.attn_point_thr
                        # pos_mask = sim > self.attn_point_thr
                        pos_mask = (sim * attn) > self.attn_point_thr
                        idx = idx * pos_mask
                        semantic_centers_split[i_img][i_obj] = torch.cat((sc, coord[idx]))

                if self.visualize:
                    self.keypoint_offset = coord_sample_org
                    self.reference = reference
                    self.assets = assets
                    self.semantic_centers_split = semantic_centers_split
                    self.attns = attns
                    self.kp_scores = kp_scores
                    self.map_cos_fg = map_cos_fg
                    self.num_parts = num_parts
                    self.vit_feat = vit_feat_rs
                    # self.pos_idx = pos_idx

                if loss_deform is None:
                    loss_deform = dict()
                    for k in loss_deform_tmp:
                        loss_deform[k] = 0

                for k in loss_deform:
                    loss_deform[k] += loss_deform_tmp[k]

            for k in loss_deform:
                loss_deform[k] /= num_imgs
            losses.update(loss_deform)

        if self.with_mask:
            # gt_masks_pseudo = self.get_pseudo_gt_masks_from_point_attn()
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    mask_point_coords, mask_point_labels,
                                                    semantic_centers=semantic_centers_split,
                                                    img_metas=img_metas)
            losses.update(mask_results['loss_mask'])

        return losses
