import torch

import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
import cv2
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import BitmapMasks

def get_multi_bboxes(cam, point, cam_thr=0.2, area_ratio=0.5, img_size=None):
    """
    cam: single image with shape (h, w, 1)
    point: one point location (x, y)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    img_h, img_w = img_size
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
            
    if len(contours) != 0:
        estimated_bbox = []
        areas = list(map(cv2.contourArea, contours))
        area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
        for idx in area_idx:
            if areas[idx] >= areas[area_idx[0]] * area_ratio:
                c = contours[idx]
                x, y, w, h = cv2.boundingRect(c)
                estimated_bbox.append([x, y, x + w, y + h])
    else:
        estimated_bbox = [[0, 0, 1, 1]]

    estimated_bbox = np.array(estimated_bbox)
    proposal_xmin = np.min(estimated_bbox[:, 0])
    proposal_ymin = np.min(estimated_bbox[:, 1])
    proposal_xmax = np.max(estimated_bbox[:, 2])
    proposal_ymax = np.max(estimated_bbox[:, 3])
    xc, yc = point

    if np.abs(xc - proposal_xmin) > np.abs(xc - proposal_xmax):
        gt_xmin = proposal_xmin
        gt_xmax = xc * 2 -  gt_xmin
        gt_xmax = gt_xmax if gt_xmax < img_w else float(img_w)
    else:
        gt_xmax = proposal_xmax
        gt_xmin = xc * 2 -  gt_xmax
        gt_xmin = gt_xmin if gt_xmin > 0 else 0.0

    if np.abs(yc - proposal_ymin) > np.abs(yc - proposal_ymax):
        gt_ymin = proposal_ymin
        gt_ymax = yc * 2 -  gt_ymin
        gt_ymax = gt_ymax if gt_ymax < img_h else float(img_h)
    else:
        gt_ymax = proposal_ymax
        gt_ymin = yc * 2 -  gt_ymax
        gt_ymin = gt_ymin if gt_ymin > 0 else 0.0
        
    estimated_bbox = [[gt_xmin, gt_ymin, gt_xmax, gt_ymax]]
    return estimated_bbox  #, thr_gray_heatmap, len(contours)

def attns_project_to_feature(attns_maps):
    attns_maps = torch.stack(attns_maps)
    residual_att = torch.eye(attns_maps.size(2)).type_as(attns_maps)
    aug_att_mat = attns_maps + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size()).type_as(aug_att_mat)
    joint_attentions[-1] = aug_att_mat[-1]
    for i in range(2, len(attns_maps) + 1):
        joint_attentions[-i] = torch.matmul(joint_attentions[-(i - 1)], aug_att_mat[-i])
    
    reverse_joint_attentions = torch.zeros(joint_attentions.size()).type_as(joint_attentions)
    
    for i in range(len(joint_attentions)):
        reverse_joint_attentions[i] = joint_attentions[-(i + 1)]
    reverse_joint_attentions = reverse_joint_attentions.permute(1, 0, 2, 3)
    return reverse_joint_attentions

# def attns_project_to_feature(attns_maps):
#     #         assert len(attns_maps[1]) == 1 
#     # [block_num], B, H, all_num, all_num
#     attns_maps = torch.stack(attns_maps)
#     # block_num, B, H, all_num, all_num
# #     attns_maps = attns_maps.mean(2)
#     # block_num, B, all_num, all_num
#     residual_att = torch.eye(attns_maps.size(2)).type_as(attns_maps)
#     aug_att_mat = attns_maps + residual_att
#     aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)

#     joint_attentions = torch.zeros(aug_att_mat.size()).type_as(aug_att_mat)
#     joint_attentions[0] = aug_att_mat[0]

#     for n in range(1, aug_att_mat.size(0)):
#         joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
#     attn_proj_map = joint_attentions[-1]
# #     return joint_attentions
#     return attn_proj_map

# def attns_project_to_feature(attns_maps, num_proposals=None, cam_layers=None, patch_size=None, pos_inds=None):
#     patch_h, patch_w = patch_size
# #     # [block_num], B, H, all_num, all_num
#     attns_maps = torch.stack(attns_maps).detach()
#     # block_num, B, H, all_num, all_num
#     attns_maps = attns_maps.mean(2)
#     # block_num, B, all_num, all_num
#     point_token_attn_maps = []
#     for layer in cam_layers:
#         attns_maps_ = attns_maps[layer:]
#         residual_att = torch.eye(attns_maps_.size(2)).type_as(attns_maps_)
#         aug_att_mat = attns_maps_ + residual_att
#         aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)

#         joint_attentions = torch.zeros(aug_att_mat.size()).type_as(aug_att_mat)
#         joint_attentions[0] = aug_att_mat[0]

#         for n in range(1, aug_att_mat.size(0)):
#             joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
#         attn_proj_map = joint_attentions[-1][-num_proposals:, 1:-num_proposals]
#         attn_proj_map = attn_proj_map.reshape(-1, 1, patch_h, patch_w)
#         attn_proj_map = F.interpolate(attn_proj_map, (patch_h * 16, patch_w * 16), 
#                                       mode='bilinear').reshape(-1, patch_h * 16, patch_w * 16).cpu().numpy() # 100, H, W
#         point_token_attn_maps.append(attn_proj_map)
#         print(attn_proj_map.size())
# #     point_token_attn_maps = torch.stack(point_token_attn_maps, dim=0)
# #     print(point_token_attn_maps.size())
#     exit()
# #     return joint_attentions
#     return attn_proj_map

@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)
            if self.train_cfg.get('point_assigner'):
                self.point_assigner = build_assigner(self.train_cfg.point_assigner)
                self.point_sampler = build_sampler(self.train_cfg.point_sampler, context=self)
            
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        if isinstance(bbox_roi_extractor, list):
            self.bbox_roi_extractor = []
            for extractor in bbox_roi_extractor:
                self.bbox_roi_extractor.append(
                    build_roi_extractor(extractor)
                )
        else:
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        if isinstance(bbox_head, list):
            self.bbox_head = nn.ModuleList([])
            for head in bbox_head:
                head = build_head(head)
                self.bbox_head.append(head)
        else:
            self.bbox_head = build_head(bbox_head)
            
    def init_mil_head(self, bbox_roi_extractor, mil_head):
        """Initialize ``mil_head``"""
#         self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.mil_head = build_head(mil_head)
    
    def init_bbox_rec_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_rec_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_rec_head = build_head(bbox_head)
        self.bbox_head=self.bbox_rec_head

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_mil:
            if hasattr(self.mil_head, 'pretrained'):
                self.mil_head.init_weights(pretrained=pretrained)
            else:
                self.bbox_head.init_weights()
        if self.with_bbox:
#             self.bbox_roi_extractor.init_weights()
            if isinstance(self.bbox_head, nn.ModuleList):
                for head in self.bbox_head:
                    if hasattr(head, 'pretrained'):
                        head.init_weights(pretrained=pretrained)
                    else:
                        head.init_weights()
            else:
                if hasattr(self.bbox_head, 'pretrained'):
                    self.bbox_head.init_weights(pretrained=pretrained)
                else:
                    self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_mae_head:
            self.mae_head.init_weights(pretrained=pretrained)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def point2bbox(self,
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
                   imgs_whwh=None,
                   attns=None,
                   scale_factor=None,
                  ):
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        num_proposals = point_cls.size(1)
        # attention maps
        joint_attentions = attns_project_to_feature(attns[self.bbox_head.cam_layer:])
#         cams = joint_attentions[-1][:, -num_proposals:, 1:-num_proposals].reshape(-1, num_proposals, patch_h, patch_w)
        cams = joint_attentions[:, -num_proposals:, 1:-num_proposals].reshape(-1, num_proposals, patch_h, patch_w)
        cams = F.interpolate(cams, (patch_h * 16, patch_w * 16), mode='bilinear')
        # seed proposal
        scores = point_cls.sigmoid()  # 获得proposal的真实得分
        scores, label_inds = scores.max(-1)  # 获得proposal label
        points_locations = imgs_whwh * point_reg  # 获得最终点的位置
        
#         pseudo_gt_scores = []
        pseudo_gt_labels = []
        pseudo_gt_bboxes = []
        
        for scores_per_img, pseudo_labels_per_img, point_locations_per_img, cam_per_img in zip(scores, label_inds, points_locations, cams):
            pseudo_inds = scores_per_img >= self.bbox_head.seed_score_thr
#             print(self.bbox_head.seed_score_thr, self.bbox_head.cam_layer)
            if sum(pseudo_inds) == 0:
                pseudo_gt_bboxes.append(torch.empty(0, 5))
                pseudo_gt_labels.append(torch.empty(0))
                continue
            
            pseudo_scores = scores_per_img[pseudo_inds]

            pseudo_labels = pseudo_labels_per_img[pseudo_inds]
            pseudo_labels_per_img = pseudo_labels.to(point_cls.device).long()
            
            pseudo_point_locations = point_locations_per_img[pseudo_inds]
            pseudo_gt_bboxes_per_img = []
            cam_ = cam_per_img[pseudo_inds]
            
            cam_ = cam_.detach().cpu().numpy()
            pseudo_point_locations = pseudo_point_locations.detach().cpu().numpy()
            for c, p in zip(cam_, pseudo_point_locations):
                c = (c - c.min()) / (c.max() - c.min())
                pseudo_gt_bbox = get_multi_bboxes(c,
                                                  p,
                                                  cam_thr=self.bbox_head.seed_thr, 
                                                  area_ratio=self.bbox_head.seed_multiple)
                pseudo_gt_bbox = torch.as_tensor(pseudo_gt_bbox).to(point_cls.device).float()
                pseudo_gt_bboxes_per_img.append(pseudo_gt_bbox)
            pseudo_gt_bboxes_per_img = torch.cat(pseudo_gt_bboxes_per_img, dim=0)
            
            del cam_
            del pseudo_point_locations
            
            if not isinstance(scale_factor, tuple):
                scale_factor = tuple([scale_factor])
            # B, 1, bboxes.size(-1)
            scale_factor = pseudo_gt_bboxes_per_img.new_tensor(scale_factor)
            pseudo_gt_bboxes_per_img /= scale_factor
            
            pseudo_gt_bboxes_per_img = torch.cat([pseudo_gt_bboxes_per_img, pseudo_scores.unsqueeze(-1)], dim=1)
#             pseudo_gt_scores.append(pseudo_scores)
            pseudo_gt_bboxes.append(pseudo_gt_bboxes_per_img)
            pseudo_gt_labels.append(pseudo_labels_per_img)
            
        return pseudo_gt_bboxes, pseudo_gt_labels

    def transfer_to_cam(self, attn_maps):
        n_layer, n_gt, h, w = attn_maps.shape
        if torch.numel(attn_maps) == 0:
            return torch.zeros(n_layer, 0, h, w, device=attn_maps.device), torch.zeros(n_layer, 0, h, w, device=attn_maps.device)

        max_val, _ = attn_maps.flatten(-2).max(dim=-1, keepdim=True)
        min_val, _ = attn_maps.flatten(-2).min(dim=-1, keepdim=True)
        max_val = max_val.unsqueeze(-1)
        min_val = min_val.unsqueeze(-1)
        attn_maps_cam = (attn_maps  - min_val) / (max_val - min_val)
        pos_idx = attn_maps_cam > self.bbox_head.seed_thr
        ignore_idx = (attn_maps_cam < self.bbox_head.seed_thr) & (attn_maps_cam > self.bbox_head.seed_thr * 0.1)
        cams = torch.zeros_like(attn_maps_cam)
        ignore_mask = torch.zeros_like(attn_maps_cam)
        ignore_mask[ignore_idx] = 1
        cams[pos_idx] = 1
        
        return cams, ignore_mask

    def seed_pseudo_gt(self,
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
                      imgs_whwh=None,
                      attns=None,
                      gt_points=None,
                      gt_points_labels=None,
                      roi_feature_map=None,
                      return_mask=False,
                     ):
        num_imgs = point_reg.size(0)
        if self.train_cfg.get('point_assigner'):
            num_proposals = point_reg.size(1)
            imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
            # 需要先获得point预测结果（point_reg sigmoid之后直接当作绝对位置,相对位置不太好弄，没有w，h无法用decoder)
            point_assign_results = []
            for i in range(num_imgs):
                normalize_point_cc = point_reg[i].detach()
                assign_result = self.point_assigner.assign(
                    normalize_point_cc, point_cls[i], gt_points[i],
                    gt_points_labels[i], img_metas[i]
                )
                point_sampling_result = self.point_sampler.sample(
                    assign_result, point_reg[i], 
                    (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2
                )
                point_assign_results.append(point_sampling_result)
            pos_inds = [sample_results.pos_inds for sample_results in point_assign_results]
            
            labels, _, point_targets, _ = self.get_targets(
                point_assign_results, gt_points, gt_points_labels, self.train_cfg,
                concat=False)
            
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        num_proposals = point_cls.size(1)
        
#         points_attn_maps = attns_project_to_feature(attns, num_proposals=num_proposals, 
#                                  cam_layers=-torch.arange(1, 13), patch_size=(patch_h, patch_w))

        points_attn_maps = attns_project_to_feature(attns[-self.bbox_head.cam_layer:]) 
        cam_maps_images = []
        layer_inds = np.arange(1, self.bbox_head.cam_layer + 1)
        # attention maps
        point_attentions_per_layer = []
        gt_scale_bboxes = []
        gt_labels = []
        for i_img in range(num_imgs):
            pos_inds_ = pos_inds[i_img]
            gt_labels.append(labels[i_img][pos_inds_])
            num_gt = len(pos_inds_)
            points_attn_maps_per_img = points_attn_maps[i_img][:, -num_proposals:, 1:-num_proposals].permute(1, 0, 2)[pos_inds_].permute(1, 0, 2)
            points_attn_maps_per_img = points_attn_maps_per_img.reshape(-1, 1, patch_h, patch_w)
            points_attn_maps_per_img = F.interpolate(points_attn_maps_per_img, (patch_h * 16, patch_w * 16), mode='bilinear').reshape(-1, num_gt, patch_h * 16, patch_w * 16) # nu_gt, H, W
            point_targets_ = point_targets[i_img][pos_inds_].unsqueeze(0).repeat(self.bbox_head.cam_layer, 1, 1)
            
            points_attn_maps_per_img_cam, points_attn_maps_per_img_ignore_mask = self.transfer_to_cam(points_attn_maps_per_img)
            cam_maps_images.append([points_attn_maps_per_img_cam.clone(), points_attn_maps_per_img_ignore_mask.clone()])
#             for ind in layer_inds:
#                 point_attentions = attns_project_to_feature(attns[-ind:])
#                 attnss = point_attentions[i_img][-num_proposals:, 1:-num_proposals]
#                 gt_cam = attnss.reshape(-1, 1, patch_h, patch_w)[pos_inds_]
#                 gt_cam = F.interpolate(gt_cam, (patch_h * 16, patch_w * 16), mode='bilinear').reshape(-1, patch_h * 16, patch_w * 16) # nu_gt, H, W
#                 cam_layers.append(gt_cam)
#             cam_layers = torch.stack(cam_layers, dim=0)
            cam_layers = points_attn_maps_per_img
            scale_bboxes_per_image = []
            for cam_per_point, point in zip(cam_layers.detach().cpu().numpy(), point_targets_.detach().cpu().numpy()):
                scale_bboxes = []
                for scale_cam, scale_point in zip(cam_per_point, point):
#                     scale_cam = cv2.resize(scale_cam, (patch_w * 16, patch_h * 16))
                    scale_cam = (scale_cam - scale_cam.min()) / (scale_cam.max() - scale_cam.min())
                    pseudo_gt_bbox = get_multi_bboxes(scale_cam,
                                                      scale_point,
                                                      cam_thr=self.bbox_head.seed_thr,
                                                      area_ratio=self.bbox_head.seed_multiple,
                                                      img_size=(patch_h * 16, patch_w * 16)
                                                     )
                    scale_bboxes.append(torch.as_tensor(pseudo_gt_bbox).type_as(gt_points[0]))
                scale_bboxes = torch.cat(scale_bboxes, dim=0)
                scale_bboxes_per_image.append(scale_bboxes)
            scale_bboxes_per_image = torch.stack(scale_bboxes_per_image, dim=0)
            gt_scale_bboxes.append(scale_bboxes_per_image)
        
        gt_bboxes = []
        for i_img in range(num_imgs):
            gt_bboxes_per_gt = []
            for i_gt in range(gt_scale_bboxes[i_img].size(1)):
                gt_bboxes_per_layer = []
                for i_layer in range(self.bbox_head.cam_layer):
                    gt_bboxes_per_layer.append(gt_scale_bboxes[i_img][i_layer][i_gt])
                gt_bboxes_per_layer = torch.stack(gt_bboxes_per_layer, dim=0)
                gt_bboxes_per_gt.append(gt_bboxes_per_layer)
            gt_bboxes_per_gt = torch.stack(gt_bboxes_per_gt, dim=0)
            gt_bboxes.append(gt_bboxes_per_gt)
            
        if self.with_mil:
            mil_out = self._mil_forward_train(roi_feature_map, None,
                                                gt_bboxes, gt_labels,
                                                img_metas, return_index=return_mask)
        if return_mask:
            gt_box_index = mil_out[2]
            pseudo_gt_mask, ignore_mask = self.get_pseudo_gt_masks_from_point_attn(cam_maps_images, gt_box_index)
            # points_attn_maps_images: list, length=#Imgs, points_attn_maps_images[i].shape: [n_layers, n_gts_i, H, W]
            # gt_box_index: tuple, length=#Imgs, gt_box_index[i]: [#gts_i, ]
            return gt_labels, mil_out[0], mil_out[1], pseudo_gt_mask, ignore_mask
        else:
            return gt_labels, mil_out[0], mil_out[1] 
    
#     def seed_pseudo_gt(self,
#                       x,
#                       img_metas,
#                       proposal_list,
#                       gt_bboxes,
#                       gt_labels,
#                       gt_bboxes_ignore=None,
#                       gt_masks=None,
#                       vit_feat=None,
#                       img=None,
#                       point_init=None,
#                       point_cls=None,
#                       point_reg=None,
#                       imgs_whwh=None,
#                       attns=None,
#                       gt_points=None,
#                       gt_points_labels=None,
#                      ):
#         num_imgs = point_reg.size(0)
#         if self.train_cfg.get('point_assigner'):
#             num_proposals = point_reg.size(1)
#             imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
#             # 需要先获得point预测结果（point_reg sigmoid之后直接当作绝对位置,相对位置不太好弄，没有w，h无法用decoder)
#             point_assign_results = []
#             for i in range(num_imgs):
#                 normalize_point_cc = point_reg[i].detach()
#                 assign_result = self.point_assigner.assign(
#                     normalize_point_cc, point_cls[i], gt_points[i],
#                     gt_points_labels[i], img_metas[i]
#                 )
#                 point_sampling_result = self.point_sampler.sample(
#                     assign_result, point_reg[i], 
#                     (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2
#                 )
#                 point_assign_results.append(point_sampling_result)
#             pos_inds = [sample_results.pos_inds for sample_results in point_assign_results]
            
#             labels, _, point_targets, _ = self.get_targets(
#                 point_assign_results, gt_points, gt_points_labels, self.train_cfg,
#                 concat=False)
            
#         patch_h, patch_w = x[2].size(-2), x[2].size(-1)
#         num_proposals = point_cls.size(1)
#         # attention maps
#         joint_attentions = attns_project_to_feature(attns[self.bbox_head.cam_layer:])
#         cams = joint_attentions[-1][:, -num_proposals:, 1:-num_proposals].reshape(-1, num_proposals, patch_h, patch_w)
#         cams = F.interpolate(cams, (patch_h * 16, patch_w * 16), mode='bilinear')
# #         joint_attentions = attns_project_to_feature(attns[self.bbox_head.cam_layer:])
# #         cams = joint_attentions[-1][:, -num_proposals:, 1:-num_proposals]
#         # seed proposal
#         scores = point_cls.sigmoid()  # 获得proposal的真实得分
#         scores, label_inds = scores.max(-1)  # 获得proposal label
#         points_locations = imgs_whwh * point_reg  # 获得最终点的位置
        
#         pseudo_gt_labels = []
#         pseudo_gt_bboxes = []
        
#         for scores_per_img, pseudo_labels_per_img, point_locations_per_img, cam_per_img, pseudo_inds, label, gt_point in zip(scores, label_inds, points_locations, cams, pos_inds, labels, point_targets):
# #             pseudo_inds = scores_per_img >= self.bbox_head.seed_score_thr
# #             if sum(pseudo_inds) == 0:
# #                 _, pseudo_inds = scores_per_img.max(0)
# #                 pseudo_inds = pseudo_inds.unsqueeze(0)
            
# #             pseudo_scores = scores_per_img[pseudo_inds]
# #             pseudo_labels = pseudo_labels_per_img[pseudo_inds]
# #             pseudo_labels_per_img = pseudo_labels.type_as(gt_points_labels[0])
#             pseudo_labels_per_img = label[pseudo_inds]
# #             pseudo_point_locations = point_locations_per_img[pseudo_inds]
#             gt_points_per_img = gt_point[pseudo_inds]
        
#             pseudo_gt_bboxes_per_img = []
#             cam_ = cam_per_img[pseudo_inds]
#             cam_ = cam_.detach().cpu().numpy()
# #             pseudo_point_locations = pseudo_point_locations.detach().cpu().numpy()
#             gt_points_per_img = gt_points_per_img.detach().cpu().numpy()
            
# #             for c, p in zip(cam_, pseudo_point_locations):
#             for c, p in zip(cam_, gt_points_per_img):
#                 c = (c - c.min()) / (c.max() - c.min())
#                 pseudo_gt_bbox = get_multi_bboxes(c,
#                                                   p, 
#                                                   cam_thr=self.bbox_head.seed_thr, 
#                                                   area_ratio=self.bbox_head.seed_multiple)
#                 pseudo_gt_bbox = torch.as_tensor(pseudo_gt_bbox).type_as(gt_points[0])
#                 pseudo_gt_bboxes_per_img.append(pseudo_gt_bbox)
            
#             del cam_
#             del pseudo_point_locations
            
#             pseudo_gt_bboxes_per_img = torch.cat(pseudo_gt_bboxes_per_img, dim=0)
            
#             pseudo_gt_bboxes.append(pseudo_gt_bboxes_per_img)
#             pseudo_gt_labels.append(pseudo_labels_per_img)
            
#         return pseudo_gt_labels, pseudo_gt_bboxes
        
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
                      imgs_whwh=None,
                      attns=None,
                      gt_points=None,
                      gt_points_labels=None,
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
                assign_result = self.point_assigner.assign(
                    normalize_point_cc, point_cls[i], gt_points[i],
                    gt_points_labels[i], img_metas[i]
                )
                point_sampling_result = self.point_sampler.sample(
                    assign_result, point_reg[i], 
                    (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2
                )
                point_assign_results.append(point_sampling_result)
            bbox_targets = self.get_targets(
                point_assign_results, gt_points, gt_points_labels, self.train_cfg,
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
        
        
        # bbox head forward and loss
#         if self.with_bbox_rec:
#             bbox_results = self._bbox_forward_train(x, sampling_results,
#                                                     gt_bboxes, gt_labels,
#                                                     img_metas)
          
#             losses.update(bbox_results['loss_bbox'])
#             losses.update(bbox_results['loss_bbox_rec'])

            
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            
        # mask head forward and loss
        if self.with_mask:
            # gt_masks_pseudo = self.get_pseudo_gt_masks_from_point_attn()
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
            
        # mae head forward and loss
        if self.with_mae_head:
            loss_rec = self.mae_head(vit_feat, img)
            losses.update(loss_rec)
        
        return losses

    def get_pseudo_gt_masks_from_point_attn(self, cams, gt_index):
        # points_attn_maps_images: list, length=#Imgs, points_attn_maps_images[i].shape: [n_layers, n_gts_i, H, W]
        # gt_box_index: tuple, length=#Imgs, gt_box_index[i]: [n_gts_i, ]
        masks = []
        ignore_mask = []
        for cam, idx in zip(cams, gt_index):
            # print(f'*****************idx: {idx}********************')
            # import pdb; pdb.set_trace()
            if torch.numel(cam[0]) == 0:
                masks.append([])
                ignore_mask.append([])
                continue

            masks.append(BitmapMasks(cam[0][idx, torch.arange(idx.shape[0])].cpu().numpy().astype(np.uint8), \
                                    height=cam[0].shape[-2], width=cam[0].shape[-1]))
            ignore_mask.append(cam[1][idx].cpu().numpy().astype(np.uint8))
        
        return masks, ignore_mask

    def _bbox_forward(self, x, rois, sampling_results=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        
        if isinstance(self.bbox_roi_extractor, list):
            rois_, restore_inds = rois_chunk([[0, 64], [64, 128], [128, 10000]], rois)
            rois_, restore_inds = check_empty(rois_, restore_inds)
            cls_scores = []
            bbox_preds = []
            for extractor, r, head in zip(self.bbox_roi_extractor, rois_, self.bbox_head):
                bbox_feats = extractor(x[:extractor.num_inputs], r)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                if self.with_bbox_rec:
                    assert True
                else:
#                     if len(bbox_feats) == 0:
# #                         bbox_feats = extractor(x[:extractor.num_inputs], rois[-1:])
# #                         cls_score, bbox_pred = head(bbox_feats) # 随机选择一个反例训练
#                         num_classes = head.num_classes
#                         out_dim_reg = 4 if head.reg_class_agnostic else 4 * num_classes
#                         cls_score = torch.empty((0, num_classes + 1), dtype=torch.float16).to(bbox_feats.device)
#                         bbox_pred = torch.empty((0, out_dim_reg), dtype=torch.float16).to(bbox_feats.device)
#                     else:
                    cls_score, bbox_pred = head(bbox_feats)
                    
                    cls_scores.append(cls_score)
                    bbox_preds.append(bbox_pred)
                    
            cls_scores = torch.cat(cls_scores)
            bbox_preds = torch.cat(bbox_preds)
            
            cls_score = torch.zeros_like(cls_scores, dtype=torch.float16).to(bbox_feats.device)
            bbox_pred = torch.zeros_like(bbox_preds, dtype=torch.float16).to(bbox_feats.device)
            
            for restore_ind, cls, bbox in zip(restore_inds, cls_scores, bbox_preds):
                cls_score[restore_ind] = cls
                bbox_pred[restore_ind] = bbox
                
            bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
                    
            return bbox_results
        
        else:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            if self.with_bbox_rec:
    #             cls_score, bbox_pred, x_rec, x_bbox, x_rec_token, x_det_token  = self.bbox_head(bbox_feats)
                cls_score, bbox_pred, rec_cls_score, rec_bbox_pred  = self.bbox_head(bbox_feats,
                                                                                     sampling_results=sampling_results)
                bbox_results = dict(
                    cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, 
                    rec_cls_score=rec_cls_score, rec_bbox_pred=rec_bbox_pred)
    #             cls_score, bbox_pred = self.bbox_head(bbox_feats)
    #             bbox_results = dict(
    #                 cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
            else:
                cls_score, bbox_pred = self.bbox_head(bbox_feats)

                bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
            return bbox_results
        
    def _mil_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, return_index=False):
        rois = bbox2roi([gt_bboxes_per_image_per_layer.reshape(-1, 4) for gt_bboxes_per_image_per_layer in gt_bboxes])
#         gt_labels = [gt_label.unsqueeze(-1).repeat()for gt_label in gt_labels]
        
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        gt_index, mil_loss = self.mil_head(bbox_feats, gt_labels=gt_labels)
        
        losses = {'mil_loss': mil_loss}
        
        split_lengths = [len(g) for g in gt_bboxes]
        gt_bboxes = torch.cat(gt_bboxes)
        gt_bboxes = torch.gather(gt_bboxes, dim=1,
                                 index=gt_index.reshape(-1, 1, 1).repeat(1, 1, 4)).reshape(-1, 4)
        gt_bboxes = list(torch.split(gt_bboxes, split_lengths, dim=0))
        
        if return_index:
            return gt_bboxes, losses, gt_index.split(split_lengths, dim=0)
        else:
            return gt_bboxes, losses
    
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        
        bbox_results = self._bbox_forward(x, rois, sampling_results=sampling_results)

        bbox_targets = self.bbox_head[0].get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg) if isinstance(self.bbox_head, nn.ModuleList) else self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg) 
        if self.with_bbox_rec:
            loss_bbox = self.bbox_head[0].loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'],
                                            rois,
                                            *bbox_targets) if isinstance(self.bbox_head, nn.ModuleList) else self.bbox_head.loss(bbox_results['cls_score'],
                                                                                    bbox_results['bbox_pred'],
                                                                                    rois,
                                                                                    *bbox_targets)
#             bbox_results.update(loss_bbox=loss_bbox)
            loss_bbox_rec = self.bbox_head[0].loss_(bbox_results['rec_cls_score'],
                                            bbox_results['rec_bbox_pred'],
                                            rois,
                                            *bbox_targets) if isinstance(self.bbox_head, nn.ModuleList) else self.bbox_head.loss_(bbox_results['rec_cls_score'],
                                                                                     bbox_results['rec_bbox_pred'],
                                                                                     rois,
                                                                                     *bbox_targets)
            bbox_results.update(loss_bbox=loss_bbox, loss_bbox_rec=loss_bbox_rec)
            
        else:   
            loss_bbox = self.bbox_head[0].loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets) if isinstance(self.bbox_head, nn.ModuleList) else self.bbox_head.loss(bbox_results['cls_score'],
                                                                                    bbox_results['bbox_pred'], rois,
                                                                                    *bbox_targets)

            bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[0].num_classes if isinstance(self.bbox_head, nn.ModuleList) else self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

        
    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.bbox_head.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples).type_as(pos_gt_labels)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 2).type_as(pos_gt_bboxes)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 2).type_as(pos_gt_bboxes)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.point_pos_weight <= 0 else cfg.point_pos_weight
            label_weights[pos_inds] = pos_weight
#             if not self.reg_decoded_bbox:
#                 pos_bbox_targets = self.bbox_coder.encode(
#                     pos_bboxes, pos_gt_bboxes)
#             else:
            pos_bbox_targets = pos_gt_bboxes 
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        """"Loss function of DIIHead, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        bg_class_ind = self.bbox_head.num_classes 
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_point_cls'] = self.bbox_head.loss_point_cls(
                    cls_score.float(),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_point_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
                if torch.isnan(losses['loss_point_cls']):
                    np.save('cls_score.npy', cls_score.cpu().detach().numpy())
                    np.save('labels.npy', labels.cpu().detach().numpy())
                    np.save('label_weights.npy', label_weights.cpu().detach().numpy())
                    print(avg_factor, num_pos)
                    print('asfqwezxc')
                    exit()
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  2)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              2)[pos_inds.type(torch.bool)]
                losses['loss_point'] = self.bbox_head.loss_point(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_point'] = bbox_pred.sum() * 0
        return losses