from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss

import pdb
from mmcv.ops import point_sample
import numpy as np
import cv2
import mmcv


class ClassAgnosticSupervisionPointGenerator(object):
    def __init__(self, point_strides=16, mask_thr=0.75, point_thr=0.75) -> None:
        self.point_strides = point_strides
        self.mask_thr = mask_thr
        self.point_thr = point_thr

    def get_pred_by_sample(self, anchor_list, pts_preds):
        '''
        Args:
            anchor_list: list of anchor points. List[Tensor(num_parts + num_obj, 2) x batch_size]
            pts_preds: list of offsets predicted on each feature points. Tensor(batch_size, num_points x 2, feat_h, feat_w)
            lvl: idx of the fpn level being used.
        
        Returns:
            preds: list of predicted points of each anchors. List[Tensor(num_parts + num_obj, num_points, 2) x batch_size]
        '''
        H, W = pts_preds[0].shape[-2:]
        H *= self.point_strides
        W *= self.point_strides
        preds = []
        for i_img, points in enumerate(anchor_list):
            points_ = points.clone()
            points_[:, 0] /= W
            points_[:, 1] /= H
            sampled_offset = point_sample(pts_preds[i_img][None], points_[None])[0] * self.point_strides
            sampled_offset = sampled_offset.unflatten(0, (-1, 2)).permute(2,0,1)
            pred_points = sampled_offset + points[:, None, :]
            preds.append(pred_points.detach())
        return preds

    def filter_with_region(self, cand_points, core_regions, ctr_pts_pred, num_parts):
        '''
        Args: 
            cand_points: List[Tensor(num_parts, num_points, 2) x batch_size]
            core_regions: List[Tensor(num_objs, H, W) x batch_size]
            num_parts: number of parts of each objects. List[List[Int x num_objs] x batch_size]
        Returns:
            cand_scores: List[Tensor(num_parts, num_points) x batch_size]
        '''
        device = ctr_pts_pred.device
        H, W = ctr_pts_pred[0].shape[-2:]
        H *= self.point_strides
        W *= self.point_strides
        ctr_cands = self.get_pred_by_sample(cand_points, ctr_pts_pred)
        cand_scores = []
        keep_cands  = []
        keep_cands_score = []
        for c_c, c_r, c_p, n_p in zip(ctr_cands, core_regions, cand_points, num_parts):
            # 这一块效率非常低，可以变成先用框截图出物体，然后resize到56x56的大小，然后再算mask. 或者用更小的分辨率，比如1/16的分辨率或者1/8的
            mask_cands = torch.tensor(self.contour_to_mask(c_c.long().cpu().numpy(), H, W), dtype=torch.bool, device=ctr_pts_pred.device)
            core_repeat= c_r.repeat_interleave(torch.tensor(n_p, device=device), dim=0)
            mask_score = (mask_cands * core_repeat).sum(dim=[-1, -2]) / core_repeat.sum(dim=[-1, -2]).clamp(1e-4) # 分母用哪个？
            keep_idx = mask_score > self.point_thr
            cand_scores.append(mask_score)
            cand_split = c_p.split(n_p)
            keep_cand_obj = []
            keep_cand_score_obj = []
            for c_s, idx, m_s in zip(cand_split, keep_idx.split(n_p), mask_score.split(n_p)):
                keep_cand_obj.append(c_s[idx])
                keep_cand_score_obj.append(m_s[idx])
            keep_cands.append(keep_cand_obj)
            keep_cands_score.append(keep_cand_score_obj)
        return cand_scores, keep_cands, keep_cands_score

    def contour_to_mask(self, ctr, H, W):
        part_masks = np.zeros((ctr.shape[0], H, W), dtype=np.uint8)
        for i, c in enumerate(ctr):
            hull = cv2.convexHull(c)
            cv2.fillConvexPoly(part_masks[i], hull, 1)
        return part_masks

    def get_core_region(self, pts, ctr_pts_pred, num_parts):
        '''
        Args:
            pts: points to be calculate core regions. List[List[Tensor(num_parts+1, 2) x num_obj] x batch_size]
            ctr_pts_pred: list of offsets predicted on each feature points. Tensor(batch_size, num_points x 2, feat_h, feat_w)
            num_parts: List[List[Int x num_parts] x num_objs]
        Return:
            core_regions: core regions of each objects. List[Tensor(num_obj, H, W) x batch_size]
        '''
        H, W = ctr_pts_pred[0].shape[-2:]
        H *= self.point_strides
        W *= self.point_strides
        pts_cat = [torch.cat(p) for p in pts]
        ctr_pts = self.get_pred_by_sample(pts_cat, ctr_pts_pred)
        core_regions = []
        for ctr, n_p in zip(ctr_pts, num_parts):
            part_masks = self.contour_to_mask(ctr.long().cpu().numpy(), H, W)
            obj_masks = ctr_pts_pred[0].new_tensor(part_masks).split(n_p, dim=0)
            obj_masks = torch.cat([m.sum(0, keepdim=True) for m  in obj_masks], dim=0)
            max_val = obj_masks.flatten(1, 2).max(dim=1)[0]
            obj_masks /= max_val[:, None, None]
            core_regions.append(obj_masks > self.mask_thr)
        return core_regions

    def get_cand_points(self, pts, sem_pts_pred, num_parts):
        '''
        Args:
            pts: points to be calculate core regions. List[List[Tensor(num_parts+1, 2) x num_obj] x batch_size]
            sem_pts_pred: list of offsets predicted on each feature points. List[Tensor(num_points x 2, feat_h, feat_w) x batch_size]
            num_parts: List[Int x num_obj]
        Return:
            sem_pts: core regions of each objects. List[Tensor(num_parts+num_objs, num_points, 2) x batch_size]
        '''
        pts_cat = [torch.cat(p) for p in pts]
        sem_pts = self.get_pred_by_sample(pts_cat, sem_pts_pred)
        # sem_pts = [p.split(n_p, dim=0) for p, n_p in zip(sem_pts, num_parts)]
        return sem_pts

    def gen_supervision_point(self, sem_pts_pred, ctr_pts_pred, init_pts, *args, **kwargs):
        '''
        Args:
            sem_pts_pred: semantic region points for each patch. Tensor(batch_size, num_points x 2, feat_h, feat_w)
            ctr_pts_pred: contour region points for each patch. Tensor(batch_size, num_points x 2, feat_h, feat_w)
            init_pts: List[List[Tensor(num_parts, 2) x num_obj] x batch_size]

        Return:
            cand_points:
        '''
        num_parts = [[p.shape[0] for p in i_p] for i_p in init_pts]
        core_regions = self.get_core_region(init_pts, ctr_pts_pred, num_parts)
        # cand_points  = self.get_cand_points(init_pts, sem_pts_pred, num_parts)
        return self.filter_with_region([torch.cat(p) for p in init_pts], core_regions, ctr_pts_pred, num_parts) + (core_regions, )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.gen_supervision_point(*args, **kwds)
    
