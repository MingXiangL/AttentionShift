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
import random
from cc_torch import connected_components_labeling
from ..utils import ObjectQueues, ObjectFactory, cosine_distance, cosine_distance_part

# from torchpq.clustering import MultiKMeans
def random_select_half(point_list_img):
    for i_img in range(len(point_list_img)):
        for i_obj in range(len(point_list_img[i_img])):
            N = point_list_img[i_img][i_obj].shape[0]
            idx = random.sample(range(N) , N//2)
            point_list_img[i_img][i_obj] = point_list_img[i_img][i_obj][idx]
    return point_list_img

def get_bbox_from_cam_fast(cam, point, cam_thr=0.2, area_ratio=0.5, 
                      img_size=None, box_method='expand', erode=False):
    img_h, img_w = img_size
    cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp(1e-6)
    # cam_thr = cam_thr * cam.max()
    cam[cam >= cam_thr] = 1 # binary the map
    cam[cam < cam_thr] = 0
        
    labeled_components = connected_components_labeling(cam.to(torch.uint8))
    labels = labeled_components.unique()
    
    areas = []
    label_masks = []
    for label in labels: # label=0 为背景 filter the area with little area
        if label == 0:
            continue
        label_mask = (labeled_components == label)
        area = label_mask.sum()
        label_masks.append(label_mask)
        areas.append(area)
    label_masks = torch.stack(label_masks)
    areas = torch.stack(areas)
    max_area = areas.max()
    remained_label_masks = label_masks[areas >= area_ratio * max_area].sum(0).bool()
    # remained_label_mask: value threshold + area threshold
    
    coordinates = torch.nonzero(remained_label_masks).to(torch.float32)
    if len(coordinates) == 0:
        estimated_bbox = cam.new_tensor([[0, 0, 1, 1]])
    else:
        proposal_xmin = coordinates[:, 1].min()
        proposal_ymin = coordinates[:, 0].min()
        proposal_xmax = coordinates[:, 1].max()
        proposal_ymax = coordinates[:, 0].max()
        if box_method == 'min_max':
            estimated_bbox = cam.new_tensor([[proposal_xmin, proposal_ymin, 
                                        proposal_xmax, proposal_ymax]])
        elif box_method == 'expand':
            xc, yc = point
            if abs(xc - proposal_xmin) > abs(xc - proposal_xmax):
                gt_xmin = proposal_xmin
                gt_xmax = xc * 2 -  gt_xmin
                gt_xmax = gt_xmax if gt_xmax < img_w else float(img_w)
            else:
                gt_xmax = proposal_xmax
                gt_xmin = xc * 2 -  gt_xmax
                gt_xmin = gt_xmin if gt_xmin > 0 else 0.0
            if abs(yc - proposal_ymin) > abs(yc - proposal_ymax):
                gt_ymin = proposal_ymin
                gt_ymax = yc * 2 -  gt_ymin
                gt_ymax = gt_ymax if gt_ymax < img_h else float(img_h)
            else:
                gt_ymax = proposal_ymax
                gt_ymin = yc * 2 -  gt_ymax
                gt_ymin = gt_ymin if gt_ymin > 0 else 0.0
            estimated_bbox = cam.new_tensor([[gt_xmin, gt_ymin, gt_xmax, gt_ymax]])
    return estimated_bbox, remained_label_masks


def update_coords_with_semantic_centers(points_coords, points_labels, semantic_centers):
    new_coords = []
    new_labels = []
    num_points = []

    for coords, labels, coord_centers in zip(points_coords, points_labels, semantic_centers):
        if len(coord_centers) == 0:
            new_coords.append(coords)
            new_labels.append(labels)
            num_points.append(new_coords[-1].shape[1])
            continue
        org_neg_labels = (~labels)
        split_list = org_neg_labels.sum(dim=1).tolist()
        org_neg_coords = torch.nn.utils.rnn.pad_sequence(coords[org_neg_labels].split(split_list, dim=0), padding_value=-1.0)
        org_neg_labels = torch.nn.utils.rnn.pad_sequence(labels[org_neg_labels].split(split_list, dim=0), padding_value=False)
        centers_coords = torch.nn.utils.rnn.pad_sequence(coord_centers, padding_value=-1.0)
        centers_labels = torch.ones(centers_coords.shape[:-1], dtype=org_neg_labels.dtype, device=org_neg_labels.device)
        new_coords.append(torch.cat((org_neg_coords, centers_coords), dim=0).transpose(0, 1))
        new_labels.append(torch.cat((org_neg_labels, centers_labels), dim=0).transpose(0, 1))
        num_points.append(new_coords[-1].shape[1])
    max_num_points = max(num_points)
    new_coords = [F.pad(c, (0, 0, 0, max_num_points - num_points[i]), value=-1) for i, c in enumerate(new_coords)]
    new_labels = [F.pad(l, (0, max_num_points - num_points[i]), value=False) for i, l in enumerate(new_labels)]
    return new_coords, new_labels


def corrosion_batch(cam, corr_size=11):
    return -F.max_pool2d(-cam, corr_size, 1, corr_size//2)

def cosine_shift(prototypes, feats, feats_org=None, tau=0.1, temp=0.1, n_shift=5):
    # prototypes.shape: n_obj, n_block, n_dim
    # feat.shape: n_patches, n_dim
    for i_s in range(n_shift):
        sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
        # weight = torch.where(sim_map >= tau, sim_map, torch.zeros_like(sim_map))
        # weight = weight / weight.sum(1, keepdim=True)
        weight = F.softmax(sim_map/(tau*temp), dim=-1)
        feat_idx = weight.argmax(0, keepdim=True)
        prot_range = torch.arange(prototypes.shape[0], device=feat_idx.device, dtype=feat_idx.dtype)[:, None]
        mask_weight = torch.where(prot_range==feat_idx, torch.ones_like(weight), torch.zeros_like(weight))
        prototypes = torch.matmul(weight * mask_weight, feats)
        tau = update_density(prototypes, feats, feat_idx[0])
    # prototypes = merge_pototypes(prototypes, thr=1-tau)
    if feats_org is not None:
        sim_map = F.cosine_similarity(prototypes[:, None, :], feats_org[None, :, :], dim=-1)
    else:
        sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
    # sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
    # weight = F.softmax(sim_map/(tau*0.1), dim=-1)
    return prototypes, sim_map

def update_density(prototypes, feats, feats_idx):
    density = torch.zeros(prototypes.shape[0], 1, dtype=prototypes.dtype, device=prototypes.device)
    for i_prot, prot in enumerate(prototypes):
        idx_feats_prot = (feats_idx == i_prot)
        num_pix = idx_feats_prot.sum()
        if num_pix > 1:
            dist = (1-F.cosine_similarity(feats[idx_feats_prot][:, None, :], prot[None, None], dim=-1)).mean()
            density[i_prot] = dist

    for i_prot, prot in enumerate(prototypes):
        idx_feats_prot = (feats_idx == i_prot)
        if num_pix <= 1:
            dist = torch.max(density)
            density[i_prot] = dist
    
    # density = density / density.mean()
    return density.clamp(1e-8)

def get_center_coord(maps, rois, obj_label, num_max_keep=50, num_max_obj=3):
    coords = []
    labels = []
    split_size = [0 for _ in range(len(maps))]
    for i_obj, map_ in enumerate(maps):
        if map_.shape[0] == 0:
            continue
        top2 = map_.flatten(1).topk(dim=1, k=1)[0][:, -1, None, None]
        coord_top2 = (map_ >= top2).nonzero().float()
        xmin, ymin, xmax, ymax = rois[i_obj]
        label = obj_label[i_obj]
        map_area_idxsort = (map_>0.9).sum(dim=[-2,-1]).argsort(descending=True, dim=0)
        for i_prot in range(map_.shape[0]):
            if i_prot > num_max_obj:
                break
            coord = (coord_top2[coord_top2[:, 0]==map_area_idxsort[i_prot]].mean(dim=0)[1:].flip(0) + 0.5) * 16 # patch坐标应该位于中心， 上采样16倍
            if (coord[0] >= xmin) & (coord[0] <= xmax) & (coord[1] >= ymin) & (coord[1] <= ymax):
                coords.append(coord)
                labels.append(label)
                split_size[i_obj] += 1

    if len(coords) == 0:
        return (torch.zeros(0, 2, dtype=rois[0].dtype, device=rois[0].device), torch.zeros(0, dtype=obj_label[0].dtype, device=obj_label[0].device)), []
    else:
        coords = torch.stack(coords)
        labels = torch.stack(labels)
        coord_split = coords.split(split_size, dim=0) # coord_split是用来监督Mask Decoder的，所以不需要有数量限制
        if coords.shape[0] > num_max_keep:
            idx_chosen = torch.randperm(coords.shape[0], device=coords.device)[:num_max_keep]
            coords = coords[idx_chosen]
            labels = labels[idx_chosen]
        return (coords, labels), coord_split


def get_center_coord_with_feat(maps, rois, obj_label, vit_feats, num_max_keep=50, num_max_obj=3):
    coords = []
    labels = []
    feats  = []
    split_size = [0 for _ in range(len(maps))]
    for i_obj, map_ in enumerate(maps):
        if map_.shape[0] == 0:
            continue
        top2 = map_.flatten(1).topk(dim=1, k=1)[0][:, -1, None, None]
        coord_top2 = (map_ >= top2).nonzero().float()
        xmin, ymin, xmax, ymax = rois[i_obj]
        label = obj_label[i_obj]
        map_area_idxsort = (map_>0.9).sum(dim=[-2,-1]).argsort(descending=True, dim=0)
        for i_prot in range(map_.shape[0]):
            if i_prot > num_max_obj:
                break
            coord_top2_mean = coord_top2[coord_top2[:, 0]==map_area_idxsort[i_prot]].mean(dim=0)[1:].flip(0)
            coord = (coord_top2_mean + 0.5) * 16 # patch坐标应该位于中心， 上采样16倍
            if (coord[0] >= xmin) & (coord[0] <= xmax) & (coord[1] >= ymin) & (coord[1] <= ymax):
                coords.append(coord)
                labels.append(label)
                feats.append(vit_feats[:, coord_top2_mean[1].long(), coord_top2_mean[0].long()])
                split_size[i_obj] += 1

    if len(coords) == 0:
        return [torch.zeros(0, 2, dtype=rois[0].dtype, device=rois[0].device), torch.zeros(0, dtype=obj_label[0].dtype, device=obj_label[0].device)], [], [], [], split_size, torch.zeros(0, 2, dtype=rois[0].dtype, device=rois[0].device), torch.zeros(0, dtype=obj_label[0].dtype, device=obj_label[0].device)
    else:
        coords = torch.stack(coords)
        labels = torch.stack(labels)
        coords_org = coords.clone()
        labels_org = labels.clone()
        feats = torch.stack(feats)
        coord_split = list(coords.split(split_size, dim=0)) # coord_split是用来监督Mask Decoder的，所以不需要有数量限制
        feats_split = list(feats.split(split_size, dim=0))
        if coords.shape[0] > num_max_keep:
            idx_chosen = torch.randperm(coords.shape[0], device=coords.device)[:num_max_keep]
            coords = coords[idx_chosen]
            labels = labels[idx_chosen]
        return [coords, labels], coord_split, feats_split, feats, split_size, coords_org, labels_org


def filter_maps(maps, pos_maps, neg_maps, pos_thr=0.85, neg_thr=0.8):
    maps_fore = torch.where(maps>0.8, torch.ones_like(maps), torch.zeros_like(maps))
    
    pos_score = (pos_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    # neg_score = (neg_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    pos_idx = (pos_score >= pos_thr)
    # neg_idx = (neg_score >= neg_thr) & (pos_score < 0.5)
    split_size = pos_idx.sum(dim=-1).tolist()
    maps_fore = maps.flatten(0,1)[pos_idx.flatten()].split(split_size, dim=0)
    # maps_back = maps.flatten(0,1)[neg_idx.flatten()]
    return maps_fore, pos_idx


def merge_maps(prototypes, thr=0.95):
    prot_ret = []
    for prot in prototypes:
        prot_obj = []
        if prot.shape[0] == 0:
            prot_ret.append([])
            continue
        sim = F.cosine_similarity(prot[None], prot[:, None], dim=-1)
        sim_triu = torch.where(torch.triu(sim, diagonal=0) >= thr, torch.ones_like(sim), torch.zeros_like(sim))
        for i_p in range(sim_triu.shape[0]):
            weight = sim_triu[i_p]
            if weight.sum() > 0:
                prot_merge = torch.matmul(weight, prot) / (weight.sum() + 1e-8)
                prot_obj.append(prot_merge)
            sim_triu[weight>0] *= 0 
        prot_ret.append(torch.stack(prot_obj))
    return prot_ret


def cal_similarity(prototypes, feat, dim=-1):
    if isinstance(prototypes, list):
        return torch.zeros(0, 0)
    sim = F.cosine_similarity(prototypes[:, None, None,:], feat[None], dim=-1)
    return sim

def box2mask(bboxes, img_size, default_val=0.5):
    N = bboxes.shape[0]
    mask = torch.zeros(N, img_size[0], img_size[1], device=bboxes.device, dtype=bboxes.dtype) + default_val
    for n in range(N):
        box = bboxes[n]
        mask[n, int(box[1]):int(box[3]+1), int(box[0]):int(box[2]+1)] = 1.0
    return mask


def idx_by_coords(map_, idx0, idx1, dim=0):
    # map.shape: N, H, W, ...
    # idx0.shape: N, k
    # idx1.shape: N, k
    N = idx0.shape[0]
    k = idx0.shape[-1]
    idx_N = torch.arange(N, dtype=torch.long, device=idx0.device).unsqueeze(1).expand_as(idx0)
    idx_N = idx_N.flatten(0)
    return map_[idx_N, idx0.flatten(0), idx1.flatten(0)].unflatten(dim=dim, sizes=[N, k])

def get_bg_points(attns, neg_thr=0.01, num_gt=10):
    bg_points = []
    for attn in attns:
        bg_points.append(get_mask_points_single_instance(attn, neg_thr=0.01, num_gt=10))
    
    return torch.stack(bg_points).flip(-1)

def norm_attns(attns):
    N, H, W = attns.shape
    max_val, _ = attns.view(N,-1,1).max(dim=1, keepdim=True)
    min_val, _ = attns.view(N,-1,1).min(dim=1, keepdim=True)
    return (attns - min_val) / (max_val - min_val)

def get_point_cos_similarity_map(point_coords, feats, ratio=1):
    feat_expand = feats.permute(0,2,3,1).expand(point_coords.shape[0], -1, -1, -1)
    point_feats = idx_by_coords(feat_expand, (point_coords[...,1].long()//16*ratio).clamp(0, feat_expand.shape[1]),( point_coords[...,0].long()//16*ratio).clamp(0, feat_expand.shape[2]))
    point_feats_mean = point_feats.mean(dim=1, keepdim=True)
    sim = F.cosine_similarity(feat_expand.flatten(1,2), point_feats_mean, dim=2)
    # sim = torch.cdist(feat_expand.flatten(1,2), point_feats_mean, p=2).squeeze(-1)
    return sim.unflatten(1, (feat_expand.shape[1], feat_expand.shape[2]))

def sample_point_grid(maps, num_points=10, thr=0.2, is_pos=False, gt_points=None):
    ret_coords = []
    for i_obj, map_ in enumerate(maps):
        factor = 1.0
        if is_pos:
            coords = (map_ >= thr*factor).nonzero(as_tuple=False).view(-1, 2)
        else:
            coords = (map_ < thr*factor).nonzero(as_tuple=False).view(-1, 2)
        # coords = coords[:0]
        num_pos_pix = coords.shape[0] 
        
        if num_pos_pix < num_points:
            if is_pos:
                coords_chosen = torch.cat((coords, gt_points[i_obj].repeat(num_points-num_pos_pix, 1)), dim=0)
                ret_coords.append(coords_chosen)
                continue
            else:
                while num_pos_pix < num_points:
                    print(f'factor adjusted from {thr * factor} to {thr * factor * 2}')
                    factor *= 2
                    coords = (map_ < thr*factor).nonzero(as_tuple=False)
                    num_pos_pix = coords.shape[0]

        step = num_pos_pix // num_points
        idx_chosen = torch.arange(0, num_pos_pix, step=step)
        idx_chosen = torch.randint(num_pos_pix, idx_chosen.shape) % num_pos_pix
        coords_chosen = coords[idx_chosen][:num_points]
        ret_coords.append(coords_chosen)
    return torch.stack(ret_coords).flip(-1)

# def sample_point_grid(maps, num_points=10, thr=0.2, is_pos=False):
#     ret_coords = []
#     for map_ in maps:
#         factor = 1.0
#         num_pos_pix = 0
#         count = 0
#         while num_pos_pix < 2*num_points: # 用死循环调整阈值，
#             if is_pos:
#                 coords = (map_ >= thr*factor).nonzero(as_tuple=False)
#             else:
#                 coords = (map_ < thr*factor).nonzero(as_tuple=False)
#             num_pos_pix = coords.shape[0] 
#             if num_pos_pix < 2*num_points:
#                 if is_pos:
#                     print(f'factor adjusted from {thr * factor} to {thr * factor * 0.5}')
#                     factor *= 0.5
#                     if count >= 10:
#                         factor = -1.0
#                     count += 1
#                 else:
#                     print(f'factor adjusted from {thr * factor} to {thr * factor * 2}')
#                     factor *= 2
#         step = num_pos_pix // num_points
#         idx_chosen = torch.arange(0, num_pos_pix, step=step)
#         idx_chosen = torch.randint(num_pos_pix, idx_chosen.shape) % num_pos_pix
#         coords_chosen = coords[idx_chosen][:num_points]
#         ret_coords.append(coords_chosen)
#     return torch.stack(ret_coords).flip(-1)

def get_mask_points_single_box_cos_map(attn_map, pos_thr=0.6, neg_thr=0.1, num_gt=10, i=0, corr_size=21):
    # Parameters:
    #     coords: num_pixels, 2
    #     attn:H, W
    #     cls: scalar,
    # Return:
    #     coords_chosen: num_gt, 2
    #     labels_chosen: num_gt
    device = attn_map.device
    # attn_pos = corrosion(attn_map.float(), corr_size=corr_size)
    coord_pos = (attn_map > pos_thr).nonzero(as_tuple=False)
    coord_neg = (attn_map < neg_thr).nonzero(as_tuple=False)
    coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
    # print(f'coord_pos.shape[0] / coord_neg.shape[0]: {coord_pos.shape[0] / coord_neg.shape[0]}')
    idx_chosen = torch.randperm(coord_pos_neg.shape[0], device=attn_map.device)[:num_gt]
    labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
                                torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
    if idx_chosen.shape[0] < num_gt:
        if idx_chosen.shape[0] == 0:
            coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
            print(f'**************一个点都没有找到!**************')
            # 这些-1的点会在point ignore里被处理掉
            return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
        else:
            idx_chosen = fill_in_idx(idx_chosen, num_gt)
    coords_chosen = coord_pos_neg[idx_chosen]
    labels_chosen = labels_pos_neg[idx_chosen]

    return coords_chosen, labels_chosen


def get_mask_points_single_box_cos_map_fg_bg(map_fg, map_bg, pos_thr=0.6, neg_thr=0.6, num_gt=10, i=0, corr_size=21):
    # Parameters:
    #     coords: num_pixels, 2
    #     attn:H, W
    #     cls: scalar,
    # Return:
    #     coords_chosen: num_gt, 2
    #     labels_chosen: num_gt
    device = map_fg.device
    # attn_pos = corrosion(attn_map.float(), corr_size=corr_size)
    coord_pos = corrosion((map_fg > map_fg.max()*pos_thr).float(), corr_size=corr_size).nonzero(as_tuple=False)
    coord_neg = (map_bg > map_bg.max()*neg_thr).nonzero(as_tuple=False)
    coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
    # print(f'coord_pos.shape[0] / coord_neg.shape[0]: {coord_pos.shape[0] / coord_neg.shape[0]}')
    idx_chosen = torch.randperm(coord_pos_neg.shape[0])[:num_gt].to(device)
    labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
                                torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
    if idx_chosen.shape[0] < num_gt:
        if idx_chosen.shape[0] == 0:
            coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
            print(f'**************一个点都没有找到!**************')
            # 这些-1的点会在point ignore里被处理掉
            return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
        else:
            idx_chosen = fill_in_idx(idx_chosen, num_gt)
    coords_chosen = coord_pos_neg[idx_chosen]
    labels_chosen = labels_pos_neg[idx_chosen]

    return coords_chosen, labels_chosen

############ debug use ############
# def get_mask_points_single_box_cos_map(attn_map, pos_thr=0.6, neg_thr=0.1, num_gt=10, i=0, corr_size=21):
#     # Parameters:
#     #     coords: num_pixels, 2
#     #     attn:H, W
#     #     cls: scalar,
#     # Return:
#     #     coords_chosen: num_gt, 2
#     #     labels_chosen: num_gt
#     device = attn_map.device
#     max_val = attn_map.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
#     # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
#     attn_pos = corrosion((attn_map > max_val * pos_thr).float(), corr_size=corr_size)
#     coord_pos = (attn_pos).nonzero(as_tuple=False)
#     coord_neg = (attn_map < max_val * neg_thr).nonzero(as_tuple=False)
#     coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
#     idx_chosen = torch.randperm(coord_pos_neg.shape[0], device=attn_map.device)[:num_gt]
#     labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
#                                 torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)

#     if idx_chosen.shape[0] < num_gt:
#         if idx_chosen.shape[0] == 0:
#             coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
#             print(f'**************一个点都没有找到!**************')
#             # 这些-1的点会在point ignore里被处理掉
#             return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
#         else:
#             idx_chosen = fill_in_idx(idx_chosen, num_gt)
#     coords_chosen = coord_pos_neg[idx_chosen]
#     labels_chosen = labels_pos_neg[idx_chosen]

#     return coords_chosen, labels_chosen
############ debug use ############


def get_rolled_sum(map_):
    # map_: num_rf, num_obj, H, W
    print(map_.shape)
    num_obj = map_.shape[1]
    map_expand = map_.unsqueeze(2).expand(-1,-1,num_obj,-1,-1)
    map_mask = torch.ones(1, map_.shape[1], map_.shape[1], 1, 1, dtype=map_.dtype, device=map_.device)
    map_mask[:,range(num_obj), range(num_obj), :, :] = 0
    return (map_ + (map_expand * map_mask).max(dim=2)[0])



# def get_refined_similarity(point_coords, feats, ratio=1, refine_times=1, tau=0.85, is_select=False):
#     cos_map = get_point_cos_similarity_map(point_coords, feats, ratio=ratio)
#     idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
#     range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
#     cos_rf = []
#     bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
#     if is_select:
#         # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
#         cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map)))
#     else:
#         cos_rf.append(cos_map.clone())
#     cos_map1 = cos_map.clone()

#     for i in range(refine_times):
#         thr = cos_map1.max() * tau
#         cos_map1[cos_map1 < thr] *= 0
#         feats_mask = feats * cos_map1.unsqueeze(1)
#         feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
#         cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
#         if is_select:
#             # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
#             cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1)))
#         else:
#             cos_rf.append(cos_map1.clone())

#     return torch.stack(cos_rf)


def get_refined_similarity_prototypes(prototypes_fg, prototypes_bg, feats, bboxes, ratio=1, refine_times=1, tau=0.85, is_select=False, return_feats=False):
    cos_sim_fg = F.cosine_similarity(feats.flatten(1).permute(1, 0)[None, :, :], prot_fg[:, None, :], dim=-1).unflatten(1, vit_feat.shape[-2:])
    cos_sim_bg = F.cosine_similarity(feats.flatten(1).permute(1, 0)[None, :, :], prot_bg[:, None, :], dim=-1).unflatten(1, vit_feat.shape[-2:])
    cos_map = torch.cat((cos_sim_fg, cos_sim_bg), dim=0)
    cos_map1 = cos_map.clone()
    cos_rf = []
    bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
    # cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
    if is_select:
        # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
        cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
        # max_val = cos_map.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        # cos_map = cos_map / (max_val + 1e-8)
        idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
        range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
        cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map.clone(), torch.zeros_like(cos_map)))
    else:
        cos_rf.append(cos_map.clone())

    for i in range(refine_times):
        # fg_map = cos_map1.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
        # cos_map1 *= fg_map
        # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
        max_val = cos_map1.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        thr = max_val * tau
        cos_map1[cos_map1 < thr] *= 0
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        if is_select:
            # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
            # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
            cos_map1[:bboxes.shape[0]] = cos_map1[:bboxes.shape[0]] * bbox_mask
            idx_max_aff = cos_map1.argmax(0, keepdim=True).expand_as(cos_map1)
            range_obj = torch.arange(cos_map1.shape[0], device=cos_map1.device)
            cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1.clone(), torch.zeros_like(cos_map1)))
        else:
            cos_rf.append(cos_map1.clone())

    return torch.stack(cos_rf)


# def get_refined_similarity(point_coords, feats, bboxes, ratio=1, refine_times=1, tau=0.85, is_select=False, return_feats=False):
#     cos_map = get_point_cos_similarity_map(point_coords, feats, ratio=ratio)
#     # fg_map = cos_map.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
#     # cos_map *= fg_map
#     cos_map1 = cos_map.clone()
#     cos_rf = []
#     bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
#     idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
#     # cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
#     if is_select:
#         # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
#         cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
#         # max_val = cos_map.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
#         # cos_map = cos_map / (max_val + 1e-8)
#         # idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
#         range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
#         cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map)))
#     else:
#         cos_rf.append(cos_map.clone())

#     for i in range(refine_times):
#         # fg_map = cos_map1.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
#         # cos_map1 *= fg_map
#         # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
#         max_val = cos_map1.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
#         thr = max_val * tau
#         cos_map1[cos_map1 < thr] *= 0
#         # weight = F.softmax(cos_map1.flatten(-2), dim=-1).unflatten(-1, cos_map1.shape[-2:]).unsqueeze(1)
#         # feats_mask = feats * weight
#         # feats_mask = feats_mask.sum([2,3], keepdim=True)
#         feats_mask = feats * cos_map1.unsqueeze(1)
#         feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
#         cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
#         # cos_map1 = -torch.cdist(feats.flatten(-2).permute(0,2,1).expand(feats_mask.shape[0],-1,-1), feats_mask.flatten(-2).permute(0,2,1), p=2).squeeze(-1).unflatten(1, feats.shape[-2:])
#         if is_select:
#             # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
#             # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
#             cos_map1[:bboxes.shape[0]] = cos_map1[:bboxes.shape[0]] * bbox_mask
#             # idx_max_aff = cos_map1.argmax(0, keepdim=True).expand_as(cos_map1)
#             range_obj = torch.arange(cos_map1.shape[0], device=cos_map1.device)
#             cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1.clone(), torch.zeros_like(cos_map1)))
#         else:
#             cos_rf.append(cos_map1.clone())

#     if return_feats:
#         return torch.stack(cos_rf), feats_mask
#     return torch.stack(cos_rf)

# def get_refined_similarity(point_coords, feats, bboxes, ratio=1, refine_times=1, tau=0.85, is_select=False):
#     cos_map = get_point_cos_similarity_map(point_coords, feats, ratio=ratio)
#     # fg_map = cos_map.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
#     # cos_map *= fg_map
#     cos_map1 = cos_map.clone()
#     cos_rf = []
#     bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
#     # cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
#     if is_select:
#         # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
#         cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
#         # max_val = cos_map.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
#         # cos_map = cos_map / (max_val + 1e-8)
#         idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
#         range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
#         cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map.clone(), torch.zeros_like(cos_map)))
#     else:
#         cos_rf.append(cos_map.clone())

#     for i in range(refine_times):
#         # fg_map = cos_map1.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
#         # cos_map1 *= fg_map
#         # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
#         max_val = cos_map1.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
#         thr = max_val * tau
#         cos_map1[cos_map1 < thr] *= 0
#         feats_mask = feats * cos_map1.unsqueeze(1)
#         feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True).clamp(1e-8))
#         cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
#         if is_select:
#             # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
#             # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
#             cos_map1[:bboxes.shape[0]] = cos_map1[:bboxes.shape[0]] * bbox_mask
#             idx_max_aff = cos_map1.argmax(0, keepdim=True).expand_as(cos_map1)
#             range_obj = torch.arange(cos_map1.shape[0], device=cos_map1.device)
#             cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1.clone(), torch.zeros_like(cos_map1)))
#         else:
#             cos_rf.append(cos_map1.clone())

#     return torch.stack(cos_rf)

def get_refined_similarity(point_coords, feats, bboxes, ratio=1, refine_times=1, tau=0.85, is_select=False):
    cos_map = get_point_cos_similarity_map(point_coords, feats, ratio=ratio)
    # fg_map = cos_map.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
    # cos_map *= fg_map
    cos_map1 = cos_map.clone()
    cos_rf = []
    bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
    # cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
    if is_select:
        # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
        cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
        # max_val = cos_map.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        # cos_map = cos_map / (max_val + 1e-8)
        idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
        range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
        cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map.clone(), torch.zeros_like(cos_map)))
    else:
        cos_rf.append(cos_map.clone())

    for i in range(refine_times):
        # fg_map = cos_map1.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
        # cos_map1 *= fg_map
        # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
        max_val = cos_map1.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        thr = max_val * tau
        cos_map1[cos_map1 < thr] *= 0
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True).clamp(1e-8))
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        if is_select:
            # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
            # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
            cos_map1[:bboxes.shape[0]] = cos_map1[:bboxes.shape[0]] * bbox_mask
            idx_max_aff = cos_map1.argmax(0, keepdim=True).expand_as(cos_map1)
            range_obj = torch.arange(cos_map1.shape[0], device=cos_map1.device)
            cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1.clone(), torch.zeros_like(cos_map1)))
        else:
            cos_rf.append(cos_map1.clone())

    return torch.stack(cos_rf)


def distance_batch(a, b):
    return sqrt(((a[None,:] - b[:,None]) ** 2).sum(2))

def feat_mean_shift(initial_points, feats, n_shift, shift_func, tau=0.1, return_feats=False, **kwargs):
    feat_prototype = get_point_cos_similarity_map_prototype(initial_points, feats) # b, n_prototype, n_patch
    feat_prototype, sim_map = shift_func(feat_prototype, feats.flatten(-2).transpose(0,1), tau=tau, n_shift=n_shift)
    if return_feats:
        return sim_map.unflatten(-1, feats.shape[-2:]), feat_prototype
    else:
        return sim_map.unflatten(-1, feats.shape[-2:])

def gaussian(dist, bandwidth):
    dist_ = -(dist**2/ (2 * bandwidth)).sum(-1)
    return F.softmax(dist_, dim=1)

def gaussian_shift(prototypes, feats, bandwidth=0.1, n_shift=5, mask=None):
    weight_pi = torch.ones(prototypes.shape[0], 1, dtype=prototypes.dtype, device=prototypes.device)
    for _ in range(n_shift):
        dist = prototypes[:,None,:] - feats[None,:,:]
        # E step
        weight = gaussian(dist, bandwidth)
        # if mask is not None:
        #     weight *= mask
        # M step
        weight = weight * weight_pi
        down = weight.sum(dim=0, keepdim=True)
        # down = torch.where(down > 0, down, torch.ones_like(down))
        weight = weight / down
        
        feat_prototype = torch.matmul(weight, feats) / (weight.sum(dim=1, keepdim=True))
        dist = torch.abs(prototypes[:,None,:] - feats[None,:,:])
        bandwidth = ((weight[..., None] * (dist ** 2)).sum(1, keepdim=True) / (weight.sum(dim=1)[:,None,None])).clamp(1)
        weight_pi = weight.mean(1, keepdim=True)
        # print(f'weight: {weight}')
        print(f'bandwidth: {bandwidth}')
        # print(f'feat_prototype: {feat_prototype}')
        # pdb.set_trace()
    sim_map = gaussian(torch.abs(prototypes[:,None,:] - feats[None,:,:]), bandwidth)
    return feat_prototype, sim_map, bandwidth

def cosine_shift(prototypes, feats, tau=0.1, n_shift=5):
    # prototypes.shape: n_obj, n_block, n_dim
    # feat.shape: n_patches, n_dim
    for _ in range(n_shift):
        sim_map = F.cosine_similarity(feats[None,None,:,:], prototypes[:,:,None,:], dim=-1)
        max_val = sim_map.max(dim=-1, keepdim=True)[0]
        weight = F.softmax(sim_map/tau, dim=-1)
        feat_prototype = (weight[:,:,:,None] * feats[None,None,:,:]).sum(dim=-2)
        # feat_prototype = feat_prototype / weight.sum(-1, keepdim=True)
    sim_map = F.cosine_similarity(feats[None,None,:,:], prototypes[:,:,None,:], dim=-1)
    return feat_prototype, sim_map

def cosine_shift_self(prototypes, feats, feats_org=None, tau=0.1, temp=0.1, n_shift=5):
    # prototypes.shape: n_block, n_dim
    # feat.shape: n_patches, n_dim
    for i_s in range(n_shift):
        sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
        # weight = torch.where(sim_map >= tau, sim_map, torch.zeros_like(sim_map))
        # weight = weight / weight.sum(1, keepdim=True)
        weight = F.softmax(sim_map/(temp), dim=-1)
        feat_idx = weight.argmax(0, keepdim=True)
        prot_range = torch.arange(prototypes.shape[0], device=feat_idx.device, dtype=feat_idx.dtype)[:, None]
        mask_weight = torch.where(prot_range==feat_idx, torch.ones_like(weight), torch.zeros_like(weight))
        prototypes = torch.matmul(weight * mask_weight, feats)
        # if i_s > n_shift // 2:
        #     prototypes = merge_pototypes(prototypes, thr=1-tau)
        # print(f'prototypes.shape: {prototypes.shape}')
        # print(f'feats.shape: {feats.shape}')
        # pdb.set_trace()
        tau = update_density(prototypes, feats, feat_idx[0])
    # prototypes = merge_pototypes(prototypes, thr=1-tau)
    if feats_org is not None:
        sim_map = F.cosine_similarity(prototypes[:, None, :], feats_org[None, :, :], dim=-1)
    else:
        sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
    # sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
    # weight = F.softmax(sim_map/(tau*0.1), dim=-1)
    return prototypes, sim_map

def cosine_shift_batch(prototypes, feats, feats_org=None, tau=0.1, temp=0.1, n_shift=5):
    # prototypes.shape: n_obj, n_block, n_dim
    # feat.shape: n_patches, n_dim
    for i_s in range(n_shift):
        sim_map = F.cosine_similarity(prototypes[:, :, None], feats[:, None], dim=-1)
        weight = F.softmax(sim_map/(temp*tau), dim=-1)
        feat_idx = weight.argmax(1, keepdim=True)
        prot_range = torch.arange(prototypes.shape[1], device=feat_idx.device, dtype=feat_idx.dtype)[None, :, None].expand(prototypes.shape[0], prototypes.shape[1], -1)
        mask_weight = torch.where(prot_range==feat_idx, torch.ones_like(weight), torch.zeros_like(weight))
        prototypes = torch.matmul(weight * mask_weight, feats)
        tau_cp = tau
        tau = update_density_batch(prototypes, feats, mask_weight)
        # print(f'i_s: {i_s}')
        # print(f'prototypes: {prototypes}')
        # print(f'tau: {tau}')
        # print(f'tau.shape: {tau.shape}')
        # pdb.set_trace()
    # prototypes = merge_pototypes(prototypes, thr=1-tau)
    if feats_org is not None:
        sim_map = F.cosine_similarity(prototypes[:, :, None, :], feats_org[None, None, :, :], dim=-1)
    else:
        sim_map = F.cosine_similarity(prototypes[:, :, None, :], feats[None, None, :, :], dim=-1)
    # sim_map = F.cosine_similarity(prototypes[:, None, :], feats[None, :, :], dim=-1)
    # weight = F.softmax(sim_map/(tau*0.1), dim=-1)
    return prototypes.flatten(0,1), sim_map.flatten(0,1)


def update_density(prototypes, feats, feats_idx):
    density = torch.zeros(prototypes.shape[0], 1, dtype=prototypes.dtype, device=prototypes.device)
    for i_prot, prot in enumerate(prototypes):
        idx_feats_prot = (feats_idx == i_prot)
        num_pix = idx_feats_prot.sum()
        if num_pix > 1:
            dist = (1-F.cosine_similarity(feats[idx_feats_prot][:, None, :], prot[None, None], dim=-1)).mean()
            # dist = F.cosine_similarity(feats[idx_feats_prot][:, None, :], prot[None, None], dim=-1).mean()
            # print(f'dist: {dist}')
            # print(f'dist: {dist}')
            # print(f'1-dist: {1-dist}')
            # pdb.set_trace()
            # dist = ((feats[idx_feats_prot] - prot[None]) ** 2).mean(-1).sqrt().mean()
            # dist = dist / torch.log(num_pix + 20)
            density[i_prot] = dist

    for i_prot, prot in enumerate(prototypes):
        idx_feats_prot = (feats_idx == i_prot)
        if num_pix <= 1:
            dist = torch.max(density)
            density[i_prot] = dist
    
    # density = density / density.mean()
    return density.clamp(1e-10)

def update_density_batch(prototypes, feats, mask_weight):
    similarity = F.cosine_similarity(prototypes[:, :, None], feats[:, None], dim=-1)
    density =(similarity * mask_weight).sum(-1)
    density = 1 - torch.where(mask_weight.sum(-1)>=1, density / mask_weight.sum(-1), torch.zeros_like(density))
    # pdb.set_trace()
    # for i_prot, prot in enumerate(prototypes):
    #     idx_feats_prot = (feats_idx == i_prot)
    #     num_pix = idx_feats_prot.sum()
    #     if num_pix > 1:
    #         dist = (1 - F.cosine_similarity(feats[idx_feats_prot][:, None, :], prot[None, None], dim=-1)).mean()
    #         # dist = F.cosine_similarity(feats[idx_feats_prot][:, None, :], prot[None, None], dim=-1).mean()
    #         # print(f'dist: {dist}')
    #         # print(f'dist: {dist}')
    #         # print(f'1-dist: {1-dist}')
    #         # pdb.set_trace()
    #         # dist = ((feats[idx_feats_prot] - prot[None]) ** 2).mean(-1).sqrt().mean()
    #         # dist = dist / torch.log(num_pix + 20)
    #         density[i_prot] = dist

    # for i_prot, prot in enumerate(prototypes):
    #     idx_feats_prot = (feats_idx == i_prot)
    #     if num_pix <= 1:
    #         dist = torch.max(density)
    #         density[i_prot] = dist
    
    # density = density / density.mean()
    return density.clamp(1e-10).unsqueeze(-1)


def merge_pototypes_bandwidth(prototypes, bandwidth, thr=0.99):
    prot_ret = []
    bandwidth_ret = []
    for prot, band in zip(prototypes, bandwidth):
        sim = F.cosine_similarity(prot[:, None, :], prot[None, :, :], dim=-1)
        prot_obj = []
        band_obj = []
        sim_triu = torch.where(torch.triu(sim, diagonal=0) >= thr, torch.ones_like(sim), torch.zeros_like(sim))
        for i_p in range(sim_triu.shape[0]):
            weight = sim_triu[i_p]
            prot_merge = torch.matmul(weight, prot) / (weight.sum() + 1e-8)
            band_merge = torch.matmul(weight, band.squeeze(1)) / (weight.sum() + 1e-8)

            if weight.sum() > 0:
                prot_obj.append(prot_merge)
                band_obj.append(band_merge)
            sim_triu[weight>0] *= 0 #
        prot_ret.append(torch.stack(prot_obj))
        bandwidth_ret.append(torch.stack(band_obj))
    return prot_ret, bandwidth_ret

def merge_pototypes(prot, thr=0.99):
    # prot_ret = []
    # for _, prot in enumerate(prototypes):
    sim = F.cosine_similarity(prot[:, None, :], prot[None, :, :], dim=-1)
    prot_obj = []
    sim_triu = torch.where(torch.triu(sim, diagonal=0) >= thr, torch.ones_like(sim), torch.zeros_like(sim))
    for i_p in range(sim_triu.shape[0]):
        weight = sim_triu[i_p]
        prot_merge = torch.matmul(weight, prot) / (weight.sum() + 1e-8)
        if weight.sum() > 0:
            prot_obj.append(prot_merge)
        sim_triu[weight>0] *= 0 #
    # prot_ret.append(torch.stack(prot_obj))
    return torch.stack(prot_obj)

# def get_point_cos_similarity_map_prototype(point_coords, feats, ratio=1, sim_type='cos'):
#     feat_expand = feats.permute(0,2,3,1).expand(point_coords.shape[0], -1, -1, -1)
#     point_feats = idx_by_coords(feat_expand, (point_coords[...,1].long()//16*ratio).clamp(0, feat_expand.shape[1]),( point_coords[...,0].long()//16*ratio).clamp(0, feat_expand.shape[2]))
#     # point_feats_mean = point_feats.mean(dim=1, keepdim=True)
#     if sim_type == 'cos':
#         sim = - F.cosine_similarity(feat_expand.flatten(1,2).unsqueeze(1), point_feats.unsqueeze(2), dim=-1)
#     elif sim_type == 'l1':
#         sim = torch.abs(feat_expand.flatten(1,2).unsqueeze(1) - point_feats.unsqueeze(2)).mean(dim=-1)
#     else:
#         raise NotImplementedError
#     return point_feats, sim

def get_refined_similarity_mean_shift(point_coords, feats, bboxes, ratio=1, refine_times=1, tau=0.85, is_select=False):
    cos_map = get_point_cos_similarity_map_prototype(point_coords, feats, ratio=ratio)
    # fg_map = cos_map.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
    # cos_map *= fg_map
    cos_map1 = cos_map.clone()
    cos_rf = []
    bbox_mask = box2mask(bboxes//16, cos_map.shape[-2:], default_val=0)
    # cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
    if is_select:
        # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
        cos_map[:bboxes.shape[0]] = cos_map[:bboxes.shape[0]] * bbox_mask
        # max_val = cos_map.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        # cos_map = cos_map / (max_val + 1e-8)
        idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
        range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
        cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map.clone(), torch.zeros_like(cos_map)))
    else:
        cos_rf.append(cos_map.clone())

    for i in range(refine_times):
        # fg_map = cos_map1.argmax(dim=0, keepdim=True)  < bboxes.shape[0]
        # cos_map1 *= fg_map
        # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
        max_val = cos_map1.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1)
        thr = max_val * tau
        cos_map1[cos_map1 < thr] *= 0
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        if is_select:
            # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
            # cos_map1[:bboxes.shape[0]] = bbox_mask * cos_map1[:bboxes.shape[0]]
            cos_map1[:bboxes.shape[0]] = cos_map1[:bboxes.shape[0]] * bbox_mask
            idx_max_aff = cos_map1.argmax(0, keepdim=True).expand_as(cos_map1)
            range_obj = torch.arange(cos_map1.shape[0], device=cos_map1.device)
            cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1.clone(), torch.zeros_like(cos_map1)))
        else:
            cos_rf.append(cos_map1.clone())

    return torch.stack(cos_rf)

def get_cosine_similarity_refined_map(attn_maps, vit_feat, bboxes, thr_pos=0.2, thr_neg=0.1, num_points=20, thr_fg=0.7, refine_times=1, obj_tau=0.85, gt_points=None):
    # attn_maps是上采样16倍之后的，vit_feat是上采样前的，实验表明，上采样后的不太好，会使cos_sim_fg ~= cos_sim_bg
    attn_norm = norm_attns(attn_maps)
    points_bg = sample_point_grid(attn_norm, thr=thr_neg, num_points=num_points)
    points_fg = sample_point_grid(attn_norm, thr=thr_pos, num_points=num_points, is_pos=True, gt_points=gt_points)
    points_bg_supp = sample_point_grid(attn_norm.mean(0, keepdim=True), thr=thr_neg, num_points=num_points)
    # points_bg_supp = torch.cat([sample_point_grid(attn_norm[0].mean(0,keepdim=True)<thr_neg, num_points=num_points) for _ in range(3)],dim=0)
    points_fg = torch.cat((points_fg, points_bg_supp), dim=0)
    cos_sim_fg = F.interpolate(get_refined_similarity(points_fg, vit_feat[None], bboxes=bboxes, refine_times=refine_times, tau=obj_tau, is_select=True), attn_maps.shape[-2:], mode='bilinear')[:,:attn_norm.shape[0]]
    cos_sim_bg = F.interpolate(get_refined_similarity(points_bg, vit_feat[None], bboxes=bboxes, refine_times=refine_times, tau=obj_tau), attn_maps.shape[-2:], mode='bilinear')
    ret_map = (1 - cos_sim_bg) * cos_sim_fg
    map_val = ret_map.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1).clamp(1e-8)
    
    cos_sim_bg = decouple_instance(cos_sim_bg.clone(), ret_map.clone())
    max_val_bg = cos_sim_bg.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1).clamp(1e-8)
#    map_fg = torch.where(ret_map < map_val * thr_fg, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    # map_bg = torch.where(ret_map > map_val * 0.1, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    return ret_map / map_val, cos_sim_bg / max_val_bg, points_fg, points_bg

def get_cos_similarity_map(point_coords, point_labels, feats, ratio=1):
    feat_expand = feats.permute(0,2,3,1).expand(point_coords.shape[0], -1, -1, -1)
    point_feats = idx_by_coords(feat_expand, (point_coords[...,1].long()//16*ratio).clamp(0, feat_expand.shape[1]),( point_coords[...,0].long()//16*ratio).clamp(0, feat_expand.shape[2]))
    point_feats_mean = (point_feats * (point_labels>0).float()).sum(1, keepdim=True) / ((point_labels>0).float().sum(1, keepdim=True) + 1e-6)
    sim = F.cosine_similarity(feat_expand.flatten(1,2), point_feats_mean, dim=2)
    return sim.unflatten(1, (feat_expand.shape[1], feat_expand.shape[2]))

def cal_entropy(map_, bboxes):
    num_obj = bboxes.shape[0]
    entropy = torch.zeros(map_.shape[:-2], dtype=map_.dtype, device=map_.device)
    map_entropy = map_ * torch.log(map_) + (1 - map_) * torch.log(1 - map_)
    for i_obj in range(num_obj):
        box = bboxes[i_obj]
        entropy[..., i_obj] = map_entropy[..., i_obj, box[1]:box[3], box[0]:box[2]].mean(dim=[-1, -2])
    return entropy

def normalize_map(map_):
    max_val = map_.flatten(-2).max(-1,keepdim=True)[0].unsqueeze(-1)
    map_ = (map_ / (max_val + 1e-8))
    return map_

def decouple_instance(map_bg, map_fg):
    map_bg = normalize_map(map_bg)
    map_fg = normalize_map(map_fg)
    map_fack_bg = 1 - (map_fg*0.5 + map_bg*0.5)
    return map_bg + map_fack_bg
    # map_fg_tot = map_fg.sum(dim=1, keepdim=True)
    # map_fg_tot = (1 - map_fg) * map_fg_tot
    # map_fg_idx = map_fg_tot > 0
    # map_bg[map_fg_idx] = map_fg_tot[map_fg_idx]*0.5 + map_bg[map_fg_idx]*0.5
    # return map_bg.clamp(0,1)

# def get_cosine_similarity_refined_map(attn_maps, vit_feat, bboxes, epoch=0, thr_pos=0.2, thr_neg=0.1, num_points=20, thr_fg=0.7, refine_times=1, gt_points=None, return_feats=False, obj_tau=0.85):
#     # attn_maps是上采样16倍之后的，vit_feat是上采样前的，实验表明，上采样后的不太好，会使cos_sim_fg ~= cos_sim_bg
#     attn_norm = norm_attns(attn_maps)
#     # if (gt_points is not None) and (epoch < 3):
#     #     points_bg = sample_point_grid(attn_norm < 0.01, num_points=num_points)
#     #     points_fg = gt_points.unsqueeze(1)
#     #     points_bg_supp = sample_point_grid(attn_norm.mean(0, keepdim=True)  < 0.1, num_points=1)
#     # else:
#     points_fg = sample_point_grid(attn_norm, thr=thr_pos, num_points=num_points, is_pos=True, gt_points=gt_points)
#     points_bg_supp = sample_point_grid(attn_norm.mean(0, keepdim=True), thr=thr_pos, num_points=num_points, is_pos=False)
#     points_bg = sample_point_grid(attn_norm, thr=thr_neg, num_points=num_points)

#     # points_bg_supp = torch.cat([sample_point_grid(attn_norm[0].mean(0,keepdim=True)<thr_neg, num_points=num_points) for _ in range(3)],dim=0)

#     points_fg = torch.cat((points_fg, points_bg_supp), dim=0)
#     # cos_sim_fg = F.interpolate(feat_mean_shift(points_fg, vit_feat, n_shift=refine_times, shift_func=cosine_shift, bboxes=bboxes, refine_times=refine_times, tau=0.1, is_select=True), attn_maps.shape[-2:], mode='bilinear')[:attn_norm.shape[0]]
#     # cos_sim_bg = F.interpolate(feat_mean_shift(points_bg, vit_feat, n_shift=refine_times, shift_func=cosine_shift, bboxes=bboxes, refine_times=refine_times, tau=0.001), attn_maps.shape[-2:], mode='bilinear')
#     # if return_feats:
#     #     cos_sim_fg, prototype_fg = get_refined_similarity(points_fg, vit_feat[None,:,:,:], bboxes=bboxes, refine_times=refine_times, tau=0.85, is_select=True, return_feats=return_feats)
#     #     cos_sim_bg, prototype_bg = get_refined_similarity(points_bg, vit_feat[None,:,:,:], bboxes=bboxes, refine_times=refine_times, tau=0.85, return_feats=return_feats)
#     # else:
#     cos_sim_fg = F.interpolate(get_refined_similarity(points_fg, vit_feat[None,:,:,:], bboxes=bboxes, refine_times=refine_times, tau=obj_tau, is_select=True), attn_maps.shape[-2:], mode='bilinear')[:,:attn_norm.shape[0]]
#     cos_sim_bg = F.interpolate(get_refined_similarity(points_bg, vit_feat[None,:,:,:], bboxes=bboxes, refine_times=refine_times, tau=obj_tau), attn_maps.shape[-2:], mode='bilinear')

#     # cos_sim_fg = F.interpolate(cos_sim_fg, attn_maps.shape[-2:], mode='bilinear')[:,:attn_norm.shape[0]]
#     # cos_sim_bg = F.interpolate(cos_sim_bg, attn_maps.shape[-2:], mode='bilinear')
#     ret_map = (1 - cos_sim_bg) * cos_sim_fg
#     # cos_sim_bg = decouple_instance(cos_sim_bg.clone(), ret_map.clone())

#     # ret_map_bg = (1 - cos_sim_fg) * cos_sim_bg
#     # bbox_mask = box2mask(bboxes, ret_map.shape[-2:], default_val=0.1)
#     # ret_map *= bbox_mask.unsqueeze(0)
#     # idx_max_aff = ret_map.argmax(1, keepdim=True).expand_as(ret_map)
#     # range_obj = torch.arange(ret_map.shape[1], device=ret_map.device)
#     # ret_map = torch.where(idx_max_aff==range_obj[None, :, None, None], ret_map, torch.zeros_like(ret_map))

#     # ret_map_bg = (1 - cos_sim_fg) * cos_sim_bg
#     map_val = ret_map.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
#     # map_val_bg = ret_map_bg.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
#     # map_fg = torch.where(ret_map < map_val * thr_fg, torch.zeros_like(ret_map), torch.ones_like(ret_map))
#     # map_bg = torch.where(ret_map > map_val * 0.1, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    
#     # return ret_map / (map_val + 1e-8), ret_map_bg / (map_val_bg + 1e-8)
#     # return ret_map / (map_val + 1e-8), cos_sim_bg
#     return ret_map / (map_val + 1e-8), cos_sim_bg, points_fg, points_bg
#     # return ret_map[-1] / map_val[-1], cos_sim_fg[-1, attn_norm.shape[0]:]


def get_cosine_similarity_refined_map_initialize_with_prototype(attn_maps, vit_feat, bboxes, prototypes_fg, prototypes_bg, epoch=0, thr_pos=0.2, thr_neg=0.1, num_points=20, thr_fg=0.7, refine_times=1, gt_points=None, return_feats=False):
    
    prot_fg = torch.cat([p.mean(dim=0, keepdim=True) for p in prototypes_fg])
    prot_bg = torch.cat([p.mean(dim=0, keepdim=True) for p in prototypes_bg])
    cos_sim_fg = F.cosine_similarity(vit_feat.flatten(1).permute(1, 0)[None, :, :], prot_fg[:, None, :], dim=-1).unflatten(1, vit_feat.shape[-2:])
    cos_sim_bg = F.cosine_similarity(vit_feat.flatten(1).permute(1, 0)[None, :, :], prot_bg[:, None, :], dim=-1).unflatten(1, vit_feat.shape[-2:])
    cos_sim_fg = F.interpolate(cos_sim_fg[None], attn_maps.shape[-2:], mode='bilinear')[0]
    cos_sim_bg = F.interpolate(cos_sim_bg[None], attn_maps.shape[-2:], mode='bilinear')[0]

    ret_map = (1 - cos_sim_bg) * cos_sim_fg
    cos_sim_bg = decouple_instance(cos_sim_bg.clone(), ret_map.clone())

    map_val = ret_map.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
    map_val_bg = cos_sim_bg.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
    map_val_fg = cos_sim_fg.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1)

    return ret_map / map_val, cos_sim_bg


def get_fgbg_likelihood_gmm(maps, 
                            vit_feat, 
                            bboxes, 
                            prototypes,
                            epoch=0, 
                            thr_pos=0.2, 
                            thr_neg=0.1, 
                            num_points=20, 
                            thr_fg=0.7, 
                            refine_times=1, 
                            gt_points=None, 
                            return_feats=False):
    GMM_models = []
    
    for idx, prot in enumerate(prototypes):
        map_ = maps[idx]
        feat = vit_feat[map_ > 0]
        gmm = GMM_Batch(K=prot.shape[0])
        gmm.fit(feat, feat.shape[0], init_centers=prot)
        GMM_models.append(gmm.clone())
        # prot_fg = torch.cat(prototypes_fg, dim=0)
        # prot_bg = torch.cat(prototypes_bg, dim=0)
        # prototypes = torch.cat((prot_fg, prot_bg), dim=0)

        # gmm_model = GMM_Batch(K=prototypes.shape[0])
        # prob, pred = gmm_model.fit(vit_feat.flatten(1).permute(1,0), batch_size=vit_feat.shape[0], init_centers=prototypes)

def fill_in_idx(idx_chosen, num_gt):
    assert idx_chosen.shape[0] != 0, '不能一个点都不选!'
    if idx_chosen.shape[0] >= num_gt / 2:
        idx_chosen = torch.cat((idx_chosen, idx_chosen[:num_gt-idx_chosen.shape[0]]), dim=0)
    else:
        repeat_times = num_gt // idx_chosen.shape[0]
        idx_chosen = idx_chosen.repeat(repeat_times, 1)
        idx_chosen = fill_in_idx(idx_chosen, num_gt)
    return idx_chosen

def get_point_coords_wrt_box(boxes_coords, point_coords):
    """
    Convert image-level absolute coordinates to box-normalized [0, 1] x [0, 1] point cooordinates.
    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    Returns:
        point_coords_wrt_box (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    """
    with torch.no_grad():
        # point_coords_wrt_box = point_coords.clone().permute(0, 2, 1)
        point_coords_wrt_box = point_coords.clone()
        point_coords_wrt_box[:, :, 0] -= boxes_coords[:, None, 0]
        point_coords_wrt_box[:, :, 1] -= boxes_coords[:, None, 1]
        point_coords_wrt_box[:, :, 0] = point_coords_wrt_box[:, :, 0] / (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_box[:, :, 1] = point_coords_wrt_box[:, :, 1] / (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
    return point_coords_wrt_box

def corrosion(cam, corr_size=11):
    if cam.ndim < 4:
        H, W = cam.shape[-2:]
        cam = cam.view(1, 1, H, W)
        return -F.max_pool2d(-cam, corr_size, 1, corr_size//2).reshape(H, W)
    return -F.max_pool2d(-cam, corr_size, 1, corr_size//2).reshape(H, W)

def expension(cam, expn_size=5):
    return F.max_pool2d(cam, expn_size, 1, expn_size//2)

def open_operation(cam, corr_size=11, expn_size=11):
    cam = corrosion(cam, corr_size)
    cam = expension(cam, expn_size)
    return cam

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
    residual_att = torch.eye(attns_maps.size(2), device=attns_maps.device, dtype=attns_maps.dtype)
    aug_att_mat = attns_maps + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size(), device=aug_att_mat.device, dtype=aug_att_mat.dtype)
    joint_attentions[-1] = aug_att_mat[-1]
    for i in range(2, len(attns_maps) + 1):
        joint_attentions[-i] = torch.matmul(joint_attentions[-(i - 1)], aug_att_mat[-i])
    
    reverse_joint_attentions = torch.zeros(joint_attentions.size(), dtype=joint_attentions.dtype, device=joint_attentions.device)
    
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
class StandardRoIHeadMaskPointSampleDeformAttnReppoints(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
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
                 pca_dim=128,
                 mean_shift_times_local=10,
                 reppoints_head=None,
                 ):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)
            
        if mil_head is not None:
            self.init_mil_head(bbox_roi_extractor, mil_head)
        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)
        if bbox_rec_head is not None:
            self.init_bbox_rec_head(bbox_roi_extractor, bbox_rec_head)
        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()
        
        if mae_head is not None:
            self.mae_head = build_head(mae_head)
            self.with_mae_head = True
        else:
            self.with_mae_head = False

        if reppoints_head is not None:
            self.reppoints_head = build_head(reppoints_head)
            self.with_reppoints_head = True
        else:
            self.with_reppoints_head = False
            
        self.visualize = visualize
        self.epoch_semantic_centers = epoch_semantic_centers
        self.epoch = epoch
        self.num_semantic_points = num_semantic_points
        self.semantic_to_token = semantic_to_token
        self.pca_dim = pca_dim
        self.mean_shift_times_local = mean_shift_times_local
        self.with_deform_sup = False

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
            # if hasattr(self.mil_head, 'pretrained'):
            #     self.mil_head.init_weights(pretrained=pretrained)
            # else:
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
                pseudo_gt_bbox = torch.as_tensor(pseudo_gt_bbox, dtype=pseudo_point_locations.dtype, device=point_cls.device)
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
        pos_idx = attn_maps_cam >= 0.2
        ignore_idx = (attn_maps_cam < 0.2) & (attn_maps_cam > 0.1)
        cams = torch.zeros_like(attn_maps_cam)
        ignore_mask = torch.zeros_like(attn_maps_cam)
        ignore_mask[ignore_idx] = 1.
        cams[pos_idx] = 1.
        
        return cams, ignore_mask

    def get_map_coords(self, h, w, device, dtype):
        range_h = torch.arange(h, device=device, dtype=dtype)
        range_w = torch.arange(w, device=device, dtype=dtype)
        coords_h, coords_w = torch.meshgrid(range_h, range_w)
        coords = torch.stack([coords_h, coords_w], dim=-1)
        return coords
    
    def get_mask_sample_points(self, coords, attn, cls_points, pos_thr=0.1, neg_thr=0.01, num_gt=10):
        # Parameters:
        #     coords: num_pixels, 2
        #     attn: num_layers, num_points, H, W
        #     cls: num_points, scalar
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        attn_map = attn.detach().clone()
        attn_map = attn_map.mean(dim=0) # num_points, num_pixels, 2
        coord_chosen = []
        label_chosen = []
        for i_p, attn_p in enumerate(attn_map):
            cls_p = cls_points[i_p]
            coor, label = self.get_mask_points_single_instance(coords, attn_p, cls_p, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt)
            coord_chosen.append(coor)
            label_chosen.append(label)
        
        coords_chosen = torch.stack(coord_chosen).float()
        labels_chosen= torch.stack(label_chosen)
        return coords_chosen, labels_chosen
        
#     def get_mask_points_single_instance(self, coords, attn_map, cls_p, pos_thr=0.1, neg_thr=0.01, num_gt=10):
#         # Parameters:
#         #     coords: num_pixels, 2
#         #     attn:H, W
#         #     cls: scalar,
#         # Return:
#         #     coords_chosen: num_gt, 2
#         #     labels_chosen: num_gt
#         attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
#         idx_pos = (attn_map > pos_thr).nonzero(as_tuple=False)
#         idx_neg = (attn_map < neg_thr).nonzero(as_tuple=False)
#         num_pos_tot = idx_pos.shape[0]
#         pos_neg_idx = torch.cat([idx_pos, idx_neg], dim=0)
#         num_anno = pos_neg_idx.shape[0]
#         idx_shuffle = np.arange(num_anno)
#         np.random.shuffle(idx_shuffle)
#         idx_shuffle_topk = torch.from_numpy(idx_shuffle[:num_gt]).to(attn_map.device)
#         idx_chosen = pos_neg_idx[idx_shuffle_topk]
#         coords_chosen = coords[idx_chosen[:,0], idx_chosen[:,1]]

#         # TODO: 这里面正例点的数量远远大于反例点的数量，是否需要调节阈值？
#         labels_chosen = (idx_shuffle_topk < num_pos_tot)
#         return coords_chosen, labels_chosen
    # def get_mask_points_single_instance(self, coords, attn_map, cls_p, pos_thr=0.1, neg_thr=0.01, num_gt=10,i=0):
    #     # Parameters:
    #     #     coords: num_pixels, 2
    #     #     attn:H, W
    #     #     cls: scalar,
    #     # Return:
    #     #     coords_chosen: num_gt, 2
    #     #     labels_chosen: num_gt
    #     device = attn_map.device
    #     attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    #     coord_pos = (attn_map > pos_thr).nonzero(as_tuple=False)
    #     coord_neg = (attn_map < neg_thr).nonzero(as_tuple=False)
    #     idx_chosen_pos = torch.randperm(coord_pos.shape[0], device=attn_map.device)[:num_gt//2]
    #     idx_chosen_neg = torch.randperm(coord_neg.shape[0], device=attn_map.device)[:num_gt//2]
    #     coords_chosen_pos = coord_pos[idx_chosen_pos]
    #     coords_chosen_neg = coord_neg[idx_chosen_neg]
    #     coords_chosen = torch.cat([coords_chosen_pos, coords_chosen_neg], dim=0)
    #     labels_chosen = torch.cat((torch.ones(coords_chosen_pos.shape[0], device=device, dtype=torch.bool),
    #                                 torch.zeros(coords_chosen_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
    #     return coords_chosen, labels_chosen

    def get_mask_sample_points_roi(self, attn, rois, pos_thr=0.2, neg_thr=0.5, num_gt=20, corr_size=21):
        # Parameters:
        #     attn: num_layers, num_points, H, W
        #     roi: num_points, 4
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        attn_map = attn.detach().clone()
        attn_map = attn_map.mean(dim=0) # num_points, num_pixels, 2
        coord_chosen = []
        label_chosen = []
        for i_p, attn_p in enumerate(attn_map):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            attn_crop = attn_p[ymin:ymax, xmin:xmax]
            coor, label = self.get_mask_points_single_instance(attn_crop, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt,i=i_p, corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            coord_chosen.append(coor)
            label_chosen.append(label)
        coords_chosen = torch.stack(coord_chosen).float()
        labels_chosen= torch.stack(label_chosen)
        return coords_chosen, labels_chosen

    def get_mask_sample_points_roi_best_attn(self, attn, rois, attn_idx, pos_thr=0.2, neg_thr=0.05, num_gt=20, corr_size=21):
        # Parameters:
        #     attn: num_layers, num_points, H, W
        #     roi: num_points, 4
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        num_points = attn.shape[1]
        attn_map = attn.detach().clone()
        attn_map = attn_map[attn_idx, torch.arange(num_points)] # num_points, num_pixels, 2
        
        coord_chosen = []
        label_chosen = []
        for i_p, attn_p in enumerate(attn_map):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = attn_p.shape[-2:]
            attn_crop = attn_p[ymin:ymax, xmin:xmax]
            coor, label = self.get_mask_points_single_instance(attn_crop, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt,i=i_p, corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            coord_chosen.append(coor)
            label_chosen.append(label)
        coords_chosen = torch.stack(coord_chosen).float()
        labels_chosen= torch.stack(label_chosen)
        return coords_chosen, labels_chosen

    def mean_shift_refine_prototype(self, map_cos_fg, map_cos_bg, prototype_fg, prototype_bg, vit_feat, rois, n_shift=5, output_size=(4,4), tau=0.1):
        # print(f'map_cos_fg.shape: {map_cos_fg.shape}')
        # print(f'map_cos_bg.shape: {map_cos_bg.shape}')
        # print(f'prototype_fg.shape: {prototype_fg.shape}')
        # print(f'prototype_bg.shape: {prototype_bg.shape}')
        # print(f'vit_feat.shape: {vit_feat.shape}')
        # print(f'rois.shape: {rois.shape}')
        # objectness = F.interpolate((map_cos_fg - map_cos_bg).unsqueeze(0), scale_factor=1/16.0, mode='bilinear')[0]
        # fg_down_sample = F.interpolate(map_cos_fg.unsqueeze(0), scale_factor=1/16.0, mode='bilinear')[0]
        # bg_down_sample = F.interpolate(map_cos_bg.unsqueeze(0), scale_factor=1/16.0, mode='bilinear')[0]
        objectness = F.interpolate(map_cos_fg.unsqueeze(0), scale_factor=1/16.0, mode='bilinear')[0]
        uncertainty = 3 - F.interpolate(torch.abs(map_cos_fg - map_cos_bg).unsqueeze(0), scale_factor=1/16.0, mode='bilinear')[0]
        # TODO: debug
        # get prototypes of earch instance
        prototypes = []
        for box, uc, ob in zip((rois/16).long().tolist(), uncertainty, objectness):
            xmin, ymin, xmax, ymax = box
            feat = vit_feat[:, ymin : ymax+1, xmin : xmax+1]
            u = uc[ymin : ymax+1, xmin : xmax+1]
            o = ob[ymin : ymax+1, xmin : xmax+1]
            h_box, w_box = u.shape[-2:]
            kernel_size = (math.ceil(h_box / output_size[0]), math.ceil(w_box / output_size[1]))
            h_pad = kernel_size[0] * output_size[0] - h_box
            w_pad = kernel_size[1] * output_size[1] - w_box
            u = F.pad(u[None, None, :, :], (0, w_pad, 0, h_pad), value=-10)
            f = F.pad(feat[:, None, :, :], (0, w_pad, 0, h_pad), value=-10)
            o = F.pad(o[None, None, :, :], (0, w_pad, 0, h_pad), value=-10)
            u_unfold = F.unfold(u, kernel_size=kernel_size, stride=kernel_size)
            o_unfold = F.unfold(o, kernel_size=kernel_size, stride=kernel_size)
            f_unfold = F.unfold(f, kernel_size=kernel_size, stride=kernel_size)
            i_max_u = u_unfold[0].argmax(dim=0)
            i_max_o = o_unfold[0].argmax(dim=0)
            i_block_hard = torch.arange(i_max_u.shape[0])
            i_block_easy = torch.arange(i_max_o.shape[0])
            prototype_hard = f_unfold[..., i_max_u, i_block_hard].transpose(0, 1)
            prototype_easy = f_unfold[..., i_max_o, i_block_easy].transpose(0, 1)
            prototypes.append(prototype_easy)
            
            # prototypes.append(torch.cat([prototype_easy, prototype_hard], dim=0))
        prototypes = torch.stack(prototypes)
        # sim = cosine_shift_self(vit_feat.flatten(-2).transpose(0,1).clone(), 
        #                                         vit_feat.flatten(-2).transpose(0,1).clone(), tau=tau, n_shift=n_shift)
        # sim = sim.unflatten(0, vit_feat.shape[-2:])
        # max_sim = sim.max(dim=-1, keepdim=True)[0]
        # sim_is_max = sim == max_sim
        # sim_is_max = sim_is_max.unflatten(-1, vit_feat.shape[-2:])
        # sim_is_max = sim_is_max[None,:,:,:,:] * (objectness>0.1)[:,None,None,:,:]
        # sim_is_max = sim_is_max.sum(dim=[-2, -1]) > 0
        # pdb.set_trace()
        # dist = F.cosine_similarity(prototypes_final[:,:,None,:], torch.cat([prototype_fg[:,None,:], prototype_bg], dim=1)[:,None,:,:], dim=-1)
        # # dist = F.cosine_similarity(prototypes_final.flatten(0,1)[:,None,:], prototype_fg[None, :, :][:, :rois.shape[0]], dim=2).unflatten(0, prototypes_final.shape[:2]).squeeze(-1).squeeze(-1)
        # # prototype_obj = dist.argmax(dim=-1)
        # sim = sim.unflatten(-1, vit_feat.shape[-2:])
        # sim = normalize_map(sim)
        # sim = torch.where(sim > 0.7, sim, torch.zeros_like(sim))
        # bg_down_sample = normalize_map(bg_down_sample)
        # sim_obj_ratio = (sim * (fg_down_sample.unsqueeze(1) > 0.3)).sum(dim=[-2, -1]) / (sim.sum(dim=[-2, -1]))
        # sim_upsample = F.interpolate(sim, scale_factor=16, mode='bilinear')
        # sim_obj_ratio = (sim_upsample * (map_cos_bg>0.85).unsqueeze(1)).sum(dim=[-2, -1]) < (sim_upsample * ((map_cos_bg<0.5).unsqueeze(1))).sum(dim=[-2, -1])
        # num_obj = rois.shape[0]

        return sim, sim_is_max
        # TODO: 将各个prototype分配给物体

    def mean_shift_grid_prototype(self, maps, vit_feat, rois=None, thr=0.35, n_shift=5, output_size=(4,4), tau=0.1, temp=0.1, n_points=20):
        # TODO: debug
        # get prototypes of earch instance
        # maps = F.interpolate(maps[None], scale_factor=1/16, mode='bilinear')[0]
        # vit_feat_pca = torch.pca_lowrank(vit_feat.flatten(1).permute(1, 0), q=64)

        prototypes = []
        select_coords = []
        for i_obj, map_ in enumerate(maps):
            pos_map = map_ >= thr
            num_pos = pos_map.sum()
            pos_idx = pos_map.nonzero()
            if num_pos >= n_points:
                grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                coords = pos_idx[grid]
            elif num_pos > 0:
                coords = pos_idx
                coords = fill_in_idx(coords, n_points)

            else: # num_pos == 0
                if rois is not None:
                    coords = ((rois[i_obj][:2] + rois[i_obj][2:]) // (2 * 16)).long().view(1, 2).flip(1)
                    coords = coords.repeat(n_points, 1)
                else:
                    pos_map = map_ >= 0
                    num_pos = pos_map.sum()
                    pos_idx = pos_map.nonzero()
                    grid = torch.arange(0, num_pos, step=num_pos//n_points)[:n_points]
                    coords = pos_idx[grid]

            select_coords.append(coords)
        select_coords = torch.stack(select_coords)
        prototypes = idx_by_coords(vit_feat[None].permute(0,2,3,1).expand(select_coords.shape[0],-1,-1,-1), select_coords[..., 0], select_coords[..., 1]).clone()
        # prototypes, sim, bandwidth = gaussian_shift(prototypes.flatten(0,1), 
        #                             vit_feat.flatten(-2).transpose(0,1).clone(),
        #                             bandwidth=tau, 
        #                             n_shift=n_shift,
        #                             mask=torch.where(maps_expand>thr, torch.ones_like(maps_expand), torch.zeros_like(maps_expand)).flatten(1))
        prot_objs = []
        sims_objs = []
        if rois is not None:
            maps_bbox = box2mask(rois//16, vit_feat.shape[-2:], default_val=0)
            # for prot, map_ in zip(prototypes, maps_bbox):
            #     prot, sim = cosine_shift_self(prot.clone(), (vit_feat*map_[None]).flatten(-2).transpose(0,1).clone(), vit_feat.flatten(-2).transpose(0,1).clone(), tau=tau, temp=temp, n_shift=n_shift)
            #     prot_objs.append(prot)
            #     sims_objs.append(sim)
            prototypes, sim = cosine_shift_batch(prototypes.clone(), (vit_feat[None]*maps_bbox[:, None]).flatten(-2).transpose(1, 2).clone(), vit_feat.flatten(-2).transpose(0,1).clone(), tau=tau, temp=temp, n_shift=n_shift)
            # prototypes = torch.cat(prot_objs)
            # sim = torch.cat(sims_objs)
        else:
            prototypes, sim = cosine_shift_self(prototypes[0], (vit_feat).flatten(-2).transpose(0,1).clone(), vit_feat.flatten(-2).transpose(0,1).clone(), tau=tau, n_shift=n_shift)

        # pdb.set_trace()
        # prototypes, bandwidth = merge_pototypes_bandwidth(prototypes.unflatten(0, select_coords.shape[:2]), bandwidth.unflatten(0, select_coords.shape[:2]))
        # sim = gaussian(torch.abs(torch.cat(prototypes)[:, None, :] - vit_feat.flatten(-2).transpose(0,1).clone()[None, :, :]), torch.cat(bandwidth).unsqueeze(1))
        # sim = sim.unflatten(-1, vit_feat.shape[-2:])
        # split_size = [p.shape[0] for p in prototypes]
        # sim = torch.split(sim, split_size, dim=0)

        # prototypes = merge_pototypes(prototypes.unflatten(0, select_coords.shape[:2]))
        # sim = F.softmax(sim, dim=-1)
        
        return prototypes, sim.unflatten(-1, vit_feat.shape[-2:]).clamp(0)


    # def get_prototypes_bg(self, bg_map, fg_map, vit_feat, n_clusters=40):
    #     max_val = bg_map.flatten(-2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    #     bg_mask = torch.where(bg_map > max_val*0.3, True, False).squeeze(1)
    #     bg_mask[fg_map.squeeze(1)>0] = False
    #     prot_bg = []
    #     for i_obj in range(bg_mask.shape[0]):
    #         mask = bg_mask[i_obj]
    #         feats = vit_feat.permute(1,2,0)[mask]
    #         cluster_ids_x, cluster_centers = kmeans(
    #             X=feats, num_clusters=n_clusters, distance='cosine', device=vit_feat.device
    #         )
    #         prot_bg.append(cluster_centers)

    #     return torch.stack(prot_bg)

    # def get_mask_sample_points_roi_best_attn_feat_refine(self, 
    #                                                     attn, 
    #                                                     rois, 
    #                                                     attn_idx, 
    #                                                     vit_feat, 
    #                                                     pos_thr=0.6, 
    #                                                     neg_thr=0.6, 
    #                                                     num_gt=20, 
    #                                                     corr_size=21, 
    #                                                     refine_times=2, 
    #                                                     gt_points=None, 
    #                                                     mean_shift_refine=False):
    #     # Parameters:
    #     #     attn: num_layers, num_points, H, W
    #     #     roi: num_points, 4
    #     # Return:
    #     #     coords_chosen: num_points, num_gt, 2
    #     #     labels_chosen: num_points, num_gt
    #     num_points = attn.shape[1]
    #     attn_map = attn.detach().clone()
    #     attn_map = attn_map[attn_idx, torch.arange(num_points)] # num_points, num_pixels, 2
    #     if mean_shift_refine:
    #         map_cos_fg, map_cos_bg = get_cosine_similarity_refined_map(attn_map, 
    #                                                                 vit_feat, 
    #                                                                 rois, 
    #                                                                 epoch=self.epoch, 
    #                                                                 thr_pos=0.1, 
    #                                                                 thr_neg=0.2, 
    #                                                                 num_points=20, 
    #                                                                 thr_fg=0.7,
    #                                                                 refine_times=5, 
    #                                                                 gt_points=gt_points)
    #         fg_inter = F.interpolate(map_cos_fg[:, None, :, :], scale_factor=1/16, mode='bilinear')
    #         bg_inter = F.interpolate(map_cos_bg[:, None, :, :], scale_factor=1/16, mode='bilinear')
    #         for _ in range(5):
    #             prototype_bg = self.get_prototypes_bg(bg_inter, fg_inter, vit_feat)
    #             prototype_fg = (vit_feat * fg_inter).sum(dim=[-2,-1]) / (fg_inter.sum(dim=[-2,-1]) + 1e-8)
    #             # prototype_bg = (vit_feat * bg_inter).sum(dim=[-2,-1]) / (bg_inter.sum(dim=[-2,-1]) + 1e-8)
    #             # prototype_bg = (vit_feat * bg_inter).sum(dim=[-2,-1]) / (bg_inter.sum(dim=[-2,-1]) + 1e-8)
    #             mean_shift_sim, mean_shift_dist  = self.mean_shift_refine_prototype(map_cos_fg, 
    #                                                                     map_cos_bg, 
    #                                                                     prototype_fg, 
    #                                                                     prototype_bg, 
    #                                                                     vit_feat,
    #                                                                     rois,
    #                                                                     n_shift=3)
                
    #         # mean_shift_sim = F.interpolate(mean_shift_sim, scale_factor=16, mode='bilinear')
    #         # prototype_is_pos = mean_shift_dist.argmax(dim=-1) == 0
    #         # prototype_is_pos = mean_shift_dist
    #         # idx_prot = torch.arange(prototype_is_pos.shape[0], device=prototype_is_pos.device)[:, None].expand(-1, prototype_is_pos.shape[0]).flatten()
    #     else:
    #         map_cos_fg, map_cos_bg = get_cosine_similarity_refined_map(attn_map, vit_feat, rois, epoch=self.epoch, thr_pos=0.2, thr_neg=0.1, num_points=20, thr_fg=0.7, refine_times=5, gt_points=gt_points, return_feats=mean_shift_refine)
        

    #     coord_chosen = []
    #     label_chosen = []
    #     # num_objs = map_cos_fg[0].shape[0]
    #     # for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg[0], map_cos_bg[0]):
    #     num_objs = map_cos_fg.shape[0]
    #     for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg, map_cos_bg):
    #         # print(f'num_objs: {num_objs}')
    #         # print(f'map_cos_fg.shape: {map_cos_fg.shape}')
    #         # pdb.set_trace()
    #         xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
    #         H, W = map_fg.shape[-2:]
    #         map_crop_fg = map_fg[ymin:ymax, xmin:xmax]
    #         map_crop_bg = map_bg[ymin:ymax, xmin:xmax]
    #         coor, label = get_mask_points_single_box_cos_map_fg_bg(map_crop_fg, map_crop_bg, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt,i=i_p, corr_size=corr_size)
    #         coor[:, 0] += ymin
    #         coor[:, 1] += xmin
            
    #         coor = coor.flip(1)
    #         coord_chosen.append(coor)
    #         label_chosen.append(label)
    #     coords_chosen = torch.stack(coord_chosen).float()
    #     labels_chosen= torch.stack(label_chosen)
    #     return coords_chosen, labels_chosen, map_cos_fg, map_cos_bg, map_cos_fg, mean_shift_dist

    def get_mask_sample_points_roi_prots_best_attn_feat_refine(self, attn, rois, attn_idx, vit_feat, pos_thr=0.6, neg_thr=0.6, num_gt=20, corr_size=21, refine_times=2, obj_tau=0.85):
        # Parameters:
        #     attn: num_layers, num_points, H, W
        #     roi: num_points, 4
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        num_points = attn.shape[1]
        attn_map = attn.detach().clone()
        attn_map = attn_map[attn_idx, torch.arange(num_points)] # num_points, num_pixels, 2
        map_cos_fg, map_cos_bg, _, _ = get_cosine_similarity_refined_map(attn_map, vit_feat, rois, thr_pos=0.3, thr_neg=0.1, num_points=20, thr_fg=0.7, refine_times=2, obj_tau=obj_tau)
        coord_chosen = []
        label_chosen = []
        num_objs = map_cos_fg[0].shape[0]
        for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg[-1], map_cos_bg[-1]):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = map_fg.shape[-2:]
            map_crop_fg = map_fg[ymin:ymax, xmin:xmax]
            map_crop_bg = map_bg[ymin:ymax, xmin:xmax]
            coor, label = get_mask_points_single_box_cos_map_fg_bg(map_crop_fg, map_crop_bg, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt,i=i_p, corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            coord_chosen.append(coor)
            label_chosen.append(label)
        coords_chosen = torch.stack(coord_chosen).float()
        labels_chosen= torch.stack(label_chosen)
        return coords_chosen, labels_chosen, map_cos_fg, map_cos_bg

    def get_mask_sample_points_roi_best_attn_feat_refine(self, attn, rois, attn_idx, vit_feat, pos_thr=0.6, neg_thr=0.6, num_gt=20, corr_size=21, refine_times=2, obj_tau=0.85, gt_points=None):
        # Parameters:
        #     attn: num_layers, num_points, H, W
        #     roi: num_points, 4
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        num_points = attn.shape[1]
        attn_map = attn.detach().clone()
        attn_map = attn_map[attn_idx, torch.arange(num_points)] # num_points, num_pixels, 2
        map_cos_fg, map_cos_bg, points_bg, points_fg = get_cosine_similarity_refined_map(attn_map, vit_feat, rois, thr_pos=0.2, thr_neg=0.1, num_points=20, thr_fg=0.7, refine_times=refine_times, obj_tau=obj_tau, gt_points=gt_points)
        coord_chosen = []
        label_chosen = []
        num_objs = map_cos_fg[0].shape[0]
        for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg[-1], map_cos_bg[-1]):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = map_fg.shape[-2:]
            map_crop_fg = map_fg[ymin:ymax, xmin:xmax]
            map_crop_bg = map_bg[ymin:ymax, xmin:xmax]
            coor, label = get_mask_points_single_box_cos_map_fg_bg(map_crop_fg, map_crop_bg, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt,i=i_p, corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coor = coor.flip(1)
            coord_chosen.append(coor)
            label_chosen.append(label)
        coords_chosen = torch.stack(coord_chosen).float()
        labels_chosen= torch.stack(label_chosen)
        return coords_chosen, labels_chosen, map_cos_fg , map_cos_bg, points_bg, points_fg

    def get_semantic_centers(self, 
                            map_cos_fg, 
                            map_cos_bg,
                            rois, 
                            vit_feat, 
                            pos_thr=0.35,
                            refine_times=5, 
                            gt_labels=None,
                            merge_thr=0.85,
                            num_semantic_points=3):
        # Parameters:
        #     attn: num_layers, num_points, H, W
        #     roi: num_points, 4
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        map_cos_fg_corr = corrosion_batch(torch.where(map_cos_fg>pos_thr, torch.ones_like(map_cos_fg), torch.zeros_like(map_cos_fg))[None], corr_size=11)[0]
        fg_inter = F.interpolate(map_cos_fg_corr.unsqueeze(0), vit_feat.shape[-2:], mode='bilinear')[0]
        bg_inter = F.interpolate(map_cos_bg.unsqueeze(0).max(dim=1, keepdim=True)[0], vit_feat.shape[-2:], mode='bilinear')[0]
        # pca = PCA(n_components=self.pca_dim)
        # pca.fit(vit_feat.flatten(1).permute(1, 0).cpu().numpy())
        # vit_feat_pca = torch.from_numpy(pca.fit_transform(vit_feat.flatten(1).permute(1, 0).cpu().numpy())).to(vit_feat.device)
        # vit_feat_pca = vit_feat_pca.permute(1, 0).unflatten(1, vit_feat.shape[-2:])

        vit_feat_pca = vit_feat
        map_fg = torch.where(fg_inter > pos_thr, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))

        prototypes_fg, sim_fg = self.mean_shift_grid_prototype(map_fg, vit_feat_pca, rois, tau=0.1, temp=0.1, n_shift=refine_times)

        sim_fg, idx_pos = filter_maps(sim_fg.unflatten(0, (sim_fg.shape[0]//20, 20)), fg_inter, bg_inter)
        split_size = idx_pos.sum(dim=-1).tolist()
        prototypes_fg = merge_maps(prototypes_fg[idx_pos.flatten()].split(split_size, dim=0), thr=merge_thr)
        sim_fg = [cal_similarity(prot, vit_feat_pca.permute(1,2,0)) for prot in prototypes_fg]
        # coord_semantic_center, coord_semantic_center_split = get_center_coord(sim_fg, rois, gt_labels, num_max_obj=num_semantic_points)
        # return coord_semantic_center, coord_semantic_center_split, sim_fg
        coord_semantic_center, coord_semantic_center_split, feat_semantic_center_split, feat_semantic_center, num_parts, coord_sc_org, label_sc_org  = get_center_coord_with_feat(sim_fg, rois, gt_labels, vit_feat, num_max_obj=num_semantic_points)
        return coord_semantic_center, coord_semantic_center_split, sim_fg, feat_semantic_center_split, feat_semantic_center, num_parts, coord_sc_org, label_sc_org


    def get_mask_sample_points_roi_best_attn_local_global(self, 
                                                        attn, 
                                                        rois, 
                                                        attn_idx, 
                                                        vit_feat, 
                                                        pos_thr=0.6, 
                                                        neg_thr=0.6, 
                                                        num_gt=20, 
                                                        corr_size=21, 
                                                        refine_times=2, 
                                                        gt_points=None, 
                                                        mean_shift_refine=False,
                                                        gt_labels=None):
        # Parameters:
        #     attn: num_layers, num_points, H, W
        #     roi: num_points, 4
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        num_points = attn.shape[1]
        attn_map = attn.detach().clone()
        attn_map = attn_map[attn_idx, torch.arange(num_points)] # num_points, num_pixels, 2

        map_cos_fg, map_cos_bg = get_cosine_similarity_refined_map(attn_map, 
                                                                vit_feat, 
                                                                rois, 
                                                                epoch=self.epoch, 
                                                                thr_pos=0.3,
                                                                thr_neg=0.1, 
                                                                num_points=20, 
                                                                thr_fg=0.7,
                                                                refine_times=2, 
                                                                gt_points=gt_points)
        fg_inter = F.interpolate(map_cos_fg[None], scale_factor=1/16, mode='bilinear')[0]
        bg_inter = F.interpolate(map_cos_bg[None].max(dim=1, keepdim=True)[0], scale_factor=1/16, mode='bilinear')[0]
        pca = PCA(n_components=128)
        pca.fit(vit_feat.flatten(1).permute(1, 0).cpu().numpy())
        vit_feat_pca = torch.from_numpy(pca.fit_transform(vit_feat.flatten(1).permute(1, 0).cpu().numpy())).to(vit_feat.device)
        vit_feat_pca = vit_feat_pca.permute(1, 0).unflatten(1, vit_feat.shape[-2:])

        map_fgbg = torch.cat((torch.where(fg_inter > 0.35, torch.ones_like(fg_inter), torch.zeros_like(fg_inter)),
                              torch.where((bg_inter > 0.35) & (fg_inter < 0.1), torch.ones_like(bg_inter), torch.zeros_like(bg_inter))))
        map_fg = torch.where(fg_inter > 0.35, torch.ones_like(fg_inter), torch.zeros_like(fg_inter))

        # print(f'map_fgbg.shape: {map_fgbg.shape}')
        # pdb.set_trace()
        prototypes_fg, sim_fg = self.mean_shift_grid_prototype(map_fg, vit_feat_pca, rois, tau=0.1, temp=1, n_shift=20)

        # prototypes_fg, sim_fg = self.mean_shift_grid_prototype(torch.where(fg_inter > 0.35, torch.ones_like(fg_inter), torch.zeros_like(fg_inter)),
        #                                 vit_feat_pca,
        #                                 rois,
        #                                 tau=25,
        #                                 n_shift=20)
        bg_filter = torch.where((bg_inter > 0.35) & (fg_inter < 0.1), torch.ones_like(bg_inter), torch.zeros_like(bg_inter))
        bg_filter = torch.sum(bg_filter, dim=0, keepdim=True).clamp(0,1)
        fg_mask = fg_inter.sum(dim=0, keepdim=True).clamp(0,1)
        bg_filter = bg_filter * fg_mask
        prototypes_bg, sim_bg = self.mean_shift_grid_prototype(bg_filter,
                                        vit_feat_pca,
                                        tau=0.1,
                                        temp=1,
                                        n_points=40,
                                        n_shift=20)
        
        sim_fg, sim_bg_0, uncertain_fg, idx_pos = filter_maps(sim_fg.unflatten(0, (sim_fg.shape[0]//20, 20)), fg_inter, bg_inter)
        sim_bg, _, uncertain_bg, _ = filter_maps(sim_bg.unflatten(0, (sim_bg.shape[0]//40, 40)), bg_inter, fg_inter.sum(0, keepdim=True))
        split_size = idx_pos.sum(dim=-1).tolist()
        
        prototypes_fg = merge_maps(prototypes_fg[idx_pos.flatten()].split(split_size, dim=0), thr=0.8)
        sim_fg = [cal_similarity(prot, vit_feat_pca.permute(1,2,0)) for prot in prototypes_fg]
        coord_semantic_center = get_center_coord(sim_fg, rois, gt_labels)

        sim_bg = torch.cat(sim_bg, dim=0)
        uncertain_map = torch.cat((uncertain_fg, uncertain_bg), dim=0)
        coord_chosen = []
        label_chosen = []
        # num_objs = map_cos_fg[0].shape[0]
        # for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg[0], map_cos_bg[0]):
        num_objs = map_cos_fg.shape[0]
        for i_p, map_fg, map_bg in zip(range(num_objs), map_cos_fg, map_cos_bg):
            # print(f'num_objs: {num_objs}')
            # print(f'map_cos_fg.shape: {map_cos_fg.shape}')
            # pdb.set_trace()
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = map_fg.shape[-2:]
            map_crop_fg = map_fg[ymin:ymax, xmin:xmax]
            map_crop_bg = map_bg[ymin:ymax, xmin:xmax]
            coor, label = get_mask_points_single_box_cos_map_fg_bg(map_crop_fg, map_crop_bg, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt,i=i_p, corr_size=corr_size)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            print(f'coor: {coor}')
            print(f'label: {label}')
            coor = coor.flip(1)
            coord_chosen.append(coor)
            label_chosen.append(label)
        coords_chosen = torch.stack(coord_chosen).float()
        labels_chosen= torch.stack(label_chosen)
        return coords_chosen, labels_chosen, map_cos_fg, map_cos_bg, sim_fg, sim_bg, uncertain_map, coord_semantic_center


    def get_mask_sample_points_roi_best_attn_with_edge(self, attn, rois, attn_idx, point_adjuster, edge_map, pos_thr=0.2, neg_thr=0.05, num_gt=20, corr_size=21):
        # Parameters:
        #     attn: num_layers, num_points, H, W
        #     roi: num_points, 4
        # Return:
        #     coords_chosen: num_points, num_gt, 2
        #     labels_chosen: num_points, num_gt
        num_points = attn.shape[1]
        attn_map = attn.detach().clone()
        attn_map = attn_map[attn_idx, torch.arange(num_points)] # num_points, num_pixels, 2
        
        coord_chosen = []
        label_chosen = []
        coords_org = []
        for i_p, attn_p in enumerate(attn_map):
            xmin, ymin, xmax, ymax = rois[i_p].int().tolist()
            H, W = attn_p.shape[-2:]
            attn_crop = attn_p[ymin:ymax, xmin:xmax]
            edge_crop = edge_map[ymin:ymax, xmin:xmax]
            coor, label, coord_org = self.get_mask_points_single_instance(attn_crop, pos_thr=pos_thr, neg_thr=neg_thr, num_gt=num_gt,i=i_p, corr_size=corr_size, edge_map=edge_crop, point_adjuster=point_adjuster)
            coor[:, 0] += ymin
            coor[:, 1] += xmin
            coord_org[:, 0] += ymin
            coord_org[:, 1] += xmin
            coor = coor.flip(1)
            coord_org = coord_org.flip(1)
            coord_chosen.append(coor)
            label_chosen.append(label)
            coords_org.append(coord_org)
        coords_chosen = torch.stack(coord_chosen).float()
        coord_org = torch.stack(coords_org).float()
        labels_chosen= torch.stack(label_chosen)

        return coords_chosen, labels_chosen, coord_org

    def get_mask_points_single_instance(self, attn_map, pos_thr=0.1, neg_thr=0.01, num_gt=10, i=0, corr_size=21, edge_map=None, point_adjuster=None):
        # Parameters:
        #     coords: num_pixels, 2
        #     attn:H, W
        #     cls: scalar,
        # Return:
        #     coords_chosen: num_gt, 2
        #     labels_chosen: num_gt
        device = attn_map.device
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        attn_pos = corrosion((attn_map > pos_thr).float(), corr_size=corr_size)
        coord_pos = (attn_pos).nonzero(as_tuple=False)
        coord_neg = (attn_map < neg_thr).nonzero(as_tuple=False)
        coord_pos_neg = torch.cat((coord_pos, coord_neg), dim=0)
        idx_chosen = torch.randperm(coord_pos_neg.shape[0], device=attn_map.device)[:num_gt]
        labels_pos_neg = torch.cat((torch.ones(coord_pos.shape[0], device=device, dtype=torch.bool),
                                    torch.zeros(coord_neg.shape[0], device=device, dtype=torch.bool)), dim=0)
        coords_chosen = coord_pos_neg[idx_chosen]
        labels_chosen = labels_pos_neg[idx_chosen]
        
        num_points = coords_chosen.shape[0]
        if num_points < num_gt:
            if idx_chosen.shape[0] == 0:
                coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
                print(f'**************一个点都没有找到!**************')
                # 这些-1的点会在point ignore里被处理掉
                return coords_chosen, torch.zeros(num_gt, dtype=torch.bool, device=device)
            else:
                idx_chosen = fill_in_idx(idx_chosen, num_gt)
                
        coords_chosen = coord_pos_neg[idx_chosen]
        labels_chosen = labels_pos_neg[idx_chosen]

        if point_adjuster is not None:
            assert edge_map is not None, 'edge_map should not be None!'
            point_coords_adjusted = point_adjuster(attn_map, edge_map, coords_chosen.flip(1), labels_chosen).flip(1)

            return point_coords_adjusted, labels_chosen, coords_chosen
        return point_coords_adjusted, labels_chosen

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
                      pos_mask_thr=0.6,
                      neg_mask_thr=0.1,
                      num_mask_point_gt=10,
                      corr_size=21,
                      point_adjuster=None,
                      edges=None,
                      obj_tau=0.85,
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
                    normalize_point_cc, point_cls[i], (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2,
                    gt_points_labels[i], img_metas[i]
                )
                point_sampling_result = self.point_sampler.sample(
                    assign_result, point_reg[i], 
                    (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2
                )
                point_assign_results.append(point_sampling_result)
            pos_inds = [sample_results.pos_inds for sample_results in point_assign_results]
            
            labels, _, point_targets, _ = self.get_targets(
                point_assign_results, (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2, gt_points_labels, self.train_cfg,
                concat=False)
            
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        num_proposals = point_cls.size(1)
        points_attn_maps = attns_project_to_feature(attns[-self.bbox_head.cam_layer:])
        # attention maps
        gt_scale_bboxes = []
        gt_labels = []
        attn_maps_dealed = []
        origin_attn_maps = []
        for i_img in range(num_imgs):
            pos_inds_ = pos_inds[i_img]
            gt_labels.append(labels[i_img][pos_inds_])
            num_gt = len(pos_inds_)
            times = self.train_cfg.point_assigner.times
            points_attn_maps_per_img = points_attn_maps[i_img][:, -num_proposals:, 1:-num_proposals].permute(1, 0, 2)[pos_inds_].permute(1, 0, 2)
            points_attn_maps_per_img = points_attn_maps_per_img.reshape(-1, 1, patch_h, patch_w)
            origin_attn_maps.append(points_attn_maps_per_img.clone().reshape(-1, times*num_gt, patch_h, patch_w))
            points_attn_maps_per_img = F.interpolate(points_attn_maps_per_img, (patch_h * 16, patch_w * 16), mode='bilinear').reshape(-1, num_gt, patch_h * 16, patch_w * 16) # nu_gt, H, W
            point_targets_ = point_targets[i_img][pos_inds_].unsqueeze(0).repeat(self.bbox_head.cam_layer, 1, 1)
            
            cam_layers = points_attn_maps_per_img
            attn_maps_dealed.append(points_attn_maps_per_img.clone())
            scale_bboxes_per_image = []
            for cam_per_point, point in zip(cam_layers.clone(), point_targets_.clone()):
                scale_bboxes = []
                for scale_cam, scale_point in zip(cam_per_point, point):
                    pseudo_gt_bbox, _ = get_bbox_from_cam_fast(scale_cam,
                                                      scale_point,
                                                      cam_thr=self.bbox_head.seed_thr,
                                                      area_ratio=self.bbox_head.seed_multiple,
                                                      img_size=(patch_h * 16, patch_w * 16)
                                                     )
                    scale_bboxes.append(pseudo_gt_bbox)
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
        gt_box_index = mil_out[2]
        gt_mask_points_coords = []
        gt_mask_points_labels = []
        gt_mask_points_coords_org = []
        # print(f'gt_box_index.shape: {gt_box_index.shape}')
        map_cos_fg_ret = []
        map_cos_bg_ret = []
        sim_fg_ret = []
        semantic_centers_ret = []
        semantic_centers_split_ret = []
        semantic_centers_feat_split = []
        semantic_centers_feat = []
        num_parts = []
        center_points = [(p[:, :2] + p[:, 2:]) / 2 for p in gt_points]
        coords_sc_org = []
        labels_sc_org = []
        pseudo_gt_masks = []
        for i_img in range(num_imgs):
            coord_point, labels_point, map_cos_fg, map_cos_bg, points_bg, points_fg = self.get_mask_sample_points_roi_best_attn_feat_refine(attn_maps_dealed[i_img], mil_out[0][i_img], gt_box_index[i_img], 
                                                        vit_feat=vit_feat[i_img].clone(), pos_thr=pos_mask_thr, neg_thr=neg_mask_thr, num_gt=num_mask_point_gt, obj_tau=obj_tau, gt_points=center_points[i_img])
            semantic_centers, semantic_centers_split, sim_fg, feat_semantic_center_split, feat_semantic_centers, num_parts_obj, coord_sc_org, label_sc_org = self.get_semantic_centers(map_cos_fg[-1].clone(), 
                                                        map_cos_bg[-1].clone(), 
                                                        mil_out[0][i_img], 
                                                        vit_feat[i_img].clone(), 
                                                        pos_thr=pos_mask_thr, 
                                                        refine_times=self.mean_shift_times_local, 
                                                        gt_labels=gt_labels[i_img],
                                                        num_semantic_points=self.num_semantic_points)
            semantic_centers_feat_split.append(feat_semantic_center_split)
            gt_mask_points_coords.append(coord_point)
            gt_mask_points_labels.append(labels_point)
            map_cos_fg_ret.append(map_cos_fg[-1])
            map_cos_bg_ret.append(map_cos_bg[-1])
            semantic_centers_ret.append(semantic_centers)
            semantic_centers_split_ret.append(semantic_centers_split)
            sim_fg_ret.append(sim_fg)
            semantic_centers_feat.append(feat_semantic_centers)
            num_parts.append(num_parts_obj)
            coords_sc_org.append(coord_sc_org)
            labels_sc_org.append(label_sc_org)
            pseudo_gt_masks.append(torch.where(map_cos_fg[-1] > map_cos_fg[-1].flatten(1).max(1)[0][:, None, None] * pos_mask_thr, 
                                              torch.ones_like(map_cos_fg[-1]), 
                                              torch.zeros_like(map_cos_fg[-1])).to(torch.uint8).detach().cpu().numpy())
        # pseudo_gt_mask, ignore_mask = self.get_pseudo_gt_masks_from_point_attn(cam_maps_images, gt_box_index)
        
        # points_attn_maps_images: list, length=#Imgs, points_attn_maps_images[i].shape: [n_layers, n_gts_i, H, W]
        # gt_box_index: tuple, length=#Imgs, gt_box_index[i]: [#gts_i, ]
        if self.visualize:
            self.semantic_centers_split = semantic_centers_split_ret
            self.attns = attns
            self.map_cos_fg = map_cos_fg
            self.map_cos_bg = map_cos_bg
            self.num_parts = num_parts
            self.best_idx = mil_out[2]

            return dict(pseudo_gt_labels=gt_labels,
                        pseudo_gt_bboxes=mil_out[0],
                        mil_losses=mil_out[1],
                        best_attn_idx=mil_out[2],
                        attns=points_attn_maps_per_img,
                        mask_points_coords=gt_mask_points_coords,
                        mask_points_labels=gt_mask_points_labels,
                        map_cos_fg=map_cos_fg_ret,
                        map_cos_bg=map_cos_bg_ret,
                        semantic_centers=semantic_centers_ret,
                        points_bg=points_bg, 
                        points_fg=points_fg,
                        sim_fg=sim_fg_ret,
                        semantic_centers_split=semantic_centers_split_ret,
                        semantic_centers_feat_split=semantic_centers_feat_split,
                        semantic_centers_feat=semantic_centers_feat,
                        num_parts=num_parts,
                        semantic_centers_org=(coords_sc_org, labels_sc_org),
                        pseudo_gt_masks=pseudo_gt_masks,
                        )
        else:
            return dict(pseudo_gt_labels=gt_labels,
                        pseudo_gt_bboxes=mil_out[0],
                        mil_losses=mil_out[1],
                        best_attn_idx=mil_out[2],
                        map_cos_fg=map_cos_fg_ret,
                        mask_points_coords=gt_mask_points_coords,
                        mask_points_labels=gt_mask_points_labels,
                        semantic_centers=semantic_centers_ret,
                        semantic_centers_split=semantic_centers_split_ret,
                        semantic_centers_feat_split=semantic_centers_feat_split,
                        semantic_centers_feat=semantic_centers_feat,
                        num_parts=num_parts,
                        semantic_centers_org=(coords_sc_org, labels_sc_org),
                        pseudo_gt_masks=pseudo_gt_masks,
                        )
    
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
                      mask_point_labels=None,
                      mask_point_coords=None,
                      semantic_centers=None,
                      semantic_centers_split=None, 
                      feats_point_tokens=None, 
                      semantic_centers_feat_split=None, 
                      semantic_centers_feat=None,
                      num_parts=None,
                      semantic_centers_org=None, #semantic_centers的数量比较少，因为有上限，而semantic_centers_org没有上限
                      map_cos_fg=None,
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
            self.gt_point_coords = semantic_centers_split
            self.point_sampling_result = point_sampling_result
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
                                                    img_metas, img=img)
            losses.update(bbox_results['loss_bbox'])
        
        if self.with_reppoints_head:
            reppoint_loss, semantic_centers_split_new = self.reppoints_head.forward_train(
                [xx.clone().detach() for xx in x], gt_bboxes, semantic_centers_org[0], img_metas, num_parts, gt_masks,
                fg_maps=map_cos_fg, gt_labels=gt_labels)

            if self.with_deform_sup:
                semantic_centers_split = semantic_centers_split_new
            #     # import copy
            #     # semantic_centers_copy = copy.deepcopy(semantic_centers_split_new)
            #     # semantic_centers_split = random_select_half(semantic_centers_split_new)
            # losses.update(reppoint_loss)
            # reppoint_loss = self.reppoints_head.forward_train(
            #     [xx.clone().detach() for xx in x], gt_bboxes, semantic_centers_org[0], img_metas, num_parts, gt_masks,
            #     fg_maps=map_cos_fg, gt_labels=gt_labels)
            losses.update(reppoint_loss)

        # mask head forward and loss
        if self.with_mask:
            # gt_masks_pseudo = self.get_pseudo_gt_masks_from_point_attn()
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    mask_point_coords, mask_point_labels,
                                                    semantic_centers=semantic_centers_split,
                                                    img_metas=img_metas)
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

            if torch.numel(cam[0]) == 0:
                masks.append([])
                ignore_mask.append([])
                continue

            masks.append(BitmapMasks(cam[0][idx, torch.arange(idx.shape[0])].cpu().numpy().astype(np.uint8), \
                                    height=cam[0].shape[-2], width=cam[0].shape[-1]))
            ignore_mask.append(BitmapMasks(cam[1][idx, torch.arange(idx.shape[0])].cpu().numpy().astype(np.uint8), \
                                    height=cam[1].shape[-2], width=cam[0].shape[-1]))        
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
            
            cls_score = torch.zeros_like(cls_scores, dtype=torch.float16, device=bbox_feats.device)
            bbox_pred = torch.zeros_like(bbox_preds, dtype=torch.float16, device=bbox_feats.device)
            
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
                cls_score, bbox_pred, rec_pred = self.bbox_head(bbox_feats)

                bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, roi_rec=rec_pred)
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
                            img_metas, img=None):
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
                                            bbox_results['bbox_pred'], 
                                            rois,
                                            *bbox_targets,
                                            recs=bbox_results['roi_rec'],
                                            rec_inds=range(bbox_results['roi_rec'].shape[0]),
                                            img=img
                                            ) if isinstance(self.bbox_head, nn.ModuleList) else self.bbox_head.loss(bbox_results['cls_score'],
                                                                                                                        bbox_results['bbox_pred'], 
                                                                                                                        rois,
                                                                                                                        *bbox_targets, 
                                                                                                                        # recs=bbox_results['roi_rec'],
                                                                                                                        # rec_inds=[range(bbox_results['roi_rec'].shape[0])], # batch size=1 only
                                                                                                                        img=img)

            bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

#     def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
#                             img_metas, ignore_mask=None):
#         """Run forward function and calculate loss for mask head in
#         training."""
#         weight = None
#         if not self.share_roi_extractor:
#             pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
#             mask_results = self._mask_forward(x, pos_rois)
#         else:
#             pos_inds = []
#             device = bbox_feats.device
#             for res in sampling_results:
#                 pos_inds.append(
#                     torch.ones(
#                         res.pos_bboxes.shape[0],
#                         device=device,
#                         dtype=torch.uint8))
#                 pos_inds.append(
#                     torch.zeros(
#                         res.neg_bboxes.shape[0],
#                         device=device,
#                         dtype=torch.uint8))
#             pos_inds = torch.cat(pos_inds)

#             mask_results = self._mask_forward(
#                 x, pos_inds=pos_inds, bbox_feats=bbox_feats)
#         mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
#                                                   self.train_cfg)
#         if ignore_mask is not None:
#             weight = self.mask_head.get_targets(sampling_results, ignore_mask,
#                                                   self.train_cfg)

#         pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
#         loss_mask = self.mask_head.loss(mask_results['mask_pred'],
#                                         mask_targets, pos_labels, weight=weight)

#         mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
#         return mask_results

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

#     def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
#         """Mask head forward function used in both training and testing."""
#         assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
#         if rois is not None:
#             mask_feats = self.mask_roi_extractor(
#                 x[: self.mask_roi_extractor.num_inputs], rois
#             )
#             if self.with_shared_head:
#                 mask_feats = self.shared_head(mask_feats)
#         else:
#             assert bbox_feats is not None
#             mask_feats = bbox_feats # 适应MAE的decoder的特性，所有的特征都输入到decoder中，只是返回的时候用pos_inds

#         mask_pred = self.mask_head(mask_feats)
#         mask_results = dict(mask_pred=mask_pred[pos_inds], mask_feats=mask_feats)
#         return mask_results

    def _mask_forward_train(
        self, x, sampling_results, bbox_feats, points_coords, points_labels, img_metas=None, semantic_centers=None, **kwargs
    ):
        # sites already sampled
        """Run forward function and calculate loss for mask head in
        training."""

        # if not self.share_roi_extractor:
        #     pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        #     mask_results = self._mask_forward(x, pos_rois)
        # else:
        pos_inds = []
        device = bbox_feats.device
        points_coords, points_labels = update_coords_with_semantic_centers(points_coords, points_labels, semantic_centers)
        for res in sampling_results:
            pos_inds.append(
                torch.ones(
                    res.pos_bboxes.shape[0], device=device, dtype=torch.bool
                )
            )
            pos_inds.append(
                torch.zeros(
                    res.neg_bboxes.shape[0], device=device, dtype=torch.bool
                )
            )
        pos_inds = torch.cat(pos_inds)
        mask_results = self._mask_forward(
            x, pos_inds=pos_inds, bbox_feats=bbox_feats
        )
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        # assert points_labels[0].shape[1] == 5
        # assert sites_img[0].shape[2] == 5
        # res=sampling_results[1]
        # res.pos_gt_labels
        # try:
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results])
        sites = torch.cat(
            [
                site_img[res.pos_assigned_gt_inds, :, :]
                for site_img, res in zip(points_coords, sampling_results)
            ]
        )
        mask_targets = torch.cat(
            [
                labels[
                    res.pos_assigned_gt_inds,
                ]
                for labels, res in zip(points_labels, sampling_results)
            ]
        )
        new_sites = get_point_coords_wrt_box(pos_bboxes, sites)
        point_ignores = (
            (new_sites[:, :, 0] < 0)
            | (new_sites[:, :, 0] > 1)
            | (new_sites[:, :, 1] < 0)
            | (new_sites[:, :, 1] > 1)
        )
        mask_targets[point_ignores] = 2
        point_preds = point_sample(
            mask_results["mask_pred"],
            new_sites,
            align_corners=False,
        )
        loss_mask = self.mask_head.loss(
            point_preds, mask_targets, pos_labels
        )

        mask_results.update(loss_mask=loss_mask)  # , mask_targets=mask_targets)
        return mask_results
        # except Exception as e:
        #     print("error in _mask_forward_train: ", e)

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

    def align_forward_train(self, semantic_centers_feat_split, tokens_pos, gt_bboxes, gt_labels):
        num_inst = 0
        corr_loss = torch.zeros(1, device=tokens_pos[0].device, dtype=tokens_pos[0].dtype)
        for i_img, tokens in enumerate(tokens_pos):
            sc_feats = semantic_centers_feat_split[i_img]
            if isinstance(sc_feats, list):
                continue
            for i_obj, token in enumerate(tokens):
                sc_feat = sc_feats[i_obj]
                if sc_feat.shape[0] == 0:
                    continue
                
                obj = ObjectFactory.create_one(
                    token[None], 
                    sc_feat, 
                    gt_bboxes[i_img][i_obj:i_obj+1], 
                    gt_labels[i_img][i_obj:i_obj+1],
                    device=token.device,
                )
                kobjs = self.object_queues.get_similar_obj(obj)
                
                if kobjs is not None and kobjs['token'].shape[0] >= 5:
                    cost_token, cosine_sim = cosine_distance(obj.token, kobjs['token'])
                    # cost_parts = cosine_distance_part(obj.part_feats, kobjs['feature'])
                    corr_loss += cost_token.min()
                    num_inst += 1
                self.object_queues.append(
                    gt_labels[i_img][i_obj], 
                    i_obj, 
                    tokens, 
                    sc_feats, 
                    gt_bboxes[i_img], 
                    device=token.device,
                )
            # pdb.set_trace()
        loss_align = corr_loss / (num_inst + 1e-6)
        return dict(loss_align=loss_align)
        
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
        label_weights = pos_bboxes.new_zeros(num_samples, dtype=pos_gt_labels.dtype, device=pos_gt_labels.device)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 2, dtype=pos_gt_bboxes.dtype, device=pos_gt_bboxes.device)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 2, dtype=pos_gt_bboxes.dtype, device=pos_gt_bboxes.device)
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
