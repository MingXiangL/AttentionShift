
import pickle
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab
import json
import pickle
import torch.nn.functional as F
import os
import pdb

import scipy
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

pylab.rcParams['figure.figsize'] = (48.0, 24.0)

from mmcv.runner.checkpoint import load_checkpoint
from mmcv import Config, DictAction
from mmdet.datasets import (build_dataloader, build_dataset,
                          replace_ImageToTensor)
from mmdet.apis import set_random_seed
set_random_seed(1)

# from edge_sample import OutmostJudger, PointAdjuster

def box2mask(bboxes, img_size, default_val=0.5):
    N = bboxes.shape[0]
    mask = torch.zeros(N, img_size[0], img_size[1], device=bboxes.device, dtype=bboxes.dtype) + default_val
    for n in range(N):
        box = bboxes[n] // 16
        mask[n, int(box[1]):int(box[3]+1), int(box[0]):int(box[2]+1)] = 1.0
    return mask

def get_fgbg_score_vote(fg_maps, bg_maps, thr=0.8, score_type='vote'):
    len_fg = fg_maps.shape[0]
    len_bg = bg_maps.shape[0]

    if score_type == 'vote':
        # score_max, _ = torch.cat((fg_maps, bg_maps), dim=0).max(dim=0, keepdim=True)
        num_fg = (fg_maps >= 0.7).sum(0)
        num_bg = (bg_maps >= 0.7).sum(0)
        ratio_fg = num_fg / (num_fg + num_bg)
        ratio_bg = num_bg / (num_fg + num_bg)
    
    return ratio_fg >= thr, ratio_bg >= thr, ((ratio_fg < thr) & (ratio_bg < thr))
        
def get_fgbg_score_mean(fg_maps, bg_maps, thr=0.5):
    fg_pos = (fg_maps>thr).sum(0)
    bg_pos = (bg_maps>thr).sum(0)
    fg_pos_mean = torch.where(fg_maps>thr, fg_maps, torch.zeros_like(fg_maps)).sum(0) / (fg_pos + 1e-8)
    bg_pos_mean = torch.where(bg_maps>thr, bg_maps, torch.zeros_like(bg_maps)).sum(0) / (bg_pos + 1e-8)
    return fg_pos_mean, bg_pos_mean, fg_pos_mean * (1-bg_pos_mean)


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
        bg_points.append(get_mask_points_single_instance(attn, neg_thr=0.01, num_gt=num_gt))
    
    return torch.stack(bg_points).flip(-1)

def norm_attns(attns):
    N, H, W = attns.shape
    max_val, _ = attns.view(1,N,-1,1).max(dim=2, keepdim=True)
    min_val, _ = attns.view(1,N,-1,1).min(dim=2, keepdim=True)
    return (attns.unsqueeze(0) - min_val) / (max_val - min_val)

def get_point_cos_similarity_map(point_coords, feats, ratio=1):
    feat_expand = feats.permute(0,2,3,1).expand(point_coords.shape[0], -1, -1, -1)
    point_feats = idx_by_coords(feat_expand, (point_coords[...,1].long()//16*ratio).clamp(0, feat_expand.shape[1]),( point_coords[...,0].long()//16*ratio).clamp(0, feat_expand.shape[2]))
    point_feats_mean = point_feats.mean(dim=1, keepdim=True)
    sim = F.cosine_similarity(feat_expand.flatten(1,2), point_feats_mean, dim=2)
    return sim.unflatten(1, (feat_expand.shape[1], feat_expand.shape[2]))

def sample_point_grid(maps, num_points=10, thr=0.2, is_pos=True):
    ret_coords = []
    for map_ in maps:
        factor = 1.0
        num_pos_pix = 0
        while num_pos_pix < num_points: # 用死循环调整阈值，
            if is_pos:
                coords = (map_ >= thr*factor).nonzero(as_tuple=False)
            else:
                coords = (map_ <= thr*factor).nonzero(as_tuple=False)
            num_pos_pix = coords.shape[0] 
            if num_pos_pix == 0:
                if is_pos:
                    print(f'factor adjusted from {thr * factor} to {thr * factor * 0.5}')
                    factor *= 0.5
                else:
                    print(f'factor adjusted from {thr * factor} to {thr * factor * 2}')
                    factor *= 2
        step = num_pos_pix // num_points
        idx_chosen = torch.arange(0, num_pos_pix, step=step)
        idx_chosen = torch.randint(num_pos_pix, idx_chosen.shape) % num_pos_pix
        coords_chosen = coords[idx_chosen][:num_points]
        ret_coords.append(coords_chosen)
    return torch.stack(ret_coords).flip(-1)

def get_rolled_sum(map_):
    # map_: num_rf, num_obj, H, W
    print(map_.shape)
    num_obj = map_.shape[1]
    map_expand = map_.unsqueeze(2).expand(-1,-1,num_obj,-1,-1)
    map_mask = torch.ones(1, map_.shape[1], map_.shape[1], 1, 1, dtype=map_.dtype, device=map_.device)
    map_mask[:,range(num_obj), range(num_obj), :, :] = 0
    return (map_ + (map_expand * map_mask).max(dim=2)[0])

def get_refined_similarity(point_coords, feats, ratio=1, refine_times=1, tau=0.85, is_select=False):
    cos_map = get_point_cos_similarity_map(point_coords, feats, ratio=ratio)
    idx_max_aff = cos_map.argmax(0, keepdim=True).expand_as(cos_map)
    range_obj = torch.arange(cos_map.shape[0], device=cos_map.device)
    cos_rf = []
    if is_select:
        # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map))
        cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map, torch.zeros_like(cos_map)))
    else:
        cos_rf.append(cos_map.clone())
    cos_map1 = cos_map.clone()

    for i in range(refine_times):
        # thr = cos_map1.max() * tau
        thr = tau
        cos_map1[cos_map1 < thr] = 0
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        if is_select:
            # cos_map_select = torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1))
            cos_rf.append(torch.where(idx_max_aff==range_obj[:,None,None], cos_map1, torch.zeros_like(cos_map1)))
        else:
            cos_rf.append(cos_map1.clone())

    return torch.stack(cos_rf)

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

def get_cosine_similarity_refined_map(attn_maps, vit_feat, bboxes, thr_pos=0.2, thr_neg=0.1, num_points=20, thr_fg=0.7):
    # attn_maps是上采样16倍之后的，vit_feat是上采样前的，实验表明，上采样后的不太好，会使cos_sim_fg ~= cos_sim_bg
    attn_norm = norm_attns(attn_maps)
    points_bg = sample_point_grid(attn_norm[0], thr=thr_neg, num_points=num_points, is_pos=False)
    points_fg = sample_point_grid(attn_norm[0], thr=thr_pos, num_points=num_points, is_pos=True)
    points_bg_supp = sample_point_grid(attn_norm[0], thr=thr_neg, num_points=num_points, is_pos=False)
    # points_bg_supp = torch.cat([sample_point_grid(attn_norm[0].mean(0,keepdim=True)<thr_neg, num_points=num_points) for _ in range(3)],dim=0)
    points_fg = torch.cat((points_fg, points_bg_supp), dim=0)
    cos_sim_fg = F.interpolate(get_refined_similarity(points_fg, vit_feat, refine_times=0, tau=0.85, is_select=False), attn_maps.shape[-2:], mode='bilinear')[:,:attn_norm.shape[1]]
    cos_sim_bg = F.interpolate(get_refined_similarity(points_bg, vit_feat, refine_times=0, tau=0.85), attn_maps.shape[-2:], mode='bilinear')
    print(f'points_fg.shape: {points_fg.shape}')
    print(f'vit_feat.shape: {vit_feat.shape}')
    print(f'attn_norm.shape: {attn_norm.shape}')
    print(f'cos_sim_fg.shape: {cos_sim_fg.shape}')
    # roll_fg = get_rolled_sum(cos_sim_fg)
    print(f'cos_sim_fg.shape: {cos_sim_fg.shape}')
    print(f'bboxes.shape: {bboxes.shape}')

    # cos_sim = (1 - (cos_sim_bg.unsqueeze(1) + roll_fg.unsqueeze(0))) * cos_sim_fg.unsqueeze(1) # num_refines_main, num_refines_select, num_obj, H, W
    # entropy_cos_sim = cal_entropy(cos_sim.clamp(min=1e-8, max=1-1e-8), bboxes.long())
    # idx_min_entropy = torch.argmin(entropy_cos_sim.flatten(0,1), dim=0)
    # 多次refine的索引
    # entropy_cos_sim[entropy_cos_sim > thr_fg] = 1
    # entropy_cos_sim[entropy_cos_sim < thr_fg] = 0
    # print(f'entropy_cos_sim.shape: {entropy_cos_sim.shape}')
    # return cos_sim.flatten(0, 1)[idx_min_entropy, range(idx_min_entropy.shape[0])]
    # bg_map = (1 - cos_sim_bg)
    # bg_map[bg_map < 0.3] = 0
    # cos_sim_fg[cos_sim_fg < 0.3] = 0
    # return bg_map * cos_sim_fg
    ret_map = (1 - cos_sim_bg) * cos_sim_fg
    map_val = ret_map.flatten(-2, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
    map_fg = torch.where(ret_map < map_val * 0.6, torch.zeros_like(ret_map), torch.ones_like(ret_map))
    map_bg = torch.where(ret_map > map_val * 0.1, torch.zeros_like(ret_map), torch.ones_like(ret_map))

    return map_fg, ret_map
    # return ((1 - (cos_sim_bg + roll_fg)) * cos_sim_fg)[0]

def get_cos_similarity_map_bg(point_coords, point_labels, feats, ratio=1):
    feat_expand = feats.permute(0,2,3,1).expand(point_coords.shape[0], -1, -1, -1)
    point_feats = idx_by_coords(feat_expand, (point_coords[...,1].long()//16*ratio).clamp(0, feat_expand.shape[1]),( point_coords[...,0].long()//16*ratio).clamp(0, feat_expand.shape[2]))
    point_feats_mean = (point_feats * (point_labels==0).float()).sum(1, keepdim=True) / ((point_labels>0).float().sum(1, keepdim=True) + 1e-6)
    sim = F.cosine_similarity(feat_expand.flatten(1,2), point_feats_mean, dim=2)
    return sim.unflatten(1, (feat_expand.shape[1], feat_expand.shape[2]))

def get_cos_similarity_map_bg_box(bboxes, feats, ratio=1):
    mask_bbox = torch.ones(bboxes.shape[0], 1, feats.shape[2], feats.shape[3], dtype=feats.dtype, device=feats.device)
    bboxes_long = bboxes.long()
    for i_b, box in enumerate(bboxes_long):
        mask_bbox[i_b,:,box[1]:box[3], box[0]:box[2]] = 0
    mean_feat = (feats * mask_bbox).sum(dim=[2,3], keepdim=True) / (mask_bbox.sum(dim=[2,3], keepdim=True) + 1e-6)
    sim = F.cosine_similarity(feats, mean_feat, dim=1)

    return sim

def get_refined_similarity_box(bboxes, feats, ratio=1, refine_times=1, tau=0.6):
    feats_expand = feats.expand(bboxes.shape[0],-1,-1,-1)
    cos_map = get_cos_similarity_map_bg_box(bboxes, feats_expand, ratio=ratio)
    cos_rf = []
    cos_rf.append(cos_map.clone())
    cos_map1 = cos_map.clone()
    for i in range(refine_times):

        cos_map1[cos_map1 < tau] = 0
        feats_mask = feats_expand * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
        cos_map1 = F.cosine_similarity(feats_expand, feats_mask, dim=1)
        cos_rf.append(cos_map1.clone())
    return cos_rf

def get_refined_similarity_bg(point_coords, point_labels, feats, ratio=1, refine_times=1, tau=0.85):
    cos_map = get_cos_similarity_map_bg(point_coords, point_labels, feats, ratio=ratio)
    cos_rf = []
    cos_rf.append(cos_map.clone())
    cos_map1 = cos_map.clone()
    
    for i in range(refine_times):
        thr = cos_map1.max() * tau
        cos_map1[cos_map1 < thr] = 0    
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        cos_rf.append(cos_map1.clone())
    return cos_rf

def get_refined_similarity_fg(point_coords, point_labels, feats, ratio=1, refine_times=1, tau=0.85):
    cos_map = get_cos_similarity_map(point_coords, point_labels, feats, ratio=ratio)
    cos_rf = []
    cos_rf.append(cos_map.clone())
    cos_map1 = cos_map.clone()
    for i in range(refine_times):
        thr = cos_map1.max() * tau
        cos_map1[cos_map1 < thr] = 0    
        feats_mask = feats * cos_map1.unsqueeze(1)
        feats_mask = feats_mask.sum([2,3], keepdim=True) / (cos_map1.unsqueeze(1).sum([2,3], keepdim=True) + 1e-6)
        cos_map1 = F.cosine_similarity(feats, feats_mask, dim=1)
        cos_rf.append(cos_map1.clone())
    return cos_rf

def fill_in_idx(idx_chosen, num_gt):
    assert idx_chosen.shape[0] != 0, '不能一个点都不选!'
    if idx_chosen.shape[0] >= num_gt / 2:
        idx_chosen = torch.cat((idx_chosen, idx_chosen[:num_gt-idx_chosen.shape[0]]), dim=0)
    else:
        repeat_times = num_gt // idx_chosen.shape[0]
        idx_chosen = idx_chosen.repeat(repeat_times, 1)
        idx_chosen = fill_in_idx(idx_chosen, num_gt)

def get_mask_points_single_instance(attn_map, neg_thr=0.1, num_gt=10):
    # Parameters:
    #     coords: num_pixels, 2
    #     attn:H, W
    #     cls: scalar,
    # Return:
    #     coords_chosen: num_gt, 2
    #     labels_chosen: num_gt
    device = attn_map.device
    coord_neg = (attn_map < neg_thr).nonzero(as_tuple=False)
    idx_chosen = torch.randperm(coord_neg.shape[0])[:num_gt].to(attn_map.device)
    coords_chosen = coord_neg[idx_chosen]
    num_points = coords_chosen.shape[0]

    if num_points < num_gt:
        if idx_chosen.shape[0] == 0:
            coords_chosen = -torch.ones(num_gt, 2, dtype=torch.float, device=device)
            print(f'***************一个点都没有找到!***************')
            # 这些-1的点会在point ignore里被处理掉
            return coords_chosen
        else:
            idx_chosen = fill_in_idx(idx_chosen, num_gt)
    coords_chosen = coord_neg[idx_chosen]
    return coords_chosen



COLORS = ('r', 'g', 'b', 'y', 'c', 'm', 'k', 'w')

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

epoch = 12
layer = 7
pth_dir =f'work_dir-proj/epoch_{epoch}.pth'
# pth_dir = f'work_dir_test/epoch_{epoch}.pth'
proposal_path = '/home/lmx/Dataset/VOC2012/Proposals/UCM-VOCAGU'
cfg = Config.fromfile('configs/mae/attnshift_deform_attn.py')
# print(cfg)
datasets = [build_dataset(cfg.data.train)]
cfg.data.samples_per_gpu = 1
cfg.model.backbone.use_checkpoint=True
cfg.data.workers_per_gpu = 1
cfg.model.roi_head.bbox_head.seed_score_thr=0.05
cfg.model.roi_head.bbox_head.cam_layer=layer
cfg.model.roi_head.mil_head.num_layers_query=layer
cfg.model.roi_head.bbox_head.seed_thr=0.1
cfg.model.roi_head.bbox_head.seed_multiple=0.5
cfg.model.corr_size=21
cfg.model.visualize = True
cfg.model.roi_head.visualize = True
cfg.model.roi_head.epoch_semantic_centers=10
cfg.model.obj_tau = 0.9
cfg.model.pos_mask_thr = 0.35
cfg.model.neg_mask_thr = 0.80
# cfg.model.roi_head.deform_attn_head.n_groups = 3
data_loaders = [
    build_dataloader(
        ds,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,   
        # cfg.gpus will be ignored if distributed
        dist=False,
        shuffle=False,
        seed=12345) for ds in datasets
]

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor')

model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()

_ = load_checkpoint(model, pth_dir, 'cpu', False)


save_dir = f'deform_attn_vis/{epoch}-no_norm'
os.makedirs(save_dir, exist_ok=True)

dataloader = iter(data_loaders[0])
thr = 0.1
dataloader._sampler_iter = iter(range(1, 1000))

from matplotlib.patches import Circle
import matplotlib.colors as mcolors

for idx in range(len(dataloader)):
    data = dataloader._dataset.__getitem__(idx)
    if idx > 200:
        break
    save_name = os.path.basename(data['img_metas'].data['filename'])[:-4]
    with torch.no_grad():
        # edges = data['edges'].cuda()
        img = data['img'].data
        # import torch.nn.functional as F
        mean=[123.675, 116.28, 103.53]
        std=[58.395, 57.12, 57.375]
        
        image = img.clone()
        for i, c in enumerate(img):
            image[i] = c * std[i] + mean[i]
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)

        img_metas = [data['img_metas'].data]
        gt_bboxes = [data['gt_bboxes'].data]
        gt_labels = [data['gt_labels'].data] 

        data_train = dict()
        for k in data:
            data_train[k] = data[k].data
        data_train['gt_bboxes'] = [data_train['gt_bboxes'].cuda()]
        data_train['gt_labels'] = [data_train['gt_labels'].cuda()]
        data_train['img'] = data_train['img'][None].cuda()
        data_train['img_metas'] = [data_train['img_metas']]
        model(**data_train)
        # embedding = model.embedding
        attns_sample = model.attns
        idx_attn = model.best_attn_idx
        attns_sample = attns_sample[idx_attn[0], range(idx_attn[0].shape[0])]
        attns_sample = norm_attns(attns_sample)
        bg_points = get_bg_points(attns_sample[0])
        attns  = [a for a in model.attns]
        bboxes = model.pseudo_gt_bboxes[0]
        num_obj = attns[0].shape[0]

    #############################
        cos_sim_model_bg = model.map_cos_bg[0]
        cos_sim_model_fg = model.map_cos_fg[0]

        cos_sim_model_bg = model.map_cos_bg[0][-1]
        cos_sim_model_fg = model.map_cos_fg[0][-1]
        sim_fg = model.sim_fg[0]
        semantic_center = model.semantic_centers[0]
        # points_fg = model.points_fg
        points_fg = model.mask_points_coords[0]
        points_label = model.mask_points_labels[0]
        attns = model.attns[[0]].detach().cpu().numpy()

        offset =  model.roi_head.keypoint_offset
        reference = model.roi_head.reference
        if len(model.roi_head.semantic_centers_split[0]) == 0:
            continue
        semantic_centers_split = torch.cat(model.roi_head.semantic_centers_split[0]).long().tolist()
        img_size = offset.new_tensor(img.shape[-2:])
        offset_coord = (offset + 1) / 2 * img_size[None, None]
        reference = (reference + 1) / 2 * img_size[None, None]

        offset_org = model.roi_head.assets[0]
        reference_org = model.roi_head.assets[1]
        point_weight = model.roi_head.assets[2].mean(0)
        # point_weight = (point_weight - point_weight.min(1, keepdim=True)[0]) / point_weight.max(1, keepdim=True)[0]
        point_weight = point_weight.repeat(1, model.roi_head.deform_attn.n_groups)
        offset_org = (offset_org + 1) / 2 * img_size[None, None]
        reference_org = (reference_org + 1) / 2 * img_size[None, None]
        map_cos_fg = model.roi_head.map_cos_fg[0].detach().cpu().numpy()
        num_parts = model.roi_head.num_parts[0]
        kp_scores = model.roi_head.kp_scores
        point_weight = kp_scores[:, None] * point_weight
        kp_scores = kp_scores.split(num_parts)

        _, axes = plt.subplots(num_obj, 3, squeeze=False)
        color_keys = list(mcolors.TABLEAU_COLORS.keys())
        offset_coord_obj = offset_coord.split(num_parts)
        point_weight_obj = point_weight.split(num_parts)
        for i_obj in range(num_obj):
            for k in range(3):
                axes[i_obj, k].imshow(image)
            axes[i_obj, 1].imshow(map_cos_fg[i_obj], cmap='jet', alpha=0.5)

            for i_s, weight_obj, off_obj, kp in zip(range(point_weight_obj[i_obj].shape[0]), point_weight_obj[i_obj].tolist(), offset_coord_obj[i_obj].long().tolist(), kp_scores[i_obj].tolist()):
                for weight, off in zip(weight_obj, off_obj):
                    color = mcolors.TABLEAU_COLORS[color_keys[i_s % len(color_keys)]]
                    circ = Circle((off[1], off[0]), radius=10*weight, color=color)
                    axes[i_obj, 2].add_patch(circ)
                    axes[i_obj, 2].text(off[1], off[0], f'{weight :.2f}, kp:{kp :.2f}')
        plt.savefig(f'{save_dir}/{idx}_{save_name}.jpg')