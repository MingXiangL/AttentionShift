
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
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import scipy
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


from mmcv.runner.checkpoint import load_checkpoint
from mmcv import Config, DictAction
from mmdet.datasets import (build_dataloader, build_dataset,
                          replace_ImageToTensor)
from mmdet.apis import set_random_seed
set_random_seed(1)
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as mcolors
from mmcv.ops import point_sample
pylab.rcParams['figure.figsize'] = (48.0, 24.0)


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
    for i_part in range(num_obj):
        box = bboxes[i_part]
        entropy[..., i_part] = map_entropy[..., i_part, box[1]:box[3], box[0]:box[2]].mean(dim=[-1, -2])
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
layer = 7


epoch = 8
layer = 7

pth_dir =f'work_dir-proj/epoch_{epoch}.pth'
save_dir = f'vis_deform_keypoints/{epoch}'
os.makedirs(save_dir, exist_ok=True)
# pth_dir =f'../work_dir-proj/epoch_12_dense_part.pth'

# pth_dir = f'work_dir_test/epoch_{epoch}.pth'
# cfg = Config.fromfile('/home/LiaoMingxiang/Workspace/psis/imted_psis_deform_attn/configs/mae/attnshift_deform_attn_focus_dense_reppoints.py')
# cfg = Config.fromfile('/home/LiaoMingxiang/Workspace/psis/imted_psis_deform_attn/configs/mae/attnshift_deform_attn_dense_reppoints.py')
cfg = Config.fromfile('configs/mae/attnshift_deform_attn_dense_contour_semantic_reppoints_attn.py')
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
cfg.model.roi_head.reppoints_head.visualize=True

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

print(f'len(data_loaders): {len(data_loaders)}')
dataloader = iter(data_loaders[0])
thr = 0.1
dataloader._sampler_iter = iter(range(1, 1000))


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


def filter_maps(maps, pos_maps, pos_thr=0.85):
    maps_fore = torch.where(maps>0.8, torch.ones_like(maps), torch.zeros_like(maps))
    
    pos_score = (pos_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    # neg_score = (neg_maps[:, None] * maps_fore).sum(dim=[-2, -1]) / maps_fore.sum(dim=[-2, -1]).clamp(1e-6)
    pos_idx = (pos_score >= pos_thr)
    # neg_idx = (neg_score >= neg_thr) & (pos_score < 0.5)
    split_size = pos_idx.sum(dim=-1).tolist()
    maps_fore = maps.flatten(0,1)[pos_idx.flatten()].split(split_size, dim=0)
    # maps_back = maps.flatten(0,1)[neg_idx.flatten()]
    return maps_fore, pos_idx

pylab.rcParams['figure.figsize'] = (48.0, 24.0)
for i_c in range(40, len(dataloader)):
    data = dataloader._dataset.__getitem__(i_c)
    
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
        img_name = os.path.basename(data['img_metas'].data['filename']).split('.')[0]

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

        sem_pred_points = model.roi_head.reppoints_head.sem_dense_points
        ctr_pred_points = model.roi_head.reppoints_head.ctr_dense_points
        sem_pred_all = model.roi_head.reppoints_head.sem_pts_preds
        ctr_pred_all = model.roi_head.reppoints_head.ctr_pts_preds
        anchor_list = model.roi_head.reppoints_head.anchor_list[0]
        dense_target = model.roi_head.reppoints_head.sem_dense_target[0]
        ctr_dense_target = model.roi_head.reppoints_head.ctr_dense_target[0]

        semantic_centers_split = torch.cat(model.roi_head.semantic_centers_split[0]).long().tolist()
        map_cos_fg = model.roi_head.map_cos_fg[-1].detach().cpu().numpy()
        num_parts = model.roi_head.num_parts[0]
        
        num_anchors = [n+1 for n in num_parts]
        sem_pred_points = torch.cat(sem_pred_points).long().detach().cpu().split(num_anchors, dim=0)
        ctr_pred_points = torch.cat(ctr_pred_points).long().detach().cpu().split(num_anchors, dim=0)
        anchors     = anchor_list.long().detach().cpu().split(num_anchors, dim=0)
        pseudo_gt_mask = model.pseudo_gt_masks[0]
        _, axes = plt.subplots(map_cos_fg.shape[0], 4, squeeze=False)
        color_keys = list(mcolors.TABLEAU_COLORS.keys())
        H, W = image.shape[:2]
        # vit_feat = model.vit_feat
        # img_size = semantic_center[0].new_tensor([W, H])
        # pred_points_cat = torch.cat(pred_points) / img_size
        # filter_maps(sim_fg.unflatten(0, (sim_fg.shape[0]//20, 20)), fg_inter, bg_inter)
        # pred_points_feat = point_sample(vit_feat, pred_points_cat)
        # sims = F.cosine_similarity(pred_points_feat[..., None, None], vit_feat, dim=1)
        # _, pos_idx = filter_maps(map_cos_fg)

        for i_f, fg in enumerate(map_cos_fg):
            axes[i_f, 0].imshow(image)
            axes[i_f, 1].imshow(image)
            axes[i_f, 2].imshow(image)
            axes[i_f, 2].imshow(fg, cmap='jet', alpha=0.5)
            axes[i_f, 3].imshow(image)
            axes[i_f, 0].imshow(pseudo_gt_mask[i_f], cmap='jet', alpha=0.5)
            axes[i_f, 1].imshow(pseudo_gt_mask[i_f], cmap='jet', alpha=0.5)

            for i_a in range(num_anchors[i_f]):
                color = mcolors.TABLEAU_COLORS[color_keys[i_a % len(color_keys)]]
                anchor = anchors[i_f][i_a].tolist()
                
                # box    = pred_bboxes[i_f][i_a].numpy()
                circ = Circle((anchor[0], anchor[1]), radius=10, color=color)
                
                # rect = Rectangle(box[:2], box[2]-box[0], box[3]-box[1], fill=False, color='blue')
                axes[i_f, 2].add_patch(circ)
                # axes[i_f, 2].add_patch(rect)
                point = dense_target[i_f].tolist()
                for p in point:
                    circ = Circle(p, radius=5, color=color)
                    axes[i_f, 0].add_patch(circ)

                point = ctr_dense_target[i_f].tolist()
                for p in point:
                    circ = Circle(p, radius=5, color=color)
                    axes[i_f, 1].add_patch(circ)

                point  = sem_pred_points[i_f][i_a].tolist()
                for p in point:
                    circ = Circle(p, radius=5, color=color)
                    axes[i_f, 2].add_patch(circ)

                point = ctr_pred_points[i_f][i_a].tolist()
                for p in point:
                    circ = Circle(p, radius=5, color=color)
                    axes[i_f, 3].add_patch(circ)
        plt.savefig(f'{save_dir}-{i_c}-{img_name}-overall.jpg')
    
    
    feat = model.roi_head.reppoints_head.feat
    cand_scores = model.roi_head.reppoints_head.cand_scores[0]
    gt_labels = model.roi_head.reppoints_head.gt_labels
    for i_obj in range(len(sem_pred_points)):
        new_points = sem_pred_points[i_obj]

        # new_points = torch.cat((sem_pred_points[0], out_points_1), dim=0)
        # new_points = out_points_1
        anchor_list = [new_points.flatten(0,1).float().to(sem_pred_all.device)]
        sub_ctr = model.roi_head.reppoints_head.get_pred_by_sample(anchor_list, ctr_pred_all)[0][0]
        sub_sem = model.roi_head.reppoints_head.get_pred_by_sample(anchor_list, sem_pred_all)[0][0]
        num_parts_keep = model.roi_head.reppoints_head.num_parts
        pdb.set_trace()
        cand_scores = model.roi_head.reppoints_head.cand_scores[0].unflatten(0, (-1, 10)).split(num_parts_keep.tolist())
        core_regions = model.roi_head.reppoints_head.core_regions[0].long().cpu().numpy()
        cls_score = model.roi_head.reppoints_head.part_score

        attn, part_cls = model.roi_head.reppoints_head.deform_attn(
            anchor_list,
            [sub_sem],
            feat,
        )
        cls_idx = gt_labels[0][i_obj]
        cls_score = part_cls.sigmoid()[:, cls_idx]
        n_r = len(cand_scores[i_obj])
        n_c = cand_scores[i_obj].shape[1]
        _, axes = plt.subplots(n_r, n_c, squeeze=False)
        for i_part, ax in enumerate(axes):
            for i_c, ctr, sem, c_s, cls_s in zip(range(sub_ctr[i_part].shape[0]) , sub_ctr.split(10, dim=0)[i_part], sub_sem.split(10, dim=0)[i_part], cand_scores[i_obj][i_part], cls_score.split(10, dim=0)[i_part]):
                img_vis = image.copy()
                hull  = cv2.convexHull(ctr.long().cpu().numpy())
                for i in range(len(hull)):
                    cv2.line(img_vis, tuple(hull[i][0]), tuple(hull[(i+1) % len(hull)][0]), (0, 255, 0), 4)
                axes[i_part, i_c].imshow(img_vis)
                axes[i_part, i_c].imshow(core_regions[i_obj], cmap='jet', alpha=core_regions[i_obj]*0.5)
                axes[i_part, i_c].scatter(new_points[0, i_c, 0].item(), new_points[0, i_c, 1].item(), marker='*')
                p = new_points[i_part, i_c].long().cpu().numpy()
                
                circ = Circle((p[0], p[1]), radius=20, color='red')
                axes[i_part, i_c].add_patch(circ)
                axes[i_part, i_c].set_title(f'overlap: {c_s :.2f}, cls: {cls_s.item() :.2f}')
                
                for p in ctr.long().tolist():
                    circ = Circle((p[0], p[1]), radius=10, color='red')
                    axes[i_part, i_c].add_patch(circ)
                    
                for p in sem.long().tolist():
                    # AECBF5
                    circ = Circle((p[0], p[1]), radius=10, color='#00B4FF')
                    axes[i_part, i_c].add_patch(circ)
        plt.savefig(f'{save_dir}-{i_c}-{img_name}-obj_{i_obj}.jpg')

