import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def cosine_similarity_matrix(feats1, feats2):
    # feats1: ..., C
    # feats2: ..., C
    feats1 = feats1 / torch.norm(feats1, p=2, dim=-1, keepdim=True)
    feats2 = feats2 / torch.norm(feats2, p=2, dim=-1, keepdim=True)
    return feats1 @ feats2.T


def cosine_similarity_feat_obj_mean(feats, objects, return_org=False):
    # feats: tensor, [N, ndim]
    # objects: OBJList
    # return: sim_o_f: n_obj x N
    feats_obj = torch.cat([
        obj.feats for obj in objects.objects
    ], dim=0)
    split_size = [obj.count() for obj in objects.objects]
    sim_o_f = F.cosine_similarity(feats_obj[:, None], feats[None], dim=-1).split(split_size)
    sim_o_f_org = sim_o_f[-1].clone()
    sim_o_f = torch.cat([
        sim.mean(0, keepdim=True) for sim in sim_o_f
    ], dim=0)

    if return_org:
        return sim_o_f, sim_o_f_org
    return sim_o_f


def cosine_similarity_feat_obj(feats, objects, return_org=False):
    # feats: tensor, [N, ndim]
    # objects: OBJList
    # return: sim_o_f: n_obj x N
    feats_obj = torch.cat([
        obj.feats for obj in objects.objects
    ], dim=0)
    split_size = [obj.count() for obj in objects.objects]
    sim_o_f = F.cosine_similarity(feats_obj[:, None], feats[None], dim=-1).split(split_size)
    sim_o_f_org = sim_o_f[-1].clone()
    try:
        sim_o_f = torch.cat([
            sim.max(0, keepdim=True)[0] for sim in sim_o_f
        ], dim=0)
    except BaseException:
        print(f'sim_o_f: {sim_o_f}')
        pdb.set_trace()

    if return_org:
        return sim_o_f, sim_o_f_org
    return sim_o_f


def filter_prots(objects, sim_f_f_objs):
    pass


def group_and_filter_bg_prots(bg_prots, fg_objs, sim_thr):
    sim_b_b = cosine_similarity_matrix(bg_prots, bg_prots)
    sim_f_b = cosine_similarity_feat_obj(bg_prots, fg_objs)
    keep_idx_fb = sim_f_b.max(0)[0] < sim_thr
    prots = bg_prots[keep_idx_fb]
    if torch.numel(prots) == 0:
        prots = bg_prots[[0]]

    # sim_b_b = sim_b_b[keep_idx_fb]
    # obj_list = OBJList()
    # keep_list = torch.ones(prots.shape[0], dtype=prots.dtype, device=prots.device)
    # while keep_list.sum() > 0:
    #     cur_idx = keep_list.argmax()
    #     sim = sim_b_b[cur_idx]
    #     chosen_idx = sim > sim_thr
    #     obj_list.add_new_obj(prots[chosen_idx])
    #     keep_list[chosen_idx] = 0
    #     sim_b_b[:, chosen_idx] = 0
    # return obj_list
    return OBJList([prots])

def multi_and_norm(s1, s2):
    s = s1 * s2
    return s / s.sum(1, keepdim=True)


def cal_obj_dist_uperbound(attns, sim_inter, spatial_weights=None):
    # attns.shape: #obj x N
    # sim_inter: N x N
    obj_idx = attns > 0
    if spatial_weights is None:
        spatial_weights = torch.ones(attns.shape[0], sim_inter.shape[0], dtype=sim_inter.dtype, device=sim_inter.device)
    inner_bound = cal_max_dist(obj_idx_1=obj_idx, obj_idx_2=obj_idx, sim_inter=sim_inter, spatial_weights=spatial_weights)
    inter_bound = cal_min_dist(obj_idx_1=obj_idx, obj_idx_2=obj_idx, sim_inter=sim_inter, spatial_weights=spatial_weights)
    
    return torch.maximum(inner_bound, inter_bound)
    

def cal_max_dist(obj_idx_1, obj_idx_2, sim_inter, spatial_weights=None):
    num_obj = obj_idx_1.shape[0]
    sim_bound = []
    for i_obj in range(num_obj):
        if spatial_weights is not None:
            sim = (sim_inter * spatial_weights[i_obj][:, None])[obj_idx_1[i_obj]][:, obj_idx_2[i_obj]]
        else:
            sim = sim_inter[obj_idx_1[i_obj]][:, obj_idx_2[i_obj]]
        if sim.shape[1] == 0:
            # Skip images with only one object for the purpose of streamlining the process in the first step
            sim_bound.append(0.8*torch.ones(1, dtype=sim.dtype, device=sim.device).squeeze())
        else:
            sim_bound.append(sim.min())
    return torch.stack(sim_bound)


def cal_mean_dist(obj_idx_1, obj_idx_2, sim_inter, spatial_weights=None):
    num_obj = obj_idx_1.shape[0]
    sim_bound = []
    for i_obj in range(num_obj):
        if spatial_weights is not None:
            sim = (sim_inter * spatial_weights[i_obj][:, None])[obj_idx_1[i_obj]][:, obj_idx_2[i_obj]]
        else:
            sim = sim_inter[obj_idx_1[i_obj]][:, obj_idx_2[i_obj]]
        pdb.set_trace()
        sim_bound.append(sim.sum() / obj_idx_1[i_obj].sum().clamp(min=1.0))
    return torch.stack(sim_bound)


def cal_min_dist(obj_idx_1, obj_idx_2, sim_inter, spatial_weights):
    num_obj = obj_idx_1.shape[0]
    sim_bound = []
    for i_obj in range(num_obj):
        idx_others = (obj_idx_2[:i_obj].sum(0) + obj_idx_2[i_obj+1:].sum(0)) > 0
        if spatial_weights is not None:
            sim = (sim_inter * spatial_weights[i_obj][:, None])[obj_idx_1[i_obj]][:, idx_others]
        else:
            sim = sim_inter[obj_idx_1[i_obj]][:, idx_others]
        
        if torch.numel(sim) == 0:
            # Skip images with only one object for the purpose of streamlining the process in the first step
            sim_bound.append(torch.zeros(1, dtype=sim.dtype, device=sim.device))
        else:
            sim_bound.append(sim.max())
    return torch.stack(sim_bound).reshape(-1)


def water_fill(feats, sim_inter, attns_in, n_iter=1):
    prots = []
    N = attns_in.sum()
    max_sim =  sim_inter.max(1, keepdim=True)[0]
    sim_inter[sim_inter<max_sim*0.8] = 0 # 这个阈值得设定得和图片内特征的紧致程度关联
    # D = sim_inter[:, attns_in>0]
    # D = torch.quantile(D, q=0.5, dim=1) / torch.max(D, dim=1)[0]
    for i in range(n_iter):
        S_in = sim_inter @ attns_in
        # S_out = sim_inter @ attns_out
        # sim_in = sim_inter[:, attns_in > 0]
        S, prot_idx = torch.max((S_in), dim=0)
        if S > 0 or i==0:
            prots.append(feats[prot_idx])
        else:
            break
        new_attn_in = torch.where(sim_inter[prot_idx]>0, torch.ones_like(sim_inter[prot_idx]), torch.zeros_like(sim_inter[prot_idx]))
        attns_in -= new_attn_in * (attns_in > 0)
        attns_in = attns_in.clamp(0, 1)
    return torch.stack(prots)


def water_fill_adaptive(feats, sim_inter, attns_in, n_iter=1, thr=0.8):
    prots = []
    N = attns_in.sum()
    # max_sim =  sim_inter.max(1, keepdim=True)[0]
    sim_inter[sim_inter <= thr] = 0 # 这个阈值得设定得和图片内特征的紧致程度关联
    # D = sim_inter[:, attns_in>0]
    # D = torch.quantile(D, q=0.5, dim=1) / torch.max(D, dim=1)[0]
    for i in range(n_iter):
        S_in = sim_inter @ attns_in
        # S_out = sim_inter @ attns_out
        # sim_in = sim_inter[:, attns_in > 0]
        S, prot_idx = torch.max((S_in), dim=0)
        if S > 0 or i==0:
            prots.append(feats[prot_idx])
        else:
            break
        new_attn_in = torch.where(sim_inter[prot_idx]>0, torch.ones_like(sim_inter[prot_idx]), torch.zeros_like(sim_inter[prot_idx]))
        attns_in -= new_attn_in * (attns_in > 0)
        attns_in = attns_in.clamp(0, 1)
    return torch.stack(prots)

def water_fill_adaptive_v1(feats, sim_inter, attns, n_iter, thr):
    prots = []
    N = attns.sum()
    # max_sim =  sim_inter.max(1, keepdim=True)[0]
    sim_inter[sim_inter <= thr] = 0 # 这个阈值得设定得和图片内特征的紧致程度关联
    # D = sim_inter[:, attns>0]
    # D = torch.quantile(D, q=0.5, dim=1) / torch.max(D, dim=1)[0]
    for i in range(n_iter):
        pdb.set_trace
        S_in = sim_inter @ attns
        S_out = sim_inter @ attns
        # sim_in = sim_inter[:, attns > 0]
        S, prot_idx = torch.max((S_in - S_out), dim=0)
        if S > 0 or i==0:
            prots.append(feats[prot_idx])
        else:
            break
        new_attn_in = torch.where(sim_inter[prot_idx]>0, torch.ones_like(sim_inter[prot_idx]), torch.zeros_like(sim_inter[prot_idx]))
        attns -= new_attn_in * (attns > 0)
        attns = attns.clamp(0, 1)
    return torch.stack(prots)


def unflatten_coordinate(idx, img_shape):
    H, W = img_shape[0], img_shape[1]
    return torch.stack((idx // W, (idx % W)), dim=0)


def norm_attns(attns):
    N, _ = attns.shape
    max_val, _ = attns.view(N,-1).max(dim=1, keepdim=True)
    min_val, _ = attns.view(N,-1).min(dim=1, keepdim=True)
    return attns / max_val


def project_to_coord(maps, k=2):
    coords = []
    H, W = maps.shape[-2:]
    for _, map_ in enumerate(maps):
        topk = map_.flatten().topk(dim=0, k=k)[0][-1]
        coord_topk = (map_ >= topk).nonzero().float()
        coord = coord_topk.mean(dim=0).flip(0)
        coord[0] /= W
        coord[1] /= H
        coords.append(coord)
    return torch.stack(coords)


class OBJ(object):
    
    def __init__(self, feats=None):
        self.feats = feats
        
    def append(self, new_feats):
        if getattr(self, 'feats', None) is None:
            self.feats = new_feats
        else:
            self.feats = torch.cat((self.feats, new_feats), dim=0)
    
    def update(self, new_feats):
        self.feats = new_feats

    def count(self):
        return self.feats.shape[0]


class OBJList(object):

    def __init__(self, objs:list=None) -> None:
        if objs is not None:
            self.objects = self.new_objs(objs)
        else:
            self.objects =  []
    
    def append(self, obj:OBJ):
        self.objects.append(obj)
    
    def add_new_obj(self, feat):
        self.append(OBJ(feat))

    def new_objs(self, feats:list):            
        return [OBJ(f) for f in feats]
    
    def cat_feats(self):
        feats = torch.cat([
            obj.feats for obj in self.objects
        ], dim=0)
        split_size = [obj.count() for obj in self.objects]
        return feats, split_size

    def update(self, new_objs):
        for obj, n in zip(self.objects, new_objs.objects):
            obj.update(n.feats)

    def concate(self, obj_list):
        self.objects += obj_list.objects
        return self

    def __len__(self):
        return len(self.objects)
        

class ObjMeanCosSim(object):

    def __call__(self, objects: OBJList):
        feats, split_size = objects.cat_feats()
        sim = F.cosine_similarity(feats[:, None], feats[None], dim=-1).split(split_size, dim=1)

        sim = torch.cat([s.mean(1, keepdim=True) for s in sim], dim=1).split(split_size, dim=0)
        sim = torch.cat([s.mean(0, keepdim=True) for s in sim], dim=0)
        # sim0 = F.cosine_similarity(feats[:, None], feats[None], dim=-1).split(split_size, dim=1)
        # pdb.set_trace()
        return  sim


class MeanFieldDecoder(object):

    def __init__(self, num_iter=20, sigma_factor=0.5):
        self._softmax = nn.Softmax(dim=0)
        self.iter = num_iter
        self.sigma_factor = sigma_factor
        self.inter_feat_sim = None
        self.inter_obj_sim = ObjMeanCosSim()

    def get_feat_grid(self, img_shape, device):
        H, W = img_shape
        range_h = torch.arange(H, dtype=torch.float16, device=device) / H
        range_w = torch.arange(W, dtype=torch.float16, device=device) / W
        grid_h, grid_w = torch.meshgrid(range_h, range_w)
        grid = torch.stack((grid_w.flatten(), grid_h.flatten()), dim=1)
        return grid

    def get_compatibility(self, prots):
        if getattr(self, 'comp', None) is None:
            weight = F.softmax(F.cosine_similarity(prots[:, None], prots[None], dim=-1), dim=1)
            self.comp = weight
        return self.comp

    def get_obj_compatibility(self, objects: OBJList):
        if getattr(self, 'comp_obj', None) is None:
            sim = self.inter_obj_sim(objects)
            # weight = F.softmax(sim, dim=0)
            # weight = (sim - 0.8).sigmoid()
            # weight = torch.where(
            #     sim < 0.8,
            #     torch.zeros_like(sim),
            #     sim
            # )
            # TODO: 这个weight是不是应该换成-sim
            weight = torch.where((torch.eye(sim.shape[0], device=sim.device)>0) & (sim > 0), sim, -sim)
            weight = F.softmax(weight, dim=1)
            self.comp_obj = weight
        return self.comp_obj

    def get_spatial_weight_prot(self, boxes, img_shape, feat_coords, coord_bg_prot):
        if getattr(self, 'spatial_weight', None) is None:
            H, W = img_shape
            centers = (boxes[:, 2:] + boxes[:, :2]) / 2
            centers[:, 0] /= W
            centers[:, 1] /= H
            centers = torch.cat((centers, coord_bg_prot), dim=0)
            dist = (feat_coords[None] - centers[:, None]) ** 2
            sigma_fg = self.get_sigma(boxes, img_shape)
            sigma = torch.cat((sigma_fg, torch.ones(coord_bg_prot.shape[0], 2, dtype=sigma_fg.dtype, device=sigma_fg.device)), dim=0)
            # weight = F.softmax(- dist.sum(-1) / (2 * sigma**2), dim=0)
            # weight = torch.exp(- dist.sum(-1) / (2 * sigma**2)) / (sigma * math.sqrt(2*math.pi))
            weight = torch.exp(- (dist / (2 * sigma.unsqueeze(1) ** 2)).sum(-1)) / (sigma[:, 0] * sigma[:, 1] * 2 * math.pi).unsqueeze(1)
            self.spatial_weight = weight
        return self.spatial_weight

    def get_spatial_weight_prot_fg_only(self, boxes, img_shape, feat_coords):
        # TODO: 背景的坐标怎么处理？
        # 要不背景的就不加这个weight了？即，所有的spatial weight权重都是1
        if getattr(self, 'spatial_weight', None) is None:
            H, W = img_shape
            centers = (boxes[:, 2:] + boxes[:, :2]) / 2
            centers[:, 0] /= W
            centers[:, 1] /= H
            dist = (feat_coords[None] - centers[:, None]) ** 2
            sigma = self.get_sigma(boxes, img_shape)
            weight = torch.exp(- (dist / (2 * sigma.unsqueeze(1) ** 2)).sum(-1)) 
            # / (sigma[:, 0] * sigma[:, 1] * 2 * math.pi).unsqueeze(1)
            # 背景的spatial weight目前设置为1
            weight = torch.cat((
                weight, torch.ones(1, weight.shape[1], dtype=weight.dtype, device=weight.device)
            ),  dim=0)
            self.spatial_weight = weight

        return self.spatial_weight

    def get_sigma(self, boxes, img_shape):
        H, W = img_shape
        wh = boxes[:, 2:] - boxes[:, :2]
        wh[:, 0] /= W
        wh[:, 1] /= H
        return wh * self.sigma_factor

    def get_spatial_weight(self, feat_coords):
        dist = (feat_coords[:, None] - feat_coords[None]) ** 2
        weight = F.softmax(- dist.sum(-1) / (2 * self.sigma_factor**2), dim=1)
        return weight

    def get_bilateral_weight(self, feats=None, org_feats=None):
        # TODO: 这个是不是考虑把原图像、不同层之间的信息都引进来？如果只用最后一层的信息，和unary的信息其实是有很多重叠的
        # weight = F.softmax(2*F.cosine_similarity(feats[:, None], feats[None], dim=-1), dim=0)
        # weight = (F.cosine_similarity(feats[:, None], feats[None], dim=-1)*10)
        if getattr(self, 'bilateral_weight', None) is None:
            assert feats is not None
            if org_feats is None:
                feats1 = feats / torch.norm(feats, p=2, dim=1, keepdim=True)
                weight = feats1 @ feats1.T * 10
            else:
                feats1 = org_feats / torch.norm(org_feats, p=2, dim=1, keepdim=True)
                weight = torch.matmul(feats1, feats1.transpose(1,2)) * 10
                weight = weight.mean(0)
            self.bilateral_weight = weight
        # T = weight.quantile(q=q, dim=-1)[:, None]
        # weight[weight<T] = 0
        return self.bilateral_weight

    def get_img_bilateral_weight(self, img):
        if getattr(self, 'img_bilateral_weight', None) is None:
            if img is None:
                self.img_bilateral_weight = None
            else:
                self.img_bilateral_weight = ((img[None] - img[:, None]) ** 2).mean(-1) / (2 * 0.25**2)
        return self.img_bilateral_weight

    def mean_field_solve(self, logits, compatibility, spatial_weight, bilateral_weight, spatial_weight_q, img_bilateral_weight=None):
        # spatial_weight: N x N 
        # bilateral_weight: N x N 
        # compatibility: p x p
        weight = spatial_weight * torch.exp(bilateral_weight)
        if img_bilateral_weight is not None:
            weight *= torch.exp(img_bilateral_weight)
        weight -= torch.diag_embed(weight.diag())
        weight = weight / weight.sum(0, keepdim=True)

        for i in range(self.iter):
            # p x N
            q_values = self._softmax(logits) * spatial_weight_q
            q_values = q_values / q_values.sum(0, keepdim=True).clamp(1e-6)
            
            msg_pass_out = torch.mm(q_values, weight)
            msg_pass_out = torch.mm(compatibility, msg_pass_out)
            logits = logits + msg_pass_out
 
        q_values = self._softmax(logits)  * spatial_weight_q
        q_values = q_values / q_values.sum(0, keepdim=True).clamp(1e-6)

        return logits
            
    def assign_feat_to_obj(self, feats, objects, boxes, img_shape, org_feats=None, img=None):
        feat_coords = self.get_feat_grid(img_shape, feats.device)
        spatial_weight_q = self.get_spatial_weight_prot_fg_only(boxes, img_shape, feat_coords)
        spatial_weight = self.get_spatial_weight(feat_coords)
        bilateral_weight = self.get_bilateral_weight(feats, org_feats)
        img_bilateral_weight = self.get_img_bilateral_weight(img)
        compatibility = self.get_obj_compatibility(objects) # TODO： iner-object similarity形式可以改变
        logits = cosine_similarity_feat_obj(feats, objects)
        logits = self.mean_field_solve(logits, compatibility, spatial_weight, bilateral_weight, spatial_weight_q, img_bilateral_weight=img_bilateral_weight)
        # print(f'spatial_weight.shape: {spatial_weight.shape}')
        # print(f'spatial_weight_q.shape: {spatial_weight_q.shape}')
        return logits

    def gen_attn_map(self, feats, prots, prot_labels, prot_coords, prot_org, boxes, img_shape):
        sim_f_p = F.cosine_similarity(feats[None, None], prots[:,:,None], dim=-1)
        sim_org = F.cosine_similarity(feats[None], prot_org[:, None], dim=-1)

        idx_prot = torch.arange(prots.shape[0], device=sim_f_p.device, dtype=sim_f_p.dtype).unsqueeze(1)
        idx_pos = prot_labels == idx_prot
        idx_bg  = prot_labels == prots.shape[0] # 背景被放在了最后一个部分
        num_pos = idx_pos.sum(1, keepdim=True)
        num_bg  = idx_bg.sum(1, keepdim=True)
        sim_obj = torch.where(idx_pos[:,:,None], sim_f_p, torch.zeros_like(sim_f_p)).sum(1) / num_pos.clamp(1e-6)
        sim_obj = torch.where(num_pos>0, sim_obj, sim_org[:-1])
        sim_bg = torch.where(idx_bg[:,:,None], sim_f_p, torch.zeros_like(sim_f_p)).sum(dim=[0, 1]) / num_bg.sum().clamp(1e-6)
        
        grid = self.get_feat_grid(img_shape=img_shape, device=sim_obj.device)
        spatial_weight = self.get_spatial_weight_prot(boxes, img_shape, grid, torch.ones(0, 2, device=sim_obj.device, dtype=sim_obj.dtype))
        spatial_weight /= spatial_weight.max(1, keepdim=True)[0]

        sim_obj *= spatial_weight
        if num_bg.sum() == 0:
            sim_bg = sim_org[-1]
        return torch.cat((sim_obj, sim_bg[None]), 0)


class AttnCRFer(object):

    def __init__(self, 
                crf_shift_iter=1, 
                mean_field_iter=10, 
                mean_field_sigma_factor=0.5,
                sim_bin_thr=0.8,
                attn_fg_thr=0.2,
                attn_bg_thr=0.1,
                ):
        self.assigner = MeanFieldDecoder(num_iter=mean_field_iter, sigma_factor=mean_field_sigma_factor)
        self.num_shift_iter = crf_shift_iter
        self.sim_bin_thr = sim_bin_thr
        self.attn_fg_thr = attn_fg_thr
        self.attn_bg_thr = attn_bg_thr

    def get_inter_feat_sim(self, feats):
        if getattr(self, 'inter_feat_sim', None) is None:
            # self.inter_feat_sim = F.cosine_similarity(feats[:, None], feats[None], dim=-1)
            prots1 = feats / torch.norm(feats, p=2, dim=1, keepdim=True)
            self.inter_feat_sim = prots1 @ prots1.T
            # print(f'qualified: {(torch.abs(self.inter_feat_sim - weight1) > 1e-3).sum() == 0}')

        return self.inter_feat_sim

    def get_obj_prots_min_geo_dist(self, feats, attns, boxes, img_shape):
        sim_f_f = self.get_inter_feat_sim(feats)
        # sim_f_f = torch.where(sim_f_f > self.sim_bin_thr, torch.ones_like(sim_f_f), torch.zeros_like(sim_f_f))
        
        bin_attn = attns.transpose(0, 1)
        max_val = bin_attn.max(1, keepdim=True)[0]
        bin_attn = torch.where((bin_attn / bin_attn.max(0, keepdim=True)[0] > self.sim_bin_thr) & (bin_attn == max_val), 
                    torch.ones_like(bin_attn), torch.zeros_like(bin_attn))
        #当一个特征占据了很大的区域时，如何避免这个区域直接主导了接下来的特征shift
        self.attn_fg = bin_attn.transpose(0, 1)
        connect = torch.mm(sim_f_f, bin_attn) / bin_attn.sum(0, keepdim=True).clamp(1e-6)
        prot_idx = connect.argmax(0)
        return feats[prot_idx], unflatten_coordinate(prot_idx, img_shape)

    def update_objects(self, feats, attns, boxes, objects, img_shape):
        # objects_bg = self.get_bg_objects(feats, attns[:-1])
        # objects_fg = self.get_fg_objects(feats, attns, boxes, img_shape)
        objects.update(self.get_prots_step(feats, attns, boxes, img_shape))
        return objects

    def get_point_label(self, prob, boxes):
        # 后面再慢慢加功能。。
        return prob.argmax(0)

    def do_one_shift(self, feats, boxes, objects, img_shape, org_feats=None, img=None):
        attns = self.assigner.assign_feat_to_obj(
            feats, 
            objects, 
            boxes,
            img_shape,
            org_feats,
            img=img,
        )
        return attns, objects

    def get_bg_objects(self, feats, attns_fg, fg_objs):
        sim_f_f = self.get_inter_feat_sim(feats)
        attn_fg = attns_fg.max(0)[0]
        attn_bg = (1 - attn_fg).clamp(max=1)
        # attn_bg[attn_bg > self.sim_bin_thr] =  1
        n_iter = 10
        sim_thr = cal_obj_dist_uperbound(torch.stack((attn_bg, attn_fg)), sim_f_f)
        prots = water_fill_adaptive(
            feats, 
            sim_f_f,
            attn_bg, 
            n_iter=n_iter,
            thr=sim_thr[0]
        )
        return group_and_filter_bg_prots(prots, fg_objs, sim_thr=0.9)
        # return OBJList([prots])

    def get_fg_objects(self, feats, attns, boxes, img_shape, fg_prot_num=1, bg_prot_num=10):
        num_obj = attns.shape[0]
        sim_f_f = self.get_inter_feat_sim(feats)
        feats_coords = self.assigner.get_feat_grid(img_shape, feats.device)
        # N x n_obj
        # spatial_weights = self.assigner.get_spatial_weight_prot_fg_only(boxes, img_shape, feats_coords, num_bg=attns.shape[0] - boxes.shape[0])
        spatial_weights = self.assigner.get_spatial_weight_prot_fg_only(boxes, img_shape, feats_coords)

        # if attns.shape[0] - boxes.shape[0] > 1:
        #     spatial_weights = torch.cat((
        #         spatial_weights, 
        #         torch.ones(attns.shape[0] - boxes.shape[0],
        #             spatial_weights.shape[1],
        #             dtype=spatial_weights.dtype, 
        #             device=spatial_weights.device)
        #         )
        #     )
        obj_list = OBJList()
        sim_thr = cal_obj_dist_uperbound(attns, sim_f_f, spatial_weights)
        for i_obj in range(num_obj):
            n_iter = fg_prot_num if i_obj < boxes.shape[0] else bg_prot_num
            prots = water_fill_adaptive(
                feats, 
                sim_f_f * spatial_weights[i_obj][:, None],
                attns[i_obj].clone(), 
                n_iter=n_iter,
                thr=sim_thr[i_obj]
            )

            # prots = water_fill_adaptive_v1(
            #     feats, 
            #     sim_f_f[None] * spatial_weights,
            #     attns,
            #     n_iter=n_iter,
            #     thr=sim_thr
            # )
            obj_list.add_new_obj(prots)
        
        
        return obj_list

    def get_initial_attn(self, feats, attns, boxes, img_shape):
        objects = self.get_initial_prots_joint(feats, attns, boxes, img_shape)
        return cosine_similarity_feat_obj(feats, objects)

    def defuse_attn(self, feats, attns, boxes, img_shape):
        max_val = attns.max(0, keepdim=True)[0]
        attns[attns != max_val] = 0 
        attns_fg = torch.where(
            norm_attns(attns) > self.attn_fg_thr, 
            torch.ones_like(attns), 
            torch.zeros_like(attns)
        )
        sim_f_f = torch.where(
            self.get_inter_feat_sim(feats) > self.sim_bin_thr, 
            torch.ones_like(sim_f_f), 
            torch.zeros_like(sim_f_f)
        )

    def get_initial_prots_joint(self, feats, attns, boxes, img_shape):
        attns = norm_attns(attns)
        max_val = attns.max(0, keepdim=True)[0]
        attns[attns != max_val] = 0
        
        attns_fg = torch.where(
            attns > self.attn_fg_thr, 
            torch.ones_like(attns), 
            torch.zeros_like(attns)
        )
        attns_bg = attns.max(0, keepdim=True)[0]
        attns_bg = torch.where(
            attns_bg < self.attn_bg_thr, 
            torch.ones_like(attns_bg), 
            torch.zeros_like(attns_bg)
        )
        objects = self.get_fg_objects(feats, attns_fg, boxes, img_shape)
        attns_fg = cosine_similarity_feat_obj(feats, objects)
        attns_fg = torch.where(attns_fg > self.sim_bin_thr, torch.ones_like(attns_fg), torch.zeros_like(attns_fg))
        objects_bg = self.get_bg_objects(feats, attns_fg, objects)
        attns_bg = cosine_similarity_feat_obj(feats, objects_bg)
        # attns_bg = torch.where(attns_bg > self.sim_bin_thr, torch.ones_like(attns_bg), torch.zeros_like(attns_bg))
        # objects = self.get_fg_objects(feats, torch.cat((attns_fg, attns_bg), dim=0), boxes, img_shape)
        return objects.concate(objects_bg)

    def update_object_prots(self, feats, attns, boxes, img_shape):
        attns = norm_attns(attns)
        max_val = attns.max(0, keepdim=True)[0]
        attns[attns != max_val] = 0
        attns_fg = torch.where(
            attns > self.attn_fg_thr, 
            torch.ones_like(attns), 
            torch.zeros_like(attns)
        )
        # when step > 1, the bg is also considered as a foreground object
        objects = self.get_fg_objects(feats, attns_fg, boxes, img_shape)
        return objects


    def get_prots_step(self, feats, attns, boxes, img_shape):
        max_val = attns.max(dim=1, keepdim=True)[0]
        attns = torch.where(
            attns > max_val * self.sim_bin_thr, 
            torch.ones_like(attns),
            torch.zeros_like(attns)
        )
        self.attns = attns
        objects = self.get_fg_objects(feats, attns, boxes, img_shape)
        return objects


    def get_initial_prots(self, feats, attns, boxes, img_shape):
        max_val = attns.max(0, keepdim=True)[0]
        attns[attns != max_val] = 0 
        attns_fg = torch.where(
            norm_attns(attns) > self.attn_fg_thr, 
            torch.ones_like(attns), 
            torch.zeros_like(attns)
        )
        self.attns_fg =  attns_fg
        object_bg = self.get_bg_objects(feats, attns_fg)
        attns_bg = cosine_similarity_feat_obj(feats, object_bg)
        attns_bg = torch.where(attns_bg > self.sim_bin_thr, torch.ones_like(attns_bg), torch.zeros_like(attns_bg))
        
        objects = self.get_fg_objects(feats, torch.cat((attns_fg, attns_bg), dim=0), boxes, img_shape)
        return objects.concate(object_bg)
    
    def reset(self):
        self.inter_feat_sim = None

    # def __call__(self, feats, attns, boxes, img_shape, org_feats=None, img=None):
    #     attns_max = norm_attns(attns).max(0, keepdim=True)[0]
    #     objects = self.get_initial_prots_joint(feats, attns, boxes, img_shape)
    #     for _ in range(1):
    #         attns, objects = self.do_one_shift(
    #             feats, 
    #             boxes,
    #             objects,
    #             img_shape,
    #             org_feats,
    #             img=img,
    #         )
    #         # objects = self.update_object_prots(feats, attns, boxes, img_shape)


    def __call__(self, obj_feats, feats, boxes, img_shape, org_feats=None, img=None):
        # obj_feats: list: N_obj,
        objects = OBJList()
        for prots in obj_feats:
            objects.add_new_obj(prots)
        attns, objects = self.do_one_shift(feats,
                boxes,
                objects,
                img_shape,
                org_feats,
                img=img,
            )
        return attns
