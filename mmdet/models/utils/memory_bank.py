import mmcv
import torch
import cv2
import pdb
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.core import multi_apply
from mmcv.ops.roi_align import RoIAlign
from mmcv import tensor2imgs
from mmcv.runner.fp16_utils import force_fp32
from ..builder import build_loss, HEADS

from torch.cuda.amp import autocast


from mmcv.cnn import bias_init_with_prob, ConvModule
import numpy as np


def relu_and_l2_norm_feat(feat, dim=1):
    feat = F.relu(feat, inplace=True)
    feat_norm = ((feat ** 2).sum(dim=dim, keepdim=True) + 1e-6) ** 0.5
    feat = feat / (feat_norm + 1e-6)
    return feat


class ObjectFactory:

    @staticmethod
    def create_one(token, parts, box, category, device='cpu'):
        object_elements = ObjectElements(size=1,
                                         token=token,
                                         part_feats=parts,
                                         box=box,
                                         device=device,
                                         category=category)
        # object_elements.part_feats[...] = relu_and_l2_norm_feat(object_elements.part_feats[0:1]) # 这个后面可以调一下
        return object_elements

    @staticmethod
    def create_queue_by_one(len_queue, category, idx, tokens, part_feats, box, device='cpu'):
        device = tokens.device
        object_elements = ObjectElements(size=len_queue,
                                         token=tokens[idx:idx+1],
                                         part_feats=part_feats[idx],
                                         box=box[idx:idx+1], 
                                         device=device,
                                         category=category)
        return object_elements


class ObjectElements:

    #@autocast(enabled=False)
    def __init__(self, size=100, token=None, part_feats=None, box=None, device='cpu', category=None):
        self.size = size
        self.token = torch.zeros(size, token.shape[1], device=device, dtype=token.dtype)
        self.part_feats = [part_feats]
        self.num_parts = torch.zeros(size, device=device, dtype=torch.long)
        self.box = torch.zeros(size, 4, device=device, dtype=token.dtype)
        self.category = int(category)
        
        self.ptr = 0
        self.device = device
        self.token[0:1] = token
        self.num_parts[0] = part_feats.shape[0]
        self.box[0] = box
        
    #@autocast(enabled=False)
    def get_box_area(self):
        box = self.box
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area
    
    #@autocast(enabled=False)
    def get_category(self):
        return self.category

    def get_feature(self):
        return self.part_feats

    def get_token(self):
        return self.token
    
    def get_ratio(self):
        box = self.box
        return (box[:, 2] - box[:, 0]) / (box[:, 3] - box[:, 1]).clamp(1e-5)

    def get_img(self):
        return self.img

    def __len__(self):
        return len(self.part_feats)

    def __getitem__(self, idx):
        if isinstance(idx, slice) or torch.is_tensor(idx) or isinstance(idx, list):
            if torch.is_tensor(idx):
                idx = idx.to(self.device).long()  # self.mask might be in cpu
            token = self.token[idx]
            feature = [self.part_feats[ii] for ii in idx.tolist()]
            box = self.box[idx]
            category = self.category
        elif isinstance(idx, int):
            token = self.token[idx:idx + 1]
            feature = self.part_feats[idx]
            box = self.box[idx:idx + 1]
            category = self.category
        else:
            raise NotImplementedError("type: {}".format(type(idx)))
        return dict(token=token, part_feats=feature, box=box, category=category)


class ObjectQueues:
    #@autocast(enabled=False)
    def __init__(self, num_class, len_queue, ratio_range, appear_thresh,
                 max_retrieval_objs):
        self.num_class = num_class
        self.queues = [None for i in range(self.num_class)]
        self.len_queue = len_queue
        self.appear_thresh = appear_thresh
        self.ratio_range = ratio_range
        self.max_retrieval_objs = max_retrieval_objs

    #@autocast(enabled=False)
    def append(self, class_idx, idx, token, part_feats, box, device='cpu'):
        with torch.no_grad():
            if self.queues[class_idx] is None:
                self.queues[class_idx] = \
                    ObjectFactory.create_queue_by_one(
                        len_queue=self.len_queue,
                        category=class_idx,
                        idx=idx,
                        tokens=token, 
                        part_feats=part_feats,
                        box=box,
                        device=device, 
                    )
                create_new_gpu_bank = True
                self.queues[class_idx].ptr += 1
                self.queues[class_idx].ptr = self.queues[class_idx].ptr % self.len_queue
            else:
                ptr = self.queues[class_idx].ptr
                if len(self.queues[class_idx]) > ptr:
                    self.queues[class_idx].part_feats[ptr] = part_feats[idx]
                else:
                    self.queues[class_idx].part_feats.append(part_feats[idx])
                self.queues[class_idx].num_parts[ptr:ptr + 1] = part_feats[idx].shape[0]
                self.queues[class_idx].box[ptr:ptr + 1] = box[idx:idx + 1]
                self.queues[class_idx].token[ptr:ptr + 1] = token[idx]
                self.queues[class_idx].ptr = (ptr + 1) % self.len_queue
                create_new_gpu_bank = False
            return create_new_gpu_bank

    #@autocast(enabled=False)
    def cal_appear_identity_sim(self, qobjs, kobjs):
        f0 = qobjs.get_feature()[0]
        f1 = torch.cat(kobjs.get_feature(), dim=0)
        f1 = f1.to(f0)  # might be in cpu
        # TODO: debug !
        sim =  f0 @ f1.T / (f0.norm(p=2, dim=1, keepdim=True) * f1.norm(p=2, dim=1, keepdim=True).T).clamp(1e-5)
        sim = sim.split(kobjs.num_parts.tolist(), dim=1)
        sim = torch.stack([s.mean() for s in sim])
        # sim = (f0 * f1).sum([1, 2, 3])
        return sim

    def cal_token_sim(self, qobjs, kobjs):
        t0 = qobjs.get_token()
        t1 = kobjs.get_token()
        sim = t0 @ t1.T / (t0.norm(p=2, dim=1, keepdim=True) * t1.norm(p=2, dim=1, keepdim=True).T).clamp(1e-5)
        return sim.clamp(0)
    
    #@autocast(enabled=False)
    def cal_shape_ratio(self, qobj, kobjs):
        ratio0 = qobj.get_ratio().unsqueeze(1)
        ratio1 = kobjs.get_ratio().unsqueeze(0)
        ratio1 = ratio1.to(ratio0)  # might be in cpu
        return ratio0 / ratio1

    #@autocast(enabled=False)
    def get_similar_obj(self, qobj: ObjectElements, return_mask=False):
        with torch.no_grad():
            category = qobj.get_category()
            if self.queues[category] is not None:
                kobjs = self.queues[qobj.category]
                appear_sim = self.cal_appear_identity_sim(qobj, kobjs)
                token_sim = self.cal_token_sim(qobj, kobjs)
                ratio = self.cal_shape_ratio(qobj, kobjs).squeeze(0)
                sim_masking = (token_sim > self.appear_thresh).float()
                ratio_masking = ((ratio >= self.ratio_range[0]).float() * (ratio <= self.ratio_range[1]).float())
                masking0 = torch.where((sim_masking * ratio_masking).bool()[0])[0]
                masking = masking0[:self.max_retrieval_objs].long()
                ret_objs = kobjs[masking]
                if return_mask:
                    return ret_objs, masking
                return ret_objs
            else:
                if return_mask:
                    return None, None
                return None
            # ObjectElements(torch.zeros([0, qmask.shape[1], qmask.shape[2]]).to(device), torch.zeros([0, qfeature.shape[1], qfeature.shape[2], qfeature.shape[3]]))

    def get_all_tokens(self):
        tokens = []
        cls_num_objs = []
        for q in self.queues:
            if q is not None:
                tokens.append(q.token[:q.ptr])
                cls_num_objs.append(q.ptr)
            else:
                cls_num_objs.append(0)
        if len(tokens) > 0:
            tokens_tensor = torch.cat(tokens)
        else:
            tokens_tensor = None
        return tokens_tensor, cls_num_objs
    

class SemanticCorrSolver:

    #@autocast(enabled=False)
    def __init__(self, exp, eps, gaussian_filter_size, low_score, num_iter, num_smooth_iter, dist_kernel):
        self.exp = exp
        self.eps = eps
        self.gaussian_filter_size = gaussian_filter_size
        self.low_score = low_score
        self.hsfilter = self.generate_gaussian_filter(gaussian_filter_size)
        self.num_iter = num_iter
        self.num_smooth_iter = num_smooth_iter
        self.count = None
        self.pairwise = None
        self.dist_kernel = dist_kernel
        self.ncells = 8192

    #@autocast(enabled=False)
    def generate_gaussian_filter(self, size=3):
        r"""Returns 2-dimensional gaussian filter"""
        dim = [size, size]

        siz = torch.LongTensor(dim)
        sig_sq = (siz.float() / 2 / 2.354).pow(2)
        siz2 = (siz - 1) / 2

        x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
        y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

        gaussian = torch.exp(-(x_axis.pow(2) / 2 / sig_sq[0] + y_axis.pow(2) / 2 / sig_sq[1]))
        gaussian = gaussian / gaussian.sum()

        return gaussian

    #@autocast(enabled=False)
    def perform_sinkhorn(self, a, b, M, reg, stopThr=1e-3, numItermax=100):
        # init data
        dim_a = a.shape[1]
        dim_b = b.shape[1]

        batch_size = b.shape[0]

        u = torch.ones((batch_size, dim_a), requires_grad=False).cuda() / dim_a
        v = torch.ones((batch_size, dim_b), requires_grad=False).cuda() / dim_b
        K = torch.exp(-M / reg)

        Kp = (1 / a).unsqueeze(2) * K
        cpt = 0
        err = 1
        KtransposeU = (K * u.unsqueeze(2)).sum(dim=1)  # has shape K.shape[1]

        while err > stopThr and cpt < numItermax:
            KtransposeU[...] = (K * u.unsqueeze(2)).sum(dim=1)  # has shape K.shape[1]
            v[...] = b / KtransposeU
            u[...] = 1. / (Kp * v.unsqueeze(1)).sum(dim=2)
            cpt = cpt + 1

        T = u.unsqueeze(2) * K * v.unsqueeze(1)
        # del u, K, v
        return T

    #@autocast(enabled=False)
    def appearance_similarityOT(self, m0, m1, sim):
        r"""Semantic Appearance Similarity"""

        pow_sim = torch.pow(torch.clamp(sim, min=0.3, max=0.7), 1.0)
        cost = 1 - pow_sim

        b, n1, n2 = sim.shape[0], sim.shape[1], sim.shape[2]
        m0, m1 = torch.clamp(m0, min=self.low_score, max=1 - self.low_score), torch.clamp(m1, min=self.low_score,
                                                                                          max=1 - self.low_score)
        mu = m0 / m0.sum(1, keepdim=True)
        nu = m1 / m1.sum(1, keepdim=True)
        with torch.no_grad():
            epsilon = self.eps
            cnt = 0
            while epsilon < 5:
                PI = self.perform_sinkhorn(mu, nu, cost, epsilon)
                if not torch.isnan(PI).any():
                    if cnt > 0:
                        print(cnt)
                    break
                else:
                    epsilon *= 2.0
                    cnt += 1
                    print(cnt, epsilon)

        if torch.isnan(PI).any():
            from IPython import embed
            embed()

        PI = n1 * PI  # re-scale PI
        exp = self.exp
        PI = torch.pow(torch.clamp(PI, min=0), exp)

        return PI

    #@autocast(enabled=False)
    def build_hspace(self, src_imsize, trg_imsize, ncells):
        r"""Build Hough space where voting is done"""
        hs_width = src_imsize[0] + trg_imsize[0]
        hs_height = src_imsize[1] + trg_imsize[1]
        hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
        nbins_x = int(hs_width / hs_cellsize) + 1
        nbins_y = int(hs_height / hs_cellsize) + 1

        return nbins_x, nbins_y, hs_cellsize

    #@autocast(enabled=False)
    def receptive_fields(self, rfsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size[3]
        height = feat_size[2]

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2).to(rfsz.device)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1).to(rfsz.device)

        box = torch.zeros(feat_ids.size()[0], 4).to(rfsz.device)
        box[:, 0] = feat_ids[:, 1] - rfsz // 2
        box[:, 1] = feat_ids[:, 0] - rfsz // 2
        box[:, 2] = feat_ids[:, 1] + rfsz // 2
        box[:, 3] = feat_ids[:, 0] + rfsz // 2
        box = box.unsqueeze(0)

        return box

    #@autocast(enabled=False)
    def pass_message(self, T, shape):
        T = T.view(T.shape[0], shape[0], shape[1], shape[0], shape[1])
        pairwise = torch.zeros_like(T).to(T)
        count = torch.zeros_like(T).to(T)
        dxs, dys = [-1, 0, 1], [-1, 0, 1]
        for dx in dxs:
            for dy in dys:
                count[:, max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1]),
                max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1])] += 1
                pairwise[:, max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1]),
                max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1])] += \
                    T[:, max(0, -dy): min(shape[0] - dy, shape[0]), max(0, -dx): min(shape[1] - dx, shape[1]),
                    max(0, -dy): min(shape[0] - dy, shape[0]), max(0, -dx): min(shape[1] - dx, shape[1])]

        T[...] = pairwise / count
        T = T.view(T.shape[0], shape[0] * shape[1], shape[0] * shape[1])
        # del pairwise, count

        return T

    #@autocast(enabled=False)
    def solve(self, qobjs, kobjs, f0):
        r"""Regularized Hough matching"""
        # Unpack hyperpixels
        m0 = qobjs.mask.float()
        f0 = f0.float()
        f1 = kobjs['feature'].to(m0).float()
        m1 = kobjs['mask'].to(m0).float()
       
        fg_mask = m0.reshape(m0.shape[0], -1, 1) * m1.reshape(m1.shape[0], 1, -1)
        bg_mask = (1 - m0).reshape(m0.shape[0], -1, 1) * (1 - m1).reshape(m1.shape[0], 1, -1)
        
        m0 = F.interpolate(m0.unsqueeze(1), (f0.shape[2], f0.shape[3]), mode='bilinear', align_corners=False).squeeze(1)
        m1 = F.interpolate(m1.unsqueeze(1), (f1.shape[2], f1.shape[3]), mode='bilinear', align_corners=False).squeeze(1)
        shape = f0.shape[2], f0.shape[3]

        m0 = m0.reshape(m0.shape[0], -1)
        m1 = m1.reshape(m1.shape[0], -1)
        f0 = f0.reshape(f0.shape[0], f0.shape[1], -1).transpose(2, 1)
        f1 = f1.reshape(f1.shape[0], f1.shape[1], -1)

        f0_norm = torch.norm(f0, p=2, dim=2, keepdim=True) + 1e-4
        f1_norm = torch.norm(f1, p=2, dim=1, keepdim=True) + 1e-4
        with autocast(enabled=False):
            Cu = torch.matmul((f0 / f0_norm), (f1 / f1_norm))

        eye = torch.eye(shape[0] * shape[1]).to(f0).reshape(1, -1, shape[0], shape[1])
        dist_mask = F.max_pool2d(eye, kernel_size=self.dist_kernel, stride=1, padding=self.dist_kernel//2).reshape(1, shape[0] * shape[1],
                                                                                    shape[0] * shape[1]).transpose(2, 1)
        with torch.no_grad():
            C = Cu.clone() * dist_mask

        for i in range(self.num_iter):
            pairwise_votes = C.clone()
            for _ in range(self.num_smooth_iter):
                pairwise_votes = self.pass_message(pairwise_votes, (shape[0], shape[1]))
                pairwise_votes = pairwise_votes / (pairwise_votes.sum(2, keepdim=True) + 1e-4)

            max_val, _ = pairwise_votes.max(2, keepdim=True)

            C = Cu + pairwise_votes
            C = C / (C.sum(2, keepdim=True) + 1e-4)

        return Cu, C, fg_mask, bg_mask


