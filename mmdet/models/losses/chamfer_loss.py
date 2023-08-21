import torch
import torch.nn as nn

# from mmdet.ops.chamfer_2d import Chamfer2D
from ..builder import LOSSES
import pdb
from mmcv.ops import point_sample
import torch.nn.functional as F


@LOSSES.register_module
class ChamferLoss2D(nn.Module):
    def __init__(self, use_cuda=True, loss_weight=1.0, eps=1e-12):
        super(ChamferLoss2D, self).__init__()
        self.use_cuda = use_cuda
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, point_set_1, point_set_2, **kwargs):
        """
        Computation of optimal transport distance via sinkhorn algorithm.
        - Input:
            - point_set_1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
            - point_set_2:	torch.Tensor	[..., num_points_2, point_dim]
                    (the dimensions of point_set_2 except the last two should be the same as point_set_1)
        - Output:
            - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
        """
        # chamfer = Chamfer2D() if self.use_cuda else ChamferDistancePytorch()
        chamfer = ChamferDistancePytorch()

        dist = 0
        n_obj = len(point_set_1)
        for set_1, set_2 in zip(point_set_1, point_set_2):
            m = set_2.sum() >= 0
            d =  chamfer(set_1.reshape(1, -1, 2), set_2) * m
            dist += d
        dist /= n_obj
        return dist * self.loss_weight


@LOSSES.register_module
class ChamferGlobalEdgeLoss2D(ChamferLoss2D):

    def forward(self, point_set_1, point_set_2, **kwargs):
        """
        Computation of optimal transport distance via sinkhorn algorithm.
        - Input:
            - point_set_1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
            - point_set_2:	torch.Tensor	[..., num_points_2, point_dim]
                    (the dimensions of point_set_2 except the last two should be the same as point_set_1)
        - Output:
            - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
        """
        # chamfer = Chamfer2D() if self.use_cuda else ChamferDistancePytorch()
        chamfer = ChamferDistancePytorch()

        dist = 0
        n_obj = len(point_set_1)
        for set_1, set_2 in zip(point_set_1, point_set_2):
            m = set_2.sum() >= 0
            d =  chamfer(set_1, set_2) * m # each point will be predicting the contour of the whole object
            dist += d
        dist /= n_obj
        return dist * self.loss_weight


@LOSSES.register_module
class SimFocusChamferLoss2D(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-12, sim_thr=0.85):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.sim_thr = sim_thr

    def forward(self, point_set_1, point_set_2, feats, img_metas, key_points, **kwargs):
        """
        Computation of optimal transport distance via sinkhorn algorithm.
        - Input:
            - point_set_1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
            - point_set_2:	torch.Tensor	[..., num_points_2, point_dim]
                    (the dimensions of point_set_2 except the last two should be the same as point_set_1)
        - Output:
            - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
        """
        # chamfer = Chamfer2D() if self.use_cuda else ChamferDistancePytorch()
        chamfer = ChamferDistancePytorch()

        # assert point_set_1.dim() == point_set_2.dim()
        # assert point_set_1.shape[-1] == point_set_2.shape[-1]
            # if self.use_cuda:
            #     dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
            #     dist1 = torch.sqrt(torch.clamp(dist1, self.eps))
            #     dist2 = torch.sqrt(torch.clamp(dist2, self.eps))
            #     dist = (dist1.mean(-1) + dist2.mean(-1)) / 2.0
            # else:
        dist = 0
        n_obj = len(point_set_1)
        n_p   = point_set_1[0].shape[0]
        img_size = feats.new_tensor(img_metas['img_shape'][:2][::-1])
        feats_set_1 = point_sample(feats, key_points[None] / img_size[None, None])[0].permute(1, 0)
        feats_set_2 = point_sample(feats.expand(len(point_set_2), -1, -1, -1), torch.cat(point_set_2, dim=0) / img_size[None, None])
        num_parts = [p.shape[0] for p in point_set_1]
        for set_1, set_2, feats_1, feats_2 in zip(point_set_1, point_set_2, feats_set_1.split(num_parts, dim=0), feats_set_2.split(1, dim=0)):
            m = set_2.sum() >= 0
            d_p = 0
            sims = F.cosine_similarity(feats_1[...,None], feats_2, dim=1)
            for s1, sim in zip(set_1, sims):
                mask = sim >= self.sim_thr
                if mask.sum() == 0:
                    d_p += 0 * chamfer(s1[None], set_2)
                else:
                    d_p += chamfer(s1[None], set_2[:, mask, :]) * m
            dist += d_p / sims.shape[0]
        dist /= n_obj
        return dist * self.loss_weight


@LOSSES.register_module
class SimWeightedChamferLoss2D(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-12, sim_thr=0.85):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.sim_thr = sim_thr

    def forward(self, point_set_1, point_set_2, feats, img_metas, key_points, **kwargs):
        """
        Computation of optimal transport distance via sinkhorn algorithm.
        - Input:
            - point_set_1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
            - point_set_2:	torch.Tensor	[..., num_points_2, point_dim]
                    (the dimensions of point_set_2 except the last two should be the same as point_set_1)
        - Output:
            - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
        """
        # chamfer = Chamfer2D() if self.use_cuda else ChamferDistancePytorch()
        chamfer = ChamferDistancePytorch()

        # assert point_set_1.dim() == point_set_2.dim()
        # assert point_set_1.shape[-1] == point_set_2.shape[-1]
            # if self.use_cuda:
            #     dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
            #     dist1 = torch.sqrt(torch.clamp(dist1, self.eps))
            #     dist2 = torch.sqrt(torch.clamp(dist2, self.eps))
            #     dist = (dist1.mean(-1) + dist2.mean(-1)) / 2.0
            # else:
        dist = 0
        n_obj = len(point_set_1)
        n_p   = point_set_1[0].shape[0]
        img_size = feats.new_tensor(img_metas['img_shape'][:2][::-1])
        feats_set_1 = point_sample(feats, key_points[None] / img_size[None, None])[0].permute(1, 0)
        feats_set_2 = point_sample(feats.expand(len(point_set_2), -1, -1, -1), torch.cat(point_set_2, dim=0) / img_size[None, None])
        num_parts = [p.shape[0] for p in point_set_1]
        for set_1, set_2, feats_1, feats_2 in zip(point_set_1, point_set_2, feats_set_1.split(num_parts, dim=0), feats_set_2.split(1, dim=0)):
            m = set_2.sum() >= 0
            d_p = 0
            sims = F.cosine_similarity(feats_1[...,None], feats_2, dim=1)
            for s1, sim in zip(set_1, sims):
                mask = sim >= self.sim_thr
                if mask.sum() == 0:
                    d_p += 0 * chamfer(s1[None], set_2)
                else:
                    d_p += chamfer(s1[None], set_2[:, mask, :]) * m
            dist += d_p / sims.shape[0]
        dist /= n_obj
        return dist * self.loss_weight


# Adapted from https://github.com/dfdazac/wassdistance
class ChamferDistancePytorch(nn.Module):
    r"""
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, reduction='mean'):
        super(ChamferDistancePytorch, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        if x.shape[0] == 0:
            return x.sum()
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function

        # compute chamfer loss
        min_x2y, _ = C.min(-1)
        d1 = min_x2y.mean(-1)
        min_y2x, _ = C.min(-2)
        d2 = min_y2x.mean(-1)
        cost = (d1 + d2) / 2.0
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.norm(x_col - y_lin, 2, -1)
        return C
