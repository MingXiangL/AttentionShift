import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module
class PtsBorderLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PtsBorderLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pts, gt_bboxes, y_first=False, *args, **kwargs):
        loss = self.loss_weight * weighted_pts_border_loss(
            pts, gt_bboxes, y_first=y_first, *args, **kwargs)
        return loss


def pts_border_loss(pts, gt_bboxes, reduction='mean', y_first=False):
    pts_reshape = pts.reshape(pts.shape[0], -1, 2)
    pts_y = pts_reshape[:, :, 0] if y_first else pts_reshape[:, :, 1]
    pts_x = pts_reshape[:, :, 1] if y_first else pts_reshape[:, :, 0]
    loss_left = (gt_bboxes[:, 0].unsqueeze(1) - pts_x).clamp_(min=0).unsqueeze(1)
    loss_right = (pts_x - gt_bboxes[:, 2].unsqueeze(1)).clamp_(min=0).unsqueeze(1)
    loss_up = (gt_bboxes[:, 1].unsqueeze(1) - pts_y).clamp_(min=0).unsqueeze(1)
    loss_bottom = (pts_y - gt_bboxes[:, 3].unsqueeze(1)).clamp_(min=0).unsqueeze(1)
    loss = torch.cat([loss_left, loss_right, loss_up, loss_bottom], dim=1)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / gt_bboxes.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_pts_border_loss(pts, gt_bboxes, avg_factor=None, y_first=False):
    # assert weight.dim() == 2
    # if avg_factor is None:
    #     avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = pts_border_loss(pts, gt_bboxes, y_first=y_first, reduction='none')  # (n, 4, num_points)
    num_points = loss.shape[2]
    loss = loss.sum(dim=2) / num_points  # (n, 4)
    return torch.mean(loss)
