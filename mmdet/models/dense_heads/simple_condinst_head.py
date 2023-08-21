import copy
from typing import Dict, List, Optional, Tuple

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS, build_loss
from mmcv.cnn import ConvModule, Scale
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, kaiming_init
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)


def aligned_bilinear(tensor: Tensor, factor: int) -> Tensor:
    """aligned bilinear, used in original implement in CondInst:

    https://github.com/aim-uofa/AdelaiDet/blob/\
    c0b2092ce72442b0f40972f7c6dda8bb52c46d16/adet/utils/comm.py#L23
    """

    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode='replicate')
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow), mode='bilinear', align_corners=True)
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0), mode='replicate')

    return tensor[:, :, :oh - 1, :ow - 1]


class MaskFeatModule(BaseModule):
    """CondInst mask feature map branch used in \
    https://arxiv.org/abs/1904.02689.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        start_level (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        end_level (int): The ending feature map level from rpn that
             will be used to predict the mask feature map.
        out_channels (int): Number of output channels of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        mask_stride (int): Downsample factor of the mask feature map output.
            Defaults to 4.
        num_stacked_convs (int): Number of convs in mask feature branch.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 start_level: int,
                 end_level: int,
                 out_channels: int,
                 mask_stride: int = 4,
                 num_stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = [
                     dict(type='Normal', layer='Conv2d', std=0.01)
                 ],
                 num_params_dynamic: int =169,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        # self.start_level = start_level
        # self.end_level = end_level
        self.mask_stride = mask_stride
        self.num_stacked_convs = num_stacked_convs
        # assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_params_dynamic = num_params_dynamic
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.convs_all_levels = nn.ModuleList()
        self.conv_in = nn.Linear(
                    self.in_channels,
                    self.feat_channels,
                    bias=False)

        conv_branch = []
        for _ in range(self.num_stacked_convs):
            conv_branch.append(
                nn.Linear(
                    self.feat_channels,
                    self.feat_channels,
                    1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=False))
        self.conv_branch = nn.Sequential(*conv_branch)

        self.conv_pred = nn.Conv2d(
            self.feat_channels, self.out_channels, 1, stride=1)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        kaiming_init(self.conv_in, a=1, distribution='uniform')
        kaiming_init(self.conv_branch, a=1, distribution='uniform')
        kaiming_init(self.conv_pred, a=1, distribution='uniform')

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            Tensor: The predicted mask feature map.
        """
        x = self.conv_in(x)
        x = self.conv_branch(x)
        x = self.conv_pred(x)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class SimpleCondInstHead(nn.Module):
    """CondInst mask head used in https://arxiv.org/abs/1904.02689.

    This head outputs the mask for CondInst.

    Args:
        part_feature_head (dict): Config of CondInstMaskFeatHead.
        num_layers (int): Number of dynamic conv layers.
        feat_channels (int): Number of channels in the dynamic conv.
        mask_out_stride (int): The stride of the mask feat.
        size_of_interest (int): The size of the region used in rel coord.
        max_masks_to_train (int): Maximum number of masks to train for
            each image.
        loss_segm (:obj:`ConfigDict` or dict, optional): Config of
            segmentation loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config
            of head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            head.
    """

    def __init__(self,
                 part_feature_head: ConfigType,
                 loss : ConfigType,
                 in_feat_channels: int = 384, 
                 num_layers: int = 3,
                 feat_channels: int = 8,
                 size_of_interest: int = 8,
                 max_masks_to_train: int = -1,
                 topk_masks_per_img: int = -1,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,) -> None:
        super().__init__()
        self.part_feature_head = MLP(**part_feature_head)
        self.in_channels = self.part_feature_head.output_dim
        self.in_feat_channels = in_feat_channels
        self.num_layers = num_layers
        self.feat_channels = feat_channels
        self.size_of_interest = size_of_interest
        self.max_masks_to_train = max_masks_to_train
        self.topk_masks_per_img = topk_masks_per_img
        # self.prior_generator = MlvlPointGenerator([self.mask_feat_stride])
        self.loss = build_loss(loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        weight_nums, bias_nums = [], []
        for i in range(self.num_layers):
            if i == 0:
                # weight_nums.append((self.in_channels + 2) * self.feat_channels)
                weight_nums.append((self.in_channels) * self.feat_channels) # 没有加rel_coord，所以不用"+2"
                bias_nums.append(self.feat_channels)
            elif i == self.num_layers - 1:
                weight_nums.append(self.feat_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.feat_channels * self.feat_channels)
                bias_nums.append(self.feat_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_params = sum(weight_nums) + sum(bias_nums)
        self.controller = nn.Linear(self.in_feat_channels, self.num_params)
        kaiming_init(self.controller, a=1, distribution='uniform')

    def parse_dynamic_params(
            self, params: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """parse the dynamic params for dynamic conv."""
        num_insts = params.size(0)
        params_splits = list(
            torch.split_with_sizes(
                params, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:self.num_layers]
        bias_splits = params_splits[self.num_layers:]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    num_insts, 1, self.in_channels, -1)
                bias_splits[i] = bias_splits[i].reshape(num_insts, 1,
                                                        self.in_channels, 1)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[i] = weight_splits[i].reshape(
                    num_insts, 1, 1, -1)
                bias_splits[i] = bias_splits[i].reshape(num_insts, 1, 1, 1)

        return weight_splits, bias_splits

    def dynamic_conv_forward(self, features: Tensor, weights: List[Tensor],
                             biases: List[Tensor], num_insts: int) -> Tensor:
        """dynamic forward, each layer follow a relu."""
        n_layers = len(weights)
        x = features.unsqueeze(-1).unsqueeze(0)
        for i, (w, b) in enumerate(zip(weights, biases)):
            # try:
            #     print(f'w.shape: {w.shape}')
            #     print(f'x.shape: {x.shape}')
            #     print(f'b.shape: {b.shape}')
            #     print(f'************************')
            x = torch.matmul(w, x) + b
            if i < n_layers - 1:
                x = F.relu(x)
            # except BaseException:
            #     print(f'-------------------------')
            #     print(f'w.shape: {w.shape}')
            #     print(f'x.shape: {x.shape}')
            #     print(f'b.shape: {b.shape}')
            #     pdb.set_trace()
        return x.reshape(num_insts, -1).transpose(0, 1)

    def forward(self, token_feats, part_feats, part_labels) -> tuple:
        """Forward feature from the upstream network to get prototypes and
        linearly combine the prototypes, using masks coefficients, into
        instance masks. Finally, crop the instance masks with given bboxes.

        Args:
            x (Tuple[Tensor]): Feature from the upstream network, which is
                a 4D-tensor.
            positive_infos (List[:obj:``InstanceData``]): Positive information
            that calculate from detect head.

        Returns:
            tuple: Predicted instance segmentation masks
        """
        dynamic_params = self.controller(token_feats)
        if isinstance(part_feats, list):
            part_feats = self.part_feature_head(torch.zeros_like(token_feats))
            return dict(loss_keypoint_align=dynamic_params.sum()*0 + part_feats.sum()*0)
        else:
            part_feats = self.part_feature_head(part_feats)
            return self.forward_single(dynamic_params, part_feats, part_labels, token_feats)

    def forward_single(self, 
                       dynamic_params: Tensor,
                       input_feats: Tensor,
                       feat_obj_labels: Tensor,
                       token_feats: Tensor,
                       ) -> Tensor:
        """Forward features of a each image."""
        num_inst = dynamic_params.shape[0]

        # ------------ original version, no rel coord is added ------------ #
        # locations = self.prior_generator.single_level_grid_priors(
        #     mask_feat.size()[2:], 0, device=mask_feat.device)
                    
        # rel_coords = relative_coordinate_maps(locations, pos_points,
        #                                       pos_strides,
        #                                       self.size_of_interest,
        #                                       mask_feat.size()[2:])
        # mask_head_inputs = torch.cat([rel_coords, mask_feat], dim=1)
        # mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        # ------------ original version, no rel coord is added ------------ #
        weights, biases = self.parse_dynamic_params(dynamic_params)
        inst_logits = self.dynamic_conv_forward(input_feats, weights, biases, num_inst)
        return dict(loss_keypoint_align=self.loss(inst_logits, feat_obj_labels) * 0.1)

    # def all_zero_loss(self):
    #     loss_all_zero = 