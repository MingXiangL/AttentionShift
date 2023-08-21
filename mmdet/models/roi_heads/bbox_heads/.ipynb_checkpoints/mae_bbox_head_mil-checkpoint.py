import os
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from collections import OrderedDict

from mmcv.runner import _load_checkpoint, load_state_dict
from mmdet.utils import get_root_logger
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from models.vision_transformer import Block, trunc_normal_
import torch.nn.functional as F


@HEADS.register_module()
class MAEBoxHeadMIL(BBoxHead):
    def __init__(self,
                 in_channels,
                 img_size=224,
                 patch_size=16, 
                 embed_dim=256, 
                 depth=4,
                 num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pretrained=False,
                 use_checkpoint=False,
                 num_layers_query=12,
                 loss_mil_factor=1.0,
                 hidden_dim=1024,
                 roi_size=7,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2
#         self.det_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.with_decoder_embed = False
        # MAE decoder specifics
        if in_channels != embed_dim:
            self.with_decoder_embed = True
            self.norm = norm_layer(in_channels)
            self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.decoder_blocks = nn.ModuleList([
#             Block(
#                 embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)
#         ])
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
#         self.decoder_box_norm = norm_layer(embed_dim)
        
#         # reconstruct fc_cls and fc_reg since input channels are changed
#         if self.with_cls:
#             self.fc_cls = nn.Linear(embed_dim, self.num_classes + 1)
#         if self.with_reg:
#             out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
#             self.fc_reg = nn.Linear(embed_dim, out_dim_reg)
        
        self.hidden_dim = hidden_dim
        self.loss_mil_factor = loss_mil_factor
        self.num_layers_query = num_layers_query
        
        self.fc1 = nn.Linear(embed_dim * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.proposal_branch = nn.Linear(hidden_dim, self.num_classes)
        self.classification_branch = nn.Linear(hidden_dim, self.num_classes)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.pretrained and isinstance(pretrained, str):
            if os.path.isfile(pretrained):
                logger.info('loading checkpoint for {}'.format(self.__class__))
                checkpoint = _load_checkpoint(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                # TODO: match the decoder blocks, norm and head in the state_dict due to the different prefix
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('patch_embed') or k.startswith('blocks'):
                        continue
                    elif k in ['pos_embed']:
                        continue
                    else:
                        new_state_dict[k] = v
                load_state_dict(self, new_state_dict, strict=False, logger=logger)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")
        elif pretrained is None:
            trunc_normal_(self.det_token, std=.02)
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        super().init_weights()

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.decoder_pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.decoder_pos_embed
        class_pos_embed = self.decoder_pos_embed[:, 0]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#     def multi_class_cross_entropy_loss(self, preds, labels, eps=1e-6):
        
#         loss = F.cross_entropy(preds, labels, reduction='none')
#         loss = loss.mean()
#         return loss

    def mil_losses(self, cls_score, labels):
        cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
        labels = labels.clamp(0, 1)
        loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
        return loss.mean()

    def forward(self, x, gt_labels=None):
        if isinstance(gt_labels, list):
            gt_labels = torch.cat(gt_labels)
            
        B, C, W, H = x.shape # list [n_gt1, 12, C, W, H], [n_gt2, 12, C, W, H] 
                             # [n_gt1 + n_gt2, 12, C, W, H] -> [(n_gt1 + n_gt2) * 12, C, W, H]
        x = x.flatten(2).transpose(1, 2) # [(n_gt1 + n_gt2) * 12, H * W, C]
        if self.with_decoder_embed:
            x = self.norm(x)
            x = self.decoder_embed(x)
#         x = torch.cat([self.det_token.expand(B, -1, -1), x], dim=1) # [(n_gt1 + n_gt2) * 12, H * W + 1, C]
#         x = x + self.interpolate_pos_encoding(x, W * self.patch_size, H * self.patch_size)
#         for blk in self.decoder_blocks:
#             if self.use_checkpoint:
#                 x = checkpoint(blk, x)
#             else:
#                 x = blk(x) # [(n_gt1 + n_gt2) * 12, H * W + 1, C]
#         x = self.decoder_box_norm(x[:, 0, :]) # [(n_gt1 + n_gt2) * 12, C]
        x = F.relu(self.fc1(x.view(B, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        
#         x = self.decoder_box_norm(x.mean(1)) # [(n_gt1 + n_gt2) * 12, C]
        classification_pred = self.classification_branch(x).reshape(-1, self.num_layers_query, self.num_classes).softmax(-1) # [(n_gt1 + n_gt2), 12, num_class]
        proposal_pred = self.proposal_branch(x).reshape(-1, self.num_layers_query, self.num_classes).softmax(-2) # [(n_gt1 + n_gt2), 12, num_class]
        bag_pred = (classification_pred * proposal_pred) # [(n_gt1 + n_gt2), 12, num_class]
        
        #  找符合gt label分类标签，并且分数最高的框作为gt
        
        bag_classification = torch.gather(bag_pred, dim=-1, 
                                          index=gt_labels.reshape(-1, 1, 1).repeat(1, self.num_layers_query, 1))[..., 0] # [(n_gt1 + n_gt2), 12]
#         bag_classification, _ = bag_pred.max(-1)
        _, gt_index = bag_classification.max(-1)  
        
        if gt_labels is not None:
            bag_pred_sum = bag_pred.sum(1)
            gt_labels_binary = torch.zeros((len(gt_labels), self.num_classes)).type_as(gt_labels)
            gt_ind = torch.arange(len(gt_labels)).type_as(gt_labels)
            gt_labels_binary[gt_ind, gt_labels] = 1
            mil_loss = self.loss_mil_factor * self.mil_losses(bag_pred_sum, gt_labels_binary)
            return gt_index, mil_loss 
#         cls_score = self.fc_cls(x) if self.with_cls else None
#         bbox_pred = self.fc_reg(x) if self.with_reg else None
#         return cls_score, bbox_pred
