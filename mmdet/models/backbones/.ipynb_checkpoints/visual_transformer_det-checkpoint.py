# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from models import VisionTransformer
from utils import trunc_normal_
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches_h = img_size[1] // patch_size

        num_patches = self.num_patches_w * self.num_patches_h
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        return self.proj(x)


@BACKBONES.register_module()
class VisionTransformerDet(VisionTransformer):
    def __init__(self,
                 img_size,
                 patch_size,
                 embed_dim,
                 in_chans=3,
                 with_fpn=True,
                 frozen_stages=-1,
                 out_indices=[3, 5, 7, 11],
                 use_checkpoint=False,
                 learnable_pos_embed=True,
                 last_feat=False,
                 recompute_last_feat=False,
                 point_tokens_num=100,
                 num_classes=20,
                 return_attention=False,
                 **kwargs):
        super(VisionTransformerDet, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim, 
            return_attention=return_attention,
            **kwargs)
        
        assert not with_fpn or (patch_size in (8, 16))
        assert not recompute_last_feat or (last_feat and recompute_last_feat)

        self.patch_size = patch_size
        self.last_feat = last_feat
        self.recompute_last_feat = recompute_last_feat

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=learnable_pos_embed,
        )
        
        self.with_fpn = with_fpn
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        
        del self.norm, self.fc_norm, self.head
        if with_fpn and patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif with_fpn and patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')

        # 点监督token
        self.point_tokens_num = point_tokens_num
        self.point_token = nn.Parameter(torch.zeros(1, point_tokens_num, embed_dim))
        self.point_pos_embed = nn.Parameter(torch.zeros(1, point_tokens_num, embed_dim))
        self.class_embed = MLP(embed_dim, embed_dim, num_classes, 3)
        self.bbox_embed = MLP(embed_dim, embed_dim, 2, 3)
        self.return_attention = return_attention
        
        trunc_normal_(self.point_token, std=.02)
        trunc_normal_(self.point_pos_embed, std=.02)
        
        
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VisionTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.pos_drop.eval()

        for i in range(1, self.frozen_stages + 1):
            if i  == len(self.blocks):
                norm_layer = getattr(self, 'norm') #f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            logger = get_root_logger()
            if  os.path.isfile(pretrained):
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                logger.info(f"checkpoint path {pretrained} is invalid, we skip it and initialize net randomly")
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 增加point token
        point_tokens = self.point_token.expand(B, -1, -1)
        point_pos_embed = self.point_pos_embed.expand(B, -1, -1)
        point_tokens = point_tokens + point_pos_embed
        x = torch.cat((x, point_tokens), dim=1)
        return self.pos_drop(x)
    
    def forward_encoder(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.prepare_tokens(x)
        if self.recompute_last_feat:
            last_feat = x
        features = []
        
        attns = []
        
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                if self.return_attention:
                    x, attn = checkpoint.checkpoint(blk, x)
                    attns.append(attn.mean(1))
                else:
                    x = checkpoint.checkpoint(blk, x)
            else:
                if self.return_attention:
                    x, attn = blk(x)
                    attns.append(attn)
                else:
                    x = blk(x)
                    
            if i in self.out_indices:
                xp = x[:, 1:, :][:, :-self.point_tokens_num].permute(0, 2, 1).reshape(B, -1, Hp, Wp)       
                features.append(xp.contiguous())
            if self.last_feat and (not self.recompute_last_feat) and i == len(self.blocks) - 1:
                last_feat = x[:, :-self.point_tokens_num]

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
        
        point_tokens = x[:, -self.point_tokens_num:]
        outputs_class = self.class_embed(point_tokens)
        outputs_coord = self.bbox_embed(point_tokens).sigmoid()
        
        if self.return_attention and self.last_feat:
            return tuple(features), last_feat, outputs_class, outputs_coord, attns
        elif self.last_feat:
            return tuple(features), last_feat, outputs_class, outputs_coord
            
        return tuple(features)
