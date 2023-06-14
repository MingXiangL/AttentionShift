import os
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.utils import get_root_logger
from mmcv.runner import BaseModule, _load_checkpoint, load_state_dict
from models.vision_transformer import Block
from ..builder import HEADS


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@HEADS.register_module()
class MAEDecoderHead(BaseModule):
    def __init__(self, 
                 in_channels,
                 mask_ratio=0.75,
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
                 norm_pix_loss=True,
                 loss_weight=1.0,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.loss_weight = loss_weight

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2

        # MAE decoder specifics
        self.norm = norm_layer(in_channels)
        self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3, bias=True) # decoder to patch

        self.mae_encoder = None

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if isinstance(pretrained, str):
            if os.path.isfile(pretrained):
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
            raise NotImplementedError
        else:
            raise TypeError('pretrained must be a str or None')
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        w, h = imgs.shape[2] // p, imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, w, p, h, p))
        x = torch.einsum('ncwphq->nwhpqc', x)
        x = x.reshape(shape=(imgs.shape[0], w * h, p**2 * 3))
        return x
    
    def masking_id(self, batch_size, num_patches, mask_ratio):
        N, L = batch_size, num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=self.mask_token.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.mask_token.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def random_masking(self, x, ids_keep):
        N, L, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked
    
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
    
    def forward_decoder(self, x, ids_restore, w, h):
        x = self.norm(x)
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.interpolate_pos_encoding(x, w, h)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x
    
    def loss(self, x, img, mask):
        target = self.patchify(img)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (x - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = self.loss_weight * (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, x, img):
        B, L, C = x.shape
        _, _, W, H = img.shape
        ids_keep, ids_restore, mask = self.masking_id(B, L - 1, self.mask_ratio)
        x = self.random_masking(x, ids_keep)
        x = self.mae_encoder.forward_encoder(x) if self.mae_encoder is not None else x
        x = self.forward_decoder(x, ids_restore, W, H)
        if not self.training:
            return x
        losses = dict(loss_mae=self.loss(x, img, mask))
        return losses
