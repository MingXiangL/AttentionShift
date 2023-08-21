# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import BACKBONES

from mmcv.cnn.utils.weight_init import trunc_normal_
from mmdet.utils import get_root_logger
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint, load_state_dict

from mmcv_custom import load_checkpoint
from einops import rearrange
from time import *

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, h=None, w=None, split_attn=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        if split_attn:
            sh = h // 14
            sw = w // 14
            qkv = rearrange(qkv, 'q b h (H W) c -> q b h H W c', H=h, W=w)
            qkv = rearrange(qkv, 'q b h (hs sh) (hw sw) c -> q b (h sh sw) (hs hw) c', hs=14, hw=14, sh=sh, sw=sw)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if split_attn:
            x = (attn @ v)
            x = rearrange(x, 'b (h sh sw) (hs hw) c -> b h (hs sh hw sw) c', hs=14, hw=14, sh=sh, sw=sw)
            x = rearrange(x, 'b h n c -> b n (h c)')
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, h=None, w=None, split_attn=False):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), h=h, w=w, split_attn=split_attn))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), h=h, w=w, split_attn=split_attn))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        _, _, H, W = x.shape
        x = self.proj(x)
        patch_num_h, patch_num_w = x.size()[-2:]
        return x.flatten(2).transpose(1, 2)
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class PretrainVisionTransformerEncoder(BaseModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 patch_size=16, 
                 in_chans=3, 
                 num_class=0, 
                 embed_dim=384, 
                 depth=12,
                 num_heads=6, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=None,
                 use_learnable_pos_emb=False,
                 split_attn_freq=0,
                 with_fpn=True,
                 out_indices=[3, 5, 7, 11]):
        super().__init__()
        self.num_class = num_class
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.split_attn_freq = split_attn_freq

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, 
                  num_heads=num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, 
                  qk_scale=qk_scale,
                  drop=drop_rate, 
                  attn_drop=attn_drop_rate, 
                  drop_path=dpr[i], 
                  norm_layer=norm_layer,
                  init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_class) if num_class > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.with_fpn = with_fpn
        self.out_indices = out_indices
        
        if with_fpn and patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)


    def forward_features(self, x):
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.patch_embed(x)
        features = []
        B, N, C = x.shape
        pos_embed = get_sinusoid_encoding_table(N, C)
        x = x + pos_embed.type_as(x).to(x.device).clone().detach()

        for i, blk in enumerate(self.blocks):
            split_attn = False
            if self.split_attn_freq > 0:
                split_attn = False if (i + 1) % self.split_attn_freq == 0 else True
                assert h % 14 == 0
                assert w % 14 == 0

                if (h > 14) and (w > 14):
                    split_attn = split_attn
                else:
                    split_attn = False

            x = blk(x, h=h, w=w, split_attn=split_attn)
            if i in self.out_indices:
                xp = self.norm(x).permute(0, 2, 1).reshape(B, -1, h, w) 
                features.append(xp.contiguous())
                
        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
        return features
    
    def forward(self, x):
        x = self.forward_features(x)
        return x


@BACKBONES.register_module()
class MAEVisionTransformer(BaseModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=384, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 split_attn_freq=0,
                 freeze_encoder=False,
                 norm_eval=True, 
                 frozen_stages=0,
                 with_fpn=True,
                 out_indices=[3, 5, 7, 11],
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_class=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            split_attn_freq=split_attn_freq,
            with_fpn=with_fpn,
            out_indices=out_indices
        )

#         self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.init_cfg = init_cfg

        self.freeze_encoder = freeze_encoder
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

#     def init_weights(self, pretrained=None):
#         logger = get_root_logger()
#         if (self.init_cfg is None) or (self.init_cfg.checkpoint is None):
#             logger.warn(f'No pre-trained weights for '
#                         f'{self.__class__.__name__}, '
#                         f'training start from scratch')
#             self.apply(self._init_weights)
#         else:
#             logger.info(f'Load pre-trained weights '
#                         f'from {self.init_cfg.checkpoint}'
#                         f' to init mae encoder')
#             checkpoint = _load_checkpoint(self.init_cfg.checkpoint, logger=logger, map_location='cpu')
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
#             load_state_dict(self, state_dict, strict=False, logger=logger)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger, map_location='cpu')
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
            
    def _freeze_stages(self):
        if self.freeze_encoder:
            self.encoder_to_decoder.eval()
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encoder_to_decoder.parameters():
                param.requires_grad = False

        if self.frozen_stages > 0:
            raise NotImplementedError

    def forward(self, x):
        x = self.encoder(x) # [B, N, C_e]
#         x = self.encoder_to_decoder(x) # [B, N, C_d]

        return x 

    def train(self, mode=True):
        """Convert the model into training mode while keep batch normalization layer freezed."""
        super(MAEVisionTransformer, self).train(mode)
        self._freeze_stages()
