import os
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from collections import OrderedDict
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner import _load_checkpoint, load_state_dict
from mmdet.utils import get_root_logger
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from models.vision_transformer import Block, trunc_normal_
from mmdet.models.losses import accuracy


def attns_project_to_feature(attns_maps):
    #         assert len(attns_maps[1]) == 1 
    # [block_num], B, H, all_num, all_num
    attns_maps = torch.stack(attns_maps)
    # block_num, B, H, all_num, all_num
    attns_maps = attns_maps.mean(2)
    # block_num, B, all_num, all_num
    residual_att = torch.eye(attns_maps.size(2)).type_as(attns_maps)
    aug_att_mat = attns_maps + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size()).type_as(aug_att_mat)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
    attn_proj_map = joint_attentions[-1]
    attn_proj_map = attn_proj_map[:, 0, 1:]
    return attn_proj_map

@HEADS.register_module()
class MAEBoxRecHead(BBoxHead):
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
                 return_attention=False,
                 keep_ratios=0.6,
                 use_checkpoint=False,
                 mask_ratio=0.4,
#                  gan_loss_weight=0.0,
                 loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.mask_ratio=mask_ratio
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2
        self.det_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.with_decoder_embed = False
        self.loss_weight=loss_weight
        # MAE decoder specifics
        if in_channels != embed_dim:
            self.with_decoder_embed = True
            self.norm = norm_layer(in_channels)
            self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_box_norm = norm_layer(embed_dim)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(embed_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = nn.Linear(embed_dim, out_dim_reg)

        self.return_attention = return_attention
        self.keep_ratios = keep_ratios
                    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        super().init_weights()
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
                print('MAEBoxRecHead load pretrained checkpoint:',pretrained)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")
        elif self.pretrained == False:
            trunc_normal_(self.det_token, std=.02)
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @force_fp32(apply_to=('cls_score', 'bbox_pred','x_rec','x_bbox'))
    def loss_(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             recs=None,
             rec_inds=None,
             img=None,
            ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls_rec'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc_rec'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox_rec'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox_rec'] = bbox_pred[pos_inds].sum()
        return losses
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred','x_rec','x_bbox'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             recs=None,
             rec_inds=None,
             img=None,
            ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        
        return losses

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

        return ids_keep, ids_restore, mask.bool()

    def random_masking(self, x, ids_keep):
        N, L, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked    
    
    def forward(self, x):
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.with_decoder_embed:
            x = self.norm(x)
            x = self.decoder_embed(x)
            
#         ids_keep, ids_restore, mask = self.masking_id(B,x.shape[1] , self.mask_ratio)
#         x_masked = self.random_masking(x, ids_keep)
        
#         # append mask tokens to sequence
#         mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - x_masked.shape[1], 1)
#         x_ = torch.cat([x_masked, mask_tokens], dim=1)  # no cls token
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         x=torch.cat([x,x_],dim=0)

        x = torch.cat([self.det_token.expand(B, -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(x, W * self.patch_size, H * self.patch_size)
        
#         x_ = x.clone() 
        
        attns = []
        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                if self.return_attention:
                    x, attn = blk(x, return_attention=self.return_attention)
                    attns.append(attn)
                else:
                    x = blk(x)
        
        x_bbox = self.decoder_box_norm(x[:B, :, :])
        cls_score = self.fc_cls(x_bbox[:,0,:]) if self.with_cls else None
        bbox_pred = self.fc_reg(x_bbox[:,0,:]) if self.with_reg else None
        
        
#         # 
        attn_proj_map = attns_project_to_feature(attns)
        print(attn_proj_map.size())
        exit()
#         keep_num = int(self.keep_ratios * attn_proj_map.size(-1))
#         keep_inds = attn_proj_map.argsort(dim=-1)[:, :keep_num]   # 降序，只取attention 低的
        
#         x_keep = torch.gather(x_[:, 1:, :], dim=1, index=keep_inds.unsqueeze(-1).repeat(1, 1, x.size(-1)))
#         x_keep = torch.cat([x_[:, :1, :], x_keep], dim=1)
        
#         for blk in self.decoder_blocks:
#             if self.use_checkpoint:
#                 x_keep = checkpoint(blk, x_keep)
#             else:
#                 x_keep = blk(x_keep)
        
#         x_rec_bbox = self.decoder_box_norm(x_keep[:B, :, :])
#         x_rec_token = x_rec_bbox[:, 0, :]
#         rec_cls_score = self.fc_cls(x_rec_token) if self.with_cls else None
#         rec_bbox_pred = self.fc_reg(x_rec_token) if self.with_reg else None
        
#         return cls_score, bbox_pred, x_rec, x_bbox[:,1:,:][mask], x_rec_token, x_bbox[:,0,:]
#         return cls_score, bbox_pred, rec_cls_score, rec_bbox_pred
        return cls_score, bbox_pred #, rec_cls_score, rec_bbox_pred

