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
import torch.nn.functional as F
from einops import rearrange

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

@HEADS.register_module()
class MAEBoxHeadRec(BBoxHead):
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
                 seed_score_thr=0.2,
                 seed_thr=0.2,
                 seed_multiple=0.5,
                 cam_layer=-1,
                 with_reconstruct=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.with_reconstruct = with_reconstruct
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2
        self.det_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.with_decoder_embed = False
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
        if self.with_reconstruct:
            self.fc_rec = nn.Linear(embed_dim, 3*patch_size*patch_size)
        self.seed_score_thr = seed_score_thr
        self.seed_thr = seed_thr
        self.seed_multiple = seed_multiple
        self.cam_layer = cam_layer
            
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
    
    def forward(self, x, img=None):
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.with_decoder_embed:
            x = self.norm(x)
            x = self.decoder_embed(x)

        x = torch.cat([self.det_token.expand(B, -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(x, W * self.patch_size, H * self.patch_size)
        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        # x = self.decoder_box_norm(x[:, 0, :])
        x = self.decoder_box_norm(x)
        cls_score = self.fc_cls(x[:, 0, :]) if self.with_cls else None
        bbox_pred = self.fc_reg(x[:, 0, :]) if self.with_reg else None
        img_rec   = self.fc_rec(x[:, 1:, :]) if self.with_reconstruct else None
        return cls_score, bbox_pred, img_rec.transpose(1,2).reshape(B, -1, W, H)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
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
## rec_loss
        if recs is not None:
            img_h, img_w = img.size()[-2:]
            patch_num_h, patch_num_w = img_h // 16, img_w // 16
            
            device = img.device
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN, device=device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD, device=device).to(device)[None, :, None, None]
            
            unnorm_images = img * std + mean  # in [0, 1]
            unnorm_images = F.interpolate(unnorm_images, (img_h, img_w), mode='bilinear')
                    
            images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', 
                                       p1=16, p2=16)
            images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            img_target = rearrange(images_norm, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', 
                                           h=patch_num_h, p1=16) # B, 3, img_h, img_w
            
            num_recs_per_img = int(len(recs) / len(rec_inds))
            num_roi_per_img = int(len(rois) / len(rec_inds))
            
            loss_rec = 0
            average_factor = 0
            for i, rec_ind in enumerate(rec_inds):
                rec = recs[i * num_recs_per_img:(i + 1) * num_recs_per_img]
                roi = rois[i * num_roi_per_img:(i + 1) * num_roi_per_img]
                # print(f'rec_inds.shape[0]: {len(rec_inds)}')
                # print(f'rois.shape: {rois.shape}')
                # import pdb; pdb.set_trace()
                rec_roi = roi[rec_ind]
                rec_target = self.crop_feature(img_target, rec_roi)
                
                for r, r_t in zip(rec, rec_target):
                    average_factor += 1
                    c, h, w = r.size()[-3:]
                    r = r.reshape(16, 16, 3, h, w).permute(0, 3, 1, 4, 2).reshape(16 * h, 16 * w, 3).permute(2, 0, 1)
                    r_size = r.size()[-2:]
                    r_t_resize = F.interpolate(r_t.unsqueeze(0), tuple(r_size), mode='nearest').squeeze(0)
                    loss_rec += ((r - r_t_resize) ** 2).mean()
                    # loss_rec += self.loss_reconstruction(r, r_t_resize)
            loss_rec /= average_factor
            losses['loss_rec'] = loss_rec
        return losses
