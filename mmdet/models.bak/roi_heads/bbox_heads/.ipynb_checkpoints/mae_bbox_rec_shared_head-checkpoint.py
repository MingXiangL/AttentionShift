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
                 ukeep_ratios=0.5,
                 use_checkpoint=False,
                 mask_ratio=0.4,
                 mask_detach=False,
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
        if mask_detach:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
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
        self.ukeep_ratios = ukeep_ratios
                    
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
        elif self.pretrained and pretrained is None:
            print('must test')
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
    
    def forward(self, x, sampling_results=None):
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
        x_ = x.clone() 
        x = x + self.interpolate_pos_encoding(x, W * self.patch_size, H * self.patch_size)
        
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

        
#         只取 top的 不进行mask token  就是mask attention value低的 
#          eg：keep ratio = 0.7  响应高的70% 保留  响应低的 30% 用mask token代替 即 只关注简单的
        attn_proj_map = attns_project_to_feature(attns)
        keep_num = int(self.keep_ratios * attn_proj_map.size(-1))
#         keep_inds = attn_proj_map.argsort(dim=-1)[:, :keep_num]   # 降序，只取attention 低的
        sort_inds = attn_proj_map.argsort(dim=-1) # sort 为升序
        restored_inds = sort_inds.argsort(dim=-1)
        keep_start_index = attn_proj_map.size(-1) - keep_num
        keep_inds = sort_inds[:, keep_start_index:]   # 只保留 attention 响应高的(简单的)
        x_keep = torch.gather(x_[:, 1:, :], dim=1, index=keep_inds.unsqueeze(-1).repeat(1, 1, x.size(-1)))
        mask_tokens = self.mask_token.repeat(B, attn_proj_map.size(-1) - keep_num, 1)
#         sort_feat = torch.cat([x_keep, mask_tokens], dim=1)
        sort_feat = torch.cat([mask_tokens, x_keep], dim=1)
        # unshuffle
        x = torch.gather(sort_feat, dim=1, index=restored_inds.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  
        # 并上token 
        x = torch.cat([x_[:, :1, :], x], dim=1)        
        

#         # 正反例不同， 正例取随机top中随机比例的填充mask token
#         # 反例，只保留top中随机比例响应高，其他全为mask token

#         if sampling_results is not None: # 不是None则为 train 是none 则为test
#             # 正例用top随机替代 botton随机代替
#             attn_proj_map = attns_project_to_feature(attns)
#             keep_num = int(self.keep_ratios * attn_proj_map.size(-1))
#             ukeep_num = int(attn_proj_map.size(-1) - keep_num)
#             neg_keep_num = int(self.neg_keep_ratios * attn_proj_map.size(-1))
#             neg_ukeep_num = int(attn_proj_map.size(-1) - neg_keep_num)        
            
#             x_final = []
#             pos_inds = [res.pos_inds for res in sampling_results]
#             attn_proj_map_batch = attn_proj_map.chunk(len(pos_inds), dim=0)
#             xs_ori_batch = x_.chunk(len(pos_inds), dim=0)
#             # 这个循环的方法建立在每个batch 的 roi是前一部分 是正例特征，
#             # 后一部分是反例特征（这个一定要保证是对的，可以看core/sample/sampler_result.py self.bboxes）
#             # 并且一定有正例（gt为正例）
#             for pos_ind, attn_batch, xs_ori_b in zip(pos_inds, attn_proj_map_batch, xs_ori_batch):
#                 # 以下为正例的mask token 替换
#                 x_pos = xs_ori_b[:len(pos_ind)]
#                 attn_pos = attn_batch[:len(pos_ind)]
                
#                 sort_inds = attn_pos.argsort(dim=-1) # 为升序
#                 restored_inds = sort_inds.argsort(dim=-1)
#                 keep_inds = sort_inds[:, :keep_num]   #正例一定保留的botton几个，即只取attention低的
#                 ukeep_inds = sort_inds[:, keep_num:]  #正例可能不保留的top几个，随机取xx%
                
#                 x_keep = torch.gather(x_pos[:, 1:, :], dim=1, index=keep_inds.unsqueeze(-1).repeat(1, 1, x_pos.size(-1)))
#                 x_ukeep = torch.gather(x_pos[:, 1:, :], dim=1, index=ukeep_inds.unsqueeze(-1).repeat(1, 1, x_pos.size(-1)))
#                 noise = torch.rand(len(pos_ind), ukeep_num, device=self.mask_token.device)
#                 ids_shuffle = torch.argsort(noise, dim=1)
#                 ids_shuffle_restore = torch.argsort(ids_shuffle, dim=1)
#                 ukeep_keep_num = int(self.ukeep_ratios * ukeep_num)
#                 ukeep_keep_inds = ids_shuffle[:, :ukeep_keep_num]
#                 x_ukeep_keep = torch.gather(x_ukeep, dim=1, index=ukeep_keep_inds.unsqueeze(-1).repeat(1, 1, x_pos.shape[-1]))
#                 mask_tokens = self.mask_token.repeat(len(pos_ind), ukeep_num - ukeep_keep_num, 1)
#                 x_ukeep_restore = torch.cat([x_ukeep_keep, mask_tokens], dim=1)
#                 x_ukeep = torch.gather(x_ukeep_restore, dim=1, 
#                                        index=ids_shuffle_restore.unsqueeze(-1).repeat(1, 1, x_pos.shape[-1])) 
#                 x = torch.cat([x_keep, x_ukeep], dim=1)
#                 x = torch.gather(x, dim=1, index=restored_inds.unsqueeze(-1).repeat(1, 1, x_pos.shape[-1]))  
#                 x_pos = torch.cat([x_pos[:, :1, :], x], dim=1)
                
#                 # 以下为反例的mask token 替换
#                 x_neg = xs_ori_b[len(pos_ind):]
#                 attn_neg = attn_batch[len(pos_ind):]
#                 num_neg = len(x_neg)        
                
#                 sort_inds = attn_neg.argsort(dim=-1) # 为升序
#                 restored_inds = sort_inds.argsort(dim=-1)
#                 mask_tokens_keep = self.mask_token.repeat(num_neg, neg_keep_num, 1)
#                 ukeep_inds = sort_inds[:, neg_keep_num:]   # 可能不mask掉的为后几个，attention 高的
                
#                 x_ukeep = torch.gather(x_neg[:, 1:, :], dim=1, index=ukeep_inds.unsqueeze(-1).repeat(1, 1, x_neg.size(-1)))
#                 noise = torch.rand(num_neg, neg_ukeep_num, device=self.mask_token.device)
#                 ids_shuffle = torch.argsort(noise, dim=1)
#                 ids_shuffle_restore = torch.argsort(ids_shuffle, dim=1)
#                 ukeep_keep_num = int(self.neg_ukeep_ratios * neg_ukeep_num)
#                 ukeep_no_keep_inds = ids_shuffle[:, ukeep_keep_num:]
                
#                 mask_tokens = self.mask_token.repeat(num_neg, ukeep_keep_num, 1)
#                 x_ukeep_no_keep = torch.gather(x_ukeep, dim=1, 
#                                                index=ukeep_no_keep_inds.unsqueeze(-1).repeat(1, 1, x_neg.shape[-1]))
#                 x_ukeep_restore = torch.cat([mask_tokens, x_ukeep_no_keep], dim=1)
#                 x_ukeep = torch.gather(x_ukeep_restore, dim=1, 
#                                        index=ids_shuffle_restore.unsqueeze(-1).repeat(1, 1, x_neg.shape[-1])) 
#                 x = torch.cat([mask_tokens_keep, x_ukeep], dim=1)
#                 x = torch.gather(x, dim=1, index=restored_inds.unsqueeze(-1).repeat(1, 1, x_neg.shape[-1]))  
#                 x_neg = torch.cat([x_neg[:, :1, :], x], dim=1)
                
#                 x = torch.cat([x_pos, x_neg], dim=0)
        

# #       正反例 取top的一部分随机一定的比例 填充mask token 

#         # 取top的，再随机top中的一部分
#         attn_proj_map = attns_project_to_feature(attns)
#         keep_num = int(self.keep_ratios * attn_proj_map.size(-1))
#         sort_inds = attn_proj_map.argsort(dim=-1)
#         restored_inds = sort_inds.argsort(dim=-1)
#         keep_inds = sort_inds[:, :keep_num]   # 降序，只取attention 低的
#         ukeep_inds = sort_inds[:, keep_num:]
#         x_keep = torch.gather(x_[:, 1:, :], dim=1, index=keep_inds.unsqueeze(-1).repeat(1, 1, x.size(-1)))
#         x_ukeep = torch.gather(x_[:, 1:, :], dim=1, index=ukeep_inds.unsqueeze(-1).repeat(1, 1, x.size(-1)))
        
#         ukeep_num = ukeep_inds.size(-1)
#         # 随机选择
#         noise = torch.rand(B, ukeep_num, device=self.mask_token.device)
#         ids_shuffle = torch.argsort(noise, dim=1)
#         ids_shuffle_restore = torch.argsort(ids_shuffle, dim=1)
#         ukeep_keep_num = int(self.ukeep_ratios * ukeep_num)
#         ukeep_keep_inds = ids_shuffle[:, :ukeep_keep_num]
#         x_ukeep_keep = torch.gather(x_ukeep, dim=1, index=ukeep_keep_inds.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
#         mask_tokens = self.mask_token.repeat(B, ukeep_num - ukeep_keep_num, 1)
#         x_ukeep_restore = torch.cat([x_ukeep_keep, mask_tokens], dim=1)
#         x_ukeep = torch.gather(x_ukeep_restore, dim=1, index=ids_shuffle_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  
#         x = torch.cat([x_keep, x_ukeep], dim=1)
#         x = torch.gather(x, dim=1, index=restored_inds.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  
#         x = torch.cat([x_[:, :1, :], x], dim=1)
        
#         只取 top的进行 mask token
#         attn_proj_map = attns_project_to_feature(attns)
#         keep_num = int(self.keep_ratios * attn_proj_map.size(-1))
# #         keep_inds = attn_proj_map.argsort(dim=-1)[:, :keep_num]   # 降序，只取attention 低的
#         sort_inds = attn_proj_map.argsort(dim=-1)
#         restored_inds = sort_inds.argsort(dim=-1)
#         keep_inds = sort_inds[:, :keep_num]   # 降序，只取attention 低的
#         x_keep = torch.gather(x_[:, 1:, :], dim=1, index=keep_inds.unsqueeze(-1).repeat(1, 1, x.size(-1)))
#         mask_tokens = self.mask_token.repeat(B, attn_proj_map.size(-1) - keep_num, 1)
#         sort_feat = torch.cat([x_keep, mask_tokens], dim=1)
#         # unshuffle
#         x = torch.gather(sort_feat, dim=1, index=restored_inds.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  
#         # 并上token 
#         x = torch.cat([x_[:, :1, :], x], dim=1)


#         正例填充mask token，反例不管    
#         if sampling_results is not None:  # 不是None则为 train 是none 则为test
#             # 只在正例上进行特征去除,反例用原来的特征代替
#             pos_inds = [res.pos_inds for res in sampling_results]
#             xs_batch = x.chunk(len(pos_inds), dim=0)  # mask token替代后的 每个batch的 roi feature 包括正反例 
#             xs_ori_batch = x_.chunk(len(pos_inds), dim=0) # 为被任何替代后的 每个batch的roi feature 的正反例

#             x_final = []
#             for pos_ind, xs_b, xs_ori_b in zip(pos_inds, xs_batch, xs_ori_batch):  # 这个方法建立在每个batch 的 roi是前一部分 是正例特征，后一部分是反例特征（这个一定要保证是对的，可以看core/sample/sampler_result.py self.bboxes）
#                 x_f = torch.zeros_like(xs_b)
#                 x_f[len(pos_ind):] = xs_ori_b[len(pos_ind):]
#                 x_f[:len(pos_ind)] = xs_b[:len(pos_ind)]
#                 x_final.append(x_f)
#             x = torch.cat(x_final, dim=0)


#         反例填充mask token，正例不管  
#         if sampling_results is not None:  # 不是None则为 train 是none 则为test
#             # 只在反例上进行特征去除,正例用原来的特征代替
#             pos_inds = [res.pos_inds for res in sampling_results]
#             xs_batch = x.chunk(len(pos_inds), dim=0)  # mask token替代后的 每个batch的 roi feature 包括正反例 
#             xs_ori_batch = x_.chunk(len(pos_inds), dim=0) # 为被任何替代后的 每个batch的roi feature 的正反例

#             x_final = []
#             for pos_ind, xs_b, xs_ori_b in zip(pos_inds, xs_batch, xs_ori_batch):  # 这个方法建立在每个batch 的 roi是前一部分 是正例特征，后一部分是反例特征（这个一定要保证是对的，可以看core/sample/sampler_result.py self.bboxes）
#                 x_f = torch.zeros_like(xs_b)
#                 x_f[:len(pos_ind)] = xs_ori_b[:len(pos_ind)]  # 原来正例还是最开始正例
#                 x_f[len(pos_ind):] = xs_b[len(pos_ind):]  # 反例 成了 修改后的反例
#                 x_final.append(x_f)
#             x = torch.cat(x_final, dim=0)
            
        # 加上位置编码
        x = x + self.interpolate_pos_encoding(x, W * self.patch_size, H * self.patch_size)
        # 
        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        
        x_rec_bbox = self.decoder_box_norm(x[:B, :, :])
        x_rec_token = x_rec_bbox[:, 0, :]
        rec_cls_score = self.fc_cls(x_rec_token) if self.with_cls else None
        rec_bbox_pred = self.fc_reg(x_rec_token) if self.with_reg else None
        
#         return cls_score, bbox_pred, x_rec, x_bbox[:,1:,:][mask], x_rec_token, x_bbox[:,0,:]
        return cls_score, bbox_pred, rec_cls_score, rec_bbox_pred
#         return cls_score, bbox_pred #, rec_cls_score, rec_bbox_pred

