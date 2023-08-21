import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS, build_loss
import pdb
from ..backbones.visual_transformer_det import MLP
from functools import partial

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


@HEADS.register_module()
class DeformableConvAttention(nn.Module):
    def __init__(self, n_heads, n_head_channels, n_groups, offset_range_factor, loss, kernel_size=3, dim_out=None, tau=1):
        super().__init__()
        self.n_heads = n_heads
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.nc = n_head_channels * n_heads
        self.dim_out = dim_out if dim_out is not None else self.nc
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.kernel_size_mul = kernel_size[0] * kernel_size[1]
        self.kernel_size = kernel_size
        self.tau = tau
        padding = [k // 2 for k in kernel_size]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.rep_pad = nn.ReplicationPad2d(padding[0])
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size, stride=1, padding=padding, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size, stride=1, padding=padding, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, out_channels=2 * kernel_size[0] * kernel_size[1], kernel_size=1, stride=1, padding=0, bias=False)
        )
        # TODO: 
        # 1.调节输出的bias为True还是False 
        # 2.调节offset的非线性变换函数，sigmoid or tanh
        # 3. kernel size 的调节
        # 4. scale_factor的调节
        # self.norm = norm_layer(self.nc)
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.dim_out,
            kernel_size=1, stride=1, padding=0
        )

        self.loss = build_loss(loss)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref

    def get_reference_unflod(self, ref):
        '''
        Arg: 
            ref: B, 2, H, W
        Return:
            ref: B, 2 * Hk *Hw
        '''
        r1 = self.rep_pad(ref)
        r1 = F.unfold(r1, self.kernel_size, padding=0, stride=1).unflatten(1, (2, self.kernel_size_mul)).permute(0, 2, 1, 3).flatten(1,2)
        return r1.unflatten(-1, ref.shape[-2:])

    def forward(self, x, idx, labels, all_zero=False, visualize=False, **kwargs):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        # x = self.norm(x.permute(0, 2,3,1)).permute(0,3,1,2)
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        q = q[..., idx[:,0], idx[:, 1]]
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1).repeat(1, self.kernel_size_mul, 1, 1) 
            #为啥要乘以1/H, 1/W, 而且还是在tanh之后, 我觉得就是限制每个点offset的范围，这么看来，这个offset_range_factor还是很重要的
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
                
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        reference = self.get_reference_unflod(reference.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # reference = reference.repeat(1, 1, 1, self.kernel_size_mul)
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        # pos_org = pos.clone()
        pos = pos.unflatten(-1, (self.kernel_size_mul, 2)).permute(0, 3, 1, 2, 4).flatten(0, 1)
        reference = reference.unflatten(-1, (self.kernel_size_mul, 2)).permute(0, 3, 1, 2, 4).flatten(0, 1) # B*n_groups*kernel_mul , Hk, Wk, 2
        # pos = pos.reshape(-1, Hk, Wk, self.kernel_size_mul, 2)
        # reference = reference.reshape(-1, Hk, Wk, 2)
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, 1, self.n_group_channels, H, W).repeat(1, self.kernel_size_mul, 1, 1, 1).flatten(0, 1), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg

        # # test code for grid sample----------------------
        # pos_single = pos_org.unflatten(-1, (self.kernel_size_mul, 2))[..., 0, :]
        # x_sampled_single = F.grid_sample(
        #     input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
        #     grid=pos_single[..., (1, 0)],
        #     mode='bilinear', align_corners=True
        # )
        # x_sampled_single = x_sampled_single.reshape(B, C, 1, Hk, Wk)

        # pos_repeat = pos_org.unflatten(-1, (self.kernel_size_mul, 2))[..., [0], :].repeat(1, 1, 1, self.kernel_size_mul, 1).permute(0, 3, 1, 2, 4).flatten(0, 1)
        # x_sampled_repeat = F.grid_sample(
        #     input=x.reshape(B * self.n_groups, 1, self.n_group_channels, H, W).repeat(1, self.kernel_size_mul, 1, 1, 1).flatten(0, 1), 
        #     grid=pos_repeat[..., (1, 0)], # y, x -> x, y
        #     mode='bilinear', align_corners=True)
        # x_sampled_repeat = x_sampled_repeat.reshape(B * self.n_groups, self.kernel_size_mul, self.n_group_channels, Hk, Wk).permute(0, 2, 1, 3, 4)
        # x_sampled_repeat = x_sampled_repeat.reshape(B, C, self.kernel_size_mul, Hk, Wk)
        # pdb.set_trace(x_sampled_repeat)
        # #-------------------------#

        x_sampled = x_sampled.reshape(B * self.n_groups, self.kernel_size_mul, self.n_group_channels, Hk, Wk).permute(0, 2, 1, 3, 4)
        x_sampled = x_sampled.reshape(B, C, self.kernel_size_mul, Hk, Wk)
        x_sampled = x_sampled[..., idx[:, 0], idx[:, 1]] # B, C, region(k x k), n_key_points
        n_sample = idx.shape[0]

        q = q.reshape(B * self.n_heads, self.n_head_channels, n_sample)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, self.kernel_size_mul, n_sample).permute(0, 1, 3, 2) # B * h, n_key_points, region(k x k)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, self.kernel_size_mul, n_sample).permute(0, 1, 3, 2)
        attn = torch.einsum('b c m, b c m n -> b m n', q, k) # B * h, n_key_points, region(k x k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn / self.tau, dim=-1)
        
        out = torch.einsum('b m n, b c m n -> b c m', attn, v)
        
        out = out.reshape(B, C, 1, n_sample)
        
        # y = self.proj_drop(self.proj_out(out))
        y = self.proj_out(out).reshape(self.dim_out, n_sample).permute(1, 0)
        print(f'labels: {labels: labels}')
        pdb.set_trace()
        loss = self.loss(y, labels)
        if all_zero:
            loss *= 0

        x_sampled_ret = F.grid_sample(
            input=x.expand(self.kernel_size_mul*self.n_groups, -1, -1, -1), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # ks * g, C, H, W
        x_sampled_ret = x_sampled_ret[..., idx[:, 0], idx[:, 1]] # ks *g, C, n_key_points
        
        return dict(deform_attn_loss=loss), \
            x_sampled_ret.permute(2,0,1), \
            pos[:, idx[:, 0], idx[:, 1]].permute(1,0,2), \
            reference[:, idx[:, 0], idx[:, 1]], \
            y[torch.arange(len(labels)), labels].sigmoid(), \
            attn.mean(0).repeat(1, self.n_groups), \
            (pos, reference, attn, y)


@HEADS.register_module()
class DeformableConvAttentionNorm(DeformableConvAttention):
    def __init__(self, n_heads, n_head_channels, n_groups, offset_range_factor, loss, kernel_size=3, dim_out=None, **kwargs):
        super().__init__(n_heads, n_head_channels, n_groups, offset_range_factor, loss, kernel_size, dim_out, **kwargs)
        self.norm = nn.LayerNorm(self.nc)
    
    def forward(self, x, idx, labels, all_zero=False, visualize=False, **kwargs):
        x = self.norm(x.permute(0,2,3,1).detach()).permute(0,3,1,2)
        return super().forward(x, idx, labels, all_zero, visualize, **kwargs)


@HEADS.register_module()
class DeformableConvAttentionClsLoc(nn.Module):
    
    def __init__(self, n_heads, n_head_channels, n_groups, offset_range_factor, loss_cls, loss_loc, kernel_size=3, dim_out=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.nc = n_head_channels * n_heads
        self.dim_out = dim_out if dim_out is not None else self.nc
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.kernel_size_mul = kernel_size[0] * kernel_size[1]
        self.kernel_size = kernel_size
        padding = [k // 2 for k in kernel_size]
        self.rep_pad = nn.ReplicationPad2d(padding[0])
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size, stride=1, padding=padding, groups=self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, out_channels=2 * kernel_size[0] * kernel_size[1], kernel_size=1, stride=1, padding=0, bias=False)
        )
        # TODO: 
        # 1.调节输出的bias为True还是False 
        # 2.调节offset的非线性变换函数，sigmoid or tanh
        # 3. kernel size 的调节
        # 4. scale_factor的调节
        
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0),
            nn.GELU()
            )
        
        # self.proj_out = nn.Conv2d(
        #     self.nc, self.dim_out,
        #     kernel_size=1, stride=1, padding=0
        # )
        self.class_embed = MLP(self.nc, self.nc, dim_out, 3)
        self.coord_embed = MLP(self.nc, self.nc, 2, 3)

        self.loss_cls = build_loss(loss_cls)
        self.loss_loc = build_loss(loss_loc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def get_reference_unflod(self, ref):
        '''
        Arg: 
            ref: B, 2, H, W
        Return:
            ref: B, 2 * Hk *Hw
        '''
        r1 = self.rep_pad(ref)
        r1 = F.unfold(r1, self.kernel_size, padding=0, stride=1).unflatten(1, (2, self.kernel_size_mul)).permute(0, 2, 1, 3).flatten(1,2)
        return r1.unflatten(-1, ref.shape[-2:])

    def forward(self, x, idx, labels, coords, all_zero=False, return_offset=False):
        B, C, H, W = x.size()
        x1 = self.input_proj(x)
        dtype, device = x.dtype, x.device
        q = self.proj_q(x1)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        q = q[..., idx[:,0], idx[:, 1]]
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
    
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1).repeat(1, self.kernel_size_mul, 1, 1) 
            #为啥要乘以1/H, 1/W, 而且还是在tanh之后, 我觉得就是限制每个点offset的范围，这么看来，这个offset_range_factor还是很重要的
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
                
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        reference = self.get_reference_unflod(reference.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # reference = reference.repeat(1, 1, 1, self.kernel_size_mul)
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        # pos_org = pos.clone()
        pos = pos.unflatten(-1, (self.kernel_size_mul, 2)).permute(0, 3, 1, 2, 4).flatten(0, 1)
        reference = reference.unflatten(-1, (self.kernel_size_mul, 2)).permute(0, 3, 1, 2, 4).flatten(0, 1) # B*n_groups*kernel_mul , Hk, Wk, 2
        # pos = pos.reshape(-1, Hk, Wk, self.kernel_size_mul, 2)
        # reference = reference.reshape(-1, Hk, Wk, 2)
        x_sampled = F.grid_sample(
            input=x1.reshape(B * self.n_groups, 1, self.n_group_channels, H, W).repeat(1, self.kernel_size_mul, 1, 1, 1).flatten(0, 1), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg

        # # test code for grid sample----------------------
        # pos_single = pos_org.unflatten(-1, (self.kernel_size_mul, 2))[..., 0, :]
        # x_sampled_single = F.grid_sample(
        #     input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
        #     grid=pos_single[..., (1, 0)],
        #     mode='bilinear', align_corners=True
        # )
        # x_sampled_single = x_sampled_single.reshape(B, C, 1, Hk, Wk)

        # pos_repeat = pos_org.unflatten(-1, (self.kernel_size_mul, 2))[..., [0], :].repeat(1, 1, 1, self.kernel_size_mul, 1).permute(0, 3, 1, 2, 4).flatten(0, 1)
        # x_sampled_repeat = F.grid_sample(
        #     input=x.reshape(B * self.n_groups, 1, self.n_group_channels, H, W).repeat(1, self.kernel_size_mul, 1, 1, 1).flatten(0, 1), 
        #     grid=pos_repeat[..., (1, 0)], # y, x -> x, y
        #     mode='bilinear', align_corners=True)
        # x_sampled_repeat = x_sampled_repeat.reshape(B * self.n_groups, self.kernel_size_mul, self.n_group_channels, Hk, Wk).permute(0, 2, 1, 3, 4)
        # x_sampled_repeat = x_sampled_repeat.reshape(B, C, self.kernel_size_mul, Hk, Wk)
        # pdb.set_trace(x_sampled_repeat)
        # #-------------------------#

        x_sampled = x_sampled.reshape(B * self.n_groups, self.kernel_size_mul, self.n_group_channels, Hk, Wk).permute(0, 2, 1, 3, 4)
        x_sampled = x_sampled.reshape(B, C, self.kernel_size_mul, Hk, Wk)
        x_sampled = x_sampled[..., idx[:, 0], idx[:, 1]] # B, C, region(k x k), n_key_points

        n_sample = idx.shape[0]

        q = q.reshape(B * self.n_heads, self.n_head_channels, n_sample)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, self.kernel_size_mul, n_sample).permute(0, 1, 3, 2) # B * h, n_key_points, region(k x k)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, self.kernel_size_mul, n_sample).permute(0, 1, 3, 2)
        attn = torch.einsum('b c m, b c m n -> b m n', q, k) # B * h, n_key_points, region(k x k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b m n, b c m n -> b c m', attn, v)
        
        out = out.reshape(B, C, n_sample).permute(0, 2, 1).flatten(0, 1)
        
        # y = self.proj_drop(self.proj_out(out))
        # y = self.proj_out(out).reshape(self.dim_out, n_sample).permute(1, 0)
        pred_cls = self.class_embed(out)
        pred_loc = self.coord_embed(out)
        loss_cls = self.loss_cls(pred_cls, labels)
        loss_loc = self.loss_loc(pred_loc, coords)

        if all_zero:
            loss_cls *= 0
            loss_loc *= 0

        return dict(deform_attn_cls_loss=loss_cls, deform_attn_loc_loss=loss_loc), \
            x_sampled, \
            pos[:, idx[:, 0], idx[:, 1]], \
            reference[:, idx[:, 0], idx[:, 1]], \
            pred_cls[torch.arange(len(labels)), labels].sigmoid(), \
            attn.mean(0), \
            (pos, reference, attn)


def batch_indexing(feat, idx):
    '''
    Args: 
        feat: tensor, B, C, H, W
        idx:  List, [(tensor, N, 1), ...] length==B
    '''
    print(f'feat.shape: {feat.shape}')
    feat_img = []
    for i_img, ii in enumerate(idx):
        feat_img.append(feat[i_img, ..., ii[:, 0], ii[:, 1]].unsqueeze(0))
    return torch.cat(feat_img, dim=-1)


@HEADS.register_module()
class DeformableConvAttentionBatch(DeformableConvAttention):
    def forward(self, x, idx, labels, all_zero=False):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
    
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1).repeat(1, self.kernel_size_mul, 1, 1) 
            #为啥要乘以1/H, 1/W, 而且还是在tanh之后, 我觉得就是限制每个点offset的范围，这么看来，这个offset_range_factor还是很重要的
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device).repeat(1, 1, 1, self.kernel_size_mul)
        if self.offset_range_factor >= 0:
            pos = offset + reference
            # pos = reference
        else:
            pos = (offset + reference).tanh()
        pos = pos.reshape(-1, Hk, Wk, 2)

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W).repeat(self.kernel_size_mul, 1, 1, 1), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        x_sampled = x_sampled.reshape(B, C, self.kernel_size_mul, Hk, Wk)
        x_sampled = batch_indexing(x_sampled, idx)
        q = batch_indexing(q, idx)
        n_sample = q.shape[-1]

        q = q.reshape(self.n_heads, self.n_head_channels, n_sample)
        k = self.proj_k(x_sampled).reshape(self.n_heads, self.n_head_channels, self.kernel_size_mul, n_sample).permute(0, 1, 3, 2) # B * h, n_key_points, region(k x k)
        v = self.proj_v(x_sampled).reshape(self.n_heads, self.n_head_channels, self.kernel_size_mul, n_sample).permute(0, 1, 3, 2)
        
        attn = torch.einsum('b c m, b c m n -> b m n', q, k) # B * h, n_key_points, region(k x k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b m n, b c m n -> b c m', attn, v)
        
        out = out.reshape(1, C, 1, n_sample)
        
        # y = self.proj_drop(self.proj_out(out))
        y = self.proj_out(out).reshape(self.dim_out, n_sample).permute(1, 0)
        
        loss = self.loss(y, labels)
        if all_zero:
            loss *= 0
        return dict(deform_attn_loss=loss)
