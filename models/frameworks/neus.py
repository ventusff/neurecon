from models.base import ImplicitSurface, NeRF, RadianceNet
from utils import rend_util, train_util

import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# def pdf_phi_s(x: torch.Tensor, s):
#     esx = torch.exp(-s*x)
#     y = s*esx / ((1+esx) ** 2)
#     return y


def cdf_Phi_s(x, s):
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x*s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = opacity_alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights

class NeuS(nn.Module):
    def __init__(self,
                 variance_init=0.05,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=-1,
                 use_outside_nerf=False,
                 obj_bounding_radius=1.0,

                 surface_cfg=dict(),
                 radiance_cfg=dict()):
        super().__init__()
        
        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor

        #------- surface network
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)
        
        #------- radiance network
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)

        #-------- outside nerf++
        if use_outside_nerf:
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)

    def forward_radiance(self, x: torch.Tensor, view_dirs: torch.Tensor):
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiance = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiance

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)


    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiances, sdf, nablas


def volume_render(
    rays_o, 
    rays_d,
    model: NeuS,
    
    obj_bounding_radius=1.0,
    
    batched = False,
    batched_info = {},

    # render algorithm config
    calc_normal = False,
    use_view_dirs = True,
    rayschunk = 65536,
    netchunk = 1048576,
    white_bkgd = False,
    near_bypass: Optional[float] = None,
    far_bypass: Optional[float] = None,

    # render function config
    detailed_output = True,
    show_progress = False,

    # sampling related
    perturb = False,   # config whether do stratified sampling
    fixed_s_recp = 1/64.,
    N_samples = 64,
    N_importance = 64,
    N_outside = 0,  # whether to use outside nerf
    
    # upsample related
    upsample_algo = 'official_solution',
    N_nograd_samples = 2048,
    N_upsample_iters = 4,
    
    **dummy_kwargs  # just place holder
):
    """
    input: 
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)
    
    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)
    
    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]
        
        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)
        
        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None
    
        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]

        # ---------------
        # Sample points on the rays
        # ---------------


        # ---------------
        # Coarse Points
        
        # [(B), N_rays, N_samples]
        # d_coarse = torch.linspace(near, far, N_samples).float().to(device)
        # d_coarse = d_coarse.view([*[1]*len(prefix_batch), 1, N_samples]).repeat([*prefix_batch, N_rays, 1])
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = near * (1 - _t) + far * _t
        
        # ---------------
        # Up Sampling
        with torch.no_grad():
            # -------- option 1: directly use
            if upsample_algo == 'direct_use':   # nerf-like
                # [(B), N_rays, N_samples, 3]
                pts_coarse = rays_o.unsqueeze(-2) + d_coarse.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # query network to get sdf
                # [(B), N_rays, N_samples]
                sdf_coarse = model.implicit_surface.forward(pts_coarse)
                # [(B), N_rays, N_samples-1]
                *_, w_coarse = sdf_to_w(sdf_coarse, 1./fixed_s_recp)
                # Fine points
                # [(B), N_rays, N_importance]
                d_fine = rend_util.sample_pdf(d_coarse, w_coarse, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)

            # -------- option 2: just using more points to calculate visibility weights for upsampling
            # used config: N_nograd_samples
            elif upsample_algo == 'direct_more':
                _t = torch.linspace(0, 1, N_nograd_samples).float().to(device)
                _d = near * (1 - _t) + far * _t
                _pts = rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # _sdf = model.implicit_surface.forward(_pts)
                _sdf = batchify_query(model.implicit_surface.forward, _pts)
                *_, _w = sdf_to_w(_sdf, 1./fixed_s_recp)
                d_fine = rend_util.sample_pdf(_d, _w, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)


            # -------- option 3: modified from NeuS official implementation: estimate sdf slopes and middle points' sdf
            # https://github.com/Totoro97/NeuS/blob/9dc9275d3a8c7266994a3b9cf9f36071621987dd/models/renderer.py#L131
            # used config: N_upsample_iters
            elif upsample_algo == 'official_solution':
                _d = d_coarse
                _sdf = batchify_query(model.implicit_surface.forward, rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2))
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = _sdf[..., :-1], _sdf[..., 1:]
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                    prev_dot_val = torch.cat([torch.zeros_like(dot_val[..., :1], device=device), dot_val[..., :-1]], dim=-1)   # jianfei: prev 斜率，但是往后shift了一格
                    dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)  # jianfei: prev 斜率和 斜率拼一起
                    dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)  # jianfei: 找到prev斜率和斜率里斜率最小的？（向前差分/向后差分，or 向前一段斜率/向后一段斜率）
                    dot_val = dot_val.clamp(-10.0, 0.0)
                    
                    dist = (next_z_vals - prev_z_vals)
                    prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5
                    
                    prev_cdf = cdf_Phi_s(prev_esti_sdf, 64 * (2**i))
                    next_cdf = cdf_Phi_s(next_esti_sdf, 64 * (2**i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    _w = alpha_to_w(alpha)
                    d_fine = rend_util.sample_pdf(_d, _w, N_importance // N_upsample_iters, det=not perturb)
                    _d = torch.cat([_d, d_fine], dim=-1)
                    
                    sdf_fine = batchify_query(model.implicit_surface.forward, rays_o.unsqueeze(-2) + d_fine.unsqueeze(-1) * rays_d.unsqueeze(-2))
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, DIM_BATCHIFY+1, d_sort_indices)
                d_all = _d
            else:
                raise NotImplementedError

        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # [(B), N_rays, N_pts-1, 3]
        # pts_mid = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        pts_mid = rays_o[..., None, :] + rays_d[..., None, :] * d_mid[..., :, None]

        # ------------------
        # Inside Scene
        # ------------------
        # sdf, nablas, _ = model.implicit_surface.forward_with_nablas(pts)
        sdf, nablas, _ = batchify_query(model.implicit_surface.forward_with_nablas, pts)
        # [(B), N_ryas, N_pts], [(B), N_ryas, N_pts-1]
        cdf, opacity_alpha = sdf_to_alpha(sdf, model.forward_s())
        # radiances = model.forward_radiance(pts_mid, view_dirs_mid)
        radiances = batchify_query(model.forward_radiance, pts_mid, view_dirs.unsqueeze(-2).expand_as(pts_mid) if use_view_dirs else None)

        # ------------------
        # Outside Scene
        # ------------------
        if N_outside > 0:
            _t = torch.linspace(0, 1, N_outside + 2)[..., 1:-1].float().to(device)
            d_vals_out = far / torch.flip(_t, dims=[-1])
            if perturb:
                _mids = .5 * (d_vals_out[..., 1:] + d_vals_out[..., :-1])
                _upper = torch.cat([_mids, d_vals_out[..., -1:]], -1)
                _lower = torch.cat([d_vals_out[..., :1], _mids], -1)
                _t_rand = torch.rand(_upper.shape).float().to(device)
                d_vals_out = _lower + (_upper - _lower) * _t_rand
            
            d_vals_out = torch.cat([d_mid, d_vals_out], dim=-1) # already sorted
            pts_out = rays_o[..., None, :] + rays_d[..., None, :] * d_vals_out[..., :, None]
            r = pts_out.norm(dim=-1, keepdim=True)
            x_out = torch.cat([pts_out/r, 1./r], dim=-1)
            views_out = view_dirs.unsqueeze(-2).expand_as(x_out[..., :3]) if use_view_dirs else None
            
            sigma_out, radiance_out = batchify_query(model.nerf_outside.forward, x_out, views_out)
            dists = d_vals_out[..., 1:] - d_vals_out[..., :-1]
            dists = torch.cat([dists, 1e10 * torch.ones(dists[..., :1].shape).to(device)], dim=-1)
            alpha_out = 1 - torch.exp(-F.softplus(sigma_out) * dists)   # use softplus instead of relu as NeuS's official repo
        
        # --------------
        # Ray Integration
        # --------------
        # [(B), N_rays, N_pts-1]
        if N_outside > 0:
            N_pts_1 = d_mid.shape[-1]
            # [(B), N_ryas, N_pts-1]
            mask_inside = (pts_mid.norm(dim=-1) <= obj_bounding_radius)
            # [(B), N_ryas, N_pts-1]
            alpha_in = opacity_alpha * mask_inside.float() + alpha_out[..., :N_pts_1] * (~mask_inside).float()
            # [(B), N_ryas, N_pts-1 + N_outside]
            opacity_alpha = torch.cat([alpha_in, alpha_out[..., N_pts_1:]], dim=-1)
            
            # [(B), N_ryas, N_pts-1, 3]
            radiance_in = radiances * mask_inside.float()[..., None] + radiance_out[..., :N_pts_1, :] * (~mask_inside).float()[..., None]
            # [(B), N_ryas, N_pts-1 + N_outside, 3]
            radiances = torch.cat([radiance_in, radiance_out[..., N_pts_1:, :]], dim=-2)
            d_final = d_vals_out
        else:
            d_final = d_mid

        # [(B), N_ryas, N_pts-1 + N_outside]
        visibility_weights = alpha_to_w(opacity_alpha)
        # [(B), N_rays]
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        # depth_map = torch.sum(visibility_weights * d_mid, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True)+1e-10) * d_final, -1)
        acc_map = torch.sum(visibility_weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),           # [(B), N_rays, 3]
            ('depth_volume', depth_map),     # [(B), N_rays]
            # ('depth_surface', d_pred_out),    # [(B), N_rays]
            ('mask_volume', acc_map)            # [(B), N_rays]
        ])

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
            normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)
            ret_i['normals_volume'] = normals_map

        if detailed_output:
            ret_i['implicit_nablas'] = nablas
            ret_i['implicit_surface'] = sdf
            ret_i['radiance'] = radiances
            ret_i['alpha'] = opacity_alpha
            ret_i['cdf'] = cdf
            ret_i['visibility_weights'] = visibility_weights
            ret_i['d_final'] = d_final
            if N_outside > 0:
                ret_i['sigma_out'] = sigma_out
                ret_i['radiance_out'] = radiance_out

        return ret_i
        
    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
            rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk]
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    return ret['rgb'], ret['depth_volume'], ret

class SingleRenderer(nn.Module):
    def __init__(self, model: NeuS):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: NeuS, device_ids=[0], batched=True):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]
    
    def forward(self, 
                args,
                indices,
                model_input,
                ground_truth,
                render_kwargs_train: dict,
                it: int,
                device='cuda'):

        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=args.data.N_rays)
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        else:
            mask_ignore = None

        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)

        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras['implicit_nablas']
        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)
        # [B, N_rays]
        mask_volume: torch.Tensor = extras['mask_volume']
        # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
        mask_volume = torch.clamp(mask_volume, 1e-3, 1-1e-3)
        extras['mask_volume_clipped'] = mask_volume

        losses = OrderedDict()

        # [B, N_rays, 3]
        losses['loss_img'] = F.l1_loss(rgb, target_rgb, reduction='none')
        # [B, N_rays, N_pts]
        losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

        if args.training.with_mask:
            # [B, N_rays]
            target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            losses['loss_mask'] = args.training.w_mask * F.binary_cross_entropy(mask_volume, target_mask.float(), reduction='mean')
            if mask_ignore is not None:
                target_mask = torch.logical_and(target_mask, mask_ignore)
            # [N_masked, 3]
            losses['loss_img'] = (losses['loss_img'] * target_mask[..., None].float()).sum() / (target_mask.sum()+1e-10)
        else:
            if mask_ignore is not None:
                losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
            else:
                losses['loss_img'] = losses['loss_img'].mean()

        loss = 0
        for k, v in losses.items():
            loss += losses[k]

        losses['total'] = loss
        extras['implicit_nablas_norm'] = nablas_norm
        extras['scalars'] = {'1/s': 1./self.model.forward_s().data}
        extras['select_inds'] = select_inds
        
        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])


def get_model(args):
    
    if not args.training.with_mask:
        assert 'N_outside' in args.model.keys() and args.model.N_outside > 0, \
            "Please specify a positive model:N_outside for neus with nerf++"
    
    model_config = {
        'obj_bounding_radius':  args.model.obj_bounding_radius,
        'W_geo_feat':       args.model.setdefault('W_geometry_feature', 256),
        'use_outside_nerf': not args.training.with_mask,
        'speed_factor':     args.training.setdefault('speed_factor', 1.0),
        'variance_init':    args.model.setdefault('variance_init', 0.05)
    }
    
    surface_cfg = {
        'use_siren':    args.model.surface.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.surface.setdefault('embed_multires', 6),
        'radius_init':  args.model.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.surface.setdefault('geometric_init', True),
        'D': args.model.surface.setdefault('D', 8),
        'W': args.model.surface.setdefault('W', 256),
        'skips': args.model.surface.setdefault('skips', [4]),
    }
        
    radiance_cfg = {
        'use_siren':    args.model.radiance.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.radiance.setdefault('use_view_dirs', True),
        'D': args.model.radiance.setdefault('D', 4),
        'W': args.model.radiance.setdefault('W', 256),
        'skips': args.model.radiance.setdefault('skips', []),
    }
    
    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg
    
    model = NeuS(**model_config)
    
    ## render kwargs
    render_kwargs_train = {
        # upsample config
        'upsample_algo':    args.model.setdefault('upsample_algo', 'official_solution'),    # [official_solution, direct_more, direct_use]
        'N_nograd_samples': args.model.setdefault('N_nograd_samples', 2048),
        'N_upsample_iters': args.model.setdefault('N_upsample_iters', 4),
        
        'N_outside': args.model.setdefault('N_outside', 0),
        'obj_bounding_radius': args.data.setdefault('obj_bounding_radius', 1.0),
        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    
    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer
