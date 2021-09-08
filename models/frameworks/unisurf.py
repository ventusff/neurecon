from models.ray_casting import root_finding_surface_points
from models.base import ImplicitSurface, RadianceNet
from utils import train_util, rend_util

import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Union, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNISURF(nn.Module):
    def __init__(self,
                 
                 input_ch=3,
                 W_geo_feat=-1,

                 surface_cfg=dict(),
                 radiance_cfg=dict()):
        super().__init__()
        
        self.implicit_surface = ImplicitSurface(
            input_ch=input_ch, W_geo_feat=W_geo_feat, **surface_cfg)
        
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)
    
    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        occ, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        normals = F.normalize(nablas)   # since the norm of the nablas is not guaranteed using OccNet.
        radiances = self.radiance_net.forward(x, view_dirs, normals, geometry_feature)
        return radiances, occ, nablas

    @staticmethod
    def get_surface_from_opacity(opacity: Union[torch.Tensor, np.ndarray], eps=1e-4):
        # NOTE: Directly use occupancy as opacity/alpha
        # modified from DVR. https://github.com/autonomousvision/differentiable_volumetric_rendering
        if isinstance(opacity, torch.Tensor):
            opacity = torch.clamp(opacity, min=eps, max=1-eps)
            imp_surface = torch.log(opacity / (1 - opacity))
        else:
            opacity = np.clip(opacity, a_min=eps, a_max=1-eps)
            imp_surface = np.log(opacity / (1 - opacity))
        # DVR'logits (+)inside (-)outside; logits here, (+)outside (-)inside.
        return (-1.) * imp_surface

    @staticmethod
    def get_opacity_from_surface(imp_surface: Union[torch.Tensor, np.ndarray]):
        # DVR'logits (+)inside (-)outside; logits here, (+)outside (-)inside.
        if isinstance(imp_surface, torch.Tensor):
            odds = torch.exp(-1. * imp_surface)
            opacity = odds / (1 + odds)
        else:
            odds = np.exp(-1. * imp_surface)
            opacity = odds / (1 + odds)
        return opacity

def volume_render(
    rays_o, 
    rays_d,
    model: UNISURF,
    
    batched = False,
    batched_info = {},

    # render algorithm config
    calc_normal = False,
    logit_tau = 0.0,
    use_view_dirs = True,
    method = 'secant',
    rayschunk = 65536,
    netchunk = 1048576,
    white_bkgd = False,
    near_bypass: Optional[float] = None,
    far_bypass: Optional[float] = None,

    # render function config
    detailed_output = True,
    show_progress = False,

    # sampling related
    radius_of_interest = 4.0,
    perturb = False,   # config whether do stratified sampling
    interval = 1.0,  # NOTE: this should be related to near/far
    too_close_threshold = 0.1,  # range from 0 to 1
    N_query = 64,
    N_freespace = 32,
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
    # NOTE: already normalize
    rays_d = F.normalize(rays_d, dim=-1)

    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)
    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]

        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=radius_of_interest, keepdim=False)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)
        d_threshold = near + (far - near) * too_close_threshold

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
        # find root
        d_pred_out, pt_pred, mask, mask_sign_change = root_finding_surface_points(
            model.implicit_surface, rays_o, rays_d, near=near, far=far, method=method, logit_tau=logit_tau, fill_inf=False, batched=batched)

        # [(B), N_rays]
        d_pred_out = torch.clamp(d_pred_out, min=near, max=far)
        d_upper = torch.clamp_max_(d_pred_out + interval, far)
        d_lower = torch.clamp_min_(d_pred_out - interval, near) 

        # ----------------
        # stratified sampling in the interval: range from d_lower to d_upper
        if perturb:
            t = torch.linspace(0.0, 1.0, steps=(N_query+1), device=device)
            d_samples_interval = d_lower.unsqueeze(-1) * (1-t) \
                + d_upper.unsqueeze(-1) * t
            _lower = d_samples_interval[..., :-1]
            _upper = d_samples_interval[..., 1:]
            t_rand = torch.rand([*prefix_batch, N_rays, N_query], device=device)
            d_samples_interval = _lower + (_upper - _lower) * t_rand
        else:
            t = torch.linspace(0.0, 1.0, steps=N_query, device=device)
            d_samples_interval = d_lower.unsqueeze(-1) * (1-t) \
                + d_upper.unsqueeze(-1) * t

        # ----------------
        # stratified sampling in free space: range from near to d_lower
        
        # TODO: decide which one is better for handling very small d_lower
        # OPTION_1: if d_lower is very small, set a min value
        d_lower = torch.clamp_min_(d_lower, d_threshold)
        # OPTION_2: if d_lower is very small, sample on the whole ray
        # d_lower[d_lower < d_threshold] = far

        # NOTE: for rays without surface intersection, sampling on the entire ray
        d_lower[mask_sign_change == 0] = far[mask_sign_change == 0]

        # NOTE: if d_lower is exactly zero, sample the whole ray instead. (consider that it's a common case in the early stages of training.) 
        d_lower[d_lower < 1e-10] = far[d_lower < 1e-10]

        if perturb:
            t = torch.linspace(0.0, 1.0, steps=(N_freespace+1), device=device)
            d_samples_freespace = torch.ones([*prefix_batch, N_rays, 1]).to(device) * near[..., None] * (1-t) \
                + d_lower.unsqueeze(-1) * t
            _lower = d_samples_freespace[..., :-1]
            _upper = d_samples_freespace[..., 1:]
            t_rand = torch.rand([*prefix_batch, N_rays, N_freespace], device=device)
            d_samples_freespace = _lower + (_upper - _lower) * t_rand
        else:
            t = torch.linspace(0.0, 1.0, steps=N_freespace, device=device)
            d_samples_freespace = torch.ones([*prefix_batch, N_rays, 1]).to(device) * near[..., None] * (1-t) \
                + d_lower.unsqueeze(-1) * t
        
        # -----------------
        # aggregate all d_all
        d_all = torch.cat([d_samples_freespace, d_samples_interval], dim=-1)
        d_all, _ = torch.sort(d_all, dim=-1)

        # ------------------
        # calculate points
        # [(B), N_rays, N_query+N_freespace, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        
        # -------------------
        # query network
        # N_pts = N_query + N_freespace
        
        radiances, logits, nablas = batchify_query(model.forward, pts, view_dirs.unsqueeze(-2).expand_as(pts) if use_view_dirs else None)
        
        # --------------
        # Ray Integration
        # --------------
        # [(B), N_rays, N_pts]
        opacity_alpha = model.get_opacity_from_surface(logits.squeeze(-1))
        # [(B), N_rays, N_pts+1]
        shifted_transparency = torch.cat(
            [
                torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
                1.0 - opacity_alpha + 1e-10,
            ], dim=-1)

        # [(B), N_rays, N_pts]
        visibility_weights = opacity_alpha *\
            torch.cumprod(shifted_transparency, dim=-1)[..., :-1]
        
        
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        # depth_map = torch.sum(visibility_weights * d_all, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True)+1e-10) * d_all, -1)
        acc_map = torch.sum(visibility_weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),           # [(B), N_rays, 3]
            ('depth_volume', depth_map),     # [(B), N_rays]
            ('mask_volume', acc_map)            # [(B), N_rays]
        ])

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
            normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)
            ret_i['normals_volume'] = normals_map

        if detailed_output:
            ret_i['surface_points'] = pt_pred   # [(B), N_rays, 3]
            ret_i['mask_surface'] = mask        # [(B), N_rays]
            ret_i['depth_surface'] = d_pred_out # [(B), N_rays]
            # [(B), N_rays, N_pts, ]
            ret_i['radiance'] = radiances
            ret_i['implicit_surface'] = logits
            ret_i['implicit_nablas'] = nablas
            ret_i['alpha'] = opacity_alpha
            ret_i['visibility_weights'] = visibility_weights
        
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

    # # NOTE: this is for debugging, which maintains computation graph. But not suitable for validation
    # ret = render_rayschunk(rays_o, rays_d)

    return ret['rgb'], ret['depth_volume'], ret

class SingleRenderer(nn.Module):
    def __init__(self, model: Union[UNISURF]):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: UNISURF, device_ids=[0], batched=True):
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
        # object_mask = model_input["object_mask"].to(device)
        c2w = model_input['c2w'].to(device)
        
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, render_kwargs_train['H'], render_kwargs_train['W'], N_rays=args.data.N_rays)
        
        target_rgb = torch.gather(ground_truth['rgb'].to(device), -2, torch.stack(3*[select_inds],-1)) 

        interval = max(args.training.delta_max * np.exp(-it * args.training.delta_beta), args.training.delta_min)
        
        rgb, depth_v, extras = self.renderer(rays_o, rays_d, interval=interval, detailed_output=True, **render_kwargs_train)
        
        # ----------- calculate loss
        losses = OrderedDict()
        
        losses['loss_img'] = F.l1_loss(rgb, target_rgb)

        losses['loss_reg'] = torch.tensor(0.).to(device)
        if args.training.w_reg > 0:
            pts_surface = extras['surface_points']
            _, nablas_surface, _ = self.model.implicit_surface.forward_with_nablas(pts_surface)
            pts_nablas_neighbor = pts_surface \
                + (torch.rand(pts_surface.shape, device=device) - 0.5) * 2. \
                    * args.training.perturb_surface_pts
            _, nablas_purturb, _ = self.model.implicit_surface.forward_with_nablas(pts_nablas_neighbor)
            
            # NOTE: normalizing the nablas before regularization is important for converging. Otherwise would lead to huge gradients and bad local minima.
            #       this is meanly because the nablas of the OccNet is different from SDF's nablas.
            losses['loss_reg'] = args.training.w_reg * F.mse_loss(F.normalize(nablas_purturb, dim=-1), F.normalize(nablas_surface, dim=-1))
        
        loss = 0
        for v in losses.values():
            loss += v
        losses['total'] = loss
        
        extras['scalars'] = {'interval': torch.tensor([interval]).to(device)} 
        
        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
                

def get_model(args):
    model_config = {
        'W_geo_feat': args.model.setdefault('W_geometry_feature', 256),
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
        # 'beta_init': 0.1
    }

    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg

    model = UNISURF(**model_config)

    ## render kwargs
    render_kwargs_train = {
        'batched': True,
        'tau': args.model.tau,
        'perturb': args.model.get('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.get('white_bkgd', False),
        'logit_tau': model.get_surface_from_opacity(args.model.tau),
        'radius_of_interest': args.model.obj_bounding_radius
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    
    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer



if __name__ == "__main__":
    def test():
        #--------------- random tests
        B = 7
        H = 3
        W = 4
        N = H * W
        near = 0.1
        far = 2.5
        
        device = 'cuda'
        
        rays_o = torch.randn([B, H, W, 3]).to(device)
        rays_d = torch.randn([B, H, W, 3]).to(device)
        
        # def dummy_network_fn(pts: torch.Tensor):
        #     # return torch.randn([B, N * N_steps])
        #     return torch.randn([*pts.shape[:-1]])
            
        model = UNISURF().to(device)
        
        volume_render(
            rays_o, rays_d, 
            model, 
            near=near, far=far, 
            batched = True, 
            use_view_dirs= True)
    test()