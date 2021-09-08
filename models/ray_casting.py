from models.base import ImplicitSurface

import numpy as np
from tqdm import tqdm
from typing import Union
from collections import OrderedDict

import torch
import torch.nn.functional as F

def run_secant_method(f_low, f_high, d_low, d_high, 
                        rays_o_masked, rays_d_masked,
                        implicit_surface_query_fn,
                        n_secant_steps, logit_tau):
    d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    for i in range(n_secant_steps):
        p_mid = rays_o_masked + d_pred.unsqueeze(-1) * rays_d_masked
        with torch.no_grad():
            # TODO: needs special design in here when the number of rays in each batch is different.
            f_mid = implicit_surface_query_fn(p_mid).squeeze(-1) - logit_tau
        ind_low = f_mid < 0
        if ind_low.sum() > 0:
            d_low[ind_low] = d_pred[ind_low]
            f_low[ind_low] = f_mid[ind_low]
        if (ind_low == 0).sum() > 0:
            d_high[ind_low == 0] = d_pred[ind_low == 0]
            f_high[ind_low == 0] = f_mid[ind_low == 0]

        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    return d_pred

def run_bisection_method():
    pass

def root_finding_surface_points(
    surface_query_fn,
    rays_o: torch.Tensor, rays_d: torch.Tensor, 
    near: Union[float, torch.Tensor]=0.0, 
    far: Union[float, torch.Tensor]=6.0,
    # function config
    batched = True, 
    batched_info = {},
    # algorithm config
    N_steps = 256,
    logit_tau=0.0, 
    method='secant',
    N_secant_steps = 8,
    fill_inf=True,
    ):
    """
    rays_o: [(B), N_rays, 3]
    rays_d: [(B), N_rays, 3]
    near: float or [(B), N_rays]
    far: float or [(B), N_rays]
    """
    # NOTE: jianfei: modified from DVR. https://github.com/autonomousvision/differentiable_volumetric_rendering
    # NOTE: DVR'logits (+)inside (-)outside; logits here, (+)outside (-)inside.
    # NOTE: rays_d needs to be already normalized
    with torch.no_grad():
        device = rays_o.device
        if not batched:
            rays_o.unsqueeze_(0)
            rays_d.unsqueeze_(0)

        B = rays_o.shape[0]
        N_rays = rays_o.shape[-2]
        
        # [B, N_rays, N_steps, 1]
        t = torch.linspace(0., 1., N_steps, device=device)[None, None, :]
        if not isinstance(near, torch.Tensor):
            near = near * torch.ones(rays_o.shape[:-1], device=device)
        if not isinstance(far, torch.Tensor):
            far = far * torch.ones(rays_o.shape[:-1], device=device)
        d_proposal = near[..., None] * (1-t) + far[..., None] * t
        
        # [B, N_rays, N_steps, 3]
        p_proposal = rays_o.unsqueeze(-2) + d_proposal.unsqueeze(-1) * rays_d.unsqueeze(-2)
    
        # only query sigma
        pts = p_proposal
        
        # query network
        # [B, N_rays, N_steps]
        val = surface_query_fn(pts)
        # [B, N_rays, N_steps]
        val = val - logit_tau   # centered at zero
        
        # mask: the first point is not occupied
        # [B, N_rays]
        mask_0_not_occupied = val[..., 0] > 0
        
        # [B, N_rays, N_steps-1]
        sign_matrix = torch.cat(
            [
                torch.sign(val[..., :-1] * val[..., 1:]),   # [B, N, N_steps-1]
                torch.ones([B, N_rays, 1], device=device) # [B, N, 1]
            ], dim=-1)

        # [B, N_rays, N_steps-1]
        cost_matrix = sign_matrix * torch.arange(N_steps, 0, -1).float().to(device)
        
        values, indices = torch.min(cost_matrix, -1)
        
        # mask: at least one sign change occured
        mask_sign_change = values < 0
        
        # mask: whether the first sign change is from pos to neg (outside surface into the surface)
        mask_pos_to_neg = val[torch.arange(B).unsqueeze(-1), torch.arange(N_rays).unsqueeze(0), indices] > 0
        
        mask = mask_sign_change & mask_pos_to_neg & mask_0_not_occupied
        
        #--------- secant method
        # [B*N_rays, N_steps, 1]
        d_proposal_flat = d_proposal.view([B*N_rays, N_steps, 1])
        val_flat = val.view([B*N_rays, N_steps, 1])
        N_secant = d_proposal_flat.shape[0]
        
        # [N_masked]
        d_high = d_proposal_flat[torch.arange(N_secant), indices.view(N_secant)].view([B, N_rays])[mask]
        f_high = val_flat[torch.arange(N_secant), indices.view(N_secant)].view([B, N_rays])[mask]
        
        indices = torch.clamp(indices + 1, max=N_steps - 1)
        d_low = d_proposal_flat[torch.arange(N_secant), indices.view(N_secant)].view([B, N_rays])[mask]
        f_low = val_flat[torch.arange(N_secant), indices.view(N_secant)].view([B, N_rays])[mask]
        
        # [N_masked, 3]
        rays_o_masked = rays_o[mask]
        rays_d_masked = rays_d[mask]
        
        # TODO: for categorical representation, mask latents here
        
        if method == 'secant' and mask.sum() > 0:
            d_pred = run_secant_method(
                f_low, f_high, d_low, d_high, 
                rays_o_masked, rays_d_masked,
                surface_query_fn,
                N_secant_steps, logit_tau)
        else:
            d_pred = torch.ones(rays_o_masked.shape[0]).to(device)
        
        # for sanity
        pt_pred = torch.ones([B, N_rays, 3]).to(device)
        pt_pred[mask] = rays_o_masked + d_pred.unsqueeze(-1) * rays_d_masked
        
        d_pred_out = torch.ones([B, N_rays]).to(device)
        d_pred_out[mask] = d_pred
        
        # Insert appropriate values for points where no depth is predicted
        if isinstance(far, torch.Tensor):
            far = far[mask == 0]
        d_pred_out[mask == 0] = np.inf if fill_inf else far # no intersections; or the first intersection is from outside to inside; or the 0-th point is occupied.
        d_pred_out[mask_0_not_occupied == 0] = 0    # if the 0-th point is occupied, the depth should be 0.
    
        if not batched:
            d_pred_out.squeeze_(0)
            pt_pred.squeeze_(0)
            mask.squeeze_(0)
            mask_sign_change.squeeze_(0)

    return d_pred_out, pt_pred, mask, mask_sign_change


def sphere_tracing_surface_points(
    implicit_surface: ImplicitSurface, 
    rays_o, rays_d, 
    # function config
    near=0.0,
    far=6.0,
    batched = True, 
    batched_info = {},
    # algorithm config
    N_iters = 20,
    ):
    device = rays_o.device
    d_preds = torch.ones([*rays_o.shape[:-1]], device=device) * near
    mask = torch.ones_like(d_preds, dtype=torch.bool, device=device)
    for _ in range(N_iters):
        pts = rays_o + rays_d * d_preds[..., :, None]
        surface_val = implicit_surface.forward(pts)
        d_preds[mask] += surface_val[mask]
        mask[d_preds > far] = False
        mask[d_preds < 0] = False
    pts = rays_o + rays_d * d_preds[..., :, None]
    return d_preds, pts, mask


def surface_render(rays_o: torch.Tensor, rays_d: torch.Tensor,
                   model, 
                   calc_normal=True,
                   rayschunk=8192, netchunk=1048576, batched=True, use_view_dirs=True, show_progress=False,
                   ray_casting_algo='',
                   ray_casting_cfgs={},
                   **not_used_kwargs):
    """
    input: 
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    with torch.no_grad():        
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

        # ---------------
        # Render a ray chunk
        # ---------------
        def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
            if use_view_dirs:
                view_dirs = rays_d
            else:
                view_dirs = None
            if ray_casting_algo == 'root_finding':
                d_pred_out, pt_pred, mask, *_ = root_finding_surface_points(
                    model.implicit_surface, rays_o, rays_d, batched=batched, **ray_casting_cfgs)
            elif ray_casting_algo == 'sphere_tracing':
                d_pred_out, pt_pred, mask = sphere_tracing_surface_points(
                    model.implicit_surface, rays_o, rays_d, batched=batched, **ray_casting_cfgs)
            else:
                raise NotImplementedError
            
            color, _, nablas = model.forward(pt_pred, view_dirs)
            color[~mask] = 0    # black
            # NOTE: all without grad. especially for nablas.
            return color.data, d_pred_out.data, nablas.data, mask.data
        
        colors = []
        depths = []
        nablas = []
        masks = []
        for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
            color_i, d_i, nablas_i, mask_i = render_rayschunk(
                rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
                rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk]
            )
            colors.append(color_i)
            depths.append(d_i)
            nablas.append(nablas_i)
            masks.append(mask_i)
        colors = torch.cat(colors, DIM_BATCHIFY)
        depths = torch.cat(depths, DIM_BATCHIFY)
        nablas = torch.cat(nablas, DIM_BATCHIFY)
        masks = torch.cat(masks, DIM_BATCHIFY)
        
        extras = OrderedDict([
            ('implicit_nablas', nablas),
            ('mask_surface', masks)
        ])

        if calc_normal:
            normals = F.normalize(nablas, dim=-1)
            normals[~masks] = 0 # grey (/2.+0.5)
            extras['normals_surface'] = normals

        return colors, depths, extras
    