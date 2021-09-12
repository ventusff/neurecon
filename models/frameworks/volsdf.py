from models.base import ImplicitSurface, NeRF, RadianceNet
from utils import io_util, train_util, rend_util
from utils.logger import Logger

import copy
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

def sdf_to_sigma(sdf: torch.Tensor, alpha, beta):
    # sdf *= -1 # NOTE: this will cause inplace opt.
    # sdf = -sdf
    # mask = sdf <= 0
    # cond1 = 0.5 * torch.exp(sdf / beta * mask.float())  # NOTE: torch.where will introduce 0*inf = nan
    # cond2 = 1 - 0.5 * torch.exp(-sdf / beta * (1-mask.float()))
    # # psi = torch.where(sdf <= 0, 0.5 * expsbeta, 1 - 0.5 / expsbeta)   # NOTE: exploding gradient
    # psi = torch.where(mask, cond1, cond2)
    
    """
    @ Section 3.1 in the paper. From sdf:d_{\Omega} to nerf's density:\sigma.
    work with arbitrary shape prefixes.
        sdf:    [...]
        
    """
    # -sdf when sdf > 0, sdf when sdf < 0
    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    psi = torch.where(sdf >= 0, exp, 1 - exp)

    return alpha * psi


def error_bound(d_vals, sdf, alpha, beta):
    """
    @ Section 3.3 in the paper. The error bound of a specific sampling.
    work with arbitrary shape prefixes.
    [..., N_pts] forms [..., N_pts-1] intervals, hence producing [..., N_pts-1] error bounds.
    Args:
        d_vals: [..., N_pts]
        sdf:    [..., N_pts]
    Return:
        bounds: [..., N_pts-1]
    """
    device = sdf.device
    sigma = sdf_to_sigma(sdf, alpha, beta)
    # [..., N_pts]
    sdf_abs_i = torch.abs(sdf)
    # [..., N_pts-1]
    # delta_i = (d_vals[..., 1:] - d_vals[..., :-1]) * rays_d.norm(dim=-1)[..., None]
    delta_i = d_vals[..., 1:] - d_vals[..., :-1]    # NOTE: already real depth
    # [..., N_pts-1]. R(t_k) of the starting point of the interval.
    R_t = torch.cat(
        [
            torch.zeros([*sdf.shape[:-1], 1], device=device), 
            torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
        ], dim=-1)[..., :-1]
    # [..., N_pts-1]
    d_i_star = torch.clamp_min(0.5 * (sdf_abs_i[..., :-1] + sdf_abs_i[..., 1:] - delta_i), 0.)
    # [..., N_pts-1]
    errors = alpha/(4*beta) * (delta_i**2) * torch.exp(-d_i_star / beta)
    # [..., N_pts-1]. E(t_{k+1}) of the ending point of the interval.
    errors_t = torch.cumsum(errors, dim=-1)
    # [..., N_pts-1]
    bounds = torch.exp(-R_t) * (torch.exp(errors_t) - 1.)
    # TODO: better solution
#     # NOTE: nan comes from 0 * inf
#     # NOTE: every situation where nan appears will also appears c * inf = "true" inf, so below solution is acceptable
    bounds[torch.isnan(bounds)] = np.inf
    return bounds


def fine_sample(implicit_surface_fn, init_dvals, rays_o, rays_d, 
                alpha_net, beta_net, far, 
                eps=0.1, max_iter:int=5, max_bisection:int=10, final_N_importance:int=64, N_up:int=128,
                perturb=True):
    """
    @ Section 3.4 in the paper.
    Args:
        implicit_surface_fn. sdf query function.
        init_dvals: [..., N_rays, N]
        rays_o:     [..., N_rays, 3]
        rays_d:     [..., N_rays, 3]
    Return:
        final_fine_dvals:   [..., N_rays, final_N_importance]
        beta:               [..., N_rays]. beta heat map
    """
    # NOTE: this algorithm is parallelized for every ray!!!
    with torch.no_grad():
        device = init_dvals.device
        prefix = init_dvals.shape[:-1]
        d_vals = init_dvals
        
        def query_sdf(d_vals_, rays_o_, rays_d_):
            pts = rays_o_[..., None, :] + rays_d_[..., None, :] * d_vals_[..., :, None]
            return implicit_surface_fn(pts)

        def opacity_invert_cdf_sample(d_vals_, sdf_, alpha_, beta_, N_importance=final_N_importance, det=not perturb):
            #-------------- final: d_vals, sdf, beta_net, alpha_net
            sigma = sdf_to_sigma(sdf_, alpha_, beta_)
            # bounds = error_bound(d_vals_, sdf_, alpha_net, beta_net)
            # delta_i = (d_vals_[..., 1:] - d_vals_[..., :-1]) * rays_d_.norm(dim=-1)[..., None]
            delta_i = d_vals_[..., 1:] - d_vals_[..., :-1]  # NOTE: already real depth
            R_t = torch.cat(
                [
                    torch.zeros([*sdf_.shape[:-1], 1], device=device), 
                    torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
                ], dim=-1)[..., :-1]
            #-------------- a fresh set of \hat{O}
            opacity_approx = 1 - torch.exp(-R_t)
            fine_dvals = rend_util.sample_cdf(d_vals_, opacity_approx, N_importance, det=det)
            return fine_dvals

        # final output storage.
        # being updated during the iterations of the algorithm
        final_fine_dvals = torch.zeros([*prefix, final_N_importance]).to(device)
        final_iter_usage = torch.zeros([*prefix]).to(device)

        #---------------- 
        # init beta+
        #---------------- 
        # [*prefix, 1]
        if not isinstance(far, torch.Tensor):
            far = far * torch.ones([*prefix, 1], device=device)
        beta = torch.sqrt((far**2) / (4 * (init_dvals.shape[-1]-1) * np.log(1+eps)))
        alpha = 1./beta
        # alpha = alpha_net
        # [*prefix, N]

        #---------------- 
        # first check of bound using network's current beta: B_{\mathcal{\tau}, \beta}
        #---------------- 
        # [*prefix]
        sdf = query_sdf(d_vals, rays_o, rays_d)
        net_bounds_max = error_bound(d_vals, sdf, alpha_net, beta_net).max(dim=-1).values
        mask = net_bounds_max > eps
        
        #---------------- 
        # first bound using beta+ : B_{\mathcal{\tau}, \beta_+}
        # [*prefix, N-1]
        bounds = error_bound(d_vals, sdf, alpha, beta)
        bounds_masked = bounds[mask]
        # NOTE: true for ANY ray that satisfy eps condition in the whole process
        final_converge_flag = torch.zeros([*prefix], device=device, dtype=torch.bool)
        
        # NOTE: these are the final fine sampling points for those rays that satisfy eps condition at the very beginning.
        if (~mask).sum() > 0:
            final_fine_dvals[~mask] = opacity_invert_cdf_sample(d_vals[~mask], sdf[~mask], alpha_net, beta_net)
            final_iter_usage[~mask] = 0
        final_converge_flag[~mask] = True
        
        cur_N = init_dvals.shape[-1]
        it_algo = 0
        #---------------- 
        # start algorithm
        #---------------- 
        while it_algo < max_iter:
            it_algo += 1
            #-----------------
            # the rays that not yet converged
            if mask.sum() > 0:
                #---------------- 
                # upsample the samples: \mathcal{\tau} <- upsample
                #---------------- 
                # [Masked, N_up]
                # NOTE: det = True should be more robust, forcing sampling points to be proportional with error bounds.
                # upsampled_d_vals_masked = rend_util.sample_pdf(d_vals[mask], bounds_masked, N_up, det=True)
                # NOTE: when using det=True, the head and the tail d_vals will always be appended, hence removed using [..., 1:-1]
                upsampled_d_vals_masked = rend_util.sample_pdf(d_vals[mask], bounds_masked, N_up+2, det=True)[..., 1:-1]
                
                # NOTE: for debugging
                # import matplotlib.pyplot as plt
                # ind = 0   # NOTE: this might not be the same ray as the 0-th rays may already converge before it reaches max_iter
                # fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15, 15))
                # ax1.plot(d_vals[mask][ind].cpu(), sdf[mask][ind].cpu(), label='sdf')
                # ax1.legend()
                # ax2.step(d_vals[mask][ind].cpu()[..., :-1], bounds_masked[ind].cpu(), label='error bounds')
                # # ax2.step(d_vals[0].cpu()[..., :-1], error, label='error')
                # ax2.scatter(upsampled_d_vals_masked[ind].cpu(), y=np.zeros([N_up]), label='up sample')
                # ax2.legend()
                # ax2.set_title("it={}, beta_net={}, beta={:.3f}".format(it_algo, beta_net, beta[mask][ind].item()))
                # plt.show()
                
                d_vals = torch.cat([d_vals, torch.zeros([*prefix, N_up]).to(device)], dim=-1)
                sdf = torch.cat([sdf, torch.zeros([*prefix, N_up]).to(device)], dim=-1)
                # NOTE. concat and sort. work with any kind of dims of mask.
                d_vals_masked = d_vals[mask]
                sdf_masked = sdf[mask]
                d_vals_masked[..., cur_N:cur_N+N_up] = upsampled_d_vals_masked
                d_vals_masked, sort_indices_masked = torch.sort(d_vals_masked, dim=-1)
                sdf_masked[..., cur_N:cur_N+N_up] = query_sdf(upsampled_d_vals_masked, rays_o[mask], rays_d[mask])
                sdf_masked = torch.gather(sdf_masked, dim=-1, index=sort_indices_masked)
                d_vals[mask] = d_vals_masked
                sdf[mask] = sdf_masked
                # NOTE: another version of the above. only work with 1-dim mask.
                # d_vals[mask, cur_N:cur_N+N_up] = upsampled_d_vals_masked
                # d_vals[mask, :cur_N+N_up], sort_indices_masked = torch.sort(d_vals[mask, :cur_N+N_up], dim=-1)
                # sdf[mask, cur_N:cur_N+N_up] = query_sdf(upsampled_d_vals_masked, rays_o[mask], rays_d[mask])
                # sdf[mask, :cur_N+N_up] = torch.gather(sdf[mask, :cur_N+N_up], dim=-1, index=sort_indices_masked)
                cur_N += N_up

                #---------------- 
                # after upsample, check the bound using network's current beta: B_{\mathcal{\tau}, \beta}
                #---------------- 
                # NOTE: for the same iteration, the number of points of input rays are the same, (= cur_N), so they can be handled parallelized. 
                net_bounds_max[mask] = error_bound(d_vals[mask], sdf[mask], alpha_net, beta_net).max(dim=-1).values
                # NOTE: mask for those rays that still remains > eps after upsampling. 
                sub_mask_of_mask = net_bounds_max[mask] > eps
                # mask-the-mask approach. below 3 lines: final_converge_flag[mask][~sub_mask_of_mask] = True (this won't work in python)
                converged_mask = mask.clone()
                converged_mask[mask] = ~sub_mask_of_mask
                
                # NOTE: these are the final fine sampling points for those rays that >eps originally but <eps after upsampling.
                if converged_mask.sum() > 0:
                    final_converge_flag[converged_mask] = True
                    final_fine_dvals[converged_mask] = opacity_invert_cdf_sample(d_vals[converged_mask], sdf[converged_mask], alpha_net, beta_net)
                    final_iter_usage[converged_mask] = it_algo
                #---------------- 
                # using bisection method to find the new beta+ s.t. B_{\mathcal{\tau}, \beta+}==eps
                #---------------- 
                if (sub_mask_of_mask).sum() > 0:
                    # mask-the-mask approach
                    new_mask = mask.clone()
                    new_mask[mask] = sub_mask_of_mask
                    # [Submasked, 1]
                    beta_right = beta[new_mask]
                    beta_left = beta_net * torch.ones_like(beta_right, device=device)
                    d_vals_tmp = d_vals[new_mask]
                    sdf_tmp = sdf[new_mask]
                    #---------------- 
                    # Bisection iterations
                    for _ in range(max_bisection):
                        beta_tmp = 0.5 * (beta_left + beta_right)
                        alpha_tmp = 1./beta_tmp
                        # alpha_tmp = alpha_net
                        # [Submasked]
                        bounds_tmp_max = error_bound(d_vals_tmp, sdf_tmp, alpha_tmp, beta_tmp).max(dim=-1).values
                        beta_right[bounds_tmp_max <= eps] = beta_tmp[bounds_tmp_max <= eps]
                        beta_left[bounds_tmp_max > eps] = beta_tmp[bounds_tmp_max > eps]
                    beta[new_mask] = beta_right
                    alpha[new_mask] = 1./beta[new_mask]
                    
                    #---------------- 
                    # after upsample, the remained rays that not yet converged.
                    #---------------- 
                    bounds_masked = error_bound(d_vals_tmp, sdf_tmp, alpha[new_mask], beta[new_mask])
                    # bounds_masked = error_bound(d_vals_tmp, rays_d_tmp, sdf_tmp, alpha_net, beta[new_mask])
                    bounds_masked = torch.clamp(bounds_masked, 0, 1e5)  # NOTE: prevent INF caused NANs
                    
                    # mask = net_bounds_max > eps   # NOTE: the same as the following
                    mask = new_mask
                else:
                    break
            else:
                break
        
        #---------------- 
        # for rays that still not yet converged after max_iter, use the last beta+
        #---------------- 
        if (~final_converge_flag).sum() > 0:
            beta_plus = beta[~final_converge_flag]
            alpha_plus = 1./beta_plus
            # alpha_plus = alpha_net
            # NOTE: these are the final fine sampling points for those rays that still remains >eps in the end. 
            final_fine_dvals[~final_converge_flag] = opacity_invert_cdf_sample(d_vals[~final_converge_flag], sdf[~final_converge_flag], alpha_plus, beta_plus)
            final_iter_usage[~final_converge_flag] = -1
        beta[final_converge_flag] = beta_net
        return final_fine_dvals, beta, final_iter_usage

class VolSDF(nn.Module):
    def __init__(self,
                 beta_init=0.1,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=-1,
                 obj_bounding_radius=3.0,
                 use_nerfplusplus=False,

                 surface_cfg=dict(),
                 radiance_cfg=dict()):
        super().__init__()
        
        self.speed_factor = speed_factor
        ln_beta_init = np.log(beta_init) / self.speed_factor
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)
        # self.beta = nn.Parameter(data=torch.Tensor([beta_init]), requires_grad=True)

        self.use_sphere_bg = not use_nerfplusplus
        self.obj_bounding_radius = obj_bounding_radius
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)

        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)

        if use_nerfplusplus:
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)

    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def forward_surface(self, x: torch.Tensor):
        sdf = self.implicit_surface.forward(x)
        if self.use_sphere_bg:
            return torch.min(sdf, self.obj_bounding_radius - x.norm(dim=-1))
        else:
            return sdf        

    def forward_surface_with_nablas(self, x: torch.Tensor):
        sdf, nablas, h = self.implicit_surface.forward_with_nablas(x)
        if self.use_sphere_bg:
            d_bg = self.obj_bounding_radius - x.norm(dim=-1)
            # outside_sphere = x_norm >= 3
            outside_sphere = d_bg < sdf # NOTE: in case the normals changed suddenly near the sphere.
            sdf[outside_sphere] = d_bg[outside_sphere]
            # nabla[outside_sphere] = normals_bg_sphere[outside_sphere] # ? NOTE: commented to ensure more eikonal constraints. 
        return sdf, nablas, h

    def forward(self, x:torch. Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = \
            self.forward_surface_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiances, sdf, nablas


def volume_render(
    rays_o, 
    rays_d,
    model: VolSDF,
    
    near=0.0,
    far=6.0,
    obj_bounding_radius=3.0,
    
    batched = False,
    batched_info = {},
    
    # render algorithm config
    calc_normal = False,
    use_view_dirs = True,
    rayschunk = 65536,
    netchunk = 1048576,
    white_bkgd = False,
    use_nerfplusplus = False,
    
    # render function config
    detailed_output = True,
    show_progress = False,
    
    # sampling related
    perturb = False,   # config whether do stratified sampling
    N_samples = 128,
    N_importance = 64,
    N_outside = 32,
    max_upsample_steps = 5,
    max_bisection_steps = 10,
    epsilon = 0.1,
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
        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None
        
        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]
        
        nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        if use_nerfplusplus:
            _, fars, mask_intersect = rend_util.get_sphere_intersection(rays_o, rays_d, r=obj_bounding_radius)
            assert mask_intersect.all()
        else:
            fars = far * torch.ones([*prefix_batch, N_rays, 1]).to(device)

        # ---------------
        # Sample points on the rays
        # ---------------
        
        # ---------------
        # Coarse Points
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        # [(B), N_rays, N_samples]
        d_coarse = nears * (1 - _t) + fars * _t
        
        # ---------------
        # Fine sampling algorithm
        alpha, beta = model.forward_ab()
        with torch.no_grad():
            # d_init = d_coarse
            
            # NOTE: setting denser d_init boost up up_sampling convergence without sacrificing much speed (since no grad here.)
            _t = torch.linspace(0, 1, N_samples*4).float().to(device) # NOTE: you might want to use less samples for faster training.
            d_init = nears * (1 - _t) + fars * _t
            
            d_fine, beta_map, iter_usage = fine_sample(
                model.forward_surface, d_init, rays_o, rays_d, 
                alpha_net=alpha, beta_net=beta, far=fars, 
                eps=epsilon, max_iter=max_upsample_steps, max_bisection=max_bisection_steps, 
                final_N_importance=N_importance, perturb=perturb, 
                N_up=N_samples*4    # NOTE: you might want to use less samples for faster training.
            )

        # ---------------
        # Gather points
        # NOTE: from the paper, should not concatenate here; 
        # NOTE: but from practice, as long as not concatenating and only using fine points, 
        #       there would be artifact emerging very fast before 10k iters, and the network converged to severe local minima (all cameras inside surface).
        d_all = torch.cat([d_coarse, d_fine], dim=-1)
        d_all, _ = torch.sort(d_all, dim=-1)
        # d_all = d_fine
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        
        # ---------------
        # Qeury network
        # [(B), N_rays, N_pts, 3],   # [(B), N_rays, N_pts]   [(B), N_rays, N_pts, W_geo]
        radiances, sdf, nablas = batchify_query(model.forward, pts, view_dirs.unsqueeze(-2).expand_as(pts) if use_view_dirs else None)
        # [(B), N_rays, N_pts]
        sigma = sdf_to_sigma(sdf, alpha, beta)

        # ---------------
        # NeRF++
        if use_nerfplusplus:
            _t = torch.linspace(0, 1, N_outside + 2)[..., 1:-1].float().to(device)
            rs = obj_bounding_radius / torch.flip(_t, dims=[-1])
            rs = rs.expand([*rays_o.shape[:-1], N_outside])
            if perturb:
                _mids = .5 * (rs[..., 1:] + rs[..., :-1])
                _upper = torch.cat([_mids, rs[..., -1:]], -1)
                _lower = torch.cat([rs[..., :1], _mids], -1)
                _t_rand = torch.rand(_upper.shape).float().to(device)
                rs = _lower + (_upper - _lower) * _t_rand
            d_out = rend_util.get_dvals_from_radius(rays_o, rays_d, rs)
            pts_out = rays_o[..., None, :] + rays_d[..., None, :] * d_out[..., :, None]
            x_out = torch.cat([pts_out/rs[..., None], 1./rs[..., None]], dim=-1)
            sigma_out, radiance_out = batchify_query(model.nerf_outside.forward, x_out, view_dirs.unsqueeze(-2).expand_as(pts_out) if use_view_dirs else None)

            # ---------------
            # Gather all input
            d_all = torch.cat([d_all, d_out], dim=-1)   # already sorted
            sigma = torch.cat([sigma, sigma_out], dim=-1)
            radiances = torch.cat([radiances, radiance_out], dim=-2)
            
        # ---------------
        # Ray integration
        # ---------------
        # [(B), N_rays, N_pts-1]
        # delta_i = (d_all[..., 1:] - d_all[..., :-1]) * rays_d.norm(dim=-1)[..., None]
        delta_i = d_all[..., 1:] - d_all[..., :-1]  # NOTE: aleardy real depth
        # [(B), N_rays, N_pts-1]
        p_i = torch.exp(-F.relu_(sigma[..., :-1] * delta_i))
        # [(B), N_rays, N_pts-1]
        # (1-p_i) * \prod_{j=1}^{i-1} p_j
        # NOTE: NOT (1-pi) * torch.cumprod(p_i)! the cumprod later should use shifted p_i! 
        #       because the cumprod ends to (i-1), not i.
        tau_i = (1 - p_i + 1e-10) * (
            torch.cumprod(
                torch.cat(
                    [torch.ones([*p_i.shape[:-1], 1], device=device), p_i], dim=-1), 
                dim=-1)[..., :-1]
            )
        # [(B), N_rays, 3]
        rgb_map = torch.sum(tau_i[..., None] * radiances[..., :-1, :], dim=-2)
        # [(B), N_rays, 1]
        depth_map = torch.sum(tau_i / (tau_i.sum(-1, keepdim=True)+1e-10) * d_all[..., :-1], dim=-1)
        acc_map = torch.sum(tau_i, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),           # [(B), N_rays, 3]
            ('depth_volume', depth_map),     # [(B), N_rays]
            ('mask_volume', acc_map)            # [(B), N_rays]
        ])

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(tau_i.shape[-1], normals_map.shape[-2])
            normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
            ret_i['normals_volume'] = normals_map

        if detailed_output:
            # [(B), N_rays, N_pts, ]
            ret_i['implicit_surface'] = sdf
            ret_i['implicit_nablas'] = nablas
            ret_i['radiance'] = radiances
            ret_i['alpha'] = 1.0 - p_i
            ret_i['p_i'] = p_i
            ret_i['visibility_weights'] = tau_i
            ret_i['d_vals'] = d_all
            ret_i['sigma'] = sigma
            # [(B), N_rays, ]
            ret_i['beta_map'] = beta_map
            ret_i['iter_usage'] = iter_usage
            if use_nerfplusplus:
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
    
    # # NOTE: this is for debugging, which maintains computation graph. But not suitable for validation
    # ret = render_rayschunk(rays_o, rays_d)

    return ret['rgb'], ret['depth_volume'], ret


class SingleRenderer(nn.Module):
    def __init__(self, model: VolSDF):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: VolSDF, device_ids=[0], batched=True):
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
             it: int):
        device = self.device
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=args.data.N_rays)
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))
        # [B, N_rays]
        # target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds)

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        else:
            mask_ignore = None
        
        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)

        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras['implicit_nablas']
        
        # [B, N_rays, ]
        #---------- OPTION1: just flatten and use all nablas
        # nablas = nablas.flatten(-3, -2)
        
        #---------- OPTION2: using only one point each ray: this may be what the paper suggests.
        # @ VolSDF section 3.5, "combine a SINGLE random uniform space point and a SINGLE point from \mathcal{S} for each pixel"
        _, _ind = extras['visibility_weights'][..., :nablas.shape[-2]].max(dim=-1)
        nablas = torch.gather(nablas, dim=-2, index=_ind[..., None, None].repeat([*(len(nablas.shape)-1)*[1], 3]))
        
        eik_bounding_box = args.model.obj_bounding_radius
        eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).to(device)
        _, nablas_eik, _ = self.model.implicit_surface.forward_with_nablas(eikonal_points)
        nablas = torch.cat([nablas, nablas_eik], dim=-2)

        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)

        losses = OrderedDict()

        losses['loss_img'] = F.l1_loss(rgb, target_rgb, reduction='none')
        losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

        if mask_ignore is not None:
            losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
        else:
            losses['loss_img'] = losses['loss_img'].mean()

        loss = 0
        for k, v in losses.items():
            loss += losses[k]
        
        losses['total'] = loss
        
        extras['implicit_nablas_norm'] = nablas_norm

        alpha, beta = self.model.forward_ab()
        alpha = alpha.data
        beta = beta.data
        extras['scalars'] = {'beta': beta, 'alpha': alpha}
        extras['select_inds'] = select_inds

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
        

    def val(self, logger: Logger, ret, to_img_fn, it, render_kwargs_test):
        #----------- plot beta heat map
        beta_heat_map = to_img_fn(ret['beta_map']).permute(0, 2, 3, 1).data.cpu().numpy()
        beta_heat_map = io_util.gallery(beta_heat_map, int(np.sqrt(beta_heat_map.shape[0])))
        _, beta = self.model.forward_ab()
        beta = beta.data.cpu().numpy().item()
        # beta_min = beta_heat_map.min()
        beta_max = beta_heat_map.max().item()
        if beta_max != beta:
            ticks = np.linspace(beta, beta_max, 10).tolist()
        else:
            ticks = [beta]
        tick_labels = ["{:.4f}".format(b) for b in ticks]
        tick_labels[0] = "beta={:.4f}".format(beta)
        
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax_im = ax.imshow(beta_heat_map, vmin=beta, vmax=beta_max)
        cbar = fig.colorbar(ax_im, ticks=ticks)
        cbar.ax.set_yticklabels(tick_labels)
        logger.add_figure(fig, 'val/beta_heat_map', it)
        
        #----------- plot iteration used for each ray
        max_iter = render_kwargs_test['max_upsample_steps']
        iter_usage_map = to_img_fn(ret['iter_usage'].unsqueeze(-1)).permute(0, 2, 3, 1).data.cpu().numpy()
        iter_usage_map = io_util.gallery(iter_usage_map, int(np.sqrt(iter_usage_map.shape[0])))
        iter_usage_map[iter_usage_map==-1] = max_iter+1
        
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax_im = ax.imshow(iter_usage_map, vmin=0, vmax=max_iter+1)
        ticks = list(range(max_iter+2))
        tick_labels = ["{:d}".format(b) for b in ticks]
        tick_labels[-1] = 'not converged'
        cbar = fig.colorbar(ax_im, ticks=ticks)
        cbar.ax.set_yticklabels(tick_labels)
        logger.add_figure(fig, 'val/upsample_iters', it)

def get_model(args):
    model_config = {
        'use_nerfplusplus': args.model.setdefault('outside_scene', 'builtin') == 'nerf++',
        'obj_bounding_radius': args.model.obj_bounding_radius,
        'W_geo_feat': args.model.setdefault('W_geometry_feature', 256),
        'speed_factor': args.training.setdefault('speed_factor', 1.0),
        'beta_init': args.training.setdefault('beta_init', 0.1)
    }
    
    surface_cfg = {
        'use_siren': args.model.surface.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.surface.setdefault('embed_multires', 6),
        'radius_init':  args.model.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.surface.setdefault('geometric_init', True),
        'D': args.model.surface.setdefault('D', 8),
        'W': args.model.surface.setdefault('W', 256),
        'skips': args.model.surface.setdefault('skips', [4]),
    }
        
    radiance_cfg = {
        'use_siren': args.model.radiance.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.radiance.setdefault('use_view_dirs', True),
        'D': args.model.radiance.setdefault('D', 4),
        'W': args.model.radiance.setdefault('W', 256),
        'skips': args.model.radiance.setdefault('skips', []),
    }
    
    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg

    model = VolSDF(**model_config)
    
    ## render_kwargs
    render_kwargs_train = {
        'near': args.data.near,
        'far': args.data.far,
        'batched': True,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
        'max_upsample_steps': args.model.setdefault('max_upsample_iter', 5),
        'use_nerfplusplus': args.model.setdefault('outside_scene', 'builtin') == 'nerf++',
        'obj_bounding_radius': args.model.obj_bounding_radius
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    
    trainer = Trainer(model, args.device_ids, batched=render_kwargs_train['batched'])
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer
