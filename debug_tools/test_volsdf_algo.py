from models.frameworks import get_model
from models.frameworks.volsdf import error_bound, sdf_to_sigma
from utils import io_util, rend_util

import torch
import numpy as np
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load_pt", type=str, default=None)
parser.add_argument("--config", type=str, default=None)
# play yourself!
parser.add_argument("--beta_net", type=float, default=0.003) 
parser.add_argument("--init_num", type=int, default=128) 
parser.add_argument("--eps", type=float, default=0.1) 
parser.add_argument("--far", type=float, default=6.0) 
parser.add_argument("--max_iter", type=int, default=5)
args = parser.parse_args()
args, unknown = parser.parse_known_args()

if args.config is not None:
    config = io_util.load_yaml(args.config)
    other_dict = vars(args)
    other_dict.pop("config")
    config.update(other_dict)
    args = config

    model, trainer, render_kwargs_train, render_kwargs_test, volume_render_fn = get_model(args)
    model.cuda()
    if args.load_pt is not None:
        state_dict = torch.load(args.load_pt, map_location='cuda')
        model.load_state_dict(state_dict['model'])

    sdf_model = model.forward_surface

    # # NOTE: you can also try this out: on real sdf model.
    def sdf1d(x: torch.Tensor, netchunk=1024):
        global sdf_model
        device = x.device
        x = x.cuda()
        # # some test rays @scene adius=3.0
        rays_o = torch.tensor([ 0.8598,  1.0232, -1.4689]).float().cuda().reshape(1, 3)
        rays_d = torch.tensor([-0.4857, -0.4841,  0.7386]).float().cuda().reshape(1, 3)
        pts = rays_o + rays_d * x[..., None]
        with torch.no_grad():
            sdf = []
            for i in range(0, pts.shape[0], netchunk):
                pts_i = pts[i:i+netchunk]
                sdf_i = sdf_model(pts_i)
                sdf.append(sdf_i)
            sdf = torch.cat(sdf, dim=0)
        return sdf.to(device)
else:
    # def sdf1d(x: torch.Tensor):
    #     # deviding point: 1.6, 1.8
        
    #     # (0, 1.8)
    #     # (1.6, 0.2)
    #     # (1.8, 0.4)
    #     # (2.2, 0.)
    #     y_cond1 = -x + 1.8
    #     y_cond2 = x - 1.4
    #     y_cond3 = -x + 2.2 
    #     cond12 = x < 1.8
    #     y = torch.zeros_like(x)
    #     y[cond12] = torch.where(x[cond12] < 1.6, y_cond1[cond12], y_cond2[cond12])
    #     y[~cond12] = y_cond3[~cond12]
    #     return y

    # NOTE: you can also try this out
    def sdf1d(x: torch.Tensor):
        # deviding point: 1.6, 1.8
        
        # (0, 1.65)
        # (1.6, 0.05)
        # (1.8, 0.25)
        # (2.05, 0.)
        y_cond1 = -x + 1.65
        y_cond2 = x - 1.55
        y_cond3 = -x + 2.05
        cond12 = x < 1.8
        y = torch.zeros_like(x)
        y[cond12] = torch.where(x[cond12] < 1.6, y_cond1[cond12], y_cond2[cond12])
        y[~cond12] = y_cond3[~cond12]
        return y

    # def sdf1d(x: torch.Tensor):
    #     return torch.ones_like(x)


def plot(x, sdf, sigma, bounds, alpha, beta, upsampled_x=None):
    device = sdf.device
    # [N-1]
    delta_i = x[..., 1:] - x[..., :-1]
    # [N]
    R_t = torch.cat(
        [
            torch.zeros([*sdf.shape[:-1], 1], device=device), 
            torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
        ], dim=-1)
    opacity_approx = 1 - torch.exp(-R_t[..., :-1])

    # -------------- ground truth compare
    # ground truth data
    dense_sigma = sdf_to_sigma(dense_sdf, alpha, beta)
    # [N-1]
    dense_delta_i = dense_x[..., 1:] - dense_x[..., :-1]
    # [N]
    dense_R_t = torch.cat(
        [
            torch.zeros([*dense_sdf.shape[:-1], 1]), 
            torch.cumsum(dense_sigma[..., :-1] * dense_delta_i, dim=-1)
        ], dim=-1)
    dense_opacity_current_beta = 1 - torch.exp(-dense_R_t[..., :-1])

    # get nearest neibor
    dis = torch.abs(x[..., :-1, None] - dense_x[..., None, :-1])
    ind = torch.argmin(dis, dim=-1)
    opaticy_real = dense_opacity_current_beta[ind]
    error = torch.abs(opacity_approx - opaticy_real)
    
    # -------------- try inverse cdf sampling
    d_fine = rend_util.sample_cdf(x, opacity_approx, 32)
    


    # plot
    # x_np = x.data.cpu().numpy()
    # sdf_np = sdf.data.cpu().numpy()
    # sigma_np = sigma.data.cpu().numpy()
    # bounds_np = bounds.data.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15, 15))
    ax1.plot(x, sdf, label='sdf')
    ax1.plot(x, sigma / sigma.max(), label='normalized sigma')
    ax1.plot(x[..., :-1], opacity_approx, label='opacity')
    ax1.plot(x[..., :-1], opaticy_real, label='opacity gt for current beta')
    ax1.plot(dense_x, opaticy_oracal, label='oracal opacity')
    ax1.scatter(d_fine, np.zeros_like(d_fine), s=20.0, label='try O^{-1} sampling')
    ax1.legend()

    ax2.step(x[..., :-1], bounds, label='error bounds')
    ax2.step(x[..., :-1], error, label='error')
    if upsampled_x is not None:
        ax2.scatter(upsampled_x, np.zeros_like(upsampled_x), label='upsampled points')
    ax2.legend()

    plt.show()

# settings or oracal data
eps = args.eps
init_num = args.init_num
beta_net = args.beta_net
alpha_net = 1./beta_net
M = args.far
max_iter = args.max_iter
x = torch.linspace(0, M, init_num)
dense_x = torch.linspace(0, M, 100001)
dense_sdf = sdf1d(dense_x)
opaticy_oracal = torch.where(dense_sdf > 0, torch.zeros_like(dense_sdf), torch.ones_like(dense_sdf))

# init
beta = np.sqrt((M**2) / (4 * (init_num-1) * np.log(1+eps)))
# beta = alpha_net * (M**2) / (4 * (init_num-1) * np.log(1+eps))
 
# algorithm
alpha = 1./beta
# alpha = alpha_net

# ------------- calculating
sdf = sdf1d(x)
sigma = sdf_to_sigma(sdf, alpha, beta)
bounds = error_bound(x, sdf, alpha, beta)
bounds_net = error_bound(x, sdf, alpha_net, beta_net)
print("init beta+ = {:.3f}".format(beta))
is_end_with_matching = False
it_algo = 0
while it_algo < max_iter and (net_bound_max := bounds_net.max()) > eps:
    print("it =", it_algo)
    print("net_bound_max = {:.6f}".format(net_bound_max.item()))

    it_algo += 1
    #------------- update: upsample
    upsampled_x = rend_util.sample_pdf(x, bounds, init_num, det=True)
    plot(x, sdf, sigma, bounds, alpha, beta, upsampled_x=upsampled_x)
    x = torch.cat([x, upsampled_x], dim=-1)
    # x, _ = torch.sort(x, dim=-1)
    # sdf = sdf1d(x)
    x, sort_indices = torch.sort(x, dim=-1)
    sdf = torch.cat([sdf, sdf1d(upsampled_x)], dim=-1)
    sdf = torch.gather(sdf, dim=-1, index=sort_indices)
    print("more samples:", x.shape[-1])

    bounds_net = error_bound(x, sdf, alpha_net, beta_net)
    if bounds_net.max() > eps:
        #-------------- find beta using bisection methods
        # left: > eps
        # right: < eps
        beta_left = beta_net
        beta_right = beta
        for _ in range(10):
            beta_tmp = 0.5 * (beta_left + beta_right)
            alpha_tmp = 1./beta_tmp
            # alpha_tmp = alpha_net
            bounds_tmp = error_bound(x, sdf, alpha_tmp, beta_tmp)
            bounds_max_tmp = bounds_tmp.max()
            if bounds_max_tmp < eps:
                beta_right = beta_tmp
            elif bounds_max_tmp == eps:
                beta_right = beta_tmp
                break
            else:
                beta_left = beta_tmp
        beta = beta_right
        alpha = 1./beta
        # alpha = alpha_net
        sigma = sdf_to_sigma(sdf, alpha, beta)
        bounds = error_bound(x, sdf, alpha, beta)
    else:
        is_end_with_matching = True
        break
    print("new beta+ = {:.3f}".format(beta))
if (not is_end_with_matching) and (it_algo != 0):
    beta_net = beta_right
    alpha_net = 1./beta_net
print("it=", it_algo)
print("final beta:", beta_net)
sigma = sdf_to_sigma(sdf, alpha_net, beta_net)
bounds = error_bound(x, sdf, alpha_net, beta_net)
print("final error bound max:", bounds.max())
plot(x, sdf, sigma, bounds, alpha_net, beta_net)

## ---------------------- backup
# def sdf_to_sigma(sdf: torch.Tensor, alpha, beta):
#     # sdf *= -1 # NOTE: this will cause inplace opt!
#     sdf = -sdf
#     expsbeta = torch.exp(sdf / beta)
#     psi = torch.where(sdf <= 0, 0.5 * expsbeta, 1 - 0.5 / expsbeta)
#     return alpha * psi


# def error_bound(d_vals, sdf, alpha, beta):
#     """
#     d_vals: [(B), N_rays, N_pts]
#     sdf:    [(B), N_rays, N_pts]
#     """
#     device = sdf.device
#     sigma = sdf_to_sigma(sdf, alpha, beta)
#     # [(B), N_rays, N_pts]
#     sdf_abs_i = torch.abs(sdf)
#     # [(B), N_rays, N_pts-1]
#     delta_i = d_vals[..., 1:] - d_vals[..., :-1]
#     # [(B), N_rays, N_pts]
#     R_t = torch.cat(
#         [
#             torch.zeros([*sdf.shape[:-1], 1], device=device), 
#             torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
#         ], dim=-1)
#     # [(B), N_rays, N_pts-1]
#     d_i_star = torch.clamp_min(0.5 * (sdf_abs_i[..., :-1] + sdf_abs_i[..., 1:] - delta_i), 0.)
#     # [(B), N_rays, N_pts-1]
#     errors = alpha/(4*beta) * (delta_i**2) * torch.exp(-d_i_star / beta)
#     # [(B), N_rays, N_pts-1]
#     errors_t = torch.cumsum(errors, dim=-1)
#     # [(B), N_rays, N_pts-1]
#     bounds = torch.exp(-R_t[..., :-1]) * (torch.exp(errors_t) - 1.)
#     # TODO: better solution
#     # NOTE: nan comes from 0 * inf
#     # NOTE: every situation where nan appears will also appears c * inf = "true" inf, so below solution is acceptable
#     bounds[torch.isnan(bounds)] = np.inf
#     return bounds
