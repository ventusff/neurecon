from models.frameworks.neus import sdf_to_w, pdf_phi_s, cdf_Phi_s

import math
import torch
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

BORDER0 = 2.13333333
BORDER1 = 3.13333333
BORDER_CENTER = 0.5 * (BORDER0 + BORDER1)


class Plotter(object):
    def __init__(self, init_num=20, near=0., far=6., min_num=2, max_num=1024, init_s=64.) -> None:
        super().__init__()
        
        assert far > BORDER0 + 0.1 and near < far and near < BORDER0 - 0.1
        
        self.near = near
        self.far = far
        scatter_size = 20.
        #-------------------
        # prepare init data
        #-------------------
        fake_dvals = np.linspace(near, far, init_num)
        fake_dvals_mid = (fake_dvals[..., 1:] + fake_dvals[..., :-1]) * 0.5
        fake_sdf = fake_1d_sdf(fake_dvals)
        
        #-------------------
        # prepare figure
        #-------------------
        fig = plt.figure(figsize=(30,12))
        #--------------------------- total
        gs = gridspec.GridSpec(50, 1)
        
        #----------------------top
        gs_top = gridspec.GridSpecFromSubplotSpec(
            nrows=1,
            ncols=2,
            subplot_spec=gs[0:40, 0])
        #---------------- left: naive w function
        pdf_coarse, cdf_coarse, alpha_coarse, w_coarse = naive_sdf2w(fake_dvals, fake_sdf, s=init_s)
        # d_pred = np.sum(w_coarse * fake_dvals)
        d_pred = np.sum((w_coarse[fake_dvals<BORDER_CENTER] / np.sum(w_coarse[fake_dvals<BORDER_CENTER], keepdims=True) * fake_dvals[fake_dvals<BORDER_CENTER]))
        error = abs(d_pred - BORDER0)
        
        ax_top_left = fig.add_subplot(gs_top[0, 0])
        self.ax_naive_sdf_scatter = ax_top_left.scatter(fake_dvals, fake_sdf, s=scatter_size, label='sdf')
        self.ax_naive_sdf_plot = ax_top_left.plot(fake_dvals, fake_sdf)
        self.ax_naive_cdf_scatter = ax_top_left.scatter(fake_dvals, cdf_coarse, s=scatter_size, label='cdf')
        self.ax_naive_cdf_plot = ax_top_left.plot(fake_dvals, cdf_coarse)
        self.ax_naive_alpha_scatter = ax_top_left.scatter(fake_dvals, alpha_coarse, s=scatter_size, label='alpha')
        self.ax_naive_alpha_plot = ax_top_left.plot(fake_dvals, alpha_coarse)
        self.ax_naive_w_scatter = ax_top_left.scatter(fake_dvals, w_coarse / w_coarse.max(), s=scatter_size*3., label='normalized w(t)/weights')
        self.ax_naive_w_plot = ax_top_left.plot(fake_dvals, w_coarse / w_coarse.max())
        self.ax_naive_pdf_scatter = ax_top_left.scatter(fake_dvals, pdf_coarse / pdf_coarse.max(), s=scatter_size, label='pdf(normalize)')
        self.ax_naive_pdf_plot = ax_top_left.plot(fake_dvals, pdf_coarse / pdf_coarse.max())
        ax_top_left.plot(np.linspace(near, far, 50), np.zeros(50), label='y=0')
        ax_top_left.axvline(x=BORDER0, linestyle='-', color='g', label='surface render depth / exact surface')
        self.ax_naive_d_pred = ax_top_left.axvline(x=d_pred, linestyle='--', color='r', label='volume render depth (max color contribution)')
        ax_top_left.legend(fontsize=12)
        ax_top_left.set_xlabel('error = {:.12f}, {:.2f} dB'.format(error, 10.*np.log(error)))
        ax_top_left.set_title('naive solution')
        
        #---------------- right: neus' w function
        cdf_coarse, alpha_coarse, w_coarse = neus_sdf2w(fake_sdf, s=init_s)
        # d_pred = np.sum(w_coarse * fake_dvals_mid)
        # NOTE: since when calculating colors, middle points' value is used.
        d_pred = np.sum((w_coarse[fake_dvals_mid<BORDER_CENTER] / np.sum(w_coarse[fake_dvals_mid<BORDER_CENTER], keepdims=True) * fake_dvals_mid[fake_dvals_mid<BORDER_CENTER]))
        error = abs(d_pred - BORDER0)
        
        ax_top_right = fig.add_subplot(gs_top[0, 1])
        self.ax_neus_sdf_scatter = ax_top_right.scatter(fake_dvals, fake_sdf, s=scatter_size, label='sdf')
        self.ax_neus_sdf_plot = ax_top_right.plot(fake_dvals, fake_sdf)
        self.ax_neus_cdf_scatter = ax_top_right.scatter(fake_dvals, cdf_coarse, s=scatter_size, label='cdf')
        self.ax_neus_cdf_plot = ax_top_right.plot(fake_dvals, cdf_coarse)
        self.ax_neus_alpha_scatter = ax_top_right.scatter(fake_dvals_mid, alpha_coarse, s=scatter_size*6., label='alpha')
        self.ax_neus_alpha_plot = ax_top_right.plot(fake_dvals_mid, alpha_coarse)
        self.ax_neus_w_scatter = ax_top_right.scatter(fake_dvals_mid, w_coarse / w_coarse.max(), s=scatter_size*3., label='normalized w(t)/weights')
        self.ax_neus_w_plot = ax_top_right.plot(fake_dvals_mid, w_coarse / w_coarse.max())
        ax_top_right.plot(np.linspace(near, far, 50), np.zeros(50), label='y=0')
        ax_top_right.axvline(x=BORDER0, linestyle='-', color='g', label='surface render depth / exact surface')
        self.ax_neus_d_pred = ax_top_right.axvline(x=d_pred, linestyle='--', color='r', label='volume render depth (max color contribution)')
        ax_top_right.legend(fontsize=12)
        ax_top_right.set_xlabel('error = {:.12f}, {:.2f} dB'.format(error, 10.*np.log(error)))
        ax_top_right.set_title('NeuS solution')
        
        self.scatters = [self.ax_naive_sdf_scatter, self.ax_naive_pdf_scatter, self.ax_naive_cdf_scatter, self.ax_naive_alpha_scatter, self.ax_naive_w_scatter,
                         self.ax_neus_sdf_scatter, self.ax_neus_cdf_scatter, self.ax_neus_alpha_scatter, self.ax_neus_w_scatter]
        
        #----------------------down
        ax_down_num = fig.add_subplot(gs[-1, 0])
        
        self.ui_slider_num = Slider(ax_down_num, 'log2(num)', np.log2(min_num), np.log2(max_num), valinit=np.log2(init_num))
        self.ui_slider_num.on_changed(self.on_update_slider_num)
        ax_down_num.set_xlabel('num = {} samples, {:.1f} points/m'.format(init_num, init_num/(self.far-self.near)))
        
        ax_down_s = fig.add_subplot(gs[-5, 0])
        self.ui_slider_s = Slider(ax_down_s, 'log2(s)', np.log2(1), np.log2(1024), valinit=np.log2(init_s))
        self.ui_slider_s.on_changed(self.on_update_slider_s)
        ax_down_s.set_xlabel('s = {:.1f}. fixed_s=64.(for sampling), learned_s>=1000.(for rendering)'.format(init_s))
        
        self.fig = fig
        self.ax_top_left = ax_top_left
        self.ax_top_right = ax_top_right
        self.ax_down_num = ax_down_num
        self.ax_down_s = ax_down_s
        self.s = init_s
        self.num = init_num
    
    def on_update_slider_num(self, val):
        self.num = int(2 ** val)
        self.refresh()
    
    def on_update_slider_s(self, val):
        self.s = 2 ** val
        self.refresh()
        
    def refresh(self):
        self.ax_down_num.set_xlabel('num = {} samples, {:.1f} points/m'.format(self.num, self.num/(self.far-self.near)))
        self.ax_down_s.set_xlabel('s = {:.1f}. fixed_s=64.(for sampling), learned_s>=1000.(for rendering)'.format(self.s))
        
        fake_dvals = np.linspace(self.near, self.far, self.num)
        fake_dvals_mid = (fake_dvals[..., 1:] + fake_dvals[..., :-1]) * 0.5
        fake_sdf = fake_1d_sdf(fake_dvals)
        
        pdf_coarse, cdf_coarse, alpha_coarse, w_coarse = naive_sdf2w(fake_dvals, fake_sdf, s=self.s)
        d_pred = np.sum((w_coarse[fake_dvals<BORDER_CENTER] / np.sum(w_coarse[fake_dvals<BORDER_CENTER], keepdims=True) * fake_dvals[fake_dvals<BORDER_CENTER]))
        error = abs(d_pred - BORDER0)

        self.ax_naive_sdf_scatter.set_offsets(np.c_[fake_dvals, fake_sdf])
        self.ax_naive_sdf_plot[0].set_data(fake_dvals, fake_sdf)
        self.ax_naive_cdf_scatter.set_offsets(np.c_[fake_dvals, cdf_coarse])
        self.ax_naive_cdf_plot[0].set_data(fake_dvals, cdf_coarse)
        self.ax_naive_alpha_scatter.set_offsets(np.c_[fake_dvals, alpha_coarse])
        self.ax_naive_alpha_plot[0].set_data(fake_dvals, alpha_coarse)
        self.ax_naive_w_scatter.set_offsets(np.c_[fake_dvals, w_coarse / w_coarse.max()])
        self.ax_naive_w_plot[0].set_data(fake_dvals, w_coarse / w_coarse.max())
        self.ax_naive_pdf_scatter.set_offsets(np.c_[fake_dvals, pdf_coarse / pdf_coarse.max()])
        self.ax_naive_pdf_plot[0].set_data(fake_dvals, pdf_coarse / pdf_coarse.max())
        self.ax_naive_d_pred.set_xdata(d_pred)
        
        # recompute the ax.dataLim
        self.ax_top_left.relim()
        # update ax.viewLim using the new dataLim
        self.ax_top_left.autoscale_view()
        self.ax_top_left.set_xlabel('error = {:.12f}, {:.2f} dB'.format(error, 10.*np.log(error)))



        cdf_coarse, alpha_coarse, w_coarse = neus_sdf2w(fake_sdf, s=self.s)
        # NOTE: since when calculating colors, middle points' value is used.
        d_pred = np.sum((w_coarse[fake_dvals_mid<BORDER_CENTER] / np.sum(w_coarse[fake_dvals_mid<BORDER_CENTER], keepdims=True) * fake_dvals_mid[fake_dvals_mid<BORDER_CENTER]))
        # d_pred = np.sum(w_coarse * fake_dvals_mid)
        error = abs(d_pred - BORDER0)

        self.ax_neus_sdf_scatter.set_offsets(np.c_[fake_dvals, fake_sdf])
        self.ax_neus_sdf_plot[0].set_data(fake_dvals, fake_sdf)
        self.ax_neus_cdf_scatter.set_offsets(np.c_[fake_dvals, cdf_coarse])
        self.ax_neus_cdf_plot[0].set_data(fake_dvals, cdf_coarse)
        self.ax_neus_alpha_scatter.set_offsets(np.c_[fake_dvals_mid, alpha_coarse])
        self.ax_neus_alpha_plot[0].set_data(fake_dvals_mid, alpha_coarse)
        self.ax_neus_w_scatter.set_offsets(np.c_[fake_dvals_mid, w_coarse / w_coarse.max()])
        self.ax_neus_w_plot[0].set_data(fake_dvals_mid, w_coarse / w_coarse.max())
        self.ax_neus_d_pred.set_xdata(d_pred)

        # recompute the ax.dataLim
        self.ax_top_right.relim()
        # update ax.viewLim using the new dataLim
        self.ax_top_right.autoscale_view()
        self.ax_top_right.set_xlabel('error = {:.12f}, {:.2f} dB'.format(error, 10.*np.log(error)))

        self.fig.canvas.draw_idle()

def neus_sdf2w(sdf, s):
    sdf = torch.from_numpy(sdf)
    
    rets = sdf_to_w(sdf, s)
    rets_py = []
    for ret in rets:
        rets_py.append(ret.data.cpu().numpy())
    return tuple(rets_py)

def naive_sdf2w(z_vals, sdf, s):
    z_vals = torch.from_numpy(z_vals)
    sdf = torch.from_numpy(sdf)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [
            dists,
            # 1e10 * torch.ones(dists[..., :1].shape).to(device)
            1e2 * torch.ones(dists[..., :1].shape)   # use 1e2, as in nerf-w
        ], dim=-1)
    
    cdf: torch.Tensor = cdf_Phi_s(sdf, s)
    pdf: torch.Tensor = pdf_phi_s(sdf, s)
    opacity_alpha = 1- torch.exp(-pdf * dists)
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1]),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)
    visibility_weights = opacity_alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]
    visibility_weights = visibility_weights/visibility_weights.sum(dim=-1, keepdim=True)
    return pdf.data.cpu().numpy(), cdf.data.cpu().numpy(), opacity_alpha.data.cpu().numpy(), visibility_weights.data.cpu().numpy()
    

def fake_1d_sdf(x, border_0=BORDER0, border_1=BORDER1):
    assert border_0 < border_1
    dis = np.min(np.stack([np.abs(x-border_0), np.abs(x-border_1)], axis=-1), axis=-1)
    inside = (border_0 < x) & (x < border_1)
    sign = np.ones_like(x)
    sign[inside] = -1
    return sign * dis


# ------------------ controllable plot
# plotter = Plotter(far=6.0)
plotter = Plotter(near=1.8, far=2.4)
plt.show()

# --------------- single plot

# # num = 1000
# num = 8
# # num = 10
# if num <=10:
#     s = 20.
# elif num <= 100:
#     s = 5.
# else:
#     s = 1.

# fake_dvals = np.linspace(0., 6., num)
# fake_dvals_mid = (fake_dvals[..., 1:] + fake_dvals[..., :-1]) * 0.5
# fake_sdf = fake_1d_sdf(fake_dvals)
 
 
# ## neus
# # fine_points = sample_pdf(fake_dvals, w_coarse, num)
# # fine_sdf = fake_1d_sdf(fine_points)

# if True:
#     cdf_coarse, alpha_coarse, w_coarse = neus_sdf2w(fake_sdf, s=FIXED_S)
    
#     plt.scatter(fake_dvals, fake_sdf, s=s, label='sdf')
#     plt.plot(fake_dvals, fake_sdf)
#     plt.scatter(fake_dvals, cdf_coarse, s=s, label='cdf')
#     plt.plot(fake_dvals, cdf_coarse)
#     plt.scatter(fake_dvals_mid, alpha_coarse, s=s, label='alpha')
#     plt.plot(fake_dvals_mid, alpha_coarse)
#     plt.scatter(fake_dvals_mid, w_coarse, s=s*3., label='w(t) / weights')
#     plt.plot(fake_dvals_mid, w_coarse)
#     plt.plot(fake_dvals, np.zeros(num), label='y=0')
#     plt.legend()  
# if True:
#     pdf_coarse, cdf_coarse, alpha_coarse, w_coarse = naive_sdf2w(fake_dvals, fake_sdf, s=FIXED_S)
    
#     plt.scatter(fake_dvals, fake_sdf, s=s, label='sdf')
#     plt.plot(fake_dvals, fake_sdf)
#     plt.scatter(fake_dvals, pdf_coarse, s=s, label='pdf')
#     plt.plot(fake_dvals, pdf_coarse)
#     plt.scatter(fake_dvals, cdf_coarse, s=s, label='cdf')
#     plt.plot(fake_dvals, cdf_coarse)
#     plt.scatter(fake_dvals, alpha_coarse, s=s*4., label='alpha')
#     plt.plot(fake_dvals, alpha_coarse)
#     plt.scatter(fake_dvals, w_coarse, s=s*2., label='w(t) / weights')
#     plt.plot(fake_dvals, w_coarse)
#     plt.plot(fake_dvals, np.zeros(num), label='y=0')
#     plt.legend()

# plt.show()










# ----------------- backup
# def pdf_phi_s(x: torch.Tensor, s):
#     esx = torch.exp(-s*x)
#     y = s*esx / ((1+esx) ** 2)
#     return y

# def cdf_Phi_s(x, s):
#     # den = 1 + torch.exp(-s*x)
#     # y = 1./den
#     # return y
#     return torch.sigmoid(x*s)

# def sdf_to_w(sdf, s):
#     device = sdf.device
#     # [(B), N_rays, N_pts]
#     cdf = cdf_Phi_s(sdf, s)
#     # [(B), N_rays, N_pts-1]
#     # TODO: check sanity.
#     opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
#     opacity_alpha = torch.clamp_min(opacity_alpha, 0)

#     # [(B), N_rays, N_pts]
#     shifted_transparency = torch.cat(
#         [
#             torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
#             1.0 - opacity_alpha + 1e-10,
#         ], dim=-1)
    
#     # [(B), N_rays, N_pts-1]
#     visibility_weights = opacity_alpha *\
#         torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

#     return cdf, opacity_alpha, visibility_weights