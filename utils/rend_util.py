import cv2
import numpy as np

import torch
from torch.nn import functional as F


def load_K_Rt_from_P(P):
    """
    modified from IDR https://github.com/lioryariv/idr
    """
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def view_matrix(
    forward: np.ndarray, 
    up: np.ndarray,
    cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def look_at(
    cam_location: np.ndarray, 
    point: np.ndarray, 
    up=np.array([0., -1., 0.])          # openCV convention
    # up=np.array([0., 1., 0.])         # openGL convention
    ):
    # Cam points in positive z direction
    forward = normalize(point - cam_location)     # openCV convention
    # forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)

def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).to(R.device)

    R00 = R[..., 0,0]
    R01 = R[..., 0, 1]
    R02 = R[..., 0, 2]
    R10 = R[..., 1, 0]
    R11 = R[..., 1, 1]
    R12 = R[..., 1, 2]
    R20 = R[..., 2, 0]
    R21 = R[..., 2, 1]
    R22 = R[..., 2, 2]

    q[...,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[..., 1]=(R21-R12)/(4*q[:,0])
    q[..., 2] = (R02 - R20) / (4 * q[:, 0])
    q[..., 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def quat_to_rot(q):
    prefix, _ = q.shape[:-1]
    q = F.normalize(q, dim=-1)
    R = torch.ones([*prefix, 3, 3]).to(q.device)
    qr = q[... ,0]
    qi = q[..., 1]
    qj = q[..., 2]
    qk = q[..., 3]
    R[..., 0, 0]=1-2 * (qj**2 + qk**2)
    R[..., 0, 1] = 2 * (qj *qi -qk*qr)
    R[..., 0, 2] = 2 * (qi * qk + qr * qj)
    R[..., 1, 0] = 2 * (qj * qi + qk * qr)
    R[..., 1, 1] = 1-2 * (qi**2 + qk**2)
    R[..., 1, 2] = 2*(qj*qk - qi*qr)
    R[..., 2, 0] = 2 * (qk * qi-qj * qr)
    R[..., 2, 1] = 2 * (qj*qk + qi*qr)
    R[..., 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def lift(x, y, z, intrinsics):
    device = x.device
    # parse intrinsics
    intrinsics = intrinsics.to(device)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    sk = intrinsics[..., 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(device)), dim=-1)


def get_rays(c2w, intrinsics, H, W, N_rays=-1):
    device = c2w.device
    if c2w.shape[-1] == 7: #In case of quaternion vector representation
        cam_loc = c2w[..., 4:]
        R = quat_to_rot(c2w[...,:4])
        p = torch.eye(4).repeat([*c2w.shape[0:-1],1,1]).to(device).float()
        p[..., :3, :3] = R
        p[..., :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = c2w[..., :3, 3]
        p = c2w

    prefix = p.shape[:-2]
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t().to(device).reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])
    j = j.t().to(device).reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])

    if N_rays > 0:
        N_rays = min(N_rays, H*W)
        # ---------- option 1: full image uniformly randomize
        # select_inds = torch.from_numpy(
        #     np.random.choice(H*W, size=[*prefix, N_rays], replace=False)).to(device)
        # select_inds = torch.randint(0, H*W, size=[N_rays]).expand([*prefix, N_rays]).to(device)
        # ---------- option 2: H/W seperately randomize
        select_hs = torch.randint(0, H, size=[N_rays]).to(device)
        select_ws = torch.randint(0, W, size=[N_rays]).to(device)
        select_inds = select_hs * W + select_ws
        select_inds = select_inds.expand([*prefix, N_rays])

        i = torch.gather(i, -1, select_inds)
        j = torch.gather(j, -1, select_inds)
    else:
        select_inds = torch.arange(H*W).to(device).expand([*prefix, H*W])

    pixel_points_cam = lift(i, j, torch.ones_like(i).to(device), intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.transpose(-1,-2)

    # NOTE: left-multiply.
    #       after the above permute(), shapes of coordinates changed from [B,N,4] to [B,4,N], which ensures correct left-multiplication
    #       p is camera 2 world matrix.
    if len(prefix) > 0:
        world_coords = torch.bmm(p, pixel_points_cam).transpose(-1, -2)[..., :3]
    else:
        world_coords = torch.mm(p, pixel_points_cam).transpose(-1, -2)[..., :3]
    rays_d = world_coords - cam_loc[..., None, :]
    # ray_dirs = F.normalize(ray_dirs, dim=2)

    rays_o = cam_loc[..., None, :].expand_as(rays_d)

    return rays_o, rays_d, select_inds


def near_far_from_sphere(ray_origins: torch.Tensor, ray_directions: torch.Tensor, r = 1.0, keepdim=True):
    """
    NOTE: modified from https://github.com/Totoro97/NeuS
    ray_origins: camera center's coordinate
    ray_directions: camera rays' directions. already normalized.
    """
    # rayso_norm_square = torch.sum(ray_origins**2, dim=-1, keepdim=True)
    # NOTE: (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    ray_cam_dot = torch.sum(ray_origins * ray_directions, dim=-1, keepdim=keepdim)
    mid = -ray_cam_dot
    # NOTE: a convservative approximation of the half chord length from ray intersections with the sphere.
    #       all half chord length < r
    near = mid - r
    far = mid + r
    
    near = near.clamp_min(0.0)
    far = far.clamp_min(r)  # NOTE: instead of clamp_min(0.0), just some trick.
    
    return near, far


def get_sphere_intersection(ray_origins: torch.Tensor, ray_directions: torch.Tensor, r = 1.0):
    """
    NOTE: modified from IDR. https://github.com/lioryariv/idr
    ray_origins: camera center's coordinate
    ray_directions: camera rays' directions. already normalized.
    """
    rayso_norm_square = torch.sum(ray_origins**2, dim=-1, keepdim=True)
    # (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    ray_cam_dot = torch.sum(ray_origins * ray_directions, dim=-1, keepdim=True)
    
    # accurate ray-sphere intersections
    near = torch.zeros([*ray_origins.shape[:-1], 1]).to(ray_origins.device)
    far = torch.zeros([*ray_origins.shape[:-1], 1]).to(ray_origins.device)
    under_sqrt = ray_cam_dot ** 2  + r ** 2 - rayso_norm_square
    mask_intersect = under_sqrt > 0
    sqrt = torch.sqrt(under_sqrt[mask_intersect])
    near[mask_intersect] = - sqrt - ray_cam_dot[mask_intersect]
    far[mask_intersect] = sqrt - ray_cam_dot[mask_intersect]

    near = near.clamp_min(0.0)
    far = far.clamp_min(0.0)

    return near, far, mask_intersect


def get_dvals_from_radius(ray_origins: torch.Tensor, ray_directions: torch.Tensor, rs: torch.Tensor, far_end=True):
    """
    ray_origins: camera center's coordinate
    ray_directions: camera rays' directions. already normalized.
    rs: the distance to the origin
    far_end: whether the point is on the far-end of the ray or on the near-end of the ray
    """
    rayso_norm_square = torch.sum(ray_origins**2, dim=-1, keepdim=True)
    # NOTE: (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    ray_cam_dot = torch.sum(ray_origins * ray_directions, dim=-1, keepdim=True)
        
    under_sqrt = rs**2 - (rayso_norm_square - ray_cam_dot ** 2)
    assert (under_sqrt > 0).all()
    sqrt = torch.sqrt(under_sqrt)
    
    if far_end:
        d_vals = -ray_cam_dot + sqrt
    else:
        d_vals = -ray_cam_dot - sqrt
        d_vals = torch.clamp_min(d_vals, 0.)
    
    return d_vals


def lin2img(tensor: torch.Tensor, H: int, W: int, batched=False, B=None):
    *_, num_samples, channels = tensor.shape
    assert num_samples == H * W
    if batched:
        if B is None:
            B = tensor.shape[0]
        else:
            tensor = tensor.view([B, num_samples//B, channels])
        return tensor.permute(0, 2, 1).view([B, channels, H, W])
    else:
        return tensor.permute(1, 0).view([channels, H, W])


#----------------------------------------------------
#-------- Sampling points from ray ------------------
#----------------------------------------------------

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf.detach(), u, right=False)

    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1]-1)
    # (batch, N_importance, 2) ==> (B, batch, N_importance, 2)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]  # fix prefix shape

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)  # fix prefix shape

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom<eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def sample_cdf(bins, cdf, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = bins.device
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf.detach(), u, right=False)

    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1]-1)
    # (batch, N_importance, 2) ==> (B, batch, N_importance, 2)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]  # fix prefix shape

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)  # fix prefix shape

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom<eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples