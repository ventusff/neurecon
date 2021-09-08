import numpy as np
import matplotlib.pyplot as plt

from utils.rend_util import get_rays
from dataio.DTU import SceneDataset

def plot_rays(rays_o: np.ndarray, rays_d: np.ndarray, ax):
    # TODO: automatic reducing number of rays
    XYZUVW = np.concatenate([rays_o, rays_d], axis=-1)
    X, Y, Z, U, V, W = np.transpose(XYZUVW)
    # X2 = X+U
    # Y2 = Y+V
    # Z2 = Z+W
    # x_max = max(np.max(X), np.max(X2))
    # x_min = min(np.min(X), np.min(X2))
    # y_max = max(np.max(Y), np.max(Y2))
    # y_min = min(np.min(Y), np.min(Y2))
    # z_max = max(np.max(Z), np.max(Z2))
    # z_min = min(np.min(Z), np.min(Z2))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_zlim(z_min, z_max)
    
    return ax

dataset = SceneDataset(False, './data/DTU/scan40', downscale=32)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
H, W = (dataset.H, dataset.W)

for i in range(dataset.n_images):
    _, model_input, _ = dataset[i]
    intrinsics = model_input["intrinsics"][None, ...]
    c2w = model_input['c2w'][None, ...]
    rays_o, rays_d, select_inds = get_rays(c2w, intrinsics, H, W, N_rays=1)
    rays_o = rays_o.data.squeeze(0).cpu().numpy()
    rays_d = rays_d.data.squeeze(0).cpu().numpy()
    ax = plot_rays(rays_o, rays_d, ax)

plt.show()