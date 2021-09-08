from utils.print_fn import log

import time
import plyfile
import skimage
import skimage.measure
import numpy as np
from tqdm import tqdm

import torch


def convert_sigma_samples_to_ply(
    input_3d_sigma_array: np.ndarray,
    voxel_grid_origin,
    volume_size,
    ply_filename_out,
    level=5.0,
    offset=None,
    scale=None,):
    """
    Convert sdf samples to .ply

    :param input_3d_sdf_array: a float array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :volume_size: a list of three floats
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        input_3d_sigma_array, level=level, spacing=volume_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))


    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    log.info("saving mesh to %s" % str(ply_filename_out))
    ply_data.write(ply_filename_out)

    log.info(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def extract_mesh(implicit_surface, volume_size=2.0, level=0.0, N=512, filepath='./surface.ply', show_progress=True, chunk=16*1024):
    s = volume_size
    voxel_grid_origin = [-s/2., -s/2., -s/2.]
    volume_size = [s, s, s]

    overall_index = np.arange(0, N ** 3, 1).astype(np.int)
    xyz = np.zeros([N ** 3, 3])

    # transform first 3 columns
    # to be the x, y, z index
    xyz[:, 2] = overall_index % N
    xyz[:, 1] = (overall_index / N) % N
    xyz[:, 0] = ((overall_index / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    xyz[:, 0] = (xyz[:, 0] * (s/(N-1))) + voxel_grid_origin[2]
    xyz[:, 1] = (xyz[:, 1] * (s/(N-1))) + voxel_grid_origin[1]
    xyz[:, 2] = (xyz[:, 2] * (s/(N-1))) + voxel_grid_origin[0]
    
    def batchify(query_fn, inputs: torch.Tensor, chunk=chunk):
        out = []
        for i in tqdm(range(0, inputs.shape[0], chunk), disable=not show_progress):
            out_i = query_fn(torch.from_numpy(inputs[i:i+chunk]).float().cuda()).data.cpu().numpy()
            out.append(out_i)
        out = np.concatenate(out, axis=0)
        return out

    out = batchify(implicit_surface.forward, xyz)
    out = out.reshape([N, N, N])
    convert_sigma_samples_to_ply(out, voxel_grid_origin, [float(v) / N for v in volume_size], filepath, level=level)
    