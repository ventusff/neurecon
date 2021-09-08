from utils import io_util
from dataio import get_data

import skimage
import skimage.measure
import numpy as np
import open3d as o3d


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors

def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


# ----------------------
# plot cameras alongside with mesh
# modified from NeRF++.  https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/extract_sfm.py
def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh', backface=False):
    things_to_draw = []
    
    if sphere_radius > 0:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
        sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        sphere.paint_uniform_color((1, 0, 0))
        things_to_draw.append(sphere)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw.append(coord_frame)
    
    idx = 0
    for camera_dict in colored_camera_dicts:
        idx += 1

        K = np.array(camera_dict['K']).reshape((4, 4))
        W2C = np.array(camera_dict['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        img_size = camera_dict['img_size']
        color = camera_dict['color']
        frustums = [get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color)]
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)
    if backface:
        o3d.visualization.RenderOption.mesh_show_back_face = True
    o3d.visualization.draw_geometries(things_to_draw)


if __name__ == "__main__":
    parser = io_util.create_args_parser()
    parser.add_argument("--scan_id", type=int, default=40)
    parser.add_argument("--mesh_file", type=str, default=None)
    parser.add_argument("--sphere_radius", type=float, default=3.0)
    parser.add_argument("--backface",action='store_true', help='render show back face')
    args = parser.parse_args()

    # load camera
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    dataset = get_data(config)

    #------------- 
    colored_camera_dicts = []
    for i in range(len(dataset)):
        (_, model_input, ground_truth) = dataset[i]
        c2w = model_input['c2w'].data.cpu().numpy()
        intrinsics = model_input["intrinsics"].data.cpu().numpy()
        
        cam_dict = {}
        cam_dict['img_size'] = (dataset.W, dataset.H)
        cam_dict['W2C'] = np.linalg.inv(c2w)
        cam_dict['K'] = intrinsics
        # cam_dict['color'] = [0, 1, 1]
        cam_dict['color'] = [1, 0, 0]
        
        # if i == 0:
        #     cam_dict['color'] = [1, 0, 0]

        # if i == 1:
        #     cam_dict['color'] = [0, 1, 0]

        # if i == 28:
        #     cam_dict['color'] = [1, 0, 0]

        colored_camera_dicts.append(cam_dict)

    visualize_cameras(colored_camera_dicts, args.sphere_radius, geometry_file=args.mesh_file, backface=args.backface)
    
    
