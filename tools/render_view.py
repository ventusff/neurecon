from utils import io_util, rend_util
from models.frameworks import get_model
from utils.checkpoints import sorted_ckpts
from utils.print_fn import log

import os
import math
import imageio
import functools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def normalize(vec, axis=-1):
    return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-9)

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

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    forward = poses[:, :3, 2].sum(0)
    up = poses[:, :3, 1].sum(0)
    c2w = view_matrix(forward, up, center)
    return c2w

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

def c2w_track_spiral(c2w, up_vec, rads, focus: float, zrate: float, rots: int, N: int, zdelta: float = 0.):
    # TODO: support zdelta
    """generate camera to world matrices of spiral track, looking at the same point [0,0,focus]

    Args:
        c2w ([4,4] or [3,4]):   camera to world matrix (of the spiral center, with average rotation and average translation)
        up_vec ([3,]):          vector pointing up
        rads ([3,]):            radius of x,y,z direction, of the spiral track
        # zdelta ([float]):       total delta z that is allowed to change 
        focus (float):          a focus value (to be looked at) (in camera coordinates)
        zrate ([float]):        a factor multiplied to z's angle
        rots ([int]):           number of rounds to rotate
        N ([int]):              number of total views
    """

    c2w_tracks = []
    rads = np.array(list(rads) + [1.])
    
    # focus_in_cam = np.array([0, 0, -focus, 1.])   # openGL convention
    focus_in_cam = np.array([0, 0, focus, 1.])      # openCV convention
    focus_in_world = np.dot(c2w[:3, :4], focus_in_cam)

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        cam_location = np.dot(
            c2w[:3, :4], 
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads    # openGL convention
            np.array([np.cos(theta), np.sin(theta), np.sin(theta*zrate), 1.]) * rads        # openCV convention
        )
        c2w_i = look_at(cam_location, focus_in_world, up=up_vec)
        c2w_tracks.append(c2w_i)
    return c2w_tracks


def smoothed_motion_interpolation(full_range, num_samples, uniform_proportion=1/3.):
    half_acc_proportion = (1-uniform_proportion) / 2.
    num_uniform_acc = max(math.ceil(num_samples*half_acc_proportion), 2)
    num_uniform = max(math.ceil(num_samples*uniform_proportion), 2)
    num_samples = num_uniform_acc * 2 + num_uniform
    seg_velocity = np.arange(num_uniform_acc)
    seg_angle = np.cumsum(seg_velocity)
    # NOTE: full angle = 2*k*x_max + k*v_max*num_uniform
    ratio = full_range / (2.0*seg_angle.max()+seg_velocity.max()*num_uniform)
    # uniform acceleration sequence
    seg_acc = seg_angle * ratio

    acc_angle = seg_acc.max()
    # uniform sequence
    seg_uniform = np.linspace(acc_angle, full_range-acc_angle, num_uniform+2)[1:-1]
    # full sequence
    all_samples = np.concatenate([seg_acc, seg_uniform, full_range-np.flip(seg_acc)])
    return all_samples


def visualize_cam_on_circle(intr, extrs, up_vec, c0):
    
    import matplotlib
    import matplotlib.pyplot as plt
    from tools.vis_camera import draw_camera
    
    cam_width = 0.2/2     # Width/2 of the displayed camera.
    cam_height = 0.1/2    # Height/2 of the displayed camera.
    scale_focal = 2000        # Value to scale the focal length.
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    ax.set_aspect("auto")    

    matplotlib.rcParams.update({'font.size': 22})
    #----------- draw cameras
    min_values, max_values = draw_camera(ax, intr, cam_width, cam_height, scale_focal, extrs, True)

    radius = np.linalg.norm(c0)
    
    #----------- draw small circle
    angles = np.linspace(0, np.pi * 2., 180)
    rots = R.from_rotvec(angles[:, None] * up_vec[None, :])
    # [180, 3]
    pts = rots.apply(c0)
    # [x, z, -y]
    ax.plot(pts[:, 0], pts[:, 2], -pts[:, 1], color='black')
    
    #----------- draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey', linewidth=0, alpha=0.1)
    
    #----------- draw axis
    axis = np.array([[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    X, Y, Z, U, V, W = zip(*axis) 
    ax.quiver(X[0], Z[0], -Y[0], U[0], W[0], -V[0], color='red')
    ax.quiver(X[1], Z[1], -Y[1], U[1], W[1], -V[1], color='green')
    ax.quiver(X[2], Z[2], -Y[2], U[2], W[2], -V[2], color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    
    plt.show()


def visualize_cam_spherical_spiral(intr, extrs, up_vec, c0, focus_center, n_rots, up_angle):
    import matplotlib
    import matplotlib.pyplot as plt
    from tools.vis_camera import draw_camera
    
    cam_width = 0.2/2     # Width/2 of the displayed camera.
    cam_height = 0.1/2    # Height/2 of the displayed camera.
    scale_focal = 2000        # Value to scale the focal length.
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    ax.set_aspect("auto")    

    matplotlib.rcParams.update({'font.size': 22})
    #----------- draw cameras
    min_values, max_values = draw_camera(ax, intr, cam_width, cam_height, scale_focal, extrs, True)

    radius = np.linalg.norm(c0)
    
    #----------- draw small circle
    # key rotations of a spherical spiral path
    num_pts = int(n_rots * 180.)
    sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, num_pts)
    sphere_phis = np.linspace(0, up_angle, num_pts)
    # first rotate about up vec
    rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
    pts = rots_theta.apply(c0)
    # then rotate about horizontal vec
    horizontal_vec = normalize(np.cross(pts-focus_center[None, :], up_vec[None, :], axis=-1))
    rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
    pts = rots_phi.apply(pts)
    # [x, z, -y]
    ax.plot(pts[:, 0], pts[:, 2], -pts[:, 1], color='black')
    
    #----------- draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey', linewidth=0, alpha=0.1)
    
    #----------- draw axis
    axis = np.array([[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    X, Y, Z, U, V, W = zip(*axis) 
    ax.quiver(X[0], Z[0], -Y[0], U[0], W[0], -V[0], color='red')
    ax.quiver(X[1], Z[1], -Y[1], U[1], W[1], -V[1], color='green')
    ax.quiver(X[2], Z[2], -Y[2], U[2], W[2], -V[2], color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    
    plt.show()


def main_function(args):
    do_render_mesh = args.render_mesh is not None
    if do_render_mesh:
        import open3d as o3d
    
    io_util.cond_mkdir('./out')

    model, trainer, render_kwargs_train, render_kwargs_test, render_fn = get_model(args)
    if args.load_pt is None:
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.training.exp_dir, 'ckpts'))[-1]
    else:
        ckpt_file = args.load_pt
    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=args.device)
    model.load_state_dict(state_dict['model'])
    model.to(args.device)
    
    if args.use_surface_render:
        assert args.use_surface_render == 'sphere_tracing' or args.use_surface_render == 'root_finding'
        from models.ray_casting import surface_render
        render_fn = functools.partial(surface_render, model=model, ray_casting_algo=args.use_surface_render)
    
    if args.alter_radiance is not None:
        state_dict = torch.load(args.alter_radiance, map_location=args.device)
        radiance_state_dict = {}
        for k, v in state_dict['model'].items():
            if 'radiance_net' in k:
                newk = k.replace('radiance_net.', '')
                radiance_state_dict[newk] = v
        model.radiance_net.load_state_dict(radiance_state_dict)

    from dataio import get_data
    dataset = get_data(args, downscale=args.downscale)

    (_, model_input, ground_truth) = dataset[0]
    intrinsics = model_input["intrinsics"].cuda()
    H, W = (dataset.H, dataset.W)
    # NOTE: fx, fy should be scalec with the same ratio. Different ratio will cause the picture itself be stretched.
    #       fx=intrinsics[0,0]                   fy=intrinsics[1,1]
    #       cy=intrinsics[1,2] for H's scal      cx=intrinsics[0,2] for W's scale
    if args.H is not None:
        intrinsics[1,2] *= (args.H/dataset.H)
        H = args.H
    if args.H_scale is not None:
        H = int(dataset.H * args.H_scale)
        intrinsics[1,2] *= (H/dataset.H)

    if args.W is not None:
        intrinsics[0,2] *= (args.W/dataset.W)
        W = args.W
    if args.W_scale is not None:
        W = int(dataset.W * args.W_scale)
        intrinsics[0,2] *= (W/dataset.W)
    log.info("=> Rendering resolution @ [{} x {}]".format(H, W))

    c2ws = torch.stack(dataset.c2w_all, dim=0).data.cpu().numpy()

    #-----------------
    # Spiral path
    #   original nerf-like spiral path
    #-----------------
    if args.camera_path == 'spiral':
        c2w_center = poses_avg(c2ws)
        up = c2ws[:, :3, 1].sum(0)
        rads = np.percentile(np.abs(c2ws[:, :3, 3]), 30, 0)
        focus_distance = np.mean(np.linalg.norm(c2ws[:, :3, 3], axis=-1))
        render_c2ws = c2w_track_spiral(c2w_center, up, rads, focus_distance*0.8, zrate=0.0, rots=1, N=args.num_views)
    #-----------------
    # https://en.wikipedia.org/wiki/Spiral#Spherical_spirals
    #   assume three input views are on a small circle, then generate a spherical spiral path based on the small circle
    #-----------------
    elif args.camera_path == 'spherical_spiral':
        up_angle = np.pi / 3.
        n_rots = 2.2
        
        view_ids = args.camera_inds.split(',')
        assert len(view_ids) == 3, 'please select three views on a small circle, in CCW order (from above)'
        view_ids = [int(v) for v in view_ids]
        centers = c2ws[view_ids, :3, 3]
        centers_norm = np.linalg.norm(centers, axis=-1)
        radius = np.max(centers_norm)
        centers = centers * radius / centers_norm
        vec0 = centers[1] - centers[0]
        vec1 = centers[2] - centers[0]
        # the axis vertical to the small circle's area
        up_vec = normalize(np.cross(vec0, vec1))
                
        # key rotations of a spherical spiral path
        sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, args.num_views)
        sphere_phis = np.linspace(0, up_angle, args.num_views)
        
        if True:
            # use the origin as the focus center
            focus_center = np.zeros([3])
        else:
            # use the center of the small circle as the focus center
            focus_center = np.dot(up_vec, centers[0]) * up_vec
        
        # first rotate about up vec
        rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
        render_centers = rots_theta.apply(centers[0])
        # then rotate about horizontal vec
        horizontal_vec = normalize(np.cross(render_centers-focus_center[None, :], up_vec[None, :], axis=-1))
        rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
        render_centers = rots_phi.apply(render_centers)
        
        render_c2ws = look_at(render_centers, focus_center[None, :], up=-up_vec)
        
        if args.debug:
            # plot camera path
            intr = intrinsics.data.cpu().numpy()
            extrs = np.linalg.inv(render_c2ws)
            visualize_cam_spherical_spiral(intr, extrs, up_vec, centers[0], focus_center, n_rots, up_angle)
            
    #------------------
    # Small Circle Path: 
    #   assume three input views are on a small circle, then interpolate along this small circle
    #------------------
    elif args.camera_path == 'small_circle':
        view_ids = args.camera_inds.split(',')
        assert len(view_ids) == 3, 'please select three views on a small circle, int CCW order (from above)'
        view_ids = [int(v) for v in view_ids]
        centers = c2ws[view_ids, :3, 3]
        centers_norm = np.linalg.norm(centers, axis=-1)
        radius = np.max(centers_norm)
        centers = centers * radius / centers_norm
        vec0 = centers[1] - centers[0]
        vec1 = centers[2] - centers[0]
        # the axis vertical to the small circle
        up_vec = normalize(np.cross(vec0, vec1))
        # length of the chord between c0 and c2
        len_chord = np.linalg.norm(vec1, axis=-1)
        # angle of the smaller arc between c0 and c1
        full_angle = np.arcsin(len_chord/2/radius) * 2.
        
        all_angles = smoothed_motion_interpolation(full_angle, args.num_views)
        
        rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
        centers = rots.apply(centers[0])
        
        # get c2w matrices
        render_c2ws = look_at(centers, np.zeros_like(centers), up=-up_vec)
        
        if args.debug:
            # plot camera path
            intr = intrinsics.data.cpu().numpy()
            extrs = np.linalg.inv(render_c2ws)
            visualize_cam_on_circle(intr, extrs, up_vec, centers[0])
    #-----------------
    # Interpolate path
    #   directly interpolate among all input views
    #-----------------
    elif args.camera_path == 'interpolation':
        # c2ws = c2ws[:25]  # NOTE: [:20] fox taxi dataset
        key_rots = R.from_matrix(c2ws[:, :3, :3])
        key_times = list(range(len(key_rots)))
        slerp = Slerp(key_times, key_rots)
        interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
        render_c2ws = []
        for i in range(args.num_views):
            time = float(i) / args.num_views * (len(c2ws) - 1)
            cam_location = interp(time)
            cam_rot = slerp(time).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = cam_rot
            c2w[:3, 3] = cam_location
            render_c2ws.append(c2w)
        render_c2ws = np.stack(render_c2ws, axis=0)
    #------------------
    # Great Circle Path: 
    #   assume two input views are on a great circle, then interpolate along this great circle
    #------------------
    elif args.camera_path == 'great_circle':
        # to interpolate along a great circle that pass through the c2w center of view0 and view1
        view01 = args.camera_inds.split(',')
        assert len(view01) == 2, 'please select two views on a great circle, in CCW order (from above)'
        view0, view1 = [int(s) for s in view01]
        c0 = c2ws[view0, :3, 3]
        c0_norm = np.linalg.norm(c0)
        c1 = c2ws[view1, :3, 3]
        c1_norm = np.linalg.norm(c1)
        # the radius of the great circle
        # radius = (c0_norm+c1_norm)/2.
        radius = max(c0_norm, c1_norm)
        # re-normalize the c2w centers to be on the exact same great circle
        c0 = c0 * radius / c0_norm
        c1 = c1 * radius / c1_norm
        # the axis vertical to the great circle
        up_vec = normalize(np.cross(c0, c1))
        # length of the chord between c0 and c1
        len_chord = np.linalg.norm(c0-c1, axis=-1)
        # angle of the smaller arc between c0 and c1
        full_angle = np.arcsin(len_chord/2/radius) * 2.
        
        all_angles = smoothed_motion_interpolation(full_angle, args.num_views)
        
        # get camera centers
        rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
        centers = rots.apply(c0)
        
        # get c2w matrices
        render_c2ws = look_at(centers, np.zeros_like(centers), up=-up_vec)
        
        if args.debug:
            # plot camera path
            intr = intrinsics.data.cpu().numpy()
            extrs = np.linalg.inv(render_c2ws)
            visualize_cam_on_circle(intr, extrs, up_vec, centers[0])
    else:
        raise RuntimeError("Please choose render type between [spiral, interpolation, small_circle, great_circle, spherical_spiral]")
    log.info("=> Camera path: {}".format(args.camera_path))

    rgb_imgs = []
    depth_imgs = []
    normal_imgs = []
    # save mesh render images
    mesh_imgs = []
    render_kwargs_test['rayschunk'] = args.rayschunk

    if do_render_mesh:
        log.info("=> Load mesh: {}".format(args.render_mesh))
        geometry = o3d.io.read_triangle_mesh(args.render_mesh)
        geometry.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=W, height=H, visible=args.debug)
        ctrl = vis.get_view_control()
        vis.add_geometry(geometry)
        # opt = vis.get_render_option()
        # opt.mesh_show_back_face = True
        
        cam = ctrl.convert_to_pinhole_camera_parameters()
        intr = intrinsics.data.cpu().numpy()
        # cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], intr[0,2], intr[1,2])
        cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], W/2-0.5, H/2-0.5)
        ctrl.convert_from_pinhole_camera_parameters(cam)
        

        
    for c2w in tqdm(render_c2ws, desc='rendering...') :
        if not args.debug and not args.disable_rgb:
            rays_o, rays_d, select_inds = rend_util.get_rays(torch.from_numpy(c2w).float().cuda()[None, ...], intrinsics[None, ...], H, W, N_rays=-1)
            with torch.no_grad():
                # NOTE: detailed_output set to False to save a lot of GPU memory.
                rgb, depth, extras = render_fn(
                    rays_o, rays_d, show_progress=True, calc_normal=True, detailed_output=False, **render_kwargs_test)
                depth = depth.data.cpu().reshape(H, W, 1).numpy()
                depth = depth/depth.max()
                rgb_imgs.append(rgb.data.cpu().reshape(H, W, 3).numpy())
                depth_imgs.append(depth)
                if args.use_surface_render:
                    normals = extras['normals_surface']
                else:
                    normals = extras['normals_volume']
                normals = normals.data.cpu().reshape(H, W, 3).numpy()
                # if True:
                #     # (c2w^(-1) @ n)^T = n^T @ c2w^(-1)^T = n^T @ c2w
                #     normals = normals @ c2w[:3, :3]
                normal_imgs.append(normals/2.+0.5)

        if do_render_mesh:
            extr = np.linalg.inv(c2w)
            cam.extrinsic = extr
            ctrl.convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()
            if not args.debug:
                rgb_mesh = vis.capture_screen_float_buffer(do_render=True)
                mesh_imgs.append(np.asarray(rgb_mesh))

    def integerify(img):
        return (img*255.).astype(np.uint8)
    
    rgb_imgs = [integerify(img) for img in rgb_imgs]
    depth_imgs = [integerify(img) for img in depth_imgs]
    normal_imgs = [integerify(img) for img in normal_imgs]
    mesh_imgs = [integerify(img) for img in mesh_imgs]
        

    if not args.debug:
        if args.outbase is None:
            outbase = args.expname
        else:
            outbase = args.outbase
        post_fix = '{}x{}_{}_{}'.format(H, W, args.num_views, args.camera_path)
        if args.use_surface_render:
            post_fix = post_fix + '_{}'.format(args.use_surface_render)
        if not args.disable_rgb:
            imageio.mimwrite(os.path.join('out', '{}_rgb_{}.mp4'.format(outbase, post_fix)), rgb_imgs, fps=args.fps, quality=10)
            imageio.mimwrite(os.path.join('out', '{}_depth_{}.mp4'.format(outbase, post_fix)), depth_imgs, fps=args.fps, quality=10)
            imageio.mimwrite(os.path.join('out', '{}_normal_{}.mp4'.format(outbase, post_fix)), normal_imgs, fps=args.fps, quality=10)
            rgb_and_normal_imgs = [np.concatenate([rgb, normal], axis=0) for rgb, normal in zip(rgb_imgs, normal_imgs)]
            imageio.mimwrite(os.path.join('out', '{}_rgb&normal_{}.mp4'.format(outbase, post_fix)), rgb_and_normal_imgs, fps=args.fps, quality=10)
        if do_render_mesh:
            vis.destroy_window()
            imageio.mimwrite(os.path.join('out', '{}_mesh_{}.mp4'.format(outbase, post_fix)), mesh_imgs, fps=args.fps, quality=10)
            if not args.disable_rgb:
                rgb_and_mesh_imgs = [np.concatenate([rgb, mesh], axis=0) for rgb, mesh in zip(rgb_imgs, mesh_imgs)]
                imageio.mimwrite(os.path.join('out', '{}_rgb&mesh_{}.mp4'.format(outbase, post_fix)), rgb_and_mesh_imgs, fps=args.fps, quality=10)
                rgb_and_normal_and_mesh_imgs = [np.concatenate([rgb, normal, mesh], axis=0) for rgb, normal, mesh in zip(rgb_imgs, normal_imgs, mesh_imgs)]
                imageio.mimwrite(os.path.join('out', '{}_rgb&normal&mesh_{}.mp4'.format(outbase, post_fix)), rgb_and_normal_and_mesh_imgs, fps=args.fps, quality=10)
                

if __name__ == "__main__":
    # Arguments
    # "./configs/neus.yaml"
    parser = io_util.create_args_parser()
    parser.add_argument("--num_views", type=int, default=200)
    parser.add_argument("--render_mesh", type=str, default=None, help='the mesh ply file to be rendered')
    parser.add_argument("--device", type=str, default='cuda', help='render device')
    parser.add_argument("--downscale", type=float, default=1)
    parser.add_argument("--rayschunk", type=int, default=4096)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--camera_path", type=str, default="interpolation", help="choose between [spiral, interpolation, small_circle, great_circle, spherical_spiral]")
    parser.add_argument("--camera_inds", type=str, help="params for generating camera paths", default='11,15')
    parser.add_argument("--load_pt", type=str, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--H_scale", type=float, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--W_scale", type=float, default=None)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--disable_rgb", action='store_true')
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--alter_radiance", type=str, default=None, help='alter the radiance net with another trained ckpt.')
    parser.add_argument("--outbase", type=str, default=None, help='base of output filename')
    parser.add_argument("--use_surface_render", type=str, default=None, help='choose between [sphere_tracing, root_finding]. \n\t Use surface rendering instead of volume rendering \n\t NOTE: way faster, but might not be the original model behavior')
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    main_function(config)