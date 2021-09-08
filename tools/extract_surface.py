from models.base import ImplicitSurface
from utils.mesh_util import extract_mesh

import torch

def main_function(args):
    N = args.N
    s = args.volume_size
    implicit_surface = ImplicitSurface(radius_init=args.init_r).cuda()
    if args.load_pt is not None:
        # --------- if load statedict
        # state_dict = torch.load("/home/PJLAB/guojianfei/latest.pt")
        # state_dict = torch.load("./dev_test/37/latest.pt")
        state_dict = torch.load(args.load_pt)
        imp_surface_state_dict = {k.replace('implicit_surface.',''):v for k, v in state_dict['model'].items() if 'implicit_surface.' in k}
        imp_surface_state_dict['obj_bounding_size'] = torch.tensor([1.0]).cuda()
        implicit_surface.load_state_dict(imp_surface_state_dict)
    if args.out is None:
        from datetime import datetime
        dt = datetime.now()
        args.out = 'surface_' + dt.strftime("%Y%m%d%H%M%S") + '.ply'
    extract_mesh(implicit_surface, s, N=N, filepath=args.out, show_progress=True, chunk=args.chunk)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help='output ply file name')
    parser.add_argument('--N', type=int, default=512, help='resolution of the marching cube algo')
    parser.add_argument('--volume_size', type=float, default=2., help='voxel size to run marching cube')
    parser.add_argument("--load_pt", type=str, default=None, help='the trained model checkpoint .pt file')
    parser.add_argument("--chunk", type=int, default=16*1024, help='net chunk when querying the network. change for smaller GPU memory.')
    parser.add_argument("--init_r", type=float, default=1.0, help='Optional. The init radius of the implicit surface.')
    args = parser.parse_args()
    
    main_function(args)