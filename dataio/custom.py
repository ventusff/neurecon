# modified from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py

import os
import json
import torch
import numpy as np
from tqdm import tqdm

from utils.io_util import load_mask, load_rgb, glob_imgs
from utils.rend_util import rot_to_quat, load_K_Rt_from_P

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""
    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 cam_file=None,
                 scale_radius=-1,
                 ):

        self.instance_dir = data_dir
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.train_cameras = train_cameras
        self.downscale = downscale

        image_dir = '{0}/images'.format(self.instance_dir)
        # image_paths = sorted(glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        # mask_paths = sorted(glob_imgs(mask_dir))
        mask_ignore_dir = '{0}/mask_out'.format(self.instance_dir)
        
        self.has_mask = os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0
        self.has_mask_out = os.path.exists(mask_ignore_dir) and len(os.listdir(mask_ignore_dir)) > 0

        self.cam_file = '{0}/cam.json'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = json.load(open(self.cam_file))
        
        self.n_images = len(camera_dict)
        
        cam_center_norms = []
        self.intrinsics_all = []
        self.c2w_all = []
        self.rgb_images = []
        self.object_masks = []
        self.masks_ignore = []
        for imgname, v in tqdm(camera_dict.items(), desc='loading dataset...'):
            world_mat = np.array(v['P'], dtype=np.float32).reshape(4,4)
            if 'SCALE' in v:
                scale_mat = np.array(v['SCALE'], dtype=np.float32).reshape(4,4)
                P = world_mat @ scale_mat
            else:
                P = world_mat
            intrinsics, c2w = load_K_Rt_from_P(P[:3, :4])
            cam_center_norms.append(np.linalg.norm(c2w[:3,3]))

            # downscale intrinsics
            intrinsics[0, 2] /= downscale
            intrinsics[1, 2] /= downscale
            intrinsics[0, 0] /= downscale
            intrinsics[1, 1] /= downscale

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.c2w_all.append(torch.from_numpy(c2w).float())

            rgb = load_rgb(os.path.join(image_dir, imgname), downscale)
            _, self.H, self.W = rgb.shape
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            fname_base = os.path.splitext(imgname)[0]
            
            if self.has_mask:
                object_mask = load_mask(os.path.join(mask_dir, "{}.png".format(fname_base)), downscale).reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

            if self.has_mask_out:
                mask_ignore = load_mask(os.path.join(mask_ignore_dir, "{}.png".format(fname_base)), downscale).reshape(-1)
                self.masks_ignore.append(torch.from_numpy(mask_ignore).to(dtype=torch.bool))

        max_cam_norm = max(cam_center_norms)
        if scale_radius > 0:
            for i in range(len(self.c2w_all)):
                self.c2w_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        # uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        # uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "intrinsics": self.intrinsics_all[idx],
        }
        if self.has_mask:
            sample["object_mask"] = self.object_masks[idx]
        if self.has_mask_out:
            sample["mask_ignore"] = self.masks_ignore[idx]

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        ground_truth["rgb"] = self.rgb_images[idx]

        if not self.train_cameras:
            sample["c2w"] = self.c2w_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_gt_pose(self, scaled=True):
        # Load gt pose without normalization to unit sphere
        camera_dict = json.load(open(self.cam_file))

        c2w_all = []
        for imgname, v in camera_dict.items():
            world_mat = np.array(v['P'], dtype=np.float32).reshape(4,4)
            if scaled and 'SCALE' in v:
                scale_mat = np.array(v['SCALE'], dtype=np.float32).reshape(4,4)
                P = world_mat @ scale_mat
            else:
                P = world_mat
            _, c2w = load_K_Rt_from_P(P[:3, :4])
            c2w_all.append(torch.from_numpy(c2w).float())

        return torch.cat([p.float().unsqueeze(0) for p in c2w_all], 0)

if __name__ == "__main__":
    # dataset = SceneDataset(False, './data/taxi/black')
    dataset = SceneDataset(False, './data/taxi/blue')
    c2w = dataset.get_gt_pose(scaled=True).data.cpu().numpy()
    extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    camera_matrix = next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()
    
    from tools.vis_camera import visualize
    visualize(camera_matrix, extrinsics)