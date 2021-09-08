from utils.io_util import glob_imgs, load_rgb

import os
import numpy as np
from tqdm import tqdm

import torch


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 scale_radius=-1):
        super().__init__()
        
        self.instance_dir = data_dir
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        
        self.train_cameras = train_cameras
        self.downscale = downscale
        
        image_dir = '{0}/blended_images'.format(self.instance_dir)
        # cam_dir = '{0}/cams'.format(self.instance_dir)
        cam_dir = '{0}/cams_normalized'.format(self.instance_dir)
        
        self.intrinsics_all = []
        self.c2w_all = []
        self.rgb_images = []
        self.basenames = []
        cam_center_norms = []
        for imgpath in tqdm(sorted(glob_imgs(image_dir)), desc='loading data...'):
            if 'masked' in imgpath:
                pass
            else:
                basename = os.path.splitext(os.path.split(imgpath)[-1])[0]
                self.basenames.append(basename)
            
                camfilepath = os.path.join(cam_dir, "{}_cam.txt".format(basename))
                assert os.path.exists(camfilepath)
                extrinsics, intrinsics = load_cam(camfilepath)
                c2w = np.linalg.inv(extrinsics)
                cam_center_norms.append(np.linalg.norm(c2w[:3,3]))
                
                # downscale intrinsics
                intrinsics[0, 2] /= downscale
                intrinsics[1, 2] /= downscale
                intrinsics[0, 0] /= downscale
                intrinsics[1, 1] /= downscale
                
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.c2w_all.append(torch.from_numpy(c2w).float())
            
                rgb = load_rgb(imgpath, downscale)
                _, self.H, self.W = rgb.shape
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())

        max_cam_norm = max(cam_center_norms)
        if scale_radius > 0:
            for i in range(len(self.c2w_all)):
                self.c2w_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)

        self.n_images = len(self.rgb_images)


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        # uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        # uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "intrinsics": self.intrinsics_all[idx],
        }

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

    def get_gt_pose(self):
        return torch.stack(self.c2w_all, dim=0)

# modified from https://github.com/YoYo000/MVSNet/blob/master/mvsnet/preprocess.py
def load_cam(filepath, interval_scale=1, original_blendedmvs=False):
    """ read camera txt file """
    cam = np.repeat(np.eye(4)[None, ...], repeats=2, axis=0)
    words = open(filepath).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if original_blendedmvs:
        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            # cam[1][3][2] = FLAGS.max_d
            cam[1][3][2] = 128 # NOTE: manually fixed here.
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

    return cam


def write_cam(filepath, cam):
    f = open(filepath, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


if __name__ == '__main__':
    def test():
        dataset = SceneDataset(False, './data/BlendedMVS/BlendedMVS/5c0d13b795da9479e12e2ee9', scale_radius=3.0)
        c2w = dataset.get_gt_pose().data.cpu().numpy()
        extrinsics = np.linalg.inv(c2w)
        camera_matrix = next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()
        from tools.vis_camera import visualize
        visualize(camera_matrix, extrinsics)
    test()