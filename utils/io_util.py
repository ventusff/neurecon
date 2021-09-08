from utils.print_fn import log

import os
import copy
import yaml
import glob
import addict
import shutil
import imageio
import argparse
import functools
import numpy as np


import torch
import skimage
from skimage.transform import rescale

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs

# def find_files(dir, exts=['*.png', '*.jpg']):
#     if os.path.isdir(dir):
#         # types should be ['*.png', '*.jpg']
#         files_grabbed = []
#         for ext in exts:
#             files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
#         if len(files_grabbed) > 0:
#             files_grabbed = sorted(files_grabbed)
#         return files_grabbed
#     else:
#         return []

def load_rgb(path, downscale=1):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        img = rescale(img, 1./downscale, anti_aliasing=False, multichannel=True)

    # NOTE: pixel values between [-1,1]
    # img -= 0.5
    # img *= 2.
    img = img.transpose(2, 0, 1)
    return img

def load_mask(path, downscale=1):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    if downscale != 1:
        alpha = rescale(alpha, 1./downscale, anti_aliasing=False, multichannel=False)
    object_mask = alpha > 127.5

    return object_mask

def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    NewCls.__name__ = cls.__name__  # to preserve old class name.

    return NewCls


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def backup(backup_dir):
    """ automatic backup codes
    """
    log.info("=> Backing up... ")
    special_files_to_copy = []
    filetypes_to_copy = [".py"]
    subdirs_to_copy = ["", "dataio/", "models/", "tools/", "debug_tools/", "utils/"]

    this_dir = "./" # TODO
    cond_mkdir(backup_dir)
    # special files
    [
        cond_mkdir(os.path.join(backup_dir, os.path.split(file)[0]))
        for file in special_files_to_copy
    ]
    [
        shutil.copyfile(
            os.path.join(this_dir, file), os.path.join(backup_dir, file)
        )
        for file in special_files_to_copy
    ]
    # dirs
    for subdir in subdirs_to_copy:
        cond_mkdir(os.path.join(backup_dir, subdir))
        files = os.listdir(os.path.join(this_dir, subdir))
        files = [
            file
            for file in files
            if os.path.isfile(os.path.join(this_dir, subdir, file))
            and file[file.rfind("."):] in filetypes_to_copy
        ]
        [
            shutil.copyfile(
                os.path.join(this_dir, subdir, file),
                os.path.join(backup_dir, subdir, file),
            )
            for file in files
        ]
    log.info("done.")


def save_video(imgs, fname, as_gif=False, fps=24, quality=8, already_np=False, gif_scale:int =512):
    """[summary]

    Args:
        imgs ([type]): [0 to 1]
        fname ([type]): [description]
        as_gif (bool, optional): [description]. Defaults to False.
        fps (int, optional): [description]. Defaults to 24.
        quality (int, optional): [description]. Defaults to 8.
    """
    gif_scale = int(gif_scale)
    # convert to np.uint8
    if not already_np:
        imgs = (255 * np.clip(
            imgs.permute(0, 2, 3, 1).detach().cpu().numpy(), 0, 1))\
            .astype(np.uint8)
    imageio.mimwrite(fname, imgs, fps=fps, quality=quality)

    if as_gif:  # save as gif, too
        os.system(f'ffmpeg -i {fname} -r 15 '
                  f'-vf "scale={gif_scale}:-1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {os.path.splitext(fname)[0] + ".gif"}')


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    # assert nindex == nrows*ncols
    if nindex > nrows*ncols:
        nrows += 1
        array = np.concatenate([array, np.zeros([nrows*ncols-nindex, height, width, intensity])])
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


# modified from tensorboardX   https://github.com/lanpa/tensorboardX
def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        # image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_hwc

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image




#-----------------------------
# configs
#-----------------------------
class ForceKeyErrorDict(addict.Dict):
    def __missing__(self, name):
        raise KeyError(name)


def load_yaml(path, default_path=None):

    with open(path, encoding='utf8') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    if default_path is not None and path != default_path:
        with open(default_path, encoding='utf8') as default_yaml_file:
            default_config_dict = yaml.load(
                default_yaml_file, Loader=yaml.FullLoader)
            main_config = ForceKeyErrorDict(**default_config_dict)

        # def overwrite(output_config, update_with):
        #     for k, v in update_with.items():
        #         if not isinstance(v, dict):
        #             output_config[k] = v
        #         else:
        #             overwrite(output_config[k], v)
        # overwrite(main_config, config)

        # simpler solution
        main_config.update(config)
        config = main_config

    return config


def save_config(datadict: ForceKeyErrorDict, path: str):
    datadict = copy.deepcopy(datadict)
    datadict.training.ckpt_file = None
    datadict.training.pop('exp_dir')
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(datadict.to_dict(), outfile, default_flow_style=False)


def update_config(config, unknown):
    # update config given args
    for idx, arg in enumerate(unknown):
        if arg.startswith("--"):
            if (':') in arg:
                k1, k2 = arg.replace("--", "").split(':')
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx+1].lower() == 'true'
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx+1])
                    else:
                        v = unknown[idx+1]
                print(f'Changing {k1}:{k2} ---- {config[k1][k2]} to {v}')
                config[k1][k2] = v
            else:
                k = arg.replace('--', '')
                v = unknown[idx+1]
                argtype = type(config[k])
                print(f'Changing {k} ---- {config[k]} to {v}')
                config[k] = v

    return config


def create_args_parser():
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default=None, help='Path to config file.')
    parser.add_argument('--resume_dir', type=str, default=None, help='Directory of experiment to load.')
    return parser


def load_config(args, unknown, base_config_path=None):
    ''' overwrite seq
    command line param --over--> args.config --over--> default config yaml
    '''
    assert (args.config is not None) != (args.resume_dir is not None), "you must specify ONLY one in 'config' or 'resume_dir' "

    # NOTE: '--local_rank=xx' is automatically given by torch.distributed.launch (if used)
    #       BUT: pytorch suggest to use os.environ['LOCAL_RANK'] instead, and --local_rank=xxx will be deprecated in the future.
    #            so we are not using --local_rank at all.
    found_k = None
    for item in unknown:
        if 'local_rank' in item:
            found_k = item
            break
    if found_k is not None:
        unknown.remove(found_k)

    print("=> Parse extra configs: ", unknown)
    if args.resume_dir is not None:
        assert args.config is None, "given --config will not be used when given --resume_dir"
        assert '--expname' not in unknown, "given --expname with --resume_dir will lead to unexpected behavior."
        #---------------
        # if loading from a dir, do not use base.yaml as the default; 
        #---------------
        config_path = os.path.join(args.resume_dir, 'config.yaml')
        config = load_yaml(config_path, default_path=None)

        # use configs given by command line to further overwrite current config
        config = update_config(config, unknown)

        # use the loading directory as the experiment path
        config.training.exp_dir = args.resume_dir
        print("=> Loading previous experiments in: {}".format(config.training.exp_dir))
    else:
        #---------------
        # if loading from a config file
        # use base.yaml as default
        #---------------
        config = load_yaml(args.config, default_path=base_config_path)

        # use configs given by command line to further overwrite current config
        config = update_config(config, unknown)

        # use the expname and log_root_dir to get the experiement directory
        if 'exp_dir' not in config.training:
            config.training.exp_dir = os.path.join(config.training.log_root_dir, config.expname)

    # add other configs in args to config
    other_dict = vars(args)
    other_dict.pop('config')
    other_dict.pop('resume_dir')
    config.update(other_dict)

    if hasattr(args, 'ddp') and args.ddp:
        if config.device_ids != -1:
            print("=> Ignoring device_ids configs when using DDP. Auto set to -1.")
            config.device_ids = -1
    else:
        args.ddp = False
        # # device_ids: -1 will be parsed as using all available cuda device
        # # device_ids: [] will be parsed as using all available cuda device
        if (type(config.device_ids) == int and config.device_ids == -1) \
                or (type(config.device_ids) == list and len(config.device_ids) == 0):
            config.device_ids = list(range(torch.cuda.device_count()))
        # # e.g. device_ids: 0 will be parsed as device_ids [0]
        elif isinstance(config.device_ids, int):
            config.device_ids = [config.device_ids]
        # # e.g. device_ids: 0,1 will be parsed as device_ids [0,1]
        elif isinstance(config.device_ids, str):
            config.device_ids = [int(m) for m in config.device_ids.split(',')]
        print("=> Use cuda devices: {}".format(config.device_ids))

    return config