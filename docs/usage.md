- [Environment](#environment)
  - [hardware](#hardware)
  - [software](#software)
    - [(optional)](#optional)
- [Dataset preparation](#dataset-preparation)
  - [DTU](#dtu)
  - [BlendedMVS](#blendedmvs)
- [Training](#training)
  - [new training](#new-training)
  - [resume training](#resume-training)
  - [monitoring & logging](#monitoring--logging)
  - [configs](#configs)
  - [training on multi-gpu or clusters](#training-on-multi-gpu-or-clusters)
- [Evaluation: mesh extraction](#evaluation-mesh-extraction)
- [Evaluation: free viewport rendering](#evaluation-free-viewport-rendering)
  - [Before rendering, debug camera trajectory by visualization](#before-rendering-debug-camera-trajectory-by-visualization)
  - [Only render RGB & depth & normal images](#only-render-rgb--depth--normal-images)
  - [Only render mesh](#only-render-mesh)
  - [Render RGB, depth image, normal image, and mesh](#render-rgb-depth-image-normal-image-and-mesh)
  - [:pushpin: Use surface rendering, instead of volume rendering](#pushpin-use-surface-rendering-instead-of-volume-rendering)
- [[WIP] to run on your own datasets](#wip-to-run-on-your-own-datasets)
  - [prerequisites](#prerequisites)

## Environment

### hardware

- Currently tested with RTX3090 with 24GiB GPU memotry, and tested with clusters.
- :pushpin: setting larger `data:val_downscale` and smaller `data:val_rayschunk` in the configs will reduce GPU memory usage.
  - model size is quite small: `~10MiB`, since the model is just several MLPs. The rendering consumes a lot of GPU.

|                | GPU Memory required<br>@ val_downscale=8 & val_rayschunk=256 |
| -------------- | ------------------------------------------------------------ |
| UNISURF        | >= 6 GiB                                                     |
| NeuS @w/o mask | >= 9 GiB                                                     |
| VolSDF         | >=11 GiB                                                     |

### software

- python>=3.6 (tested on python=3.8.5 using conda)
- pytorch>=1.6.0
- `pip install tqdm scikit-image opencv-python pandas tensorboard addict imageio imageio-ffmpeg pyquaternion scikit-learn pyyaml seaborn PyMCubes trimesh plyfile`

#### (optional)
- visualization of meshes: `pip install open3d`



## Dataset preparation

### DTU

Follow the download_data.sh script from IDR repo: [[click here]](https://github.com/lioryariv/idr/blob/main/data/download_data.sh).

> NOTE: For NeuS experiments, you can also use their versions of DTU data, see [[here]](https://drive.google.com/drive/folders/1Nlzejs4mfPuJYORLbDEUDWlc9IZIbU0C?usp=sharing)。
>
> The camera normalizations/scaling are similar, except they seems to additionally adjust the rotations so that objects are in the same canonical frame.
>
> And then, add `data: cam_file: cameras_sphere.npz` to the configs (this is already by default for neus_xxx.yaml in this repo.

### BlendedMVS

From the BlendedMVS repo: [[click here]](https://github.com/YoYo000/BlendedMVS), Download the low-res set `BlendedMVS.zip`(27.5GB) of BlendedMVS dataset.

Download the `BlendedMVS_norm_cam.tar.gz` from [[GoogleDrive]](https://drive.google.com/drive/folders/1B7y-nMFO9noVI0byU34yPTRtqqzMdMIQ?usp=sharing) or [[Baidu, code: `reco`]](https://pan.baidu.com/s/10g1IWwrGrpE--VJ5XLuRFw), and extract them into the same folder of the extracted `BlendedMVS.zip`.

The final file structure would be:

```python
BlendedMVS
├── 5c0d13b795da9479e12e2ee9
│   ├── blended_images      # original. not changed.
│   ├── cams                # original. not changed.
│   ├── rendered_depth_maps # original. not changed.
│   ├── cams_normalized     # newly appended normalized cameras.
├── ...
```

> NOTE: normalization of some of the instances of blendedMVS failed. You can refer to [[dataio/blendedmvs_normalized.txt]](dataio/blendedmvs_normalized.txt) for the succeeded list.

> :warning: WARNING: the normalization method is currently not fully tested on all BlendedMVS instances, and the normalized camera file may be updated.

## Training

### new training

```shell
python -m train --config configs/volsdf.yaml
```

or you can use any of the configs in the [configs](../configs) folder;

or you can create new configs on your own.

For training on multi-GPU or clusters, see section [[training on multi-gpu or clusters]](#training-on-multi-gpu-or-clusters)。

### resume training

```shell
# replace xxx with specific expname
python -m train --resume_dir ./logs/xxx/
```

**Or**, simply re-use the original config file:

```shell
# replace xxx with specific filename
python -m train --config configs/xxx.yaml
```

### monitoring & logging

```shell
# replace xxx with specific expname
tensorboard --logdir logs/xxx/events/
```

the whole logging directory is structured as follows:

```python
logs
├── exp1
│   ├── backup      # backup codes
│   ├── ckpts       # saved checkpoints
│   ├── config.yaml # the training config
│   ├── events      # tensorboard events
│   ├── imgs        # validation image output    # NOTE: default validation image is 4 time down-sampled.
│   └── stats.p     # saved scalars stats (lr, losses, value max/mins, etc.)
├── exp2
└── ...
```

### configs

You can run different experiments by running different configs files. All of the config files of implemented papers (NeuS, VolSDF and UNISURF) can be found in the [[configs]](../configs) folder.

```python
configs
├── neus.yaml                   # NeuS, training with mask
├── neus_nomask.yaml            # NeuS, training without mask, using NeRF++ as background
├── neus_nomask_blended.yaml    # NeuS, training without mask, using NeRF++ as background, for training with BlendedMVS dataset.
├── unisurf.yaml                # UNISURF
├── volsdf.yaml                 # VolSDF
├── volsdf_nerfpp.yaml          # VolSDF, with NeRF++ as background
├── volsdf_nerfpp_blended.yaml  # VolSDF, with NeRF++ as background, for training with BlendedMVS dataset.
└── volsdf_siren.yaml           # VolSDF, with SIREN replaces ReLU activation.
```

### training on multi-gpu or clusters

This repo has full tested support for the following training conditions: 

- single GPU
- `nn.DataParallel` 
- `DistributedDataParallel` with `torch.distributed.launch` 
- `DistributedDataParallel` with `SLURM`.

#### single process with single GPU

```shell
python -m train --config xxx
```

Or force to run on one GPU if you have multiple GPUs

```shell
python -m train --config xxx --device_ids 0
```

#### single process with multiple GPU (`nn.DataParallel`)

Automatically supported since default `device_ids` is set to `-1`, which means using all available GPUs.

Or, you can specify used GPUs manually: (for example)

```shell
python -m train --config xxx --device_ids 1,0,5,3
```

#### multi-process with multiple local GPU (`DistributedDataParallel` with `torch.distributed.launch`)

Add `--ddp` when calling `train.py`.

```shell
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py --ddp --config xxx
```

#### multi-process with clusters (`DistributedDataParallel` with `SLURM`)

Add `--ddp` when calling `train.py`.

```shell
srun --partition your_partition --mpi pmi2 --gres=gpu:4 --cpus-per-task=7 --ntasks-per-node=4 -n4 \
  --kill-on-bad-exit=1 python -m train --ddp --config xxx --expname cluster_test --training:monitoring none --training:i_val 2000
```



## Evaluation: mesh extraction

```shell
python -m tools.extract_surface --load_pt /path/to/xxx.pt --N 512 --volume_size 2.0 --out /path/to/surface.ply
```



## Evaluation: free viewport rendering

### Before rendering, debug camera trajectory by visualization

| camera trajectory type | example                                                    | explanation and command line options                         |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| `small_circle`         | ![cam_small_circle](../media/cam_small_circle.gif)         | select 3 view ids, in CCW order (from above);<br>when rendering, will interpolate camera paths along the **small circle** that pass through the selected 3 camera center locations<br>`--camera_path small_circle --camera_inds 11,14,17` |
| `great_circle`         | ![cam_great_circle](../media/cam_great_circle.gif)         | select 2 view ids, in CCW order (from above);<br>when rendering, will interpolate camera paths along the **great circle** that pass through the selected 2 camera center locations<br>`--camera_path great_circle --camera_inds 11,17` |
| `spherical_spiral`     | ![cam_spherical_spiral](../media/cam_spherical_spiral.gif) | select 3 view ids, in CCW order (from above);<br>when rendering, will interpolate camera paths along the **spherical spiral path** that starts from the **small circle** that pass through these 3 selected camera centers <br>`--camera_path spherical_spiral --camera_inds 11,14,17` |

Add `--debug` option.

```shell
python -m tools.render_view --config trained_models/volsdf_114/config.yaml --load_pt trained_models/volsdf_114/final_00100000.pt --camera_path spherical_spiral --camera_inds 48,43,38 --debug --num_views 20
```

You can replace the `camera_path` and `camera_inds` with any of the camera path configurations you like.

> NOTE: remember to remove the --debug option after debugging.

### Only render RGB & depth & normal images

For GPUs with smaller GPU memory, use smaller `rayschunk`, and larger `downscale`.

```shell
python -m tools.render_view --num_views 60 --downscale 4 --config trained_models/volsdf_24/config.yaml \
--load_pt trained_models/volsdf_24/final_00100000.pt --camera_path small_circle --rayschunk 1024 \
--camera_inds 27,24,21 --H_scale 1.2
```

### Only render mesh

Add `--disable_rgb` option.

### Render RGB, depth image, normal image, and mesh

Add `--render_mesh /path/to/xxx.ply`. Example:

```shell
python -m tools.render_view --num_views 60 --downscale 4 --config trained_models/volsdf_24/config.yaml \
--load_pt trained_models/volsdf_24/final_00100000.pt --camera_path small_circle --rayschunk 1024 \
--render_mesh trained_models/volsdf_24/surface.ply --camera_inds 27,24,21 --H_scale 1.2
```

### :pushpin: Use surface rendering, instead of volume rendering

Since the underlying shape representation is a implicit surface, one can use surface rendering techniques to render the image. For each ray, only the point intersected with the surface will have contribution to its pixel color, instead of considering neighboring points along the ray as in volume rendering.

This will boost rendering speed **100x** faster using `sphere_tracing`.

- Specifically, for NeuS/VolSDF which utilize SDF representation, you can render with `sphere_tracing` or `root_finding` along the ray.

Just add `--use_surface_render sphere_tracing`.

```shell
python -m tools.render_view --device_ids 0 --num_views 60 --downscale 4 --num_views 60 --downscale 4 \
--config trained_models/neus_65_nomask/config.yaml --load_pt trained_models/neus_65_nomask/final_00300000.pt \
--camera_path small_circle --camera_inds 11,13,15  --H_scale 1.2 --outbase neus_st_65 \
--use_surface_render sphere_tracing --rayschunk 1000000
```

> NOTE: in this case, the `rayschunk` can be very very large, 1000000 for example, since only ONE point on the ray is queried.

Example @NeuS @DTU-65 @[360 x 400] resolution @60 frames rendering.

|                 | Original volume rendering<br>& Integrated normals of volume rendering | Surface rendering using `sphere_tracing`<br>& Normals from the ray-traced surface points |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| rendering time  | 28 minutes                                                   | 18 seconds                                                   |
| rendered result | ![neus_65_nomask_new_rgb&normal_360x400_60_small_circle_None](../media/neus_65_nomask_new_rgb&normal_360x400_60_small_circle_None.gif) | ![neus_65_nomask_new_rgb&normal_360x400_60_small_circle_sphere_tracing_None](../media/neus_65_nomask_new_rgb&normal_360x400_60_small_circle_sphere_tracing_None.gif) |

> NOTE:  the NeRF++ background is removed when sphere tracing.



- For UNISURF which utilize OccupancyNet representation, you can render with `--use_surface_render root_finding`.



## [WIP] to run on your own datasets

### prerequisites

- [COLMAP](https://github.com/colmap/colmap) for extracting camera extrinsics
- To run on your own masks:
  - annotation tool: [CVAT](https://github.com/openvinotoolkit/cvat) or their online annotation site: [cvat.org](https://cvat.org/)
  - load coco mask: `pip install pycocotools`
