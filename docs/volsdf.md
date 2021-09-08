

# Notes on the up-sampling algorithm and error bound of VolSDF

In VolSDF, they prove a error bound of the discontinuous Riemann Sum's approximations in the opacity's calculation, and derive a ray point up-sampling algorithm to control the error bound to keep smaller than manually set  `episilon`, which is set to `0.1`.

## 1. up sampling algorithm's visualization in tensorboard when training

|                                      | @0k                                                          | @4k                                                          | @10k                                                         | @200k                                                        |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| up sample iterations until converged | ![volsdf_up_iter_00000000](../media/volsdf_up_iter_00000000.png) | ![volsdf_up_iter_00004000](../media/volsdf_up_iter_00004000.png) | ![volsdf_up_iter_00010000](../media/volsdf_up_iter_00010000.png) | ![volsdf_up_iter_00200000](../media/volsdf_up_iter_00200000.png) |
| beta heat map                        | ![volsdf_beta_00000000](../media/volsdf_beta_00000000.png)   | ![volsdf_beta_00004000](../media/volsdf_beta_00004000.png)   | ![volsdf_beta_00010000](../media/volsdf_beta_00010000.png)   | ![volsdf_beta_00200000](../media/volsdf_beta_00200000.png)   |



## 2. up sampling algorithm single ray testing

- network's beta = 0.001
- eps = 0.1
- To try it yourself, run:

```shell
python -m debug_tools.test_volsdf_algo
```



### 0-th iteration

- 128 uniform sample points.
- If use the network's `beta`, then `error_bound.max`=inf, which does not satisfy `<eps=0.1`.
- for `beta+`=0.431, the plots are

![image-20210809041923683](../media/image-20210809041923683.png)

### 1-st iteration

- 256 sampling points.
- If use the network's `beta`, then `error_bound.max`=inf, which does not satisfy `<eps=0.1`.
- for `beta+`=0.049, the plots are

![image-20210809041939732](../media/image-20210809041939732.png)

### 2-nd iteration

- 384 sampling points
- If use the network's `beta`, then `error_bound.max = 1.3e5`, which does not satisfy `<eps=0.1`.
- for `beta+`=0.023, the plots are

![image-20210809042000880](../media/image-20210809042000880.png)

### 3-rd iteraion

- 512 sampling points
- If use the network's `beta`, then `error_bound.max = 0.570`, which does not satisfy `<eps=0.1`.
- for `beta+`=0.013, the plots are

![image-20210809042044843](../media/image-20210809042044843.png)

### 4-th iteration

- 640 sampling points
- If use the network's `beta`, then `error_bound.max = 0.116`, which does not satisfy `<eps=0.1`.
- for `beta+`=0.001, the plots are

![image-20210809042204675](../media/image-20210809042204675.png)

### 5-th iteration

- 768 sampling points
- If use the network's `beta`, then `error_bound.max = 0.0220`, which **satisfies** `<eps=0.1`. The plots are:

![image-20210809042455605](../media/image-20210809042455605.png)