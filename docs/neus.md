# Notes on the unbiased property of NeuS

In NeuS's solution, the maximum color contribution (visibility weights) has no first order bias with the exact surface.

- See this [[video]](https://longtimenohack.com/hosted/nerf-surface/neus_unbiased.mp4) (1.3MiB).
- To try it yourself, run: 

```shell
python -m debug_tools.plot_neus_bias
```

