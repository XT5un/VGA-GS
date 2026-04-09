# VGA-GS

Optimizing Sparse-view Gaussian Splatting in View Space with Global Monocular Prior Alignment

## Environment

Git clone this repository.

``` bash
git clone https://github.com/Ru1zhi/VGA-GS.git
cd VGA-GS
git submodule update --init --recursive
```

Create conda environment.

``` bash
conda create -n vgags python=3.10.14 -y
conda activate vgags
conda install -c conda-forge ffmpeg
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install submodules

``` sh
cd submodules/GSRasterizerLib
./install.sh
cd ../..
```

``` sh
pip install submodules/simple-knn
```

Download image matcher and depth estimator weights to `~/.cache/torch/hub/checkpoints`.

| Model | Link |
| ----- | ---- |
| gim_dkm | [Link](https://github.com/xuelunshen/gim?tab=readme-ov-file#-usage) |
| Metric3Dv2 (DINO2reg-ViT-giant2) | [Link](https://github.com/YvanYin/Metric3D#%EF%B8%8F-inference) |
| Depth-AnythinV2 (Depth-Anything-V2-Large) | [Link](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models) |

## Dataset

You can download our processed data here.

## Run

To run the optimization, use the following command:

``` bash
base ./scripts/train_replica.sh
```

## Acknowledgments

This work use the rasterizer of `2dgs` and `RaDe-GS`, monocular depth estimation `DepthAnythingV2`, `Metric3DV2` and `DepthPro`, and image matcher `GIM`. And the authors are thanked for their open source behaviour.
