# GSRasterizerLib

不同GS可微光栅器的集合，用于测试不同光栅器的性能

## Environment

``` sh
conda create -n rasterizer_test python=3.10 -y
conda activate rasterizer_test
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

## Install

``` sh
git clone https://github.com/Ru1zhi/GSRasterizerLib.git
./install.sh
```

### Usage

``` python
# origin 3DGS Rasterizer
from diff_gaussian_rasterization_origin import GaussianRasterizationSettings as Origin
```

## Rasterizers

| Methods | image | alpha | depth | normal | mid_depth | point_map | mid_point_map |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3DGS (origin) | :white_check_mark: |  |  |  |  |  |  |
| 3DGS (depth) | :white_check_mark: | :white_check_mark: | :white_check_mark: |  |  |  |  |
| 2DGS | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |  |  |
| RaDe | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

### 3DGS(Origin)

url: <https://github.com/graphdeco-inria/diff-gaussian-rasterization>

commit: 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d

paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering

### 3DGS(depth)

来自DreamGaussian的GS光栅器实现

url: <https://github.com/ashawkey/diff-gaussian-rasterization>

commit: 8829d14f814fccdaf840b7b0f3021a616583c0a1

paper: DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

### 2DGS

来自2DGS的光栅器实现

url: <https://github.com/hbb1/diff-surfel-rasterization>

commit: e0ed0207b3e0669960cfad70852200a4a5847f61

paper: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

### RaDe

来自RaDe的光栅器实现

url: <https://github.com/BaowenZ/RaDe-GS>

commit: ec9199f5eebdec92b22e2d6e7314cbb316101d59

paper: RaDe-GS: Rasterizing Depth in Gaussian Splatting
