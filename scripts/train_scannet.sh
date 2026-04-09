#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
rasterizer_backend=$2
depth_estimation_method=$3

dataset_name="ScanNet"
data_root_dir="/hdd24T/sunxt/dataset/few_shot_datasets/Scannet_DDP"
output_root_dir="/hdd24T/sunxt/VGA-GS_comparisons/ours_ps15"
exp_name="${rasterizer_backend}_${depth_estimation_method}"

scene_list=("scene0710_00" "scene0758_00" "scene0781_00")
for scene in ${scene_list[@]}; do

    python train/vga_trainer/trainer.py \
        --gin_file "exp_configs/ScanNet.gin" \
        --gin_param "VGAGSConfigs.scene_name='$scene'" \
        --gin_param "VGAGSConfigs.exp_name='$exp_name'" \
        --gin_param "VGAGSConfigs.output_root_dir='$output_root_dir'" \
        --gin_param "VGAGSConfigs.rasterizer_backend='$rasterizer_backend'" \
        --gin_param "VGAGSConfigs.depth_estimation_method='$depth_estimation_method'" \
        --gin_param "VGAGSConfigs.data_root_dir='$data_root_dir'" \
        --gin_param "VGAGSConfigs.patch_size_for_sample=15" \

done

