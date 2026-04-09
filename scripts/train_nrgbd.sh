#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
rasterizer_backend=$2
depth_estimation_method=$3

dataset_name="NRGBD"
data_root_dir="/hdd24T/sunxt/dataset/neural_rgbd"
output_root_dir="/hdd24T/sunxt/VGA-GS_comparisons/ours_ps15"
exp_name="${rasterizer_backend}_${depth_estimation_method}"

scene_list=("breakfast_room" "complete_kitchen" "green_room" "grey_white_room" "kitchen" "morning_apartment" "staircase" "thin_geometry" "whiteroom")
for scene in ${scene_list[@]}; do

    python train/vga_trainer/trainer.py \
        --gin_file "exp_configs/NRGBD.gin" \
        --gin_param "VGAGSConfigs.scene_name='$scene'" \
        --gin_param "VGAGSConfigs.exp_name='$exp_name'" \
        --gin_param "VGAGSConfigs.output_root_dir='$output_root_dir'" \
        --gin_param "VGAGSConfigs.rasterizer_backend='$rasterizer_backend'" \
        --gin_param "VGAGSConfigs.depth_estimation_method='$depth_estimation_method'" \
        --gin_param "VGAGSConfigs.data_root_dir='$data_root_dir'" \
        --gin_param "VGAGSConfigs.patch_size_for_sample=15" \

done

