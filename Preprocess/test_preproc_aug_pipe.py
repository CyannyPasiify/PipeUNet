#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理和数据增强结合测试脚本

功能概述：
    读取nii.gz格式的图像和蒙版文件
    先应用数据预处理管线处理
    再应用数据增强管线处理
    保存每步处理结果

使用方法：
    python test_preproc_aug_pipe.py -i image.nii.gz -m mask.nii.gz -o output_dir -n 21
"""

import os
import sys
import argparse
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, SaveImaged
from Preprocess.preproc_pipe import DataPreprocessingPipeline
from Augmentation.aug_pipe import DataAugmentationPipeline


def save_processed_result(data_dict, output_dir, file_name, step_name):
    """
    保存处理后的图像和蒙版
    
    Args:
        data_dict: 包含'image'和可选'mask'的字典
        output_dir: 输出目录
        file_name: 原始文件名(不含扩展名)
        step_name: 处理步骤名称
    """
    # 复制数据字典以避免修改原始数据
    save_dict = {}
    
    # 为图像和蒙版创建元数据
    if 'image' in data_dict:
        save_dict['image'] = data_dict['image'].clone()
        # 创建或更新图像元数据
        if 'image_meta_dict' not in data_dict:
            save_dict['image_meta_dict'] = {}
        else:
            save_dict['image_meta_dict'] = data_dict['image_meta_dict'].copy()
        save_dict['image_meta_dict']['filename_or_obj'] = f"{file_name}_image"
    
    if 'mask' in data_dict:
        save_dict['mask'] = data_dict['mask'].clone()
        # 创建或更新蒙版元数据
        if 'mask_meta_dict' not in data_dict:
            save_dict['mask_meta_dict'] = {}
        else:
            save_dict['mask_meta_dict'] = data_dict['mask_meta_dict'].copy()
        save_dict['mask_meta_dict']['filename_or_obj'] = f"{file_name}_mask"
    
    # 确定要保存的键
    keys_to_save = []
    if 'image' in save_dict:
        keys_to_save.append('image')
    if 'mask' in save_dict:
        keys_to_save.append('mask')
    
    # 使用 SaveImaged 保存数据
    if keys_to_save:
        saver = SaveImaged(
            keys=keys_to_save,
            output_dir=output_dir,
            output_postfix=step_name,
            output_ext=".nii.gz",
            output_dtype=torch.float32,
            squeeze_end_dims=True,
            separate_folder=False,
            print_log=True,
            allow_missing_keys=True
        )
        saver(save_dict)


def process_callback(output_dir, base_file_name, process_type):
    """
    创建回调函数，用于保存每步处理结果
    
    Args:
        output_dir: 输出目录
        base_file_name: 基础文件名
        process_type: 处理类型（preprocess或augment）
    
    Returns:
        回调函数
    """
    # 步骤计数器
    step_count = 0
    
    def callback(step_name, params, result):
        """
        回调函数实现
        
        Args:
            step_name: 处理步骤名称
            params: 处理参数
            result: 处理结果
        """
        nonlocal step_count
        step_count += 1
        # 格式化步骤计数为3位数，不足补0
        formatted_count = f"{step_count:03d}"
        # 构建带计数的步骤名称
        count_step_name = f"{process_type}_{formatted_count}_{step_name}"
        # 保存当前步骤的结果
        save_processed_result(result, output_dir, base_file_name, count_step_name)
    
    return callback


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='预处理和数据增强结合测试脚本')
    parser.add_argument('-i', '--image', required=True, help='输入图像文件路径(.nii.gz)')
    parser.add_argument('-m', '--mask', required=False, help='输入蒙版文件路径(.nii.gz)')
    parser.add_argument('-o', '--output_dir', required=True, help='输出目录路径')
    parser.add_argument('-n', '--n_radiomics', type=int, default=21, help='影像组学特征数量')
    parser.add_argument('-p', '--prob', type=float, default=1.0, help='每个增强步骤的概率')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出模式')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取基础文件名
    base_file_name = os.path.splitext(os.path.splitext(os.path.basename(args.image))[0])[0]
    
    # 创建读取变换
    transforms_list = [
        LoadImaged(keys=['image'] + (['mask'] if args.mask else [])),
        EnsureChannelFirstd(keys=['image'] + (['mask'] if args.mask else []))
    ]
    load_transform = Compose(transforms_list)
    
    # 准备数据字典
    data_dict = {'image': args.image}
    if args.mask:
        data_dict['mask'] = args.mask
    
    # 加载数据
    print(f"加载数据: 图像={args.image}")
    if args.mask:
        print(f"加载数据: 蒙版={args.mask}")
    
    loaded_data = load_transform(data_dict)
    
    # 保存原始数据
    print("保存原始数据...")
    save_processed_result(loaded_data, args.output_dir, base_file_name, "original")
    
    # 准备预处理回调函数
    preprocess_callback = process_callback(args.output_dir, base_file_name, "preprocess")
    
    # 创建预处理管线
    print("初始化数据预处理管线...")
    preproc_pipeline = DataPreprocessingPipeline(callbacks=[preprocess_callback] if args.verbose else [])
    
    # 应用预处理
    print("开始数据预处理...")
    preprocessed_data = preproc_pipeline(loaded_data)
    
    # 保存预处理结果
    print("保存预处理结果...")
    save_processed_result(preprocessed_data, args.output_dir, base_file_name, "preprocessed_final")
    
    # 准备增强参数，设置每个步骤的概率
    custom_params = {
        "Rand3DElasticd": {"prob": args.prob},
        "RandGaussianSmoothd": {"prob": args.prob},
        "RandGaussianNoised": {"prob": args.prob},
        "RandShiftIntensityd": {"prob": args.prob},
        "RandScaleIntensityFixedMeand": {"prob": args.prob},
        "RandAdjustContrastd": {"prob": args.prob},
        "RandSimulateLowResolutiond": {"prob": args.prob},
        "RandCoarseDropoutd": {"prob": args.prob}
    }
    
    # 准备增强回调函数
    augment_callback = process_callback(args.output_dir, base_file_name, "augment")
    
    # 创建增强管线
    print("初始化数据增强管线...")
    aug_pipeline = DataAugmentationPipeline(
        callbacks=[augment_callback] if args.verbose else [],
        transform_params=custom_params
    )
    
    # 应用增强
    print("开始数据增强处理...")
    augmented_data = aug_pipeline(preprocessed_data)
    
    # 保存最终增强结果
    print("保存最终增强结果...")
    save_processed_result(augmented_data, args.output_dir, base_file_name, "augmented_final")
    
    print("处理完成！")
    print(f"所有中间结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()