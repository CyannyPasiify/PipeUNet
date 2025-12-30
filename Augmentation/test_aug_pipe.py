#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强管线测试脚本

功能概述：
    读取nii.gz格式的图像和蒙版文件
    应用数据增强管线处理
    保存每步处理结果
    测试数据增强的可复现性
    测试状态保存和恢复功能

使用方法：
    python test_aug_pipe.py -i image.nii.gz -m mask.nii.gz -o output_dir -p 0.5 -s 42 -t basic

测试模式：
    - basic: 基本处理，保存每步结果
    - reproducibility: 可复现性测试，验证相同种子产生相同结果
    - state_management: 状态管理测试，验证状态保存和恢复功能
"""

import os
import argparse
import torch
import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, SaveImaged
from aug_pipe import DataAugmentationPipeline


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
        save_dict['image'].meta['filename_or_obj'] = f"{file_name}_image"
    
    if 'mask' in data_dict:
        save_dict['mask'] = data_dict['mask'].clone()
        # 创建或更新蒙版元数据
        save_dict['mask'].meta['filename_or_obj'] = f"{file_name}_mask"
    
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


def process_callback(output_dir, base_file_name):
    """
    创建回调函数，用于保存每步处理结果
    
    Args:
        output_dir: 输出目录
        base_file_name: 基础文件名
    
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
        count_step_name = f"{formatted_count}_{step_name}"
        # 保存当前步骤的结果
        save_processed_result(result, output_dir, base_file_name, count_step_name)
    
    return callback


def are_results_equal(result1, result2):
    """
    比较两个增强结果是否相等
    
    Args:
        result1: 第一个增强结果字典
        result2: 第二个增强结果字典
    
    Returns:
        bool: 如果所有对应的值都相等，返回True
    """
    # 检查键是否相同
    if set(result1.keys()) != set(result2.keys()):
        print(f"键不匹配: {set(result1.keys())} vs {set(result2.keys())}")
        return False
    
    # 比较每个键对应的值
    for key in result1.keys():
        # 跳过元数据字典的比较
        if key.endswith('_meta_dict'):
            continue
        
        # 比较张量
        if isinstance(result1[key], torch.Tensor) and isinstance(result2[key], torch.Tensor):
            # 检查形状
            if result1[key].shape != result2[key].shape:
                print(f"{key} 形状不匹配: {result1[key].shape} vs {result2[key].shape}")
                return False
            
            # 检查值是否相等
            if not torch.allclose(result1[key], result2[key], atol=1e-6):
                max_diff = torch.max(torch.abs(result1[key] - result2[key])).item()
                print(f"{key} 值不匹配，最大差异: {max_diff}")
                # 找出最大差异的位置
                max_diff_idx = torch.argmax(torch.abs(result1[key] - result2[key])).item()
                print(f"  最大差异位置: {max_diff_idx}")
                print(f"  值1: {result1[key].flatten()[max_diff_idx]}")
                print(f"  值2: {result2[key].flatten()[max_diff_idx]}")
                return False
        
    return True


def run_basic_test(loaded_data, args, base_file_name):
    """
    运行基本测试
    
    Args:
        loaded_data: 加载的数据字典
        args: 命令行参数
        base_file_name: 基础文件名
    """
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
    
    # 创建回调函数
    callback = process_callback(args.output_dir, base_file_name)
    
    # 创建增强管线
    print("初始化数据增强管线...")
    aug_pipeline = DataAugmentationPipeline(
        callbacks=[callback], 
        transform_params=custom_params,
        random_seed=args.seed if args.seed is not None else None
    )
    
    # 应用增强
    print("开始数据增强处理...")
    augmented_data = aug_pipeline(loaded_data)
    
    # 保存最终结果
    print("保存最终结果...")
    save_processed_result(augmented_data, args.output_dir, base_file_name, "final")
    
    return augmented_data


def run_reproducibility_test(loaded_data, args, base_file_name):
    """
    运行可复现性测试
    
    Args:
        loaded_data: 加载的数据字典
        args: 命令行参数
        base_file_name: 基础文件名
    """
    # 设置固定随机种子
    seed = args.seed if args.seed is not None else 42
    print(f"使用固定随机种子: {seed} 进行可复现性测试")
    
    # 准备增强参数
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
    
    # 创建第一个增强管线实例
    print("创建第一个增强管线实例...")
    pipeline1 = DataAugmentationPipeline(
        transform_params=custom_params,
        random_seed=seed
    )
    
    # 应用第一次增强
    print("应用第一次增强...")
    result1 = pipeline1(loaded_data.copy())
    
    # 保存第一次增强结果
    save_processed_result(result1, args.output_dir, f"{base_file_name}_run1", "final")
    
    # 创建第二个增强管线实例（使用相同的种子）
    print("创建第二个增强管线实例（使用相同种子）...")
    pipeline2 = DataAugmentationPipeline(
        transform_params=custom_params,
        random_seed=seed
    )
    
    # 应用第二次增强
    print("应用第二次增强...")
    result2 = pipeline2(loaded_data.copy())
    
    # 保存第二次增强结果
    save_processed_result(result2, args.output_dir, f"{base_file_name}_run2", "final")
    
    # 比较两次结果
    print("比较两次增强结果...")
    if are_results_equal(result1, result2):
        print("[PASS] 可复现性测试通过！相同种子产生了完全相同的增强结果。")
        return True
    else:
        print("[FAILED] 可复现性测试失败！相同种子产生了不同的增强结果。")
        return False


def run_state_management_test(loaded_data, args, base_file_name):
    """
    运行状态管理测试
    
    Args:
        loaded_data: 加载的数据字典
        args: 命令行参数
        base_file_name: 基础文件名
    """
    # 设置固定随机种子
    seed = args.seed if args.seed is not None else 42
    print(f"使用固定随机种子: {seed} 进行状态管理测试")
    
    # 准备增强参数
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
    
    # 创建第一个增强管线实例
    print("创建第一个增强管线实例...")
    pipeline1 = DataAugmentationPipeline(
        transform_params=custom_params,
        random_seed=seed
    )
    
    # 应用第一次增强
    print("应用第一次增强...")
    result1_before_save = pipeline1(loaded_data.copy())
    
    # 获取并保存当前状态
    print("获取并保存当前状态...")
    state = pipeline1.get_state()
    print(f"状态字典包含 {len(state)} 个变换的状态信息")
    
    # 应用第二次增强
    print("应用第二次增强...")
    result1_before_save = pipeline1(loaded_data.copy())

    # 保存增强结果
    save_processed_result(result1_before_save, args.output_dir, f"{base_file_name}_before_save", "final")
    
    # 创建第二个增强管线实例
    print("创建第二个增强管线实例...")
    pipeline2 = DataAugmentationPipeline(
        transform_params=custom_params,
        random_seed=seed  # 使用相同的初始种子
    )
    
    # 加载保存的状态
    print("加载保存的状态到第二个实例...")
    pipeline2.set_state(state)
    
    # 使用第二个实例应用增强
    print("使用第二个实例应用增强...")
    result2_after_load = pipeline2(loaded_data.copy())
    
    # 保存第二个增强结果
    save_processed_result(result2_after_load, args.output_dir, f"{base_file_name}_after_load", "final")
    
    # 比较结果
    print("比较状态恢复前后的增强结果...")
    if are_results_equal(result1_before_save, result2_after_load):
        print("[PASS] 状态管理测试通过！状态保存和恢复功能正常工作。")
        return True
    else:
        print("[FAILED] 状态管理测试失败！状态恢复后结果不匹配。")
        return False


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据增强管线测试脚本')
    parser.add_argument('-i', '--image', required=True, help='输入图像文件路径(.nii.gz)')
    parser.add_argument('-m', '--mask', required=False, help='输入蒙版文件路径(.nii.gz)')
    parser.add_argument('-o', '--output_dir', required=True, help='输出目录路径')
    parser.add_argument('-p', '--prob', type=float, default=1.0, help='每个增强步骤的概率')
    parser.add_argument('-s', '--seed', type=int, default=42, help='随机种子，用于可复现性测试')
    parser.add_argument('-t', '--test_mode', type=str, default='basic', 
                        choices=['basic', 'reproducibility', 'state_management'],
                        help='测试模式：basic（基本处理）、reproducibility（可复现性测试）、state_management（状态管理测试）')
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
    
    # 根据测试模式运行不同的测试
    if args.test_mode == 'basic':
        print("\n=== 运行基本处理测试 ===")
        run_basic_test(loaded_data, args, base_file_name)
    elif args.test_mode == 'reproducibility':
        print("\n=== 运行可复现性测试 ===")
        run_reproducibility_test(loaded_data, args, base_file_name)
    elif args.test_mode == 'state_management':
        print("\n=== 运行状态管理测试 ===")
        run_state_management_test(loaded_data, args, base_file_name)
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()