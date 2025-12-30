#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
数据集清单序列类预处理和数据增强测试脚本

功能概述：
    测试DatasetManifestWithLabelSequential和DatasetManifestNoLabelSequential类在应用预处理和数据增强时的功能
    配置preprocess_augment参数使用预处理管线和数据增强管线
    通过回调函数打印处理步骤记录并输出中间结果
    验证数据处理流程和结果的正确性

使用方法：
    python test_ds_preproc_aug_pipe.py -r data_root_dir -m manifest_file.xlsx -n 21 -o output_dir -w -s 5
    python d:\CBIB\Storages\DevelopmentSoftwares\Trae\EsophagusModelProject\Dataset\test_ds_preproc_aug_pipe.py -r c:\Users\User\Desktop\test_files -m c:\Users\User\Desktop\test_files\manifest_tongji.xlsx -o c:\Users\User\Desktop\test_files\transformed -p 1
"""

import os
import argparse
import torch
from monai.transforms import SaveImaged
from typing import Dict, Any
from Dataset.ds_pipe import DatasetManifestWithLabelSequential, DatasetManifestNoLabelSequential
from Preprocess.preproc_pipe import DataPreprocessingPipeline
from Augmentation.aug_pipe import DataAugmentationPipeline


def save_intermediate_result(output_dir, pid, step_name, data_dict):
    """
    保存中间处理结果到文件
    
    Args:
        output_dir: 输出目录
        pid: 患者ID
        step_name: 处理步骤名称
        data_dict: 包含'image'和可选'mask'的字典
    """
    # 为当前患者创建子目录
    patient_dir = os.path.join(output_dir, str(pid))
    os.makedirs(patient_dir, exist_ok=True)
    
    # 复制数据字典以避免修改原始数据
    save_dict = {}
    
    # 为图像和蒙版创建元数据
    for key in ['pre_img', 'pre_mask', 'post_img', 'post_mask']:
        if key in data_dict and data_dict[key] is not None:
            save_dict[key] = data_dict[key].clone()
            # 创建或更新元数据
            meta_key = f"{key}_meta_dict"
            save_dict[meta_key] = {}
            save_dict[meta_key]['filename_or_obj'] = f"{pid}_{key}"
    
    # 确定要保存的键
    keys_to_save = []
    for key in ['pre_img', 'pre_mask', 'post_img', 'post_mask']:
        if key in save_dict:
            keys_to_save.append(key)
    
    # 使用 SaveImaged 保存数据
    if keys_to_save:
        saver = SaveImaged(
            keys=keys_to_save,
            output_dir=patient_dir,
            output_postfix=step_name,
            output_ext=".nii.gz",
            output_dtype=torch.float32,
            squeeze_end_dims=True,
            separate_folder=False,
            print_log=True,
            allow_missing_keys=True
        )
        saver(save_dict)
        print(f"中间结果已保存: {patient_dir}/{pid}_*_{step_name}.nii.gz")


def create_preprocessing_callback(output_dir, pid):
    """
    创建预处理回调函数
    
    Args:
        output_dir: 输出目录
        pid: 患者ID
    
    Returns:
        预处理回调函数
    """
    step_count = 0
    
    def preprocess_callback(name: str, params: Dict[str, Any], result: Dict[str, torch.Tensor]) -> None:
        """
        预处理回调函数实现
        
        Args:
            name: 处理步骤名称
            params: 处理参数
            result: 处理结果
        """
        nonlocal step_count
        step_count += 1
        formatted_count = f"{step_count:03d}"
        step_name = f"preprocess_{formatted_count}_{name}"
        
        print(f"患者 {pid} - 预处理步骤 {step_count}: {name}")
        print(f"  参数: {params}")
        
        # 保存中间结果
        save_intermediate_result(output_dir, pid, step_name, result)
    
    return preprocess_callback


def create_augmentation_callback(output_dir, pid):
    """
    创建数据增强回调函数
    
    Args:
        output_dir: 输出目录
        pid: 患者ID
    
    Returns:
        数据增强回调函数
    """
    step_count = 0
    
    def augment_callback(name: str, params: Dict[str, Any], result: Dict[str, torch.Tensor]) -> None:
        """
        数据增强回调函数实现
        
        Args:
            name: 增强步骤名称
            params: 增强参数
            result: 增强结果
        """
        nonlocal step_count
        step_count += 1
        formatted_count = f"{step_count:03d}"
        step_name = f"augment_{formatted_count}_{name}"
        
        print(f"患者 {pid} - 增强步骤 {step_count}: {name}")
        print(f"  参数: {params}")
        
        # 保存中间结果
        save_intermediate_result(output_dir, pid, step_name, result)
    
    return augment_callback


def create_combined_preprocess_augment(output_dir, pid, prob=0.5):
    """
    创建预处理和数据增强的组合列表
    
    Args:
        output_dir: 输出目录
        pid: 患者ID
        prob: 增强概率
    
    Returns:
        预处理和数据增强的组合列表
    """
    # 创建预处理回调函数
    preprocess_callbacks = [create_preprocessing_callback(output_dir, pid)]
    
    # 创建数据增强回调函数
    augment_callbacks = [create_augmentation_callback(output_dir, pid)]
    
    # 创建预处理参数
    preprocess_params = {
        "ScaleIntensityRanged": {
            "keys": ["pre_img", "post_img"],
            "a_min": -1000.0,
            "a_max": 1000.0,
            "b_min": 0.0,
            "b_max": 1.0,
            "clip": True
        },
        "ResizeWithPadOrCropd": {
            "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
            "spatial_size": (128, 128, 128),
            "mode": "constant",
            "method": "symmetric",
            "allow_missing_keys": True
        }
    }
    
    # 创建数据增强参数
    augment_params = {
        "Rand3DElasticd": {
            "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
            "prob": prob,
            "mode": ["bilinear", "nearest", "bilinear", "nearest"]
        },
        "RandGaussianSmoothd": {
            "keys": ["pre_img", "post_img"],
            "prob": prob
        },
        "RandGaussianNoised": {
            "keys": ["pre_img", "post_img"],
            "prob": prob
        },
        "RandShiftIntensityd": {
            "keys": ["pre_img", "post_img"],
            "prob": prob
        },
        "RandScaleIntensityFixedMeand": {
            "keys": ["pre_img", "post_img"],
            "prob": prob
        },
        "RandAdjustContrastd": {
            "keys": ["pre_img", "post_img"],
            "prob": prob
        },
        "RandSimulateLowResolutiond": {
            "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
            "prob": prob,
        },
        "RandCoarseDropoutd": {
            "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
            "prob": prob,
        }
    }
    
    # 创建预处理管线
    preprocess_pipeline = DataPreprocessingPipeline(
        callbacks=preprocess_callbacks,
        transform_params=preprocess_params
    )
    
    # 创建数据增强管线
    augment_pipeline = DataAugmentationPipeline(
        callbacks=augment_callbacks,
        transform_params=augment_params
    )
    
    # 返回组合的处理列表
    return [preprocess_pipeline, augment_pipeline]


def dataset_callback(output_dir, pid):
    """
    数据集处理回调函数
    
    Args:
        output_dir: 输出目录
        pid: 患者ID
    
    Returns:
        回调函数
    """
    def callback(idx: int, pid: str, image_paths: Dict[str, str], 
                 mask_paths: Dict[str, str], data: Dict[str, Any]) -> None:
        """
        数据集回调函数实现
        
        Args:
            idx: 样本索引
            pid: 患者ID
            image_paths: 图像路径字典
            mask_paths: 蒙版路径字典
            data: 处理和包装后的数据
        """
        print(f"患者 {pid} - 数据集处理: 样本 {idx}")
        print(f"  图像路径: {image_paths}")
        print(f"  蒙版路径: {mask_paths}")
        print(f"  数据键: {data.keys()}")
        print(f"  数据形状: {data['pre_img'].shape}")
        print(f"  影像组学特征: {data['radiomics']}")
        
        # 保存最终处理结果
        save_intermediate_result(output_dir, pid, f"final_{idx}", data)
    
    return callback


def test_dataset_with_preprocessing_augmentation(dataset_class, root_dir, n_radiomics, manifest_file, 
                                               output_dir, num_samples=3, n_len=None, augment_prob=0.5):
    """
    测试带有预处理和数据增强的数据集类
    
    Args:
        dataset_class: 要测试的数据集类
        root_dir: 数据集根目录
        n_radiomics: 影像组学特征数量
        manifest_file: 清单文件路径
        output_dir: 输出目录
        num_samples: 要测试的样本数量
        n_len: 数据集长度限制
        augment_prob: 数据增强概率
    """
    print(f"\n测试数据集类 (带预处理和数据增强): {dataset_class.__name__}")
    print(f"------------------------------------")
    
    try:
        # 创建数据集实例
        dataset = dataset_class(
            root_dir=root_dir,
            n_radiomics=n_radiomics,
            manifest_file=manifest_file,
            preprocess_augment=None,  # 初始为None，将在每个样本处理时动态创建
            callbacks=None,  # 初始为None，将在每个样本处理时动态创建
            n_len=n_len
        )
        
        # 打印数据集信息
        print(f"数据集大小: {len(dataset)}")
        print(f"影像组学特征数量: {n_radiomics}")
        
        # 限制测试样本数量
        test_samples = min(num_samples, len(dataset))
        print(f"测试样本数量: {test_samples}")
        print(f"数据增强概率: {augment_prob}")
        print(f"中间结果输出目录: {output_dir}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试样本加载
        for i in range(test_samples):
            print(f"\n=========================================")
            print(f"加载样本 {i+1}/{test_samples}")
            
            # 获取当前患者ID
            pid = dataset.manifest.iloc[i]['pid']
            print(f"样本PID: {pid}")
            
            # 为当前患者创建预处理和数据增强组合
            preprocess_augment = create_combined_preprocess_augment(output_dir, pid, augment_prob)
            
            # 为当前患者创建数据集回调函数
            callbacks = [dataset_callback(output_dir, pid)]
            
            # 更新数据集实例的预处理、增强和回调
            dataset.preprocess_augment = preprocess_augment
            dataset.callbacks = callbacks
            
            # 加载并处理样本
            print(f"开始处理样本...")
            sample = dataset[i]
            
            # 验证返回数据结构
            required_keys = ['pre_img', 'pre_mask', 'post_img', 'post_mask', 'radiomics']
            if dataset_class.__name__ == 'DatasetManifestWithLabelSequential':
                required_keys.append('label')
            
            for key in required_keys:
                assert key in sample, f"样本缺少键: {key}"
            
            # 打印处理后的数据形状
            print(f"\n处理后的数据形状:")
            for key in ['pre_img', 'pre_mask', 'post_img', 'post_mask']:
                if sample[key] is not None:
                    print(f"{key} 形状: {sample[key].shape}")
            
            # 打印影像组学特征信息
            print(f"影像组学特征形状: {sample['radiomics'].shape}")
            
            # 打印标签信息（如果有）
            if 'label' in sample:
                print(f"标签: {sample['label']}")
            
            print(f"样本 {i+1}/{test_samples} 处理完成!")
        
        print(f"\n=========================================")
        print(f"测试成功完成！所有样本已处理并保存中间结果。")
        print(f"中间结果位置: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集清单序列类预处理和数据增强测试脚本')
    parser.add_argument('-r', '--root_dir', required=True, help='数据集根目录')
    parser.add_argument('-m', '--manifest_file', required=True, help='清单文件路径')
    parser.add_argument('-n', '--n_radiomics', type=int, default=21, help='影像组学特征数量')
    parser.add_argument('-o', '--output_dir', required=True, help='中间结果输出目录')
    parser.add_argument('-w', '--with_label', action='store_true', default=False, help='数据集是否带标签')
    parser.add_argument('-s', '--samples', type=int, default=3, help='要测试的样本数量')
    parser.add_argument('-l', '--n_len', type=int, default=None, help='数据集长度限制')
    parser.add_argument('-p', '--prob', type=float, default=0.5, help='数据增强概率')
    args = parser.parse_args()
    
    # 确保根目录存在
    if not os.path.exists(args.root_dir):
        print(f"错误：数据集根目录不存在: {args.root_dir}")
        return 1
    
    # 确保清单文件存在
    if not os.path.exists(args.manifest_file):
        print(f"错误：清单文件不存在: {args.manifest_file}")
        return 1
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 选择数据集类
    dataset_class = DatasetManifestWithLabelSequential if args.with_label else DatasetManifestNoLabelSequential
    
    # 执行测试
    success = test_dataset_with_preprocessing_augmentation(
        dataset_class=dataset_class,
        root_dir=args.root_dir,
        n_radiomics=args.n_radiomics,
        manifest_file=args.manifest_file,
        output_dir=args.output_dir,
        num_samples=args.samples,
        n_len=args.n_len,
        augment_prob=args.prob
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())