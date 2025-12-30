#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理管线

功能概述：
    封装基于MONAI的3D图像数据预处理变换，支持同时处理图像和蒙版
    实现了2种数据预处理操作：强度缩放和大小调整/裁剪
    支持自定义每个预处理步骤的参数
    提供回调函数接口，实时获取处理过程信息

核心功能：
    1. 输入验证：确保图像和蒙版格式正确，维度匹配
    2. 数据预处理：执行强度缩放、大小调整/裁剪等预处理操作
    3. 回调机制：每个预处理步骤完成后触发回调，返回详细信息
    4. 参数配置：支持为每个预处理操作配置详细参数

参数说明：
    - callbacks: 回调函数列表，每个函数接收(name, params, result)作为参数
    - transform_params: 包含所有变换参数的字典，可以覆盖默认参数

使用示例：
    # 创建数据预处理管线实例
    preproc_pipeline = DataPreprocessingPipeline(callback=callback_function)
    
    # 处理图像和蒙版字典
    data = {'image': image_tensor, 'mask': mask_tensor}
    preprocessed_data = preproc_pipeline(data)
    
    # 自定义变换参数
    custom_params = {
        "ScaleIntensityRanged": {"a_min": 0, "a_max": 1000, "b_min": 0, "b_max": 1},
        "ResizeWithPadOrCropd": {"spatial_size": (128, 128, 128)}
    }
    preproc_pipeline = DataPreprocessingPipeline(callback=callback_function, transform_params=custom_params)
"""

import numpy as np
import torch
import monai.transforms as mT
from typing import Callable, Dict, Any, Optional, Union, Tuple, List


class DataPreprocessingPipeline:
    """
    数据预处理管线类，封装了一系列基于MONAI的3D图像预处理变换。
    支持同时处理图像和蒙版，确保两者同步变换。
    输入为字典格式，包含'image'和'mask'键，值为通道优先的四维张量 (C, D, H, W)。
    使用字典包装器管理所有变换操作。
    """
    
    # 类属性类型注解
    callbacks: List[Callable]
    transform_defaults: Dict[str, Dict[str, Any]]
    transform_classes: Dict[str, Callable]
    transform_params: Dict[str, Dict[str, Any]]
    transform_order: List[str]
    transforms: List[Tuple[str, mT.Transform]]
    
    def __init__(self,
                 callbacks: Optional[List[Callable]] = None,
                 transform_params: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        初始化数据预处理管线
        
        Args:
            callback: 回调函数，每次预处理步骤完成后调用，接收(name, params, result)作为参数
            transform_params: 包含所有变换参数的字典，键为变换名称，值为参数字典
        """
        self.callbacks: List[Callable] = callbacks if callbacks is not None else []
        
        # 默认变换参数字典
        self.transform_defaults: Dict[str, Dict[str, Any]] = {
            "ScaleIntensityRanged": {
                "keys": ["image"],
                "a_min": -1000.0,
                "a_max": 1000.0,
                "b_min": 0.0,
                "b_max": 1.0,
                "clip": True
            },
            "ResizeWithPadOrCropd": {
                "keys": ["image", "mask"],
                "spatial_size": (128, 128, 128),
                "mode": "constant",
                "method": "symmetric",
                "allow_missing_keys": True
            }
        }
        
        # 变换类映射
        self.transform_classes: Dict[str, Callable] = {
            "ScaleIntensityRanged": mT.ScaleIntensityRanged,
            "ResizeWithPadOrCropd": mT.ResizeWithPadOrCropd
        }
        
        # 应用用户自定义参数
        self.transform_params: Dict[str, Dict[str, Any]] = self._merge_transform_params(transform_params)
        
        # 定义变换执行顺序
        self.transform_order: List[str] = [
            "ScaleIntensityRanged",
            "ResizeWithPadOrCropd"
        ]
        
        # 初始化所有变换
        self.transforms: List[Tuple[str, mT.Transform]] = self._initialize_transforms()
    
    def _merge_transform_params(self, custom_params: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        合并默认参数和自定义参数
        
        Args:
            custom_params: 自定义变换参数字典
            
        Returns:
            Dict[str, Dict[str, Any]]: 合并后的参数字典
        """
        merged_params: Dict[str, Dict[str, Any]] = {}
        for transform_name, default_params in self.transform_defaults.items():
            default_params: Dict[str, Any] = default_params
            merged_params[transform_name] = default_params.copy()
            if custom_params and transform_name in custom_params:
                merged_params[transform_name].update(custom_params[transform_name])
        return merged_params
    
    def _initialize_transforms(self) -> List[Tuple[str, mT.Transform]]:
        """
        初始化所有变换实例
        
        Returns:
            List[Tuple[str, mT.Transform]]: 包含变换名称和实例的元组列表
        """
        transforms_list: List[Tuple[str, mT.Transform]] = []
        for transform_name in self.transform_order:
            params: Dict[str, Any] = self.transform_params[transform_name].copy()
            transform_class: Callable = self.transform_classes[transform_name]
            transform: mT.Transform = transform_class(**params)
            transforms_list.append((transform_name, transform))
        return transforms_list
    
    def validate_input(self, data: Dict[str, torch.Tensor]) -> None:
        """
        验证输入数据的有效性
        
        Args:
            data: 包含张量对象的字典
            
        Raises:
            ValueError: 当输入不符合要求时抛出
        """
        # 验证字典格式
        if not isinstance(data, dict):
            raise ValueError(f"输入必须是字典格式，当前类型: {type(data)}")
        
        # 验证字典不为空
        if not data:
            raise ValueError("输入字典不能为空")
        
        # 存储第一个张量的空间维度作为参考
        reference_shape: Optional[Tuple[int, ...]] = None
        
        # 对所有键进行验证
        for key, value in data.items():
            # 验证是否为Tensor类型
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"'{key}'必须是torch.Tensor类型，当前类型: {type(value)}")
            
            # 验证是否为4维张量
            if value.dim() != 4:
                raise ValueError(f"'{key}'必须是4维张量，当前维度: {value.dim()}")
            
            # 设置参考形状（第一个张量）
            if reference_shape is None:
                reference_shape = value.shape[1:]
            # 验证所有张量的空间维度是否匹配
            elif value.shape[1:] != reference_shape:
                raise ValueError(f"'{key}'的空间维度与参考维度不匹配: 当前{value.shape[1:]}, 参考{reference_shape}")
    
    def apply_callbacks(self, name: str, params: Dict[str, Any], result: Dict[str, torch.Tensor]) -> None:
        """
        应用所有注册的回调函数
        
        Args:
            name: 预处理操作名称
            params: 预处理操作参数
            result: 预处理操作结果
        """
        for callback in self.callbacks:
            callback(name, params, result)
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        执行数据预处理流程
        
        Args:
            data: 包含'image'和可选'mask'的字典，张量形状为 (C, D, H, W)
            
        Returns:
            预处理后的字典数据
        """
        # 验证输入
        self.validate_input(data)
        
        # 复制数据以避免修改原始数据
        result_data: Dict[str, torch.Tensor] = {k: v.clone() for k, v in data.items()}
        
        # 按顺序执行所有变换
        for transform_name, transform in self.transforms:
            # 获取当前变换的参数
            params: Dict[str, Any] = self.transform_params[transform_name].copy()
            
            # 应用变换
            result_data = transform(result_data)
            
            # 应用回调
            self.apply_callbacks(transform_name, params, result_data)
        
        return result_data


def example_callback(name: str, params: Dict[str, Any], result: Dict[str, torch.Tensor]) -> None:
    """
    示例回调函数，用于打印预处理操作的信息
    
    Args:
        name: 预处理操作名称
        params: 预处理操作参数
        result: 预处理操作结果
    """
    print(f"执行变换: {name}")
    print(f"变换参数: {params}")
    if 'image' in result:
        print(f"图像结果形状: {result['image'].shape}")
        print(f"图像结果范围: [{result['image'].min():.4f}, {result['image'].max():.4f}]")
    if 'mask' in result:
        print(f"蒙版结果形状: {result['mask'].shape}")
        print(f"蒙版结果范围: [{result['mask'].min():.4f}, {result['mask'].max():.4f}]")
    print("---")


def example_usage() -> None:
    """
    示例用法
    """
    print("\n=== 示例1: 处理图像和蒙版 ===")
    # 创建随机的4D张量 (C, D, H, W)
    sample_image: torch.Tensor = torch.rand((3, 32, 64, 64)) * 1000  # 3通道图像，范围0-1000
    sample_mask: torch.Tensor = torch.zeros((1, 32, 64, 64))  # 单通道蒙版
    
    # 在蒙版中添加一些随机区域
    for _ in range(5):
        d, h, w = np.random.randint(0, 16, size=3)
        sample_mask[0, d:d+16, h:h+32, w:w+32] = 1
    
    # 创建数据字典
    data: Dict[str, torch.Tensor] = {
        'image': sample_image,
        'mask': sample_mask
    }
    
    # 创建预处理管线实例
    preproc_pipeline: DataPreprocessingPipeline = DataPreprocessingPipeline(callbacks=[example_callback])
    
    # 应用预处理
    preprocessed_data: Dict[str, torch.Tensor] = preproc_pipeline(data)
    
    print(f"原始图像形状: {sample_image.shape}")
    print(f"原始蒙版形状: {sample_mask.shape}")
    print(f"预处理后图像形状: {preprocessed_data['image'].shape}")
    print(f"预处理后蒙版形状: {preprocessed_data['mask'].shape}")
    print(f"原始图像范围: [{sample_image.min():.4f}, {sample_image.max():.4f}]")
    print(f"预处理后图像范围: [{preprocessed_data['image'].min():.4f}, {preprocessed_data['image'].max():.4f}]")
    
    print("\n=== 示例2: 使用自定义参数 ===")
    # 定义自定义参数
    custom_params: Dict[str, Dict[str, Any]] = {
        "ScaleIntensityRanged": {"a_min": 0, "a_max": 500, "b_min": 0, "b_max": 1},
        "ResizeWithPadOrCropd": {"spatial_size": (64, 64, 64)}
    }
    
    # 使用自定义参数创建预处理管线
    custom_preproc_pipeline: DataPreprocessingPipeline = DataPreprocessingPipeline(
        callbacks=[example_callback],
        transform_params=custom_params
    )
    
    # 应用预处理
    custom_preprocessed_data: Dict[str, torch.Tensor] = custom_preproc_pipeline(data)
    
    print(f"使用自定义参数预处理后图像形状: {custom_preprocessed_data['image'].shape}")
    print(f"使用自定义参数预处理后蒙版形状: {custom_preprocessed_data['mask'].shape}")
    print(f"使用自定义参数预处理后图像范围: [{custom_preprocessed_data['image'].min():.4f}, {custom_preprocessed_data['image'].max():.4f}]")


if __name__ == "__main__":
    example_usage()