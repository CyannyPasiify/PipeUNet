#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强管线

功能概述：
    封装基于MONAI的3D图像数据增强变换，支持同时处理图像和蒙版
    实现了9种数据增强操作，按指定顺序执行
    支持自定义每个增强步骤的参数
    提供回调函数接口，实时获取处理过程信息

核心功能：
    1. 输入验证：确保图像和蒙版格式正确，维度匹配
    2. 数据增强：执行弹性变换、高斯平滑、噪声添加等多种增强操作
    3. 回调机制：每个增强步骤完成后触发回调，返回详细信息
    4. 参数配置：支持为每个增强操作配置详细参数

参数说明：
    - callbacks: 回调函数列表，每个函数接收(name, params, result)作为参数
    - transform_params: 包含所有变换参数的字典，可以覆盖默认参数

使用示例：
    # 创建数据增强管线实例
    aug_pipeline = DataAugmentationPipeline(callback=callback_function)
    
    # 处理图像和蒙版字典
    data = {'image': image_tensor, 'mask': mask_tensor}
    augmented_data = aug_pipeline(data)
    
    # 自定义变换参数
    custom_params = {
        "Rand3DElasticd": {"max_displacement": 5, "prob": 0.5},
        "RandGaussianNoised": {"std": (0.0, 0.2), "prob": 0.4}
    }
    aug_pipeline = DataAugmentationPipeline(callback=callback_function, transform_params=custom_params)
"""

import numpy as np
import torch
import monai.transforms as mT
from typing import Callable, Dict, Any, Optional, Union, Tuple, List


class DataAugmentationPipeline:
    """
    数据增强管线类，封装了一系列基于MONAI的3D图像增强变换。
    支持同时处理图像和蒙版，确保两者同步变换。
    输入为字典格式，包含'image'和'mask'键，值为通道优先的四维张量 (C, D, H, W)。
    使用字典包装器管理所有变换操作。
    """
    
    # 类属性类型注解
    callbacks: List[Callable]
    transform_defaults: Dict[str, Dict[str, Any]]
    transform_classes: Dict[str, Callable]
    transform_params: Dict[str, Dict[str, Any]]
    random_seed: Optional[int]
    transform_order: List[str]
    transforms: List[Tuple[str, mT.Transform]]
    
    def __init__(self,
                 callbacks: Optional[List[Callable]] = None,
                 transform_params: Optional[Dict[str, Dict[str, Any]]] = None,
                 random_seed: Optional[int] = None):
        """
        初始化数据增强管线
        
        Args:
            callback: 回调函数，每次增强步骤完成后调用，接收(name, params, result)作为参数
            transform_params: 包含所有变换参数的字典，键为变换名称，值为参数字典
            random_seed: 随机种子，用于确保增强过程可复现，默认值为None
        """
        self.callbacks: List[Callable] = callbacks if callbacks is not None else []
        
        # 默认变换参数字典
        self.transform_defaults: Dict[str, Dict[str, Any]] = {
            "Rand3DElasticd": {
                "keys": ["image", "mask"],
                "sigma_range": (5.0, 7.0),
                "magnitude_range": (50.0, 150.0),
                "prob": 0.3,
                "mode": ["bilinear", "nearest"],
                "padding_mode": "border",
                "allow_missing_keys": True
            },
            "RandGaussianSmoothd": {
                "keys": ["image"],
                "sigma_x": (0.5, 1.0),
                "sigma_y": (0.5, 1.0),
                "sigma_z": (0.5, 1.0),
                "prob": 0.3
            },
            "RandGaussianNoised": {
                "keys": ["image"],
                "mean": 0.0,
                "std": 0.1,
                "prob": 0.3
            },
            "RandShiftIntensityd": {
                "keys": ["image"],
                "offsets": (-5.0, 5.0),
                "prob": 0.3
            },
            "RandScaleIntensityFixedMeand": {
                "keys": ["image"],
                "factors": (0.8, 1.2),
                "prob": 0.3
            },
            "RandAdjustContrastd": {
                "keys": ["image"],
                "gamma": (0.8, 1.2),
                "invert_image": False,
                "prob": 0.3
            },
            "RandSimulateLowResolutiond": {
                "keys": ["image", "mask"],
                "zoom_range": (0.5, 1.0),
                "prob": 0.3,
                "allow_missing_keys": True
            },
            "RandCoarseDropoutd": {
                "keys": ["image", "mask"],
                "holes": 8,
                "spatial_size": (16, 16, 16),
                "fill_value": 0.0,
                "prob": 0.3,
                "allow_missing_keys": True
            }
        }
        
        # 变换类映射
        self.transform_classes: Dict[str, Callable] = {
            "Rand3DElasticd": mT.Rand3DElasticd,
            "RandGaussianSmoothd": mT.RandGaussianSmoothd,
            "RandGaussianNoised": mT.RandGaussianNoised,
            "RandShiftIntensityd": mT.RandShiftIntensityd,
            "RandScaleIntensityFixedMeand": mT.RandScaleIntensityFixedMeand,
            "RandAdjustContrastd": mT.RandAdjustContrastd,
            "RandSimulateLowResolutiond": mT.RandSimulateLowResolutiond,
            "RandCoarseDropoutd": mT.RandCoarseDropoutd
        }
        
        # 应用用户自定义参数
        self.transform_params: Dict[str, Dict[str, Any]] = self._merge_transform_params(transform_params)
        
        # 随机种子
        self.random_seed: Optional[int] = random_seed
        
        # 定义变换执行顺序
        self.transform_order: List[str] = [
            "Rand3DElasticd",
            "RandGaussianSmoothd",
            "RandGaussianNoised",
            "RandShiftIntensityd",
            "RandScaleIntensityFixedMeand",
            "RandAdjustContrastd",
            "RandSimulateLowResolutiond",
            "RandCoarseDropoutd"
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
            merged_params[transform_name] = default_params.copy()
            if custom_params and transform_name in custom_params:
                merged_params[transform_name].update(custom_params[transform_name])
        return merged_params
    
    def _initialize_transforms(self) -> List[Tuple[str, mT.Transform]]:
        """
        初始化所有变换实例，并为每个实例设置随机状态
        
        Returns:
            List[Tuple[str, mT.Transform]]: 包含变换名称和实例的元组列表
            
        Raises:
            ValueError: 当变换类没有set_random_state方法时抛出
        """
        transforms_list: List[Tuple[str, mT.Transform]] = []
        for i, transform_name in enumerate(self.transform_order):
            params: Dict[str, Any] = self.transform_params[transform_name].copy()
            transform_class: Callable = self.transform_classes[transform_name]
            transform: mT.Transform = transform_class(**params)
            
            # 如果提供了随机种子，为每个变换创建独立的随机状态并设置
            if self.random_seed is not None:
                # 为每个变换生成唯一的种子
                transform_seed: int = self.random_seed + hash(transform_name) % 10000
                
                # 创建独立的RandomState
                random_state: np.random.RandomState = np.random.RandomState(transform_seed)
                
                # 检查变换是否有set_random_state方法
                if not hasattr(transform, 'set_random_state'):
                    raise ValueError(f"变换类 {transform_name} 没有set_random_state方法")
                
                try:
                    transform.set_random_state(state=random_state)
                except Exception as e:
                    raise ValueError(f"设置 {transform_name} 的随机状态失败: {str(e)}")
            
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
            name: 增强操作名称
            params: 增强操作参数
            result: 增强操作结果
        """
        for callback in self.callbacks:
            callback(name, params, result)
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        执行数据增强流程
        
        Args:
            data: 包含'image'和可选'mask'的字典，张量形状为 (C, D, H, W)
            
        Returns:
            增强后的字典数据
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
    
    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有增强步骤的随机状态
        
        Returns:
            Dict[str, Dict[str, Any]]: 包含每个变换随机状态的字典
            
        Raises:
            ValueError: 当变换类没有get_random_state方法时抛出
        """
        state_dict: Dict[str, Dict[str, Any]] = {}
        for transform_name, transform in self.transforms:
            try:
                state_dict[transform_name] = {
                    'random_state': transform.R.get_state()
                }
            except Exception as e:
                raise ValueError(f"获取 {transform_name} 的随机状态失败: {str(e)}")
            
        return state_dict
    
    def set_state(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        设置所有增强步骤的随机状态
        
        Args:
            state_dict: 包含每个变换随机状态的字典
            
        Raises:
            ValueError: 当变换类没有set_random_state方法或状态设置失败时抛出
        """
        for transform_name, transform in self.transforms:
            if transform_name not in state_dict:
                raise ValueError(f"状态字典中缺少 {transform_name} 的状态信息")
                
            if not hasattr(transform, 'set_random_state'):
                raise ValueError(f"变换类 {transform_name} 没有set_random_state方法")
                
            random_state: np.random.RandomState = np.random.RandomState()
            random_state.set_state(state_dict[transform_name].get('random_state'))

            if random_state is None:
                raise ValueError(f"{transform_name} 的状态信息中缺少random_state")
                
            try:
                transform.set_random_state(state=random_state)
            except Exception as e:
                raise ValueError(f"设置 {transform_name} 的随机状态失败: {str(e)}")


def example_callback(name: str, params: Dict[str, Any], result: Dict[str, torch.Tensor]) -> None:
    """
    示例回调函数，用于打印增强操作的信息
    
    Args:
        name: 增强操作名称
        params: 增强操作参数
        result: 增强操作结果
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
    sample_image: torch.Tensor = torch.rand((3, 32, 64, 64)) * 255  # 3通道图像
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
    
    # 创建增强管线实例
    aug_pipeline: DataAugmentationPipeline = DataAugmentationPipeline(callbacks=[example_callback])
    
    # 应用增强
    augmented_data: Dict[str, torch.Tensor] = aug_pipeline(data)
    
    print(f"原始图像形状: {sample_image.shape}")
    print(f"原始蒙版形状: {sample_mask.shape}")
    print(f"增强后图像形状: {augmented_data['image'].shape}")
    print(f"增强后蒙版形状: {augmented_data['mask'].shape}")
    print(f"原始图像范围: [{sample_image.min():.4f}, {sample_image.max():.4f}]")
    print(f"增强后图像范围: [{augmented_data['image'].min():.4f}, {augmented_data['image'].max():.4f}]")
    
    print("\n=== 示例2: 仅处理图像 ===")
    # 仅包含图像的数据字典
    image_only_data: Dict[str, torch.Tensor] = {
        'image': torch.rand((1, 32, 64, 64)) * 255
    }
    
    # 应用增强
    augmented_image_only: Dict[str, torch.Tensor] = aug_pipeline(image_only_data)
    
    print(f"增强后图像形状: {augmented_image_only['image'].shape}")
    
    print("\n=== 示例3: 使用自定义参数 ===")
    # 定义自定义参数
    custom_params: Dict[str, Dict[str, Any]] = {
        "Rand3DElasticd": {"prob": 0.5},
        "RandGaussianNoised": {"std": 1.0, "prob": 0.4},
        "RandCoarseDropoutd": {"holes": 12, "spatial_size": (10, 10, 10)}
    }
    
    # 使用自定义参数创建增强管线
    custom_aug_pipeline: DataAugmentationPipeline = DataAugmentationPipeline(
        callbacks=[example_callback],
        transform_params=custom_params
    )
    
    # 应用增强
    custom_augmented_data: Dict[str, torch.Tensor] = custom_aug_pipeline(data)
    
    print(f"使用自定义参数增强后图像形状: {custom_augmented_data['image'].shape}")
    print(f"使用自定义参数增强后蒙版形状: {custom_augmented_data['mask'].shape}")


if __name__ == "__main__":
    example_usage()