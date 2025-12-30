#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集管理模块

功能概述：
    基于PyTorch Lightning的DataModule实现
    支持训练集、验证集、测试集和预测集的管理
    实现样本加权采样，确保训练集正负样本均衡
    提供标准的数据加载器接口，支持单独配置batch_size
    所有参数均提供默认值，增强代码灵活性
    完善的路径和清单文件验证机制

核心功能：
    1. 多数据集支持：分别配置训练集、验证集、测试集和预测集
    2. 加权采样：使用WeightedRandomSampler均衡训练集正负样本分布
    3. 灵活的batch_size配置：每个数据集可单独设置batch_size，包括预测集
    4. 数据加载器：提供标准的train_dataloader/val_dataloader/test_dataloader/predict_dataloader接口
    5. 空值检查：在setup阶段验证必要的路径和清单文件不为None
    6. 参数默认值：所有参数均提供合理的默认值，提高使用便捷性
    7. 预处理和增强管线：各数据集支持独立配置预处理和增强流程
    8. 预测集专用功能：使用DatasetManifestNoLabelSequential处理无标签数据

参数说明：
    - root_dir_train/val/test/predict: 各数据集根目录，默认值为None
    - manifest_file_train/val/test/predict: 各数据集清单文件，默认值为None
    - n_radiomics: 影像组学特征数量，默认值为21
    - train_len: 训练集长度限制，默认值为None
    - batch_size_train: 训练集批量大小，默认值为2
    - batch_size_val: 验证集批量大小，默认值为1
    - batch_size_test: 测试集批量大小，默认值为1
    - batch_size_predict: 预测集批量大小，默认值为1
    - num_workers: 数据加载工作线程数，默认值为4

使用示例：
    # 创建DataModule实例（最小化配置）
    data_module = EsophagusDataModule(
        root_dir_train='./data/train',
        root_dir_val='./data/val',
        root_dir_test='./data/test',
        manifest_file_train='manifest_train.xlsx',
        manifest_file_val='manifest_val.xlsx',
        manifest_file_test='manifest_test.xlsx'
    )

    # 创建DataModule实例（完整配置）
    data_module = EsophagusDataModule(
        root_dir_train='./data/train',
        root_dir_val='./data/val',
        root_dir_test='./data/test',
        root_dir_predict='./data/predict',
        manifest_file_train='manifest_train.xlsx',
        manifest_file_val='manifest_val.xlsx',
        manifest_file_test='manifest_test.xlsx',
        manifest_file_predict='manifest_predict.xlsx',
        n_radiomics=21,
        train_len=100,
        batch_size_train=2,
        batch_size_val=1,
        batch_size_test=1,
        batch_size_predict=1,
        sample_weight_train=[0.25, 0.75],
        num_workers=4,
        random_seed=42,
    )

    # 在Lightning Trainer中使用
    trainer = L.Trainer()
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L
from typing import Dict, Any, Optional, Callable, Union, List, Sequence

# 导入自定义数据集类
from Dataset.ds_pipe import DatasetManifestWithLabelSequential, DatasetManifestNoLabelSequential

# 定义预处理和增强管线
from Preprocess.preproc_pipe import DataPreprocessingPipeline
from Augmentation.aug_pipe import DataAugmentationPipeline

class EsophagusDataModule(L.LightningDataModule):
    """
    食管数据集管理模块，基于PyTorch Lightning DataModule实现
    支持训练、验证、测试和预测数据集的管理和加载
    
    主要功能：
    - 训练集：支持加权采样，预处理和数据增强
    - 验证集：支持预处理，用于模型验证
    - 测试集：支持预处理，用于模型评估
    - 预测集：使用无标签数据集类，支持预处理，用于模型推理
    """
    
    # 前置成员变量类型声明
    root_dirs: Dict[str, Optional[str]]
    manifest_files: Dict[str, Optional[str]]
    n_radiomics: int
    train_len: Optional[int]
    batch_sizes: Dict[str, int]
    num_workers: int
    datasets: Dict[str, Union[DatasetManifestWithLabelSequential, DatasetManifestNoLabelSequential]]
    train_weights: Optional[torch.Tensor]
    random_seed: Optional[int]
    sampler_rng: Optional[torch.Generator]
    sample_weight_train: Union[bool, Sequence[float]]
    
    def __init__(self,
                 root_dir_train: Optional[str] = None,
                 root_dir_val: Optional[str] = None,
                 root_dir_test: Optional[str] = None,
                 root_dir_predict: Optional[str] = None,
                 manifest_file_train: Optional[str] = None,
                 manifest_file_val: Optional[str] = None,
                 manifest_file_test: Optional[str] = None,
                 manifest_file_predict: Optional[str] = None,
                 n_radiomics: int = 21,
                 train_len: Optional[int] = None,
                 batch_size_train: int = 2,
                 batch_size_val: int = 1,
                 batch_size_test: int = 1,
                 batch_size_predict: int = 1,
                 sample_weight_train: Union[bool, Sequence[float]] = False,
                 num_workers: int = 4,
                 random_seed: Optional[int] = None):
        """
        初始化数据模块
        
        Args:
            root_dir_train: 训练集根目录，默认值为None
            root_dir_val: 验证集根目录，默认值为None
            root_dir_test: 测试集根目录，默认值为None
            root_dir_predict: 预测集根目录，默认值为None
            manifest_file_train: 训练集清单文件路径，默认值为None
            manifest_file_val: 验证集清单文件路径，默认值为None
            manifest_file_test: 测试集清单文件路径，默认值为None
            manifest_file_predict: 预测集清单文件路径，默认值为None
            n_radiomics: 影像组学特征数量，默认值为21
            train_len: 训练集长度限制，默认值为None
            batch_size_train: 训练集批量大小，默认值为2
            batch_size_val: 验证集批量大小，默认值为1
            batch_size_test: 测试集批量大小，默认值为1
            batch_size_predict: 预测集批量大小，默认值为1
            num_workers: 数据加载工作线程数，默认值为4
            random_seed: 随机种子，用于数据增强和数据加载器，默认值为None
        """
        super().__init__()
        
        # 数据集路径配置
        self.root_dirs: Dict[str, Optional[str]] = {
            'train': root_dir_train,
            'val': root_dir_val,
            'test': root_dir_test,
            'predict': root_dir_predict
        }
        
        # 清单文件路径配置
        self.manifest_files: Dict[str, Optional[str]] = {
            'train': manifest_file_train,
            'val': manifest_file_val,
            'test': manifest_file_test,
            'predict': manifest_file_predict
        }
        
        # 其他参数
        self.n_radiomics: int = n_radiomics
        self.train_len: Optional[int] = train_len
        self.batch_sizes: Dict[str, int] = {
            'train': batch_size_train,
            'val': batch_size_val,
            'test': batch_size_test,
            'predict': batch_size_predict
        }
        self.num_workers: int = num_workers
        
        # 数据集实例
        self.datasets: Dict[str, Union[DatasetManifestWithLabelSequential, DatasetManifestNoLabelSequential]] = {}
        
        # 缓存训练集权重
        self.train_weights: Optional[torch.Tensor] = None
        
        # 随机种子
        self.random_seed: Optional[int] = random_seed
        
        # 数据加载器生成器
        self.sampler_rng: Optional[torch.Generator] = None
        
        # 样本权重配置
        self.sample_weight_train: Union[bool, Sequence[float]] = sample_weight_train
        
    
    def prepare_data(self) -> None:
        """
        数据准备阶段，验证数据集和清单文件是否存在
        注意：此方法在所有进程中只执行一次
        """
        # 验证目录和文件
        for split in ['train', 'val', 'test', 'predict']:
            root_dir: Optional[str] = self.root_dirs[split]
            manifest_file: Optional[str] = self.manifest_files[split]
            
            # 当需要初始化某个数据集时，检查root_dir和manifest_file是否存在
            # 在setup中会调用，这里先不做None检查
            if root_dir and manifest_file:
                if not os.path.exists(root_dir):
                    raise ValueError(f"数据集根目录不存在: {root_dir}")
                if not os.path.exists(manifest_file):
                    raise ValueError(f"清单文件不存在: {manifest_file}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        数据集设置阶段，初始化数据集实例
        
        Args:
            stage: 当前训练阶段，可以是'fit', 'validate', 'test', 'predict'
            
        Raises:
            ValueError: 当尝试初始化包含None值的数据集路径或清单文件时抛出
        """
        # 根据当前阶段初始化对应数据集
        if stage == 'fit' or stage is None:
            # 初始化训练集
            if 'train' in self.root_dirs and 'train' in self.manifest_files:
                root_dir: Optional[str] = self.root_dirs['train']
                manifest_file: Optional[str] = self.manifest_files['train']
                
                if root_dir is None:
                    raise ValueError("训练集根目录不能为None")
                if manifest_file is None:
                    raise ValueError("训练集清单文件不能为None")
                
                # 定义预处理参数
                preproc_params: Dict[str, Dict[str, Any]] = {
                    "ScaleIntensityRanged": {
                        "keys": ["pre_img", "post_img"],
                        "a_min": -400,
                        "a_max": 400.0,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True
                    },
                    "ResizeWithPadOrCropd": {
                        "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
                        "spatial_size": (128, 128, 128),
                        "mode": "constant",
                        "method": "symmetric"
                    }
                }
                
                # 定义增强参数
                aug_params: Dict[str, Dict[str, Any]] = {
                    "Rand3DElasticd": {
                        "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
                        "sigma_range": (5.0, 7.0),
                        "magnitude_range": (50.0, 150.0),
                        "prob": 0.3,
                        "mode": ["bilinear", "nearest", "bilinear", "nearest"],
                        "padding_mode": "border"
                    },
                    "RandGaussianSmoothd": {
                        "keys": ["pre_img", "post_img"],
                        "sigma_x": (0.5, 1.0),
                        "sigma_y": (0.5, 1.0),
                        "sigma_z": (0.5, 1.0),
                        "prob": 0.3
                    },
                    "RandGaussianNoised": {
                        "keys": ["pre_img", "post_img"],
                        "mean": 0.0,
                        "std": 0.1,
                        "prob": 0.3
                    },
                    "RandShiftIntensityd": {
                        "keys": ["pre_img", "post_img"],
                        "offsets": (-0.2, 0.2),
                        "prob": 0.3
                    },
                    "RandScaleIntensityFixedMeand": {
                        "keys": ["pre_img", "post_img"],
                        "factors": (0.9, 1.1),
                        "prob": 0.3
                    },
                    "RandAdjustContrastd": {
                        "keys": ["pre_img", "post_img"],
                        "gamma": (0.8, 1.2),
                        "invert_image": False,
                        "prob": 0.3
                    },
                    "RandSimulateLowResolutiond": {
                        "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
                        "zoom_range": (0.5, 1.0),
                        "prob": 0.3
                    },
                    "RandCoarseDropoutd": {
                        "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
                        "holes": 1,
                        "spatial_size": (4, 4, 4),
                        "fill_value": 0.0,
                        "max_holes": 16,
                        "max_spatial_size": (32, 32, 32),
                        "prob": 0.3
                    }
                }
                
                # 创建预处理和增强管线实例
                preprocess_pipeline: DataPreprocessingPipeline = DataPreprocessingPipeline(
                    callbacks=[],  # 如需调试，可添加回调函数
                    transform_params=preproc_params
                )
                
                # 为增强管线设置随机种子
                augmentation_pipeline: DataAugmentationPipeline = DataAugmentationPipeline(
                    callbacks=[],  # 如需调试，可添加回调函数
                    transform_params=aug_params,
                    random_seed=self.random_seed
                )
                
                # 组合预处理和增强管线
                preprocess_augment: List[Union[DataPreprocessingPipeline, DataAugmentationPipeline]] = [preprocess_pipeline, augmentation_pipeline]
                # preprocess_augment: List[Union[DataPreprocessingPipeline, DataAugmentationPipeline]] = [preprocess_pipeline] # 禁用数据增强

                self.datasets['train'] = DatasetManifestWithLabelSequential(
                    root_dir=self.root_dirs['train'],
                    n_radiomics=self.n_radiomics,
                    manifest_file=self.manifest_files['train'],
                    preprocess_augment=preprocess_augment,
                    callbacks=[],  # 如需调试，可添加回调函数
                    n_len=self.train_len
                )
                # 根据sample_weight_train决定是否计算权重
                if self.sample_weight_train is True:
                    # 自动计算权重
                    self._calculate_train_weights()
                elif isinstance(self.sample_weight_train, Sequence):
                    # 使用指定的权重
                    self.train_weights = torch.tensor(self.sample_weight_train, dtype=torch.float32)
                    # 归一化权重
                    if self.train_weights.sum() > 0:
                        self.train_weights = self.train_weights / self.train_weights.sum()
                
                # 初始化采样器的随机数生成器
                self.sampler_rng = torch.Generator()
                self.sampler_rng.manual_seed(self.random_seed)
            
            # 初始化验证集
            if 'val' in self.root_dirs and 'val' in self.manifest_files:
                root_dir_val: Optional[str] = self.root_dirs['val']
                manifest_file_val: Optional[str] = self.manifest_files['val']
                
                if root_dir_val is None:
                    raise ValueError("验证集根目录不能为None")
                if manifest_file_val is None:
                    raise ValueError("验证集清单文件不能为None")
                
                # 定义预处理参数
                preproc_params: Dict[str, Dict[str, Any]] = {
                    "ScaleIntensityRanged": {
                        "keys": ["pre_img", "post_img"],
                        "a_min": -400,
                        "a_max": 400.0,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True
                    },
                    "ResizeWithPadOrCropd": {
                        "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
                        "spatial_size": (128, 128, 128),
                        "mode": "constant",
                        "method": "symmetric"
                    }
                }

                # 创建预处理和增强管线实例
                preprocess_pipeline: DataPreprocessingPipeline = DataPreprocessingPipeline(
                    callbacks=[],  # 如需调试，可添加回调函数
                    transform_params=preproc_params
                )

                self.datasets['val'] = DatasetManifestWithLabelSequential(
                    root_dir=self.root_dirs['val'],
                    n_radiomics=self.n_radiomics,
                    manifest_file=self.manifest_files['val'],
                    preprocess_augment=[preprocess_pipeline],
                    callbacks=[],  # 如需调试，可添加回调函数
                    n_len=None
                )
        
        if stage == 'test' or stage is None:
            # 初始化测试集
            if 'test' in self.root_dirs and 'test' in self.manifest_files:
                root_dir_test: Optional[str] = self.root_dirs['test']
                manifest_file_test: Optional[str] = self.manifest_files['test']
                
                if root_dir_test is None:
                    raise ValueError("测试集根目录不能为None")
                if manifest_file_test is None:
                    raise ValueError("测试集清单文件不能为None")
                
                # 定义预处理参数
                preproc_params: Dict[str, Dict[str, Any]] = {
                    "ScaleIntensityRanged": {
                        "keys": ["pre_img", "post_img"],
                        "a_min": -400,
                        "a_max": 400.0,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True
                    },
                    "ResizeWithPadOrCropd": {
                        "keys": ["pre_img", "pre_mask", "post_img", "post_mask"],
                        "spatial_size": (128, 128, 128),
                        "mode": "constant",
                        "method": "symmetric"
                    }
                }
                
                # 创建预处理和增强管线实例
                preprocess_pipeline: DataPreprocessingPipeline = DataPreprocessingPipeline(
                    callbacks=[],  # 如需调试，可添加回调函数
                    transform_params=preproc_params
                )

                self.datasets['test'] = DatasetManifestWithLabelSequential(
                    root_dir=self.root_dirs['test'],
                    n_radiomics=self.n_radiomics,
                    manifest_file=self.manifest_files['test'],
                    preprocess_augment=[preprocess_pipeline],
                    callbacks=[],  # 如需调试，可添加回调函数
                    n_len=None
                )
        
        # 初始化预测集
        if stage == 'predict' or stage is None:
            if 'predict' in self.root_dirs and 'predict' in self.manifest_files:
                root_dir_predict: Optional[str] = self.root_dirs['predict']
                manifest_file_predict: Optional[str] = self.manifest_files['predict']
                
                if root_dir_predict is None:
                    raise ValueError("预测集根目录不能为None")
                if manifest_file_predict is None:
                    raise ValueError("预测集清单文件不能为None")
                
                # 定义预处理参数（与测试集相同）
                preproc_params: Dict[str, Dict[str, Any]] = {
                    "ScaleIntensityRanged": {
                        "keys": ["pre_img", "post_img"],
                        "a_min": -400,
                        "a_max": 400.0,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True
                    },
                    "ResizeWithPadOrCropd": {
                        "keys": ["pre_img", "post_img"],
                        "spatial_size": (128, 128, 128),
                        "mode": "constant",
                        "method": "symmetric"
                    }
                }
                
                # 创建预处理和增强管线实例
                preprocess_pipeline: DataPreprocessingPipeline = DataPreprocessingPipeline(
                    callbacks=[],  # 如需调试，可添加回调函数
                    transform_params=preproc_params
                )
                
                self.datasets['predict'] = DatasetManifestNoLabelSequential(
                    root_dir=self.root_dirs['predict'],
                    n_radiomics=self.n_radiomics,
                    manifest_file=self.manifest_files['predict'],
                    preprocess_augment=[preprocess_pipeline],
                    callbacks=[],  # 如需调试，可添加回调函数
                    n_len=None
                )
    
    def _calculate_train_weights(self) -> None:
        """
        计算训练集样本权重，用于WeightedRandomSampler
        基于pCR标签计算权重，确保正负样本均衡
        """
        if 'train' not in self.datasets:
            return
        
        # 从训练集manifest中获取pCR标签
        train_manifest: pd.DataFrame = self.datasets['train'].manifest
        labels: pd.Series = train_manifest['pCR'].values
        
        # 计算正负样本数量
        n_positive: int = (labels == 1).sum()
        n_negative: int = (labels == 0).sum()
        n_total: int = len(labels)
        
        # 计算权重
        weights: torch.Tensor = torch.zeros(n_total)
        if n_positive > 0:
            weights[labels == 1] = 1.0 / n_positive
        if n_negative > 0:
            weights[labels == 0] = 1.0 / n_negative
        
        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        self.train_weights = weights
    
    def train_dataloader(self) -> DataLoader:
        """
        返回训练集数据加载器
        根据sample_weight_train参数决定是否使用加权随机采样器
        
        Returns:
            DataLoader: 训练集数据加载器
        """
        if 'train' not in self.datasets:
            raise ValueError("训练集未初始化")
        
        # 根据sample_weight_train决定是否使用权重采样器
        if self.sample_weight_train and self.train_weights is not None:
            # 使用加权随机采样器
            sampler: WeightedRandomSampler = WeightedRandomSampler(
                weights=self.train_weights,
                num_samples=len(self.datasets['train']),
                replacement=True,
                generator=self.sampler_rng
            )
            
            return DataLoader(
                self.datasets['train'],
                batch_size=self.batch_sizes['train'],
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,  # 使用sampler时shuffle应设为False
                drop_last=False,
                generator=self.sampler_rng,
                persistent_workers=True if self.num_workers > 0 else False
            )
        else:
            # 不使用权重采样器，使用普通的shuffle
            return DataLoader(
                self.datasets['train'],
                batch_size=self.batch_sizes['train'],
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,  # 不使用sampler时开启shuffle
                drop_last=False,
                generator=self.sampler_rng,
                persistent_workers=True if self.num_workers > 0 else False
            )
    
    def val_dataloader(self) -> DataLoader:
        """
        返回验证集数据加载器
        
        Returns:
            DataLoader: 验证集数据加载器
        """
        if 'val' not in self.datasets:
            raise ValueError("验证集未初始化")
        
        return DataLoader(
            self.datasets['val'],
            batch_size=self.batch_sizes['val'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        返回测试集数据加载器
        
        Returns:
            DataLoader: 测试集数据加载器
        """
        if 'test' not in self.datasets:
            raise ValueError("测试集未初始化")
        
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_sizes['test'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """
        返回预测集数据加载器
        
        Returns:
            DataLoader: 预测集数据加载器
            
        Raises:
            ValueError: 当预测集未初始化时抛出
        """
        if 'predict' not in self.datasets:
            raise ValueError("预测集未初始化")
        
        return DataLoader(
            self.datasets['predict'],
            batch_size=self.batch_sizes['predict'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def state_dict(self) -> Dict[str, Any]:
        """
        保存数据模块的状态，用于PyTorch Lightning的检查点功能
        
        Returns:
            Dict[str, Any]: 包含数据模块状态的字典
        """
        state_dict: Dict[str, Any] = {
            'random_seed': self.random_seed,
            'train_weights': self.train_weights,
            'sampler_rng': self.sampler_rng.get_state()
        }
        
        # 尝试保存增强管线的状态
        if 'train' in self.datasets and hasattr(self.datasets['train'], 'preprocess_augment'):
            for transform in self.datasets['train'].preprocess_augment:
                transform: Any = transform
                if hasattr(transform, 'get_state'):
                    try:
                        state_dict[f'transform_state_{type(transform).__name__}'] = transform.get_state()
                    except Exception as e:
                        raise ValueError(f"获取 {type(transform).__name__} 的状态失败: {str(e)}")
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载数据模块的状态，用于PyTorch Lightning的检查点恢复功能
        
        Args:
            state_dict: 包含数据模块状态的字典
        """
        # 恢复随机种子
        self.random_seed = state_dict.get('random_seed')
        
        # 恢复训练权重
        if 'train_weights' in state_dict:
            self.train_weights = state_dict['train_weights']
        
        # 恢复生成器状态
        if 'sampler_rng' in state_dict:
            self.sampler_rng.set_state(state_dict['sampler_rng'])
        
        # 尝试恢复增强管线的状态
        if 'train' in self.datasets and hasattr(self.datasets['train'], 'preprocess_augment'):
            for transform in self.datasets['train'].preprocess_augment:
                transform_name: str = f'transform_state_{type(transform).__name__}'
                if transform_name in state_dict and hasattr(transform, 'set_state'):
                    try:
                        transform.set_state(state_dict[transform_name])
                    except Exception as e:
                        raise ValueError(f"设置 {type(transform).__name__} 的状态失败: {str(e)}")


def example_usage() -> None:
    """
    示例用法
    """
    print("EsophagusDataModule示例用法")
    print("--------------------------------")
    print("以下是如何使用此类的示例代码:")
    print("""
    # 导入必要的库
    import lightning as L
    from DataModule.dm_pipe import EsophagusDataModule
    
    # 基础用法 - 创建训练和验证数据模块
    data_module = EsophagusDataModule(
        root_dir_train='./data/train',
        root_dir_val='./data/val',
        manifest_file_train='manifest_train.xlsx',
        manifest_file_val='manifest_val.xlsx'
    )
    
    # 完整用法 - 包含测试集和预测集
    data_module = EsophagusDataModule(
        root_dir_train='./data/train',
        root_dir_val='./data/val',
        root_dir_test='./data/test',
        root_dir_predict='./data/predict',
        manifest_file_train='manifest_train.xlsx',
        manifest_file_val='manifest_val.xlsx',
        manifest_file_test='manifest_test.xlsx',
        manifest_file_predict='manifest_predict.xlsx',
        n_radiomics=21,
        train_len=100,
        batch_size_train=2,
        batch_size_val=1,
        batch_size_test=1,
        batch_size_predict=1,
        num_workers=4
    )
    
    # 在Lightning Trainer中使用
    trainer = L.Trainer()
    
    # 训练和验证
    trainer.fit(model, data_module)
    
    # 测试
    trainer.test(model, data_module)
    
    # 预测
    predictions = trainer.predict(model, data_module)
    
    # 或者直接使用数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    predict_loader = data_module.predict_dataloader()
    """)
    print("\n预测集使用说明:")
    print("1. 预测集使用DatasetManifestNoLabelSequential类处理无标签数据")
    print("2. 配置与测试集类似，但不包含mask相关预处理")
    print("3. 通过predict_dataloader()方法获取数据加载器")
    print("4. 可使用trainer.predict()或直接使用数据加载器进行推理")


if __name__ == "__main__":
    example_usage()