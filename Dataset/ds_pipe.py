#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集清单序列类

功能概述：
    提供两个数据集类：
    1. DatasetManifestWithLabelSequential：基于清单文件（Excel）加载包含标签的数据集
    2. DatasetManifestNoLabelSequential：基于清单文件（Excel）加载不包含标签的数据集
    
    两者均支持预处理和数据增强，实现标准Dataset接口，提供回调函数接口

核心功能：
    1. 清单文件解析：从Excel读取数据集信息
    2. 数据验证：校验影像组学特征数量和存在性
    3. 图像加载：使用MONAI加载图像和蒙版
    4. 预处理/增强：支持应用预处理和数据增强操作
    5. 回调机制：在数据获取过程中触发回调函数

参数说明：
    - root_dir: 数据集根目录
    - n_radiomics: 影像组学特征数量
    - manifest_file: 清单文件路径（Excel格式）
    - preprocess_augment: 预处理或数据增强类的实例对象列表
    - callbacks: 回调函数列表
    - n_len: 数据集长度限制

使用示例：
    # 创建带标签的数据集实例
    dataset_with_label = DatasetManifestWithLabelSequential(
        root_dir='./data',
        n_radiomics=21,
        manifest_file='manifest_with_label.xlsx',
        preprocess_augment=[preprocess_pipeline, augmentation_pipeline],
        callbacks=[callback_function]
    )
    
    # 创建无标签的数据集实例
    dataset_no_label = DatasetManifestNoLabelSequential(
        root_dir='./data',
        n_radiomics=21,
        manifest_file='manifest_no_label.xlsx',
        preprocess_augment=[preprocess_pipeline, augmentation_pipeline],
        callbacks=[callback_function]
    )
    
    # 获取单个样本（带标签）
    sample_with_label = dataset_with_label[0]
    image = sample_with_label['image']
    mask = sample_with_label['mask']
    radiomics = sample_with_label['radiomics']
    label = sample_with_label['label']
    
    # 获取单个样本（无标签）
    sample_no_label = dataset_no_label[0]
    image = sample_no_label['image']
    mask = sample_no_label['mask']
    radiomics = sample_no_label['radiomics']
"""

import os
import pandas as pd
import torch
import monai.transforms as transforms
from typing import Callable, Dict, Any, Optional, List, Union, Tuple, cast


class DatasetManifestWithLabelSequential(torch.utils.data.Dataset):
    """
    带标签的数据集清单序列类，基于Excel清单文件加载和处理包含标签的数据。
    实现标准Dataset接口，支持预处理和数据增强操作。
    """
    
    # 类属性类型注解
    root_dir: str
    n_radiomics: int
    manifest_file: str
    preprocess_augment: List[Callable]
    callbacks: List[Callable]
    n_len: Optional[int]
    manifest: pd.DataFrame
    loader: transforms.LoadImaged
    
    def __init__(self,
                 root_dir: str,
                 n_radiomics: int,
                 manifest_file: str,
                 preprocess_augment: Optional[List[Callable]] = None,
                 callbacks: Optional[List[Callable]] = None,
                 n_len: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            n_radiomics: 影像组学特征数量
            manifest_file: 清单文件路径（Excel格式）
            preprocess_augment: 预处理或数据增强类的实例对象列表
            callbacks: 回调函数列表
            n_len: 数据集长度限制
            
        Raises:
            ValueError: 当清单文件格式不正确或影像组学特征数量不匹配时抛出
        """
        self.root_dir: str = root_dir
        self.n_radiomics: int = n_radiomics
        self.manifest_file: str = manifest_file
        self.preprocess_augment: List[Callable] = preprocess_augment if preprocess_augment is not None else []
        self.callbacks: List[Callable] = callbacks if callbacks is not None else []
        self.n_len: Optional[int] = n_len
        
        # 验证目录和文件
        if not os.path.exists(root_dir):
            raise ValueError(f"数据集根目录不存在: {root_dir}")
        
        if not os.path.exists(manifest_file):
            raise ValueError(f"清单文件不存在: {manifest_file}")
        
        # 加载和验证清单文件
        self.manifest: pd.DataFrame = self._load_and_validate_manifest()
        
        # 初始化图像加载器
        self.loader: transforms.LoadImaged = transforms.LoadImaged(
            keys=['pre_img', 'pre_mask', 'post_img', 'post_mask'],
            image_only=False,
            ensure_channel_first=True
        )
    
    def _load_and_validate_manifest(self) -> pd.DataFrame:
        """
        加载并验证带标签的清单文件
        
        Returns:
            pd.DataFrame: 验证后的清单数据
            
        Raises:
            ValueError: 当清单文件格式不正确、缺少标签列或影像组学特征数量不匹配时抛出
        """
        # 加载Excel文件
        try:
            manifest: pd.DataFrame = pd.read_excel(self.manifest_file, dtype={'pid': str})
        except Exception as e:
            raise ValueError(f"无法读取清单文件: {str(e)}")
        
        # 验证必要列是否存在（包含标签列）
        required_columns: List[str] = ['pid', 'pCR', 'center', 'pre_img', 'pre_mask', 'post_img', 'post_mask']
        for col in required_columns:
            if col not in manifest.columns:
                raise ValueError(f"清单文件缺少必要列: {col}")
        
        # 验证影像组学特征列数量
        radiomics_columns: List[str] = [col for col in manifest.columns if col not in required_columns]
        if len(radiomics_columns) != self.n_radiomics:
            raise ValueError(
                f"影像组学特征数量不匹配: 预期 {self.n_radiomics}, 实际 {len(radiomics_columns)}"
            )
        
        # 验证图像和蒙版路径是否存在
        for _, row in manifest.iterrows():
            for img_col in ['pre_img', 'pre_mask', 'post_img', 'post_mask']:
                if pd.notna(row[img_col]):
                    full_path: str = os.path.join(self.root_dir, row[img_col].replace('\\', '/'))
                    if not os.path.exists(full_path):
                        raise ValueError(f"文件不存在: {full_path}")
        
        # 验证pCR标签值
        if not manifest['pCR'].isin([0, 1]).all():
            raise ValueError("pCR标签必须为0或1")
        
        return manifest
    
    def apply_callbacks(self, idx: int, pid: str, image_paths: Dict[str, str], 
                       mask_paths: Dict[str, str], data: Dict[str, Any]) -> None:
        """
        应用所有注册的回调函数
        
        Args:
            idx: 样本索引
            pid: 样本标识符
            image_paths: 图像路径字典
            mask_paths: 蒙版路径字典
            data: 处理和包装后的数据
        """
        callback: Callable[[int, str, Dict[str, str], Dict[str, str], Dict[str, Any]], None]
        for callback in self.callbacks:
            callback(idx, pid, image_paths, mask_paths, data)
    
    def __len__(self) -> int:
        """
        返回数据集样本数量
        
        Returns:
            int: 样本数量
        """
        if self.n_len is not None and self.n_len > 0:
            return self.n_len
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        根据索引获取带标签的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict[str, torch.Tensor]: 包含处理后的图像、蒙版、影像组学特征和标签的字典
        """
        # 获取清单记录
        record: pd.Series = self.manifest.iloc[idx]
        pid: str = str(record['pid'])
        
        # 准备图像和蒙版路径
        image_paths: Dict[str, str] = {}
        mask_paths: Dict[str, str] = {}
        data_dict: Dict[str, str] = {}
        
        # 处理预处理图像和蒙版
        for key in ['pre_img', 'pre_mask']:
            if pd.notna(record[key]):
                full_path: str = os.path.join(self.root_dir, record[key].replace('\\', '/'))
                data_dict[key] = full_path
                if key.endswith('img'):
                    image_paths[key] = full_path
                else:
                    mask_paths[key] = full_path
        
        # 处理后处理图像和蒙版
        for key in ['post_img', 'post_mask']:
            if pd.notna(record[key]):
                full_path: str = os.path.join(self.root_dir, record[key].replace('\\', '/'))
                data_dict[key] = full_path
                if key.endswith('img'):
                    image_paths[key] = full_path
                else:
                    mask_paths[key] = full_path
        
        # 加载图像和蒙版
        loaded_data: Dict[str, Any] = self.loader(data_dict)
        
        # 准备处理数据（只保留图像和蒙版）
        process_data: Dict[str, torch.Tensor] = {}
        for key in loaded_data:
            if isinstance(loaded_data[key], torch.Tensor):
                process_data[key] = loaded_data[key]
        
        # 应用预处理和数据增强
        for transform in self.preprocess_augment:
            process_data = cast(Dict[str, torch.Tensor], transform(process_data))
        
        # 提取影像组学特征
        radiomics_columns: List[str] = [col for col in self.manifest.columns if col not in 
                           ['pid', 'pCR', 'center', 'pre_img', 'pre_mask', 'post_img', 'post_mask']]
        radiomics_features: Any = record[radiomics_columns].values.astype(float)
        radiomics_tensor: torch.Tensor = torch.tensor(radiomics_features, dtype=torch.float32)
        
        # 创建one-hot编码标签
        pcr_label: int = int(record['pCR'])
        label_tensor: torch.Tensor = torch.zeros(2, dtype=torch.float32)
        label_tensor[pcr_label] = 1.0
        
        # 准备返回数据
        result: Dict[str, torch.Tensor] = {
            'idx': torch.tensor(idx, dtype=torch.int64),
            'pre_img': process_data.get('pre_img'),
            'pre_mask': process_data.get('pre_mask'),
            'post_img': process_data.get('post_img'),
            'post_mask': process_data.get('post_mask'),
            'radiomics': radiomics_tensor,
            'label': label_tensor,
        }
        
        # 应用回调函数
        self.apply_callbacks(idx, pid, image_paths, mask_paths, cast(Dict[str, Any], result))
        
        return result


class DatasetManifestNoLabelSequential(torch.utils.data.Dataset):
    """
    无标签的数据集清单序列类，基于Excel清单文件加载和处理不包含标签的数据。
    实现标准Dataset接口，支持预处理和数据增强操作。
    """
    
    # 类属性类型注解
    root_dir: str
    n_radiomics: int
    manifest_file: str
    preprocess_augment: List[Callable]
    callbacks: List[Callable]
    n_len: Optional[int]
    manifest: pd.DataFrame
    loader: transforms.LoadImaged
    
    def __init__(self,
                 root_dir: str,
                 n_radiomics: int,
                 manifest_file: str,
                 preprocess_augment: Optional[List[Callable]] = None,
                 callbacks: Optional[List[Callable]] = None,
                 n_len: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            n_radiomics: 影像组学特征数量
            manifest_file: 清单文件路径（Excel格式）
            preprocess_augment: 预处理或数据增强类的实例对象列表
            callbacks: 回调函数列表
            n_len: 数据集长度限制
            
        Raises:
            ValueError: 当清单文件格式不正确或影像组学特征数量不匹配时抛出
        """
        self.root_dir: str = root_dir
        self.n_radiomics: int = n_radiomics
        self.manifest_file: str = manifest_file
        self.preprocess_augment: List[Callable] = preprocess_augment if preprocess_augment is not None else []
        self.callbacks: List[Callable] = callbacks if callbacks is not None else []
        self.n_len: Optional[int] = n_len
        
        # 验证目录和文件
        if not os.path.exists(root_dir):
            raise ValueError(f"数据集根目录不存在: {root_dir}")
        
        if not os.path.exists(manifest_file):
            raise ValueError(f"清单文件不存在: {manifest_file}")
        
        # 加载和验证清单文件
        self.manifest: pd.DataFrame = self._load_and_validate_manifest()
        
        # 初始化图像加载器
        self.loader: transforms.LoadImaged = transforms.LoadImaged(
            keys=['pre_img', 'pre_mask', 'post_img', 'post_mask'],
            image_only=False,
            ensure_channel_first=True
        )
    
    def _load_and_validate_manifest(self) -> pd.DataFrame:
        """
        加载并验证无标签的清单文件
        
        Returns:
            pd.DataFrame: 验证后的清单数据
            
        Raises:
            ValueError: 当清单文件格式不正确或影像组学特征数量不匹配时抛出
        """
        # 加载Excel文件
        try:
            manifest: pd.DataFrame = pd.read_excel(self.manifest_file, dtype={'pid': str})
        except Exception as e:
            raise ValueError(f"无法读取清单文件: {str(e)}")
        
        # 验证必要列是否存在（不包含标签列）
        required_columns: List[str] = ['pid', 'center', 'pre_img', 'pre_mask', 'post_img', 'post_mask']
        for col in required_columns:
            if col not in manifest.columns:
                raise ValueError(f"清单文件缺少必要列: {col}")
        
        # 验证影像组学特征列数量
        radiomics_columns: List[str] = [col for col in manifest.columns if col not in required_columns + ['pCR']]
        if len(radiomics_columns) != self.n_radiomics:
            raise ValueError(
                f"影像组学特征数量不匹配: 预期 {self.n_radiomics}, 实际 {len(radiomics_columns)}"
            )
        
        # 验证图像和蒙版路径是否存在
        for _, row in manifest.iterrows():
            for img_col in ['pre_img', 'pre_mask', 'post_img', 'post_mask']:
                if pd.notna(row[img_col]):
                    full_path: str = os.path.join(self.root_dir, row[img_col].replace('\\', '/'))
                    if not os.path.exists(full_path):
                        raise ValueError(f"文件不存在: {full_path}")
        
        return manifest
    
    def apply_callbacks(self, idx: int, pid: str, image_paths: Dict[str, str], 
                       mask_paths: Dict[str, str], data: Dict[str, Any]) -> None:
        """
        应用所有注册的回调函数
        
        Args:
            idx: 样本索引
            pid: 样本标识符
            image_paths: 图像路径字典
            mask_paths: 蒙版路径字典
            data: 处理和包装后的数据
        """
        for callback in self.callbacks:
            callback(idx, pid, image_paths, mask_paths, data)
    
    def __len__(self) -> int:
        """
        返回数据集样本数量
        
        Returns:
            int: 样本数量
        """
        if self.n_len is not None and self.n_len > 0:
            return self.n_len
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        根据索引获取无标签的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict[str, torch.Tensor]: 包含处理后的图像、蒙版和影像组学特征的字典（不包含标签）
        """
        # 获取清单记录
        record: pd.Series = self.manifest.iloc[idx]
        pid: str = str(record['pid'])
        
        # 准备图像和蒙版路径
        image_paths: Dict[str, str] = {}
        mask_paths: Dict[str, str] = {}
        data_dict: Dict[str, str] = {}
        
        # 处理预处理图像和蒙版
        for key in ['pre_img', 'pre_mask']:
            if pd.notna(record[key]):
                full_path: str = os.path.join(self.root_dir, record[key].replace('\\', '/'))
                data_dict[key] = full_path
                if key.endswith('img'):
                    image_paths[key] = full_path
                else:
                    mask_paths[key] = full_path
        
        # 处理后处理图像和蒙版
        for key in ['post_img', 'post_mask']:
            if pd.notna(record[key]):
                full_path: str = os.path.join(self.root_dir, record[key].replace('\\', '/'))
                data_dict[key] = full_path
                if key.endswith('img'):
                    image_paths[key] = full_path
                else:
                    mask_paths[key] = full_path
        
        # 加载图像和蒙版
        loaded_data: Dict[str, Any] = self.loader(data_dict)
        
        # 准备处理数据（只保留图像和蒙版）
        process_data: Dict[str, torch.Tensor] = {}
        for key in loaded_data:
            if isinstance(loaded_data[key], torch.Tensor):
                process_data[key] = loaded_data[key]
        
        # 应用预处理和数据增强
        for transform in self.preprocess_augment:
            process_data = cast(Dict[str, torch.Tensor], transform(process_data))
        
        # 提取影像组学特征
        radiomics_columns: List[str] = [col for col in self.manifest.columns if col not in 
                           ['pid', 'pCR', 'center', 'pre_img', 'pre_mask', 'post_img', 'post_mask']]
        radiomics_features: Any = record[radiomics_columns].values.astype(float)
        radiomics_tensor: torch.Tensor = torch.tensor(radiomics_features, dtype=torch.float32)
        
        # 准备返回数据（不包含标签）
        result: Dict[str, torch.Tensor] = {
            'pre_img': process_data.get('pre_img'),
            'pre_mask': process_data.get('pre_mask'),
            'post_img': process_data.get('post_img'),
            'post_mask': process_data.get('post_mask'),
            'radiomics': radiomics_tensor,
        }
        
        # 应用回调函数
        self.apply_callbacks(idx, pid, image_paths, mask_paths, cast(Dict[str, Any], result))
        
        return result


def example_callback(idx: int, pid: str, image_paths: Dict[str, str], 
                    mask_paths: Dict[str, str], data: Dict[str, Any]) -> None:
    """
    示例回调函数，用于打印数据加载和处理信息
    
    Args:
        idx: 样本索引
        pid: 样本标识符
        image_paths: 图像路径字典
        mask_paths: 蒙版路径字典
        data: 处理和包装后的数据
    """
    print(f"加载样本: 索引={idx}, PID={pid}")
    print(f"图像路径: {image_paths}")
    print(f"蒙版路径: {mask_paths}")
    
    # 打印处理后的数据信息
    if 'pre_img' in data and data['pre_img'] is not None:
        print(f"治疗前图像形状: {data['pre_img'].shape}")
    if 'pre_mask' in data and data['pre_mask'] is not None:
        print(f"治疗前蒙版形状: {data['pre_mask'].shape}")
    if 'post_img' in data and data['post_img'] is not None:
        print(f"治疗后图像形状: {data['post_img'].shape}")
    if 'post_mask' in data and data['post_mask'] is not None:
        print(f"治疗后蒙版形状: {data['post_mask'].shape}")
    
    if 'radiomics' in data:
        print(f"影像组学特征规格: {data['radiomics'].shape}")
    if 'label' in data:
        print(f"标签: {data['label']}")
    print("---")


def example_usage() -> None:
    """
    示例用法
    """
    # 注意：这个示例需要实际的数据集和清单文件才能运行
    print("数据集类示例用法")
    print("--------------------------------")
    print("以下是如何使用这些类的示例代码:")
    print("""
    # 导入必要的库
    from Preprocess.preproc_pipe import DataPreprocessingPipeline
    from Augmentation.aug_pipe import DataAugmentationPipeline
    
    # 创建预处理和增强实例
    preprocess_pipeline = DataPreprocessingPipeline(callbacks=[])
    augmentation_pipeline = DataAugmentationPipeline(callbacks=[])
    
    # 示例1: 创建带标签的数据集实例
    dataset_with_label = DatasetManifestWithLabelSequential(
        root_dir='./data',
        n_radiomics=21,
        manifest_file='manifest_with_label.xlsx',
        preprocess_augment=[preprocess_pipeline, augmentation_pipeline],
        callbacks=[example_callback],
        n_len=None  # 不限制数据集长度，使用完整数据集
    )
    
    # 获取带标签数据集大小
    print(f"带标签数据集大小: {len(dataset_with_label)}")
    
    # 获取第一个带标签样本
    if len(dataset_with_label) > 0:
        sample_with_label = dataset_with_label[0]
        print(f"带标签样本包含的键: {list(sample_with_label.keys())}")
        if 'label' in sample_with_label:
            print(f"样本标签: {sample_with_label['label']}")
    
    # 示例2: 创建无标签的数据集实例
    dataset_no_label = DatasetManifestNoLabelSequential(
        root_dir='./data',
        n_radiomics=21,
        manifest_file='manifest_no_label.xlsx',
        preprocess_augment=[preprocess_pipeline, augmentation_pipeline],
        callbacks=[example_callback],
        n_len=None  # 不限制数据集长度，使用完整数据集
    )
    
    # 获取无标签数据集大小
    print(f"无标签数据集大小: {len(dataset_no_label)}")
    
    # 获取第一个无标签样本
    if len(dataset_no_label) > 0:
        sample_no_label = dataset_no_label[0]
        print(f"无标签样本包含的键: {list(sample_no_label.keys())}")
        print(f"是否包含标签: {'label' in sample_no_label}")
    """)


if __name__ == "__main__":
    example_usage()