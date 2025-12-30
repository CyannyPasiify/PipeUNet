#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EsophagusDataModule训练模式可重复性测试脚本

功能概述：
    测试EsophagusDataModule的训练模式功能的可重复性
    运行数据加载管线两次，比较其结果一致性
    验证加权随机采样器和数据增强在相同随机种子下的行为一致性
    确保数据加载过程的可复现性

使用方法：
    python test_dm_pipe_train_reproducibility.py -r data_root_dir -m manifest_file.xlsx -n 21 -b 2 -w 4 -s 42
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict, Any, List, Tuple

# 导入自定义数据模块类
from DataModule.dm_pipe import EsophagusDataModule


def create_weighted_sampler_hook():
    """
    创建一个Hook来捕获WeightedRandomSampler选择的索引
    
    Returns:
        包含hook函数和索引列表的字典
    """
    selected_indices = []
    
    def sampler_hook(indices: List[int]) -> None:
        """
        捕获并记录采样器选择的索引
        
        Args:
            indices: 采样器选择的索引列表
        """
        selected_indices.extend(indices)
    
    return {
        'hook': sampler_hook,
        'indices': selected_indices
    }


def monkey_patch_weighted_sampler():
    """
    猴子补丁WeightedRandomSampler以捕获采样的索引
    
    Returns:
        包含hook信息的字典
    """
    # 获取原始的__iter__方法
    original_iter = torch.utils.data.WeightedRandomSampler.__iter__
    
    # 创建hook
    hook_info = create_weighted_sampler_hook()
    hook_info['original_iter'] = original_iter
    
    # 定义新的__iter__方法
    def new_iter(self):
        # 调用原始方法生成索引
        for idx in original_iter(self):
            # 记录索引
            hook_info['hook']([idx])
            yield idx
    
    # 替换方法
    torch.utils.data.WeightedRandomSampler.__iter__ = new_iter
    
    return hook_info


def reset_weighted_sampler_hook(hook_info: Dict) -> None:
    """
    重置hook中的索引列表
    
    Args:
        hook_info: hook信息字典
    """
    hook_info['indices'] = []


def extract_batch_data(batch: Dict[str, Any]) -> Tuple[List[int], Dict[str, torch.Tensor]]:
    """
    提取批次数据中的索引和关键张量数据
    
    Args:
        batch: 批次数据字典
    
    Returns:
        索引列表和张量数据字典的元组
    """
    # 获取索引
    indices = batch['idx'].tolist()
    
    # 提取关键张量数据
    tensor_data = {}
    for key in ['pre_img', 'pre_mask', 'post_img', 'post_mask', 'radiomics', 'label']:
        if key in batch:
            tensor_data[key] = batch[key].clone().detach()
    
    return indices, tensor_data


def compare_tensor_data(data1: Dict[str, torch.Tensor], data2: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    """
    比较两个批次的张量数据是否完全一致
    
    Args:
        data1: 第一组张量数据
        data2: 第二组张量数据
    
    Returns:
        各张量键值对的比较结果字典
    """
    results = {}
    
    # 检查键是否一致
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    # 检查共同键的数据是否一致
    common_keys = keys1.intersection(keys2)
    for key in common_keys:
        tensor1 = data1[key]
        tensor2 = data2[key]
        
        # 检查形状是否一致
        same_shape = tensor1.shape == tensor2.shape
        
        # 检查数值是否一致
        same_values = torch.allclose(tensor1, tensor2) if same_shape else False
        
        results[key] = same_values
    
    # 检查是否有缺失的键
    missing_keys1 = keys2 - keys1
    missing_keys2 = keys1 - keys2
    
    if missing_keys1:
        results['missing_keys_in_run1'] = missing_keys1
    
    if missing_keys2:
        results['missing_keys_in_run2'] = missing_keys2
    
    return results


def run_data_pipeline(root_dir: str, manifest_file: str, n_radiomics: int, 
                     batch_size: int, num_workers: int, random_seed: int, 
                     num_batches: int, hook_info: Dict) -> Tuple[List[List[int]], List[Dict[str, torch.Tensor]]]:
    """
    运行数据加载管线并收集结果
    
    Args:
        root_dir: 训练集根目录
        manifest_file: 训练集清单文件
        n_radiomics: 影像组学特征数量
        batch_size: 批量大小
        num_workers: 数据加载工作线程数
        random_seed: 随机种子
        num_batches: 要测试的批次数量
        hook_info: WeightedRandomSampler的hook信息
    
    Returns:
        索引列表和张量数据列表的元组
    """
    # 重置hook中的索引列表
    reset_weighted_sampler_hook(hook_info)
    
    # 创建数据模块实例
    dm = EsophagusDataModule(
        root_dir_train=root_dir,
        manifest_file_train=manifest_file,
        root_dir_val=root_dir,
        manifest_file_val=manifest_file,
        n_radiomics=n_radiomics,
        batch_size_train=batch_size,
        num_workers=num_workers,
        random_seed=random_seed
    )
    
    # 准备数据
    dm.prepare_data()
    
    # 设置训练阶段
    dm.setup(stage='fit')
    
    # 获取训练数据加载器
    train_loader = dm.train_dataloader()
    
    # 收集批次数据
    batch_indices = []
    batch_data = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # 提取批次数据
        indices, tensor_data = extract_batch_data(batch)
        batch_indices.append(indices)
        batch_data.append(tensor_data)
    
    return batch_indices, batch_data


def test_reproducibility(root_dir: str, manifest_file: str, n_radiomics: int, 
                         batch_size: int, num_workers: int, random_seed: int, 
                         num_batches: int = 5):
    """
    测试数据模块的训练模式可重复性
    
    Args:
        root_dir: 训练集根目录
        manifest_file: 训练集清单文件
        n_radiomics: 影像组学特征数量
        batch_size: 批量大小
        num_workers: 数据加载工作线程数
        random_seed: 随机种子
        num_batches: 要测试的批次数量
    """
    print(f"\n测试EsophagusDataModule训练模式可重复性")
    print(f"------------------------------------------")
    print(f"随机种子: {random_seed}")
    print(f"批量大小: {batch_size}")
    print(f"数据加载线程数: {num_workers}")
    print(f"测试批次数量: {num_batches}")
    print(f"")
    
    try:
        # 猴子补丁WeightedRandomSampler以捕获索引
        hook_info = monkey_patch_weighted_sampler()
        
        # 第一次运行数据加载管线
        print(f"运行1 - 开始加载数据...")
        run1_indices, run1_data = run_data_pipeline(
            root_dir=root_dir,
            manifest_file=manifest_file,
            n_radiomics=n_radiomics,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
            num_batches=num_batches,
            hook_info=hook_info
        )
        print(f"运行1 - 数据加载完成，共加载 {len(run1_indices)} 个批次")
        
        # 第二次运行数据加载管线
        print(f"\n运行2 - 开始加载数据...")
        run2_indices, run2_data = run_data_pipeline(
            root_dir=root_dir,
            manifest_file=manifest_file,
            n_radiomics=n_radiomics,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
            num_batches=num_batches,
            hook_info=hook_info
        )
        print(f"运行2 - 数据加载完成，共加载 {len(run2_indices)} 个批次")
        
        # 验证两次运行加载的批次数量是否一致
        if len(run1_indices) != len(run2_indices):
            print(f"\n警告: 两次运行加载的批次数量不一致")
            print(f"运行1: {len(run1_indices)} 批次")
            print(f"运行2: {len(run2_indices)} 批次")
            return False
        
        # 比较两次运行的结果
        print(f"\n开始比较两次运行的结果...")
        all_consistent = True
        batch_comparison_results = []
        
        for i in range(len(run1_indices)):
            print(f"\n批次 {i+1}/{len(run1_indices)} 比较:")
            
            # 比较索引是否一致
            indices_consistent = run1_indices[i] == run2_indices[i]
            print(f"  索引列表一致: {indices_consistent}")
            print(f"  运行1索引: {run1_indices[i]}")
            print(f"  运行2索引: {run2_indices[i]}")
            
            # 比较张量数据是否一致
            tensor_comparison = compare_tensor_data(run1_data[i], run2_data[i])
            tensors_consistent = all(result for key, result in tensor_comparison.items() 
                                    if isinstance(result, bool))
            
            # 打印张量比较结果
            for key, consistent in tensor_comparison.items():
                if isinstance(consistent, bool):
                    print(f"  {key} 数据一致: {consistent}")
                else:
                    print(f"  {key}: {consistent}")
            
            # 记录此批次的一致性结果
            batch_consistent = indices_consistent and tensors_consistent
            batch_comparison_results.append(batch_consistent)
            print(f"  批次 {i+1} 整体一致: {batch_consistent}")
            
            if not batch_consistent:
                all_consistent = False
        
        # 恢复原始方法
        torch.utils.data.WeightedRandomSampler.__iter__ = hook_info['original_iter']
        
        # 打印总结
        print(f"\n=========================================")
        print(f"一致性测试总结:")
        print(f"总批次数量: {len(run1_indices)}")
        print(f"一致的批次数量: {sum(batch_comparison_results)}")
        print(f"不一致的批次数量: {len(batch_comparison_results) - sum(batch_comparison_results)}")
        print(f"整体一致性: {'通过' if all_consistent else '失败'}")
        
        if all_consistent:
            print(f"\n[PASS] 测试成功！在相同随机种子下，两次运行的结果完全一致。")
        else:
            print(f"\n[FAIL] 测试失败！两次运行的结果存在不一致。")
        
        return all_consistent
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 确保恢复原始方法
        if 'hook_info' in locals() and 'original_iter' in hook_info:
            torch.utils.data.WeightedRandomSampler.__iter__ = hook_info['original_iter']
        
        return False


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='EsophagusDataModule训练模式可重复性测试脚本')
    parser.add_argument('-r', '--root_dir', required=True, help='训练集根目录')
    parser.add_argument('-m', '--manifest_file', required=True, help='训练集清单文件路径')
    parser.add_argument('-n', '--n_radiomics', type=int, default=21, help='影像组学特征数量')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='批量大小')
    parser.add_argument('-w', '--num_workers', type=int, default=4, help='数据加载工作线程数')
    parser.add_argument('-s', '--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('-k', '--num_batches', type=int, default=2, help='要测试的批次数量')
    args = parser.parse_args()
    
    # 确保根目录存在
    if not os.path.exists(args.root_dir):
        print(f"错误：训练集根目录不存在: {args.root_dir}")
        return 1
    
    # 确保清单文件存在
    if not os.path.exists(args.manifest_file):
        print(f"错误：训练集清单文件不存在: {args.manifest_file}")
        return 1
    
    # 执行测试
    success = test_reproducibility(
        root_dir=args.root_dir,
        manifest_file=args.manifest_file,
        n_radiomics=args.n_radiomics,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        num_batches=args.num_batches
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())