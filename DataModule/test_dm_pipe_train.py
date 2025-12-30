#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EsophagusDataModule训练模式测试脚本

功能概述：
    测试EsophagusDataModule的训练模式功能
    验证加权随机采样器的工作情况
    打印每个batch中选取的样本索引列表
    检查数据加载器的正常运行

使用方法：
    python test_dm_pipe_train.py -r data_root_dir -m manifest_file.xlsx -n 21 -b 2 -w 4 -s 42
"""

import os
import argparse
import torch
from typing import Dict, Any, List

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


def test_dm_train_mode(root_dir: str, manifest_file: str, n_radiomics: int, 
                      batch_size: int, num_workers: int, random_seed: int, 
                      num_batches: int = 5):
    """
    测试数据模块的训练模式
    
    Args:
        root_dir: 训练集根目录
        manifest_file: 训练集清单文件
        n_radiomics: 影像组学特征数量
        batch_size: 批量大小
        num_workers: 数据加载工作线程数
        random_seed: 随机种子
        num_batches: 要测试的批次数量
    """
    print(f"\n测试EsophagusDataModule训练模式")
    print(f"------------------------------------")
    
    try:
        # 猴子补丁WeightedRandomSampler以捕获索引
        hook_info = monkey_patch_weighted_sampler()
        
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
        
        # 打印数据加载器信息
        print(f"训练集大小: {len(dm.datasets['train'])}")
        print(f"批量大小: {batch_size}")
        print(f"数据加载线程数: {num_workers}")
        print(f"随机种子: {random_seed}")
        print(f"样本权重形状: {dm.train_weights.shape if dm.train_weights is not None else None}")
        print(f"训练集样本权重: {dm.train_weights}")
        
        # 限制测试批次数量
        test_batches = min(num_batches, len(train_loader))
        print(f"训练数据加载器长度: {len(train_loader)}")
        print(f"测试批次数量: {test_batches}")
        
        # 测试数据加载
        print(f"\n开始加载批次数据...")
        
        for batch_idx, batch in enumerate(train_loader):
            # 超过测试次数后跳出
            if batch_idx == test_batches: break

            print(f"\n批次 {batch_idx + 1}/{test_batches}")
            
            # 记录当前索引列表长度
            current_len = batch_size * batch_idx
            
            # 获取当前批次的索引
            batch_indices = hook_info['indices'][current_len:]
            
            # 打印批次信息
            print(f"  批次中选取的索引列表: {batch_indices}")
            print(f"  批次索引数量: {len(batch_indices)}")
            
            # 验证批次中的数据结构
            print(f"  idx: {batch['idx']}")
            required_keys = ['pre_img', 'pre_mask', 'post_img', 'post_mask', 'radiomics', 'label']
            for key in required_keys:
                assert key in batch, f"批次缺少键: {key}"
                print(f"  {key} 形状: {batch[key].shape}")
            
            print(f"  批次处理完成")
        
        # 恢复原始方法
        torch.utils.data.WeightedRandomSampler.__iter__ = hook_info['original_iter']
        
        print(f"\n=========================================")
        print(f"测试成功完成！所有批次已正常加载并打印索引信息。")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='EsophagusDataModule训练模式测试脚本')
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
    success = test_dm_train_mode(
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