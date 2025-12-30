#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集清单序列类测试脚本

功能概述：
    测试DatasetManifestWithLabelSequential和DatasetManifestNoLabelSequential类的功能
    加载数据集并检查数据结构和内容
    验证清单文件解析和数据验证功能

使用方法：
    python test_ds_pipe.py -r data_root_dir -m manifest_file.xlsx -n 21 -w -s 10
"""

import os
import argparse
from ds_pipe import DatasetManifestWithLabelSequential, DatasetManifestNoLabelSequential, example_callback


def test_dataset(dataset_class, root_dir, n_radiomics, manifest_file, 
                 num_samples=5, n_len=None, callbacks=None):
    """
    测试数据集类
    
    Args:
        dataset_class: 要测试的数据集类
        root_dir: 数据集根目录
        n_radiomics: 影像组学特征数量
        manifest_file: 清单文件路径
        num_samples: 要测试的样本数量
        n_len: 数据集长度限制
        callbacks: 回调函数列表
    """
    print(f"\n测试数据集类: {dataset_class.__name__}")
    print(f"------------------------------------")
    
    try:
        # 创建数据集实例（不使用预处理和数据增强）
        dataset = dataset_class(
            root_dir=root_dir,
            n_radiomics=n_radiomics,
            manifest_file=manifest_file,
            preprocess_augment=None,  # 不使用预处理和数据增强
            callbacks=callbacks,
            n_len=n_len
        )
        
        # 打印数据集信息
        print(f"数据集大小: {len(dataset)}")
        print(f"影像组学特征数量: {n_radiomics}")
        
        # 限制测试样本数量
        test_samples = min(num_samples, len(dataset))
        print(f"测试样本数量: {test_samples}")
        
        # 测试样本加载
        for i in range(test_samples):
            print(f"\n加载样本 {i+1}/{test_samples}")
            sample = dataset[i]
            
            # 验证返回数据结构
            required_keys = ['pre_img', 'pre_mask', 'post_img', 'post_mask', 'radiomics']
            if dataset_class.__name__ == 'DatasetManifestWithLabelSequential':
                required_keys.append('label')
            
            for key in required_keys:
                assert key in sample, f"样本缺少键: {key}"
            
            # 打印样本信息
            pid = dataset.manifest.iloc[i]['pid']
            print(f"样本PID: {pid}")
            print(f"返回的键: {list(sample.keys())}")
            
            # 打印数据形状
            for key in ['pre_img', 'pre_mask', 'post_img', 'post_mask']:
                if sample[key] is not None:
                    print(f"{key} 形状: {sample[key].shape}")
            
            # 打印影像组学特征信息
            print(f"影像组学特征形状: {sample['radiomics'].shape}")
            
            # 打印标签信息（如果有）
            if 'label' in sample:
                print(f"标签: {sample['label']}")
        
        print(f"\n测试成功完成！")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集清单序列类测试脚本')
    parser.add_argument('-r', '--root_dir', required=True, help='数据集根目录')
    parser.add_argument('-m', '--manifest_file', required=True, help='清单文件路径')
    parser.add_argument('-n', '--n_radiomics', type=int, default=21, help='影像组学特征数量')
    parser.add_argument('-w', '--with_label', action='store_true', default=False, help='数据集是否带标签')
    parser.add_argument('-s', '--samples', type=int, default=3, help='要测试的样本数量')
    parser.add_argument('-l', '--n_len', type=int, default=None, help='数据集长度限制')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出模式')
    args = parser.parse_args()
    
    # 确保根目录存在
    if not os.path.exists(args.root_dir):
        print(f"错误：数据集根目录不存在: {args.root_dir}")
        return 1
    
    # 确保清单文件存在
    if not os.path.exists(args.manifest_file):
        print(f"错误：清单文件不存在: {args.manifest_file}")
        return 1
    
    # 准备回调函数
    callbacks = [example_callback] if args.verbose else None
    
    # 选择数据集类
    dataset_class = DatasetManifestWithLabelSequential if args.with_label else DatasetManifestNoLabelSequential
    
    # 执行测试
    success = test_dataset(
        dataset_class=dataset_class,
        root_dir=args.root_dir,
        n_radiomics=args.n_radiomics,
        manifest_file=args.manifest_file,
        num_samples=args.samples,
        n_len=args.n_len,
        callbacks=callbacks
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())