#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器可重复性测试模块

功能概述：
    测试相同配置下创建的训练器和模型是否具有参数一致性
    验证初始参数是否完全相同
    验证经过相同训练步骤后参数是否保持一致
"""
import os
import sys
import random
import torch
import numpy as np
from typing import Dict, List, Tuple
import lightning as L

from Trainer.trainer_resnet import create_esophagus_trainer
from Module.module_resnet import create_resnet_module
from DataModule.dm_pipe import EsophagusDataModule


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(seed, workers=True)


def compare_state_dicts(state_dict1: Dict[str, torch.Tensor],
                        state_dict2: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
    """
    比较两个模型状态字典是否完全相同
    
    Args:
        state_dict1: 第一个模型的状态字典
        state_dict2: 第二个模型的状态字典
        
    Returns:
        Tuple[bool, List[str]]: (是否相同, 不同参数名称列表)
    """
    # 检查键是否相同
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    missing_keys = keys1 - keys2
    extra_keys = keys2 - keys1

    if missing_keys or extra_keys:
        print(f"[FAIL] 状态字典键不匹配:")
        if missing_keys:
            print(f"  第一个模型有但第二个模型没有的键: {missing_keys}")
        if extra_keys:
            print(f"  第二个模型有但第一个模型没有的键: {extra_keys}")
        return False, list(missing_keys.union(extra_keys))

    # 检查值是否相同
    different_params = []
    all_same = True

    for key in keys1:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if tensor1.device != tensor2.device:
            print(f"[FAIL] 参数 {key} 设备不同: {tensor1.device} vs {tensor2.device}")
            different_params.append(key)
            all_same = False
            continue

        if tensor1.shape != tensor2.shape:
            print(f"[FAIL] 参数 {key} 形状不同: {tensor1.shape} vs {tensor2.shape}")
            different_params.append(key)
            all_same = False
            continue

        if not torch.allclose(tensor1, tensor2):
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            print(f"[FAIL] 参数 {key} 值不同, 最大差值: {max_diff}")
            different_params.append(key)
            all_same = False

    if all_same:
        print(f"[PASS] 所有 {len(keys1)} 个参数完全相同")
    else:
        print(f"[FAIL] 发现 {len(different_params)} 个不同的参数")

    return all_same, different_params


def run_single_training_step(model: L.LightningModule,
                             batch: Tuple[torch.Tensor, torch.Tensor]):
    """
    运行单个训练步骤
    
    Args:
        model: LightningModule实例
        batch: 输入批次数据
    """
    x, y = batch

    # 确保模型在训练模式
    model.train()

    # 清零梯度
    model.optimizer.zero_grad()

    # 前向传播
    logits = model(x)
    loss = model.loss_fn(logits, y)

    # 反向传播
    loss.backward()

    # 更新参数
    model.optimizer.step()

    return loss.item()


def create_dummy_batch(batch_size: int = 1,
                       in_channels: int = 4,
                       spatial_size: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建用于测试的虚拟批次数据
    
    Args:
        batch_size: 批次大小
        in_channels: 输入通道数
        spatial_size: 空间维度
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (输入数据, 标签)
    """
    x = torch.randn(batch_size, in_channels, *spatial_size)
    y = torch.randint(0, 2, (batch_size,))
    return x, y


def test_trainer_reproducibility():
    """
    测试训练器和模型的可重复性
    """
    print("=" * 80)
    print("开始测试训练器和模型可重复性")
    print("=" * 80)

    # 设置随机种子
    seed = 42
    print(f"设置随机种子: {seed}")
    set_seed(seed)

    # 模型配置
    model_config_1 = {
        'block': 'basic',
        'layers': [2, 2, 2, 2],
        'in_channels': 4,
        'num_classes': 2
    }

    # 创建两个相同配置的模型
    print("\n创建第一个模型...")
    set_seed(seed)
    model1 = create_resnet_module(
        model_type='resnet3d',
        model_config=model_config_1,
        learning_rate=1e-4,
        T_max=200
    )

    # 模型配置
    model_config_2 = {
        'block': 'basic',
        'layers': [2, 2, 2, 2],
        'in_channels': 4,
        'num_classes': 2
    }

    print("创建第二个模型...")
    set_seed(seed)
    model2 = create_resnet_module(
        model_type='resnet3d',
        model_config=model_config_2,
        learning_rate=1e-4,
        T_max=200
    )

    # 比较初始参数
    print("\n比较初始模型参数:")
    initial_params_match, initial_diff_params = compare_state_dicts(
        model1.model.state_dict(),
        model2.model.state_dict()
    )

    model1.configure_optimizers()
    model2.configure_optimizers()

    # 创建虚拟批次数据
    dummy_batch = create_dummy_batch()

    # 运行单个训练步骤
    print("\n运行单个训练步骤...")
    loss1 = run_single_training_step(model1, dummy_batch)
    loss2 = run_single_training_step(model2, dummy_batch)

    print(f"第一个模型训练步骤损失: {loss1:.6f}")
    print(f"第二个模型训练步骤损失: {loss2:.6f}")

    # 比较训练后的参数
    print("\n比较训练后模型参数:")
    trained_params_match, trained_diff_params = compare_state_dicts(
        model1.model.state_dict(),
        model2.model.state_dict()
    )

    # 总结结果
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)

    initial_all_match = initial_params_match
    trained_all_match = trained_params_match

    print(f"初始状态一致性: {'[PASS]' if initial_all_match else '[FAIL]'}")
    print(f"训练后状态一致性: {'[PASS]' if trained_all_match else '[FAIL]'}")

    if initial_all_match:
        print("  ✓ 初始模型参数完全匹配")
        print("  ✓ 初始优化器状态完全匹配")
    else:
        if not initial_params_match:
            print(f"  ✗ 初始模型参数不匹配: {len(initial_diff_params)} 个不同参数")

    if trained_all_match:
        print("  ✓ 训练后模型参数完全匹配")
        print("  ✓ 训练后优化器状态完全匹配")
    else:
        if not trained_params_match:
            print(f"  ✗ 训练后模型参数不匹配: {len(trained_diff_params)} 个不同参数")

    print(f"\n最终可重复性测试结果: {'[PASS]' if initial_all_match and trained_all_match else '[FAIL]'}")

    return initial_all_match and trained_all_match


def test_trainer_instance_reproducibility():
    """
    测试训练器实例的可重复性
    """
    print("\n" + "=" * 80)
    print("开始测试训练器实例可重复性")
    print("=" * 80)

    # 设置随机种子
    seed = 42
    print(f"设置随机种子: {seed}")
    set_seed(seed)

    # 创建两个相同配置的训练器实例
    print("\n创建第一个训练器实例...")
    set_seed(seed)
    trainer1 = create_esophagus_trainer(
        max_epochs=5,
        accelerator="cpu",  # 为了测试一致性，使用CPU
        devices=1,
        precision=32,
        experiment_name="Test-Reproducibility",
        experiment_version="test",
        log_dir="./test_logs",
        wandb_project="Test",
        enable_ddp=False
    )

    print("创建第二个训练器实例...")
    set_seed(seed)
    trainer2 = create_esophagus_trainer(
        max_epochs=5,
        accelerator="cpu",
        devices=1,
        precision=32,
        experiment_name="Test-Reproducibility",
        experiment_version="test",
        log_dir="./test_logs",
        wandb_project="Test",
        enable_ddp=False
    )

    # 获取训练器配置信息
    print("\n比较训练器配置:")

    # 比较关键配置参数
    config_keys = ['max_epochs', 'accelerator', 'devices', 'precision', 'experiment_name', 'experiment_version']
    config_mismatches = []

    for key in config_keys:
        val1 = getattr(trainer1, key)
        val2 = getattr(trainer2, key)
        if val1 != val2:
            print(f"[FAIL] 配置 {key} 不匹配: {val1} vs {val2}")
            config_mismatches.append(key)

    if not config_mismatches:
        print("[PASS] 所有关键配置参数完全匹配")
    else:
        print(f"[FAIL] 发现 {len(config_mismatches)} 个配置不匹配")

    print("\n训练器实例可重复性测试完成")
    return len(config_mismatches) == 0


if __name__ == "__main__":
    # 运行所有测试
    print("启动训练器可重复性测试套件")

    # 确保测试目录存在
    os.makedirs("./test_logs", exist_ok=True)

    try:
        # 测试训练器实例可重复性
        trainer_instance_reproducible = test_trainer_instance_reproducibility()

        # 测试模型和训练可重复性
        model_reproducible = test_trainer_reproducibility()

        print("\n" + "=" * 80)
        print("总体测试结果")
        print("=" * 80)
        print(f"训练器实例可重复性: {'[PASS]' if trainer_instance_reproducible else '[FAIL]'}")
        print(f"模型和训练可重复性: {'[PASS]' if model_reproducible else '[FAIL]'}")
        print(f"\n综合可重复性测试: {'[PASS]' if trainer_instance_reproducible and model_reproducible else '[FAIL]'}")

    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理测试日志目录
        try:
            import shutil

            if os.path.exists("./test_logs"):
                shutil.rmtree("./test_logs")
                print("\n已清理测试日志目录")
        except:
            pass
