#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化器配置模块

功能概述：
    提供两个优化器创建函数，核心目的是隐藏大部分不常用参数以提供快速配置接口
    1. SGD：创建配置好的SGD优化器，内部使用预设的常用参数
    2. AdamW：创建配置好的AdamW优化器，内部使用预设的常用参数
    
核心功能：
    1. 快速配置：通过隐藏大部分不常用参数，简化优化器创建过程
    2. 参数封装：将经过验证的常用参数组合封装，减少重复代码

参数说明：
    - SGD参数：
      * params: 需要优化的模型参数
      * lr: 学习率
    - AdamW参数：
      * params: 需要优化的模型参数
      * lr: 学习率
      * amsgrad: 是否使用AMSGrad变体，默认为False
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Iterable, Dict, Any, Callable


def SGD(params: Iterable[torch.nn.parameter.Parameter],
        lr: float) -> torch.optim.SGD:
    """
    创建配置好的SGD优化器实例
    
    Args:
        params: 需要优化的模型参数
        lr: 学习率
        
    Returns:
        torch.optim.SGD: 配置好的SGD优化器实例
    """
    # 创建并返回SGD优化器
    return optim.SGD(
        params=params,
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )


def AdamW(params: Iterable[torch.nn.parameter.Parameter],
          lr: float,
          amsgrad: bool = False) -> torch.optim.AdamW:
    """
    创建配置好的AdamW优化器实例
    
    Args:
        params: 需要优化的模型参数
        lr: 学习率
        amsgrad: 是否使用AMSGrad变体，默认为False
        
    Returns:
        torch.optim.AdamW: 配置好的AdamW优化器实例
        
    Raises:
        ValueError: 当参数值无效时抛出
        TypeError: 当参数类型不正确时抛出
    """
    # 创建并返回AdamW优化器
    return optim.AdamW(
        params=params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=amsgrad
    )


def test_optimizer_consistency() -> None:
    """
    测试优化器的一致性：相同配置下两次优化过程的结果应该相同
    构造和初始化只有一个线性层的神经网络以及优化器，优化3步，记录结果；
    然后构造和初始化新的神经网络以及优化器，优化3步，比较两次优化的结果是否相同。
    """
    print("===== 优化器一致性测试 =====")

    # 设置随机种子以确保可复现性
    torch.manual_seed(42)

    # 定义模型和优化器参数
    input_dim: int = 10
    output_dim: int = 1
    batch_size: int = 8

    # 创建测试数据
    def create_test_data() -> Tuple[torch.Tensor, torch.Tensor]:
        """创建相同的测试数据"""
        torch.manual_seed(42)  # 确保每次创建相同的数据
        x: torch.Tensor = torch.randn(batch_size, input_dim)
        y: torch.Tensor = torch.randn(batch_size, output_dim)
        return x, y

    # 测试函数：训练模型3步并返回最终权重
    def train_model_steps(optimizer_fn: Callable, 
                         *optimizer_args: Any, 
                         **optimizer_kwargs: Any) -> Dict[str, torch.Tensor]:
        """训练模型3步并返回最终权重"""
        # 创建新模型
        model: nn.Linear = nn.Linear(input_dim, output_dim)
        # 初始化权重（使用固定种子确保两次初始化相同）
        torch.manual_seed(42)
        for param in model.parameters():
            nn.init.normal_(param)

        # 创建优化器
        optimizer: optim.Optimizer = optimizer_fn(model.parameters(), *optimizer_args, **optimizer_kwargs)

        # 训练3步
        for step in range(3):
            x: torch.Tensor
            y: torch.Tensor
            x, y = create_test_data()

            # 前向传播
            outputs: torch.Tensor = model(x)
            loss: torch.Tensor = nn.MSELoss()(outputs, y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"  第 {step + 1} 步损失: {loss.item():.6f}")

        # 返回最终权重
        return {
            'weight': model.weight.clone().detach(),
            'bias': model.bias.clone().detach()
        }

    # 测试SGD优化器
    print("\n=== SGD优化器一致性测试 ===")
    print("第一次训练:")
    weights1: Dict[str, torch.Tensor] = train_model_steps(
        SGD,
        lr=0.01
    )

    print("\n第二次训练:")
    weights2: Dict[str, torch.Tensor] = train_model_steps(
        SGD,
        lr=0.01
    )

    # 比较两次训练的权重是否相同
    weight_diff: torch.Tensor = torch.norm(weights1['weight'] - weights2['weight'])
    bias_diff: torch.Tensor = torch.norm(weights1['bias'] - weights2['bias'])

    print(f"\nSGD优化器权重差异: {weight_diff.item():.10f}")
    print(f"SGD优化器偏置差异: {bias_diff.item():.10f}")
    print(f"SGD优化器一致性测试: {'通过' if weight_diff < 1e-6 and bias_diff < 1e-6 else '失败'}")

    # 测试AdamW优化器
    print("\n=== AdamW优化器一致性测试 ===")
    print("第一次训练:")
    weights3: Dict[str, torch.Tensor] = train_model_steps(
        AdamW,
        lr=0.001
    )

    print("\n第二次训练:")
    weights4: Dict[str, torch.Tensor] = train_model_steps(
        AdamW,
        lr=0.001
    )

    # 比较两次训练的权重是否相同
    weight_diff: torch.Tensor = torch.norm(weights3['weight'] - weights4['weight'])
    bias_diff: torch.Tensor = torch.norm(weights3['bias'] - weights4['bias'])

    print(f"\nAdamW优化器权重差异: {weight_diff.item():.10f}")
    print(f"AdamW优化器偏置差异: {bias_diff.item():.10f}")
    print(f"AdamW优化器一致性测试: {'通过' if weight_diff < 1e-6 and bias_diff < 1e-6 else '失败'}")

    print("\n===== 优化器一致性测试完成 =====")


if __name__ == "__main__":
    # 运行优化器一致性测试
    test_optimizer_consistency()
