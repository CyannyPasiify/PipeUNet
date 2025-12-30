#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习率调度器配置器

功能概述：
    提供五个学习率调度器创建函数：
    1. LinearLR：创建配置好的线性学习率调度器，支持参数验证
    2. CosineAnnealingLR：创建配置好的余弦退火学习率调度器，支持参数验证
    3. CosineAnnealingWarmRestarts：创建配置好的余弦退火重启动学习率调度器，支持参数验证
    4. OneCycleLR：创建配置好的OneCycle学习率调度器，支持参数验证
    5. ReduceLROnPlateau：创建配置好的性能衰减学习率调度器，支持参数验证
    
核心功能：
    1. 快速配置：通过隐藏大部分不常用参数，简化优化器创建过程
    2. 参数封装：将经过验证的常用参数组合封装，减少重复代码
    3. 命令行测试：支持通过命令行参数测试不同调度器的学习率变化曲线

参数说明：

使用示例：
    # 创建线性调度器
    linear_scheduler = LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=100
    )
    
    # 创建余弦退火调度器
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=100,
        eta_min=0.0001
    )
    
    # 命令行测试
    # python schedule_configurer.py --scheduler LinearLR --step 100 --lr 0.1
    # python schedule_configurer.py --scheduler CosineAnnealingLR --step 100 --lr 0.1
    # python schedule_configurer.py --scheduler CosineAnnealingWarmRestarts --step 100 --lr 0.1
    # python schedule_configurer.py --scheduler OneCycleLR --step 100 --lr 0.1
    # python schedule_configurer.py --scheduler ReduceLROnPlateau --step 100 --lr 0.1
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Union, List, Tuple, Literal


def LinearLR(optimizer: torch.optim.Optimizer,
             start_factor: float,
             end_factor: float,
             total_iters: int,
             last_epoch: int = -1) -> torch.optim.lr_scheduler.LinearLR:
    """
    创建配置好的线性学习率调度器实例
    
    Args:
        optimizer: 需要调整学习率的优化器
        start_factor: 起始因子
        end_factor: 结束因子
        total_iters: 总迭代次数
        last_epoch: 上一个训练轮次，默认为-1
        
    Returns:
        torch.optim.lr_scheduler.LinearLR: 配置好的线性学习率调度器实例
    """
    # 创建并返回LinearLR调度器
    return lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters,
        last_epoch=last_epoch
    )

def CosineAnnealingLR(optimizer: torch.optim.Optimizer,
                      T_max: int,
                      eta_min: float = 0.0,
                      last_epoch: int = -1) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """
    创建配置好的余弦退火学习率调度器实例
    
    Args:
        optimizer: 需要调整学习率的优化器
        T_max: 最大迭代次数
        eta_min: 最小学习率，默认为0
        last_epoch: 上一个训练轮次，默认为-1
        
    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR: 配置好的余弦退火学习率调度器实例
    """
    # 创建并返回CosineAnnealingLR调度器
    return lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=T_max,
        eta_min=eta_min,
        last_epoch=last_epoch
    )

def CosineAnnealingWarmRestarts(optimizer: torch.optim.Optimizer,
                                T_0: int,
                                T_mult: int = 1,
                                eta_min: float = 0.0,
                                last_epoch: int = -1) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """
    创建配置好的余弦退火重启动学习率调度器实例
    
    Args:
        optimizer: 需要调整学习率的优化器
        T_0: 初始周期
        T_mult: 周期乘数，默认为1
        eta_min: 最小学习率，默认为0
        last_epoch: 上一个训练轮次，默认为-1
        
    Returns:
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts: 配置好的余弦退火重启动学习率调度器实例
    """
    # 创建并返回CosineAnnealingWarmRestarts调度器
    return lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        last_epoch=last_epoch
    )

def OneCycleLR(optimizer: torch.optim.Optimizer,
               max_lr: Union[float, List[float]],
               total_steps: int,
               epochs: int,
               steps_per_epoch: int,
               pct_start: float = 0.3,
               div_factor: float = 25.0,
               final_div_factor: float = 10000.0,
               last_epoch: int = -1) -> torch.optim.lr_scheduler.OneCycleLR:
    """
    创建配置好的OneCycle学习率调度器实例
    
    Args:
        optimizer: 需要调整学习率的优化器
        max_lr: 最大学习率
        total_steps: 周期中的总步数。注意，如果此处未提供值，则必须通过提供 epochs 和 steps_per_epoch 的值来推断
        epochs: 训练的 epoch 数。此参数与 steps_per_epoch 一起用于推断周期中的总步数，前提是未提供 total_steps 的值
        steps_per_epoch: 每个 epoch 的训练步数。此参数与 epochs 一起用于推断周期中的总步数，前提是未提供 total_steps 的值
        pct_start: 预热比例，默认为0.3
        div_factor: 最大学习率与初始学习率的比值，默认为25.0
        final_div_factor: 初始学习率与最小学习率的比值，默认为10000.0
        last_epoch: 上一个训练轮次，默认为-1
        
    Returns:
        torch.optim.lr_scheduler.OneCycleLR: 配置好的OneCycle学习率调度器实例
    """
    # 创建并返回OneCycleLR调度器
    return lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        last_epoch=last_epoch
    )


def ReduceLROnPlateau(optimizer: torch.optim.Optimizer,
                      mode: Literal["min", "max"] = 'min',
                      factor: float = 0.1,
                      patience: int = 10,
                      threshold: float = 1e-4,
                      threshold_mode: Literal["rel", "abs"] = 'rel',
                      cooldown: int = 0,
                      min_lr: float = 0) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    创建配置好的性能衰减学习率调度器实例

    Args:
        optimizer: 需要调整学习率的优化器
        mode: 模式选择，'min'表示当指标停止下降时降低学习率，'max'表示当指标停止上升时降低学习率，默认为'min'
        factor: 学习率降低的因子，新学习率 = 旧学习率 * factor，默认为0.1
        patience: 在降低学习率之前允许验证指标不改善的轮数，默认为10
        threshold: 衡量新最佳值的阈值，默认为1e-4
        threshold_mode: 阈值模式，'rel'表示相对，'abs'表示绝对，默认为'rel'
        cooldown: 降低学习率后等待的轮数，在此期间不进行新的测量，默认为0
        min_lr: 学习率的下限，默认为0

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: 配置好的性能衰减学习率调度器实例
    """
    # 创建并返回ReduceLROnPlateau调度器
    return lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr
    )

def plot_learning_rate_curve(scheduler_type: str, total_steps: int, initial_lr: float) -> None:
    """
    绘制学习率曲线
    
    Args:
        scheduler_type: 调度器类型
        total_steps: 总迭代步数
        initial_lr: 初始学习率
    """
    import matplotlib.pyplot as plt
    import torch.optim as optim

    # 创建一个简单的模型和优化器
    model = torch.nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    
    # 创建对应的调度器
    if scheduler_type == 'LinearLR':
        scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=total_steps
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=0
        )
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=total_steps // 4,
            T_mult=2,
            eta_min=0
        )
    elif scheduler_type == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=initial_lr * 10,
            total_steps=total_steps,
            epochs=1,
            steps_per_epoch=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=3,
            min_lr=0
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    # 记录学习率变化
    lrs = []
    for i in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        # 对于ReduceLROnPlateau，我们模拟指标下降和停滞的情况
        if scheduler_type == 'ReduceLROnPlateau':
            # 前200步模拟指标下降
            if i < 200:
                scheduler.step(1.0 - i * 0.004)  # 指标从1.0降到0.2
            # 接下来400步模拟指标不变
            elif i < 600:
                scheduler.step(0.2)  # 指标保持不变
            # 之后模拟指标略微上升后又下降
            else:
                # 模拟偶尔的上升和下降来触发学习率调整
                if i % 30 == 0:
                    scheduler.step(0.3)  # 小幅上升
                else:
                    scheduler.step(0.2 - (i % 20) * 0.01)  # 缓慢下降
        else:
            scheduler.step()
    
    # 绘制学习率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.title(f'{scheduler_type} LR Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # 显示图片
    plt.show()

def main():
    """
    主函数，处理命令行参数并绘制学习率曲线
    """
    import argparse

    parser = argparse.ArgumentParser(description='学习率调度器测试工具')
    parser.add_argument('--scheduler', type=str, required=True, 
                       choices=['LinearLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'ReduceLROnPlateau'],
                       help='调度器类型')
    parser.add_argument('--step', type=int, required=True, help='迭代步数')
    parser.add_argument('--lr', type=float, required=True, help='初始学习率')
    
    args = parser.parse_args()
    
    try:
        plot_learning_rate_curve(args.scheduler, args.step, args.lr)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()

