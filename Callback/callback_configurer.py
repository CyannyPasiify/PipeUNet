#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回调函数配置模块

功能概述：
    提供PyTorch Lightning回调函数的包装接口，简化常用回调的配置过程
    每个包装函数隐藏了大部分不常用参数，提供快速配置接口

核心功能：
    1. 快速配置：通过隐藏大部分不常用参数，简化回调函数创建过程
    2. 参数封装：将经过验证的常用参数组合封装，减少重复代码
    3. 统一接口：提供一致的API风格，便于使用和维护
"""

import lightning.pytorch.callbacks as callbacks
from typing import Optional, Union, List, Literal, Dict, Any
from pathlib import Path


def DeviceStatsMonitor(cpu_stats: Optional[bool] = None) -> callbacks.DeviceStatsMonitor:
    """
    创建设备状态监控回调实例
    
    Args:
        cpu_stats: 是否监控CPU状态。如果为None，则仅在使用CPU加速器时监控；
                  如果为True，无论使用什么加速器都会监控CPU状态；
                  如果为False，不监控CPU状态
    
    Returns:
        callbacks.DeviceStatsMonitor: 配置好的设备状态监控回调实例
    """
    return callbacks.DeviceStatsMonitor(cpu_stats=cpu_stats)


def EarlyStopping(monitor: str,
                  min_delta: float = 0.0,
                  patience: int = 3,
                  mode: Literal['min', 'max'] = 'min',
                  verbose: bool = False) -> callbacks.EarlyStopping:
    """
    创建早停回调实例
    
    Args:
        monitor: 要监控的指标名称
        min_delta: 被认为是改进的最小变化量
        patience: 连续多少次检查没有改进后停止训练
        mode: 监控指标的模式，'min'表示越小越好，'max'表示越大越好
        verbose: 是否输出详细信息
    
    Returns:
        callbacks.EarlyStopping: 配置好的早停回调实例
    """
    return callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode,
        verbose=verbose
    )


def LearningRateMonitor(logging_interval: Optional[Literal['step', 'epoch']] = None,
                        log_momentum: bool = False,
                        log_weight_decay: bool = False) -> callbacks.LearningRateMonitor:
    """
    创建学习率监控回调实例
    
    Args:
        logging_interval: 日志记录间隔，可以是'step'、'epoch'或None
        log_momentum: 是否记录优化器的动量值
        log_weight_decay: 是否记录优化器的权重衰减值
    
    Returns:
        callbacks.LearningRateMonitor: 配置好的学习率监控回调实例
    """
    return callbacks.LearningRateMonitor(
        logging_interval=logging_interval,
        log_momentum=log_momentum,
        log_weight_decay=log_weight_decay
    )


def ModelCheckpoint(dirpath: Optional[Union[str, Path]] = None,
                    filename: Optional[str] = None,
                    monitor: Optional[str] = None,
                    save_top_k: int = 1,
                    mode: Literal['min', 'max'] = 'min',
                    save_last: Optional[Union[bool, Literal['link']]] = None,
                    every_n_epochs: Optional[int] = None) -> callbacks.ModelCheckpoint:
    """
    创建模型检查点回调实例
    
    Args:
        dirpath: 保存检查点的目录路径
        filename: 检查点文件名模板
        monitor: 监控的指标名称，None表示仅保存最后一个epoch的检查点
        save_top_k: 保存前k个最佳模型，0表示不保存，-1表示保存所有
        mode: 监控指标的模式，'min'表示越小越好，'max'表示越大越好
        save_last: 是否保存最后一个检查点，'link'表示创建符号链接
        every_n_epochs: 每n个周期检查一次是否保存检查点
    
    Returns:
        callbacks.ModelCheckpoint: 配置好的模型检查点回调实例
    """
    return callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        save_top_k=save_top_k,
        mode=mode,
        save_last=save_last,
        auto_insert_metric_name=False,
        every_n_epochs=every_n_epochs
    )


def ModelSummary(max_depth: int = 1) -> callbacks.ModelSummary:
    """
    创建模型摘要回调实例
    
    Args:
        max_depth: 层嵌套的最大深度，0表示关闭层摘要
    
    Returns:
        callbacks.ModelSummary: 配置好的模型摘要回调实例
    """
    return callbacks.ModelSummary(max_depth=max_depth)


def RichModelSummary(max_depth: int = 1) -> callbacks.RichModelSummary:
    """
    创建Rich格式的模型摘要回调实例
    
    Args:
        max_depth: 层嵌套的最大深度，0表示关闭层摘要
    
    Returns:
        callbacks.RichModelSummary: 配置好的Rich模型摘要回调实例
    """
    return callbacks.RichModelSummary(max_depth=max_depth)


def RichProgressBar(refresh_rate: int = 1,
                    leave: bool = False) -> callbacks.RichProgressBar:
    """
    创建Rich格式的进度条回调实例
    
    Args:
        refresh_rate: 进度条更新频率（批次数量）
        leave: 训练结束后是否在终端保留进度条
    
    Returns:
        callbacks.RichProgressBar: 配置好的Rich进度条回调实例
    """
    return callbacks.RichProgressBar(refresh_rate=refresh_rate, leave=leave)


def TQDMProgressBar(refresh_rate: int = 1,
                    process_position: int = 0,
                    leave: bool = False) -> callbacks.TQDMProgressBar:
    """
    创建TQDM进度条回调实例
    
    Args:
        refresh_rate: 进度条更新频率（批次数量）
        process_position: 进度条在终端中的位置偏移
        leave: 训练结束后是否在终端保留进度条
    
    Returns:
        callbacks.TQDMProgressBar: 配置好的TQDM进度条回调实例
    """
    return callbacks.TQDMProgressBar(
        refresh_rate=refresh_rate,
        process_position=process_position,
        leave=leave
    )


# 导出所有回调函数
__all__ = [
    'DeviceStatsMonitor',
    'EarlyStopping',
    'LearningRateMonitor',
    'ModelCheckpoint',
    'ModelSummary',
    'RichModelSummary',
    'RichProgressBar',
    'TQDMProgressBar'
]
