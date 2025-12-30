#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多分类指标测试与可视化工具 - 使用包装类版本

该脚本用于生成模拟多分类数据，使用metric_configurer.py中的包装类计算各种评估指标，
并生成可视化图表。支持自定义样本数量、类别数量和错误率分布。
同时检测包装类与原始torchmetrics类的结果一致性。
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchmetrics
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy.typing as npt

# 导入metric_configurer中的包装类
from Metric.metric_configurer import (
    # 多分类指标
    MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision, MulticlassConfusionMatrix,
    MulticlassF1Score, MulticlassPrecision, MulticlassPrecisionRecallCurve, MulticlassRecall,
    MulticlassSpecificity, MulticlassROC,
    # 二分类指标
    BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision, BinaryConfusionMatrix,
    BinaryF1Score, BinaryPrecision, BinaryPrecisionRecallCurve, BinaryRecall,
    BinarySpecificity, BinaryROC,
    get_metric_configurer, create_metric
)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间
    """
    parser = argparse.ArgumentParser(description='分类指标测试与可视化工具 - 使用包装类版本')
    parser.add_argument('-s', '--samples', type=int, default=1000,
                        help='样本数量，默认值为1000')
    parser.add_argument('-c', '--classes', type=int, default=3,
                        help='类别数量，默认值为3')
    parser.add_argument('-p', '--error_rate', type=float, default=0.2,
                        help='期望错误率，默认值为0.2')
    parser.add_argument('-o', '--save_dir', type=str, default=None,
                        help='保存可视化结果的目录路径，默认为不保存')
    parser.add_argument('-x', '--compare', action='store_true', default=False,
                        help='比较包装类与原始类的结果一致性')
    parser.add_argument('-b', '--binary', action='store_true', default=False,
                        help='启用二分类测试模式')
    return parser.parse_args()


def generate_binary_data(num_samples: int, error_rate: float) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    生成二分类数据，模拟指定错误率的分布
    确保预测值有error_rate的概率与标签不符
    
    Args:
        num_samples: 样本数量
        error_rate: 目标错误率
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - y_true: 真实标签 (num_samples,)
            - y_pred_probs: 预测概率 (num_samples, 1) 或 (num_samples,)，取决于需要
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成真实标签（二分类，0或1）
    y_true: npt.NDArray[np.int64] = np.random.randint(0, 2, size=num_samples)

    # 初始化预测概率
    y_pred_probs: npt.NDArray[np.float64] = np.zeros(num_samples)

    # 为每个样本生成预测概率
    for i in range(num_samples):
        true_class = y_true[i]

        # 随机决定当前样本是正确还是错误
        if np.random.random() < error_rate:
            # 错误情况：预测概率偏向于错误类别
            # 如果真实类别是1，则预测概率在0-0.45之间
            # 如果真实类别是0，则预测概率在0.55-1.0之间
            if true_class == 1:
                y_pred_probs[i] = np.random.uniform(0, 0.45)
            else:
                y_pred_probs[i] = np.random.uniform(0.55, 1.0)
        else:
            # 正确情况：预测概率偏向于正确类别
            # 如果真实类别是1，则预测概率在0.55-1.0之间
            # 如果真实类别是0，则预测概率在0-0.45之间
            if true_class == 1:
                y_pred_probs[i] = np.random.uniform(0.55, 1.0)
            else:
                y_pred_probs[i] = np.random.uniform(0, 0.45)

    # 添加少量噪声以增加随机性
    noise = np.random.normal(0, 0.05, size=num_samples)
    y_pred_probs += noise

    # 确保概率在0-1范围内
    y_pred_probs = np.clip(y_pred_probs, 0, 1)

    return torch.tensor(y_true, dtype=torch.int64), torch.tensor(y_pred_probs, dtype=torch.float32)


def generate_multiclass_data(num_samples: int, num_classes: int, error_rate: float) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    生成多分类数据，模拟指定错误率的分布
    确保预测值有error_rate的概率与标签不符（不是所有类预测值中最大的那一个）
    
    Args:
        num_samples: 样本数量
        num_classes: 类别数量
        error_rate: 目标错误率
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - y_true: 真实标签 (num_samples,)
            - y_pred_probs: 预测概率 (num_samples, num_classes)
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成真实标签（均匀分布）
    y_true: npt.NDArray[np.int64] = np.random.randint(0, num_classes, size=num_samples)

    # 初始化预测概率
    y_pred_probs: npt.NDArray[np.float64] = np.zeros((num_samples, num_classes))

    # 为每个样本生成预测概率
    for i in range(num_samples):
        true_class = y_true[i]

        # 随机决定当前样本是正确还是错误
        if np.random.random() < error_rate:
            # 错误情况：确保正确类别不是概率最大的
            # 处理类别数为1的特殊情况
            if num_classes <= 1:
                # 只有一个类别时，无法产生错误预测，设置概率为1.0
                y_pred_probs[i, 0] = 1.0
                continue

            # 生成除正确类别外的所有类别索引
            other_classes = np.setdiff1d(np.arange(num_classes), [true_class])

            # 确保other_classes不为空
            if len(other_classes) == 0:
                # 如果只有一个类别，设置概率为1.0
                y_pred_probs[i, true_class] = 1.0
                continue

            # 随机选择一个错误类别作为概率最大的类别
            max_class = np.random.choice(other_classes)

            # 随机生成最大概率值（范围：0.55-0.95），确保明显大于其他类别
            max_prob = np.random.uniform(0.55, 0.95)
            # 设置最大概率到错误类别
            y_pred_probs[i, max_class] = max_prob

            # 剩余概率分配给其他类别，包括正确类别
            remaining_prob = 1.0 - max_prob

            # 为其他类别分配概率
            if num_classes > 2:
                # 为所有非max_class类别生成随机权重
                non_max_classes = np.setdiff1d(np.arange(num_classes), [max_class])
                other_probs = np.random.random(len(non_max_classes))

                # 归一化并分配概率
                other_probs_normalized = other_probs / np.sum(other_probs)
                for j, cls in enumerate(non_max_classes):
                    y_pred_probs[i, cls] = remaining_prob * other_probs_normalized[j]
            else:
                # 二分类情况，直接分配剩余概率给正确类别
                y_pred_probs[i, true_class] = remaining_prob
        else:
            # 正确情况：确保正确类别是概率最大的
            # 处理类别数为1的特殊情况
            if num_classes <= 1:
                y_pred_probs[i, 0] = 1.0
                continue

            # 随机生成最大概率值（范围：0.55-0.95）
            max_prob = np.random.uniform(0.55, 0.95)
            # 设置最大概率到正确类别
            y_pred_probs[i, true_class] = max_prob

            # 剩余概率随机分配给其他类别
            remaining_prob = 1.0 - max_prob
            if num_classes > 1:
                # 为其他类别生成随机权重
                other_classes = np.setdiff1d(np.arange(num_classes), [true_class])
                if len(other_classes) > 0:
                    other_probs = np.random.random(len(other_classes))
                    other_probs_normalized = other_probs / np.sum(other_probs)
                    for j, cls in enumerate(other_classes):
                        y_pred_probs[i, cls] = remaining_prob * other_probs_normalized[j]

    # 添加少量噪声以增加随机性
    noise = np.random.normal(0, 0.05, size=(num_samples, num_classes))
    y_pred_probs += noise

    # 确保概率归一化
    y_pred_probs = np.clip(y_pred_probs, 0, None)  # 避免负概率
    y_pred_probs = y_pred_probs / y_pred_probs.sum(axis=1, keepdims=True)

    return torch.tensor(y_true, dtype=torch.int64), torch.tensor(y_pred_probs, dtype=torch.float32)


def compute_binary_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor) -> Dict[str, Any]:
    """
    使用包装后的TorchMetrics计算各种二分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred_probs: 预测概率
    
    Returns:
        Dict[str, Any]: 包含指标对象和计算值的字典
    """
    # 确保在合适的设备上运行
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # 获取预测类别
    y_pred: torch.Tensor = (y_pred_probs >= 0.5).long()

    # 初始化二分类指标计算器
    accuracy = BinaryAccuracy().to(device)
    auroc = BinaryAUROC().to(device)
    average_precision = BinaryAveragePrecision().to(device)
    conf_matrix = BinaryConfusionMatrix().to(device)
    f1_score = BinaryF1Score().to(device)
    prc = BinaryPrecisionRecallCurve().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    roc = BinaryROC().to(device)
    specificity = BinarySpecificity().to(device)

    # 计算指标值
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # 更新混淆矩阵，PRC曲线，ROC曲线
    conf_matrix(y_pred_probs, y_true)
    prc(y_pred_probs, y_true)
    roc(y_pred_probs, y_true)

    metrics = {
        # 指标值
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # 指标对象（用于可视化）
        'accuracy': accuracy,
        'auroc': auroc,
        'average_precision': average_precision,
        'conf_matrix': conf_matrix,
        'f1_score': f1_score,
        'precision': precision,
        'prc': prc,
        'recall': recall,
        'roc': roc,
        'specificity': specificity,
        # 数据
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred,
        'num_classes': 2
    }

    return metrics


def compute_multiclass_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, num_classes: int) -> Dict[str, Any]:
    """
    使用包装后的TorchMetrics计算各种评估指标，包括总体指标和逐类别指标
    
    Args:
        y_true: 真实标签
        y_pred_probs: 预测概率
        num_classes: 类别数量
    
    Returns:
        Dict[str, Any]: 包含指标对象和计算值的字典
    """
    # 确保在合适的设备上运行
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # 获取预测类别
    y_pred: torch.Tensor = y_pred_probs.argmax(dim=1)

    # 初始化指标计算器 - 总体指标（micro平均）
    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    auroc = MulticlassAUROC(num_classes=num_classes).to(device)
    average_precision = MulticlassAveragePrecision(num_classes=num_classes).to(device)
    conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    f1_score = MulticlassF1Score(num_classes=num_classes).to(device)
    prc = MulticlassPrecisionRecallCurve(num_classes=num_classes).to(device)
    precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall = MulticlassRecall(num_classes=num_classes).to(device)
    roc = MulticlassROC(num_classes=num_classes).to(device)
    specificity = MulticlassSpecificity(num_classes=num_classes).to(device)

    # 初始化逐类别指标计算器
    class_auroc = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
    class_average_precision = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    class_f1_score = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    class_precision = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    class_recall = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    class_specificity = MulticlassSpecificity(num_classes=num_classes, average=None).to(device)

    # 计算总体指标（标量）
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # 计算逐类别指标
    class_auroc_values: npt.NDArray[np.float64] = class_auroc(y_pred_probs, y_true).cpu().numpy()
    class_ap_values: npt.NDArray[np.float64] = class_average_precision(y_pred_probs, y_true).cpu().numpy()
    class_f1_values: npt.NDArray[np.float64] = class_f1_score(y_pred_probs, y_true).cpu().numpy()
    class_precision_values: npt.NDArray[np.float64] = class_precision(y_pred_probs, y_true).cpu().numpy()
    class_recall_values: npt.NDArray[np.float64] = class_recall(y_pred_probs, y_true).cpu().numpy()
    class_specificity_values: npt.NDArray[np.float64] = class_specificity(y_pred_probs, y_true).cpu().numpy()

    # 混淆矩阵，PRC曲线，ROC曲线
    conf_matrix(y_pred_probs, y_true)
    prc(y_pred_probs, y_true)
    roc(y_pred_probs, y_true)

    metrics = {
        # 总体指标值
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # 逐类别指标值
        'class_auroc_values': class_auroc_values,
        'class_ap_values': class_ap_values,
        'class_f1_values': class_f1_values,
        'class_precision_values': class_precision_values,
        'class_recall_values': class_recall_values,
        'class_specificity_values': class_specificity_values,
        # 指标对象（用于可视化）
        'accuracy': accuracy,
        'auroc': auroc,
        'average_precision': average_precision,
        'conf_matrix': conf_matrix,
        'f1_score': f1_score,
        'precision': precision,
        'prc': prc,
        'recall': recall,
        'roc': roc,
        'specificity': specificity,
        # 逐类别指标对象（用于可视化）
        'class_auroc': class_auroc,
        'class_average_precision': class_average_precision,
        'class_f1_score': class_f1_score,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_specificity': class_specificity,
        # 数据
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred,
        'num_classes': num_classes
    }

    return metrics


def compute_original_binary_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor) -> Dict[str, Any]:
    """
    使用原始TorchMetrics计算各种二分类评估指标，用于与包装类进行比较
    
    Args:
        y_true: 真实标签
        y_pred_probs: 预测概率
    
    Returns:
        Dict[str, Any]: 包含指标对象和计算值的字典
    """
    # 确保在合适的设备上运行
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # 获取原始的TorchMetrics二分类指标类
    from torchmetrics.classification import (
        BinaryAccuracy as OriginalBinaryAccuracy,
        BinaryAUROC as OriginalBinaryAUROC,
        BinaryAveragePrecision as OriginalBinaryAveragePrecision,
        BinaryF1Score as OriginalBinaryF1Score,
        BinaryPrecision as OriginalBinaryPrecision,
        BinaryRecall as OriginalBinaryRecall,
        BinarySpecificity as OriginalBinarySpecificity
    )

    # 初始化二分类指标计算器
    accuracy = OriginalBinaryAccuracy().to(device)
    auroc = OriginalBinaryAUROC().to(device)
    average_precision = OriginalBinaryAveragePrecision().to(device)
    f1_score = OriginalBinaryF1Score().to(device)
    precision = OriginalBinaryPrecision().to(device)
    recall = OriginalBinaryRecall().to(device)
    specificity = OriginalBinarySpecificity().to(device)

    # 计算指标值
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    metrics = {
        # 指标值
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        'num_classes': 2
    }

    return metrics


def compute_original_multiclass_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, num_classes: int) -> Dict[str, Any]:
    """
    使用原始TorchMetrics计算各种评估指标，用于与包装类进行比较
    
    Args:
        y_true: 真实标签
        y_pred_probs: 预测概率
        num_classes: 类别数量
    
    Returns:
        Dict[str, Any]: 包含指标对象和计算值的字典
    """
    # 确保在合适的设备上运行
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # 获取原始的TorchMetrics类
    from torchmetrics.classification import (
        MulticlassAccuracy as OriginalAccuracy,
        MulticlassAUROC as OriginalAUROC,
        MulticlassAveragePrecision as OriginalAveragePrecision,
        MulticlassF1Score as OriginalF1Score,
        MulticlassPrecision as OriginalPrecision,
        MulticlassRecall as OriginalRecall,
        MulticlassSpecificity as OriginalSpecificity
    )

    # 初始化指标计算器 - 总体指标（micro平均）
    accuracy = OriginalAccuracy(num_classes=num_classes, average='micro').to(device)
    auroc = OriginalAUROC(num_classes=num_classes).to(device)
    average_precision = OriginalAveragePrecision(num_classes=num_classes).to(device)
    f1_score = OriginalF1Score(num_classes=num_classes).to(device)
    precision = OriginalPrecision(num_classes=num_classes).to(device)
    recall = OriginalRecall(num_classes=num_classes).to(device)
    specificity = OriginalSpecificity(num_classes=num_classes).to(device)

    # 初始化逐类别指标计算器
    class_auroc = OriginalAUROC(num_classes=num_classes, average=None).to(device)
    class_average_precision = OriginalAveragePrecision(num_classes=num_classes, average=None).to(device)
    class_f1_score = OriginalF1Score(num_classes=num_classes, average=None).to(device)
    class_precision = OriginalPrecision(num_classes=num_classes, average=None).to(device)
    class_recall = OriginalRecall(num_classes=num_classes, average=None).to(device)
    class_specificity = OriginalSpecificity(num_classes=num_classes, average=None).to(device)

    # 计算总体指标（标量）
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # 计算逐类别指标
    class_auroc_values: npt.NDArray[np.float64] = class_auroc(y_pred_probs, y_true).cpu().numpy()
    class_ap_values: npt.NDArray[np.float64] = class_average_precision(y_pred_probs, y_true).cpu().numpy()
    class_f1_values: npt.NDArray[np.float64] = class_f1_score(y_pred_probs, y_true).cpu().numpy()
    class_precision_values: npt.NDArray[np.float64] = class_precision(y_pred_probs, y_true).cpu().numpy()
    class_recall_values: npt.NDArray[np.float64] = class_recall(y_pred_probs, y_true).cpu().numpy()
    class_specificity_values: npt.NDArray[np.float64] = class_specificity(y_pred_probs, y_true).cpu().numpy()

    metrics = {
        # 总体指标值
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # 逐类别指标值
        'class_auroc_values': class_auroc_values,
        'class_ap_values': class_ap_values,
        'class_f1_values': class_f1_values,
        'class_precision_values': class_precision_values,
        'class_recall_values': class_recall_values,
        'class_specificity_values': class_specificity_values,
        'num_classes': num_classes
    }

    return metrics


def compare_binary_metrics(wrapped_metrics: Dict[str, Any], original_metrics: Dict[str, Any], tolerance: float = 1e-6) -> bool:
    """
    比较二分类包装类与原始类计算的指标是否一致
    
    Args:
        wrapped_metrics: 使用包装类计算的指标
        original_metrics: 使用原始类计算的指标
        tolerance: 浮点数比较的容差
    
    Returns:
        bool: 是否所有指标都一致
    """
    print("\n=== 二分类包装类与原始类指标一致性检测 ===")

    # 标量指标比较
    scalar_metrics = ['accuracy_value', 'auroc_value', 'average_precision_value',
                      'f1_score_value', 'precision_value', 'recall_value', 'specificity_value']

    all_consistent = True

    for metric_name in scalar_metrics:
        wrapped_val = wrapped_metrics[metric_name]
        original_val = original_metrics[metric_name]
        is_consistent = abs(wrapped_val - original_val) < tolerance

        print(f"{metric_name}: 包装类={wrapped_val:.6f}, 原始类={original_val:.6f}, 一致性: {is_consistent}")

        if not is_consistent:
            all_consistent = False

    if all_consistent:
        print("\n[PASS] 所有二分类指标计算结果一致!")
    else:
        print("\n[FAIL] 发现不一致的二分类指标计算结果!")

    print("=====================================\n")
    return all_consistent


def compare_multiclass_metrics(wrapped_metrics: Dict[str, Any], original_metrics: Dict[str, Any], tolerance: float = 1e-6) -> bool:
    """
    比较多分类包装类与原始类计算的指标是否一致
    
    Args:
        wrapped_metrics: 使用包装类计算的指标
        original_metrics: 使用原始类计算的指标
        tolerance: 浮点数比较的容差
    
    Returns:
        bool: 是否所有指标都一致
    """
    print("\n=== 包装类与原始类指标一致性检测 ===")

    # 标量指标比较
    scalar_metrics = ['accuracy_value', 'auroc_value', 'average_precision_value',
                      'f1_score_value', 'precision_value', 'recall_value', 'specificity_value']

    all_consistent = True

    for metric_name in scalar_metrics:
        wrapped_val = wrapped_metrics[metric_name]
        original_val = original_metrics[metric_name]
        is_consistent = abs(wrapped_val - original_val) < tolerance

        print(f"{metric_name}: 包装类={wrapped_val:.6f}, 原始类={original_val:.6f}, 一致性: {is_consistent}")

        if not is_consistent:
            all_consistent = False

    # 逐类别指标比较
    class_metrics = ['class_auroc_values', 'class_ap_values', 'class_f1_values', 'class_precision_values',
                     'class_recall_values', 'class_specificity_values']

    for metric_name in class_metrics:
        wrapped_vals = wrapped_metrics[metric_name]
        original_vals = original_metrics[metric_name]

        # 检查数组长度是否一致
        if len(wrapped_vals) != len(original_vals):
            print(f"{metric_name}: 长度不一致! 包装类={len(wrapped_vals)}, 原始类={len(original_vals)}")
            all_consistent = False
            continue

        # 逐元素比较
        is_consistent = np.allclose(wrapped_vals, original_vals, atol=tolerance)

        print(f"{metric_name}: 一致性: {is_consistent}")
        if not is_consistent:
            # 找出不一致的位置
            diff_indices = np.where(np.abs(wrapped_vals - original_vals) >= tolerance)[0]
            for idx in diff_indices:
                print(f"  类别 {idx}: 包装类={wrapped_vals[idx]:.6f}, 原始类={original_vals[idx]:.6f}")
            all_consistent = False

    if all_consistent:
        print("\n[PASS] 所有指标计算结果一致!")
    else:
        print("\n[FAIL] 发现不一致的指标计算结果!")

    print("=================================\n")
    return all_consistent


def plot_binary_metrics_with_wrapped_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    使用包装后的TorchMetrics的plot方法可视化二分类指标
    利用包装类的增强功能进行更灵活的控制
    
    Args:
        metrics: 包含指标对象和计算值的字典
        save_dir: 保存目录路径
    """

    # 将所有标量特征放在一张图中可视化
    def plot_all_scalar_metrics() -> None:
        # 准备标量指标数据
        metric_names: List[str] = ['Accuracy', 'AUROC', 'Average Precision', 'F1 Score',
                                   'Precision', 'Recall', 'Specificity']
        metric_values: List[float] = [
            metrics['accuracy_value'],
            metrics['auroc_value'],
            metrics['average_precision_value'],
            metrics['f1_score_value'],
            metrics['precision_value'],
            metrics['recall_value'],
            metrics['specificity_value']
        ]

        # 使用不同的颜色
        colors = ['skyblue', 'yellow', 'orange', 'purple', 'red', 'brown', 'lightgreen']

        # 创建图表
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylim(0, 1)
        plt.ylabel('Value')
        plt.title('Binary Classification Metrics')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'binary_all_scalar_metrics.png'))
        else:
            plt.show()
        plt.close()

    # 绘制所有标量指标在一张图中
    plot_all_scalar_metrics()

    # 使用包装类的plot方法绘制混淆矩阵
    try:
        conf_matrix: BinaryConfusionMatrix = metrics['conf_matrix']
        fig, ax = conf_matrix.plot(
            title='Binary Confusion Matrix',
            figsize=(10, 8)
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'binary_confusion_matrix.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")

    # 使用包装类的plot方法绘制ROC曲线
    try:
        roc: BinaryROC = metrics['roc']
        fig, ax = roc.plot(
            score=True,
            title='Binary ROC Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': '--', 'alpha': 0.3}
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'binary_roc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")

    # 使用包装类的plot方法绘制PR曲线
    try:
        prc: BinaryPrecisionRecallCurve = metrics['prc']
        fig, ax = prc.plot(
            score=True,
            title='Binary Precision-Recall Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': ':', 'alpha': 0.7}
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'binary_prc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制PR曲线时出错: {e}")


def plot_multiclass_metrics_with_wrapped_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    使用包装后的TorchMetrics的plot方法可视化指标，包括总体指标和逐类别指标
    利用包装类的增强功能进行更灵活的控制
    
    Args:
        metrics: 包含指标对象和计算值的字典
        save_dir: 保存目录路径
    """

    # 将所有标量特征放在一张图中可视化
    def plot_all_scalar_metrics() -> None:
        # 准备标量指标数据
        metric_names: List[str] = ['Accuracy', 'Macro-AUROC', 'Macro-Average Precision', 'Macro-F1 Score',
                                   'Macro-Precision', 'Macro-Recall', 'Macro-Specificity']
        metric_values: List[float] = [
            metrics['accuracy_value'],
            metrics['auroc_value'],
            metrics['average_precision_value'],
            metrics['f1_score_value'],
            metrics['precision_value'],
            metrics['recall_value'],
            metrics['specificity_value']
        ]

        # 使用不同的颜色
        colors = ['skyblue', 'yellow', 'orange', 'purple', 'red', 'brown', 'lightgreen']

        # 创建图表
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylim(0, 1)
        plt.ylabel('Value')
        plt.title('Classification Metrics')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'all_scalar_metrics.png'))
        else:
            plt.show()
        plt.close()

    # 绘制逐类别指标统计图，利用包装类的增强plot功能
    def plot_classwise_metric(metric_name: str, metric: torchmetrics.Metric, title: str, ylabel: str) -> None:
        """
        使用包装类的plot方法绘制单个指标的逐类别统计图
        利用包装类提供的额外控制选项
        
        Args:
            metric_name: 指标名称（用于保存文件）
            metric: 包装后的指标对象
            title: 图表标题
            ylabel: y轴标签
        """
        try:
            # 使用包装类的plot方法，利用其增强功能
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = metric.plot(
                title=title,
                ylabel=ylabel,
                add_data_labels=True,
                figsize=(10, 6)
            )

            if save_dir:
                fig.savefig(os.path.join(save_dir, f'classwise_{metric_name}.png'))
            else:
                plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"绘制{metric_name}时出错: {e}")

    # 绘制所有标量指标在一张图中
    plot_all_scalar_metrics()

    # 绘制逐类别指标统计图，使用包装类的增强功能
    plot_classwise_metric('auroc', metrics['class_auroc'],
                          'Class-wise AUROC', 'AUROC')
    plot_classwise_metric('average_precision', metrics['class_average_precision'],
                          'Class-wise Average Precision', 'Average Precision')
    plot_classwise_metric('f1_score', metrics['class_f1_score'],
                          'Class-wise F1 Score', 'F1 Score')
    plot_classwise_metric('precision', metrics['class_precision'],
                          'Class-wise Precision', 'Precision')
    plot_classwise_metric('recall', metrics['class_recall'],
                          'Class-wise Recall', 'Recall')
    plot_classwise_metric('specificity', metrics['class_specificity'],
                          'Class-wise Specificity', 'Specificity')

    # 使用包装类的plot方法绘制混淆矩阵
    try:
        conf_matrix: MulticlassConfusionMatrix = metrics['conf_matrix']
        fig, ax = conf_matrix.plot(
            title='Confusion Matrix',
            figsize=(10, 8)
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")

    # 使用包装类的plot方法绘制ROC曲线，利用grid_kwargs参数
    try:
        roc: MulticlassROC = metrics['roc']
        fig, ax = roc.plot(
            score=True,
            title='ROC Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': '--', 'alpha': 0.3},
            legend_title='Classes'
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'roc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")

    # 使用包装类的plot方法绘制PR曲线，利用grid_kwargs参数
    try:
        prc: MulticlassPrecisionRecallCurve = metrics['prc']
        fig, ax = prc.plot(
            score=True,
            title='Precision-Recall Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': ':', 'alpha': 0.7},
            legend_title='Classes'
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'prc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制PR曲线时出错: {e}")


def visualize_binary_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    使用包装类可视化所有二分类指标
    
    Args:
        metrics: 包含指标对象和计算值的字典
        save_dir: 保存目录路径
    """
    # 如果指定了保存目录，确保它存在
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 使用包装类的plot方法绘制所有二分类指标
    plot_binary_metrics_with_wrapped_metrics(metrics, save_dir)

    print("所有二分类指标可视化完成！")
    if save_dir:
        print(f"可视化结果已保存到：{save_dir}")


def visualize_multiclass_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    使用包装类可视化所有指标
    
    Args:
        metrics: 包含指标对象和计算值的字典
        save_dir: 保存目录路径
    """
    # 如果指定了保存目录，确保它存在
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 使用包装类的plot方法绘制所有指标
    plot_multiclass_metrics_with_wrapped_metrics(metrics, save_dir)

    print("所有指标可视化完成！")
    if save_dir:
        print(f"可视化结果已保存到：{save_dir}")


def print_binary_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    打印二分类指标摘要
    
    Args:
        metrics: 包含指标对象和计算值的字典
    """
    print("\n=== 二分类指标摘要 ===")
    print(f"Accuracy: {metrics['accuracy_value']:.4f}")
    print(f"AUROC: {metrics['auroc_value']:.4f}")
    print(f"Average Precision: {metrics['average_precision_value']:.4f}")
    print(f"F1 Score: {metrics['f1_score_value']:.4f}")
    print(f"Precision: {metrics['precision_value']:.4f}")
    print(f"Recall: {metrics['recall_value']:.4f}")
    print(f"Specificity: {metrics['specificity_value']:.4f}")
    print("====================\n")


def print_multiclass_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    打印多分类指标摘要，包括总体指标和逐类别指标
    
    Args:
        metrics: 包含指标对象和计算值的字典
    """
    print("\n=== 多分类指标摘要 ===")
    print(f"Accuracy: {metrics['accuracy_value']:.4f}")
    print(f"AUROC: {metrics['auroc_value']:.4f}")
    print(f"Average Precision: {metrics['average_precision_value']:.4f}")
    print(f"F1 Score: {metrics['f1_score_value']:.4f}")
    print(f"Precision: {metrics['precision_value']:.4f}")
    print(f"Recall: {metrics['recall_value']:.4f}")
    print(f"Specificity: {metrics['specificity_value']:.4f}")

    # 打印逐类别指标
    print("\n=== 逐类别指标 ===")
    num_classes = metrics['num_classes']

    print("\n类别AUROC:")
    for i in range(num_classes):
        print(f"  类别 {i}: {metrics['class_auroc_values'][i]:.4f}")

    print("\n类别Average Precision:")
    for i in range(num_classes):
        print(f"  类别 {i}: {metrics['class_ap_values'][i]:.4f}")

    print("\n类别F1 Score:")
    for i in range(num_classes):
        print(f"  类别 {i}: {metrics['class_f1_values'][i]:.4f}")

    print("\n类别Precision:")
    for i in range(num_classes):
        print(f"  类别 {i}: {metrics['class_precision_values'][i]:.4f}")

    print("\n类别Recall:")
    for i in range(num_classes):
        print(f"  类别 {i}: {metrics['class_recall_values'][i]:.4f}")

    print("\n类别Specificity:")
    for i in range(num_classes):
        print(f"  类别 {i}: {metrics['class_specificity_values'][i]:.4f}")

    print("====================\n")


def main() -> None:
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()

    # 二分类测试模式
    if args.binary:
        print(f"生成 {args.samples} 个样本，错误率为 {args.error_rate} 的二分类数据...")
        
        # 生成二分类数据
        y_true, y_pred_probs = generate_binary_data(args.samples, args.error_rate)
        
        print("\n使用包装的TorchMetrics计算二分类指标...")
        # 使用包装类计算二分类指标
        wrapped_metrics = compute_binary_metrics(y_true, y_pred_probs)
        
        # 打印指标摘要
        print_binary_metrics_summary(wrapped_metrics)
        
        # 比较包装类与原始类的结果一致性
        if args.compare:
            print("使用原始TorchMetrics计算二分类指标进行比较...")
            original_metrics = compute_original_binary_metrics(y_true, y_pred_probs)
            compare_binary_metrics(wrapped_metrics, original_metrics)
        
        # 可视化所有二分类指标
        visualize_binary_metrics(wrapped_metrics, args.save_dir)
    else:
        # 多分类测试模式（原有逻辑）
        print(f"生成 {args.samples} 个样本，{args.classes} 个类别，错误率为 {args.error_rate} 的多分类数据...")
        
        # 生成多分类数据
        y_true, y_pred_probs = generate_multiclass_data(args.samples, args.classes, args.error_rate)
        
        print("\n使用包装的TorchMetrics计算多分类指标...")
        # 使用包装类计算指标
        wrapped_metrics = compute_multiclass_metrics(y_true, y_pred_probs, args.classes)
        
        # 打印指标摘要
        print_multiclass_metrics_summary(wrapped_metrics)
        
        # 比较包装类与原始类的结果一致性
        if args.compare:
            print("使用原始TorchMetrics计算指标进行比较...")
            original_metrics = compute_original_multiclass_metrics(y_true, y_pred_probs, args.classes)
            compare_multiclass_metrics(wrapped_metrics, original_metrics)
        
        # 可视化所有指标
        visualize_multiclass_metrics(wrapped_metrics, args.save_dir)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
