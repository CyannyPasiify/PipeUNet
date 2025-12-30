#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多分类指标测试与可视化工具

该脚本用于生成模拟多分类数据，使用TorchMetrics计算各种评估指标，并生成可视化图表。
支持自定义样本数量、类别数量和错误率分布。
使用TorchMetrics内置的plot方法实现可视化。
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchmetrics
import torchmetrics.classification
from torchmetrics import Metric, Accuracy, AUROC, AveragePrecision, ConfusionMatrix, F1Score, Precision, PrecisionRecallCurve, Recall, Specificity, ROC
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy.typing as npt


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间
    """
    parser = argparse.ArgumentParser(description='多分类指标测试与可视化工具')
    parser.add_argument('-s', '--samples', type=int, default=1000, 
                        help='样本数量，默认值为1000')
    parser.add_argument('-c', '--classes', type=int, default=3, 
                        help='类别数量，默认值为3')
    parser.add_argument('-p', '--error_rate', type=float, default=0.2, 
                        help='期望错误率，默认值为0.2')
    parser.add_argument('-o', '--save_dir', type=str, default=None, 
                        help='保存可视化结果的目录路径，默认为不保存')
    return parser.parse_args()


def generate_multiclass_data(num_samples: int, num_classes: int, error_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
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
    print(f'预测概率: {y_pred_probs}')
    print(f'真实标签: {y_true}')
    
    return torch.tensor(y_true, dtype=torch.int64), torch.tensor(y_pred_probs, dtype=torch.float32)


def compute_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, num_classes: int) -> Dict[str, Any]:
    """
    使用TorchMetrics计算各种评估指标，包括总体指标和逐类别指标
    
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
    
    # 初始化指标计算器 - 总体指标（macro平均）
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    auroc = AUROC(task="multiclass", num_classes=num_classes).to(device)
    average_precision = AveragePrecision(task="multiclass", num_classes=num_classes).to(device)
    conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)
    prc = PrecisionRecallCurve(task="multiclass", num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=num_classes).to(device)
    roc = ROC(task="multiclass", num_classes=num_classes).to(device)
    specificity = Specificity(task="multiclass", num_classes=num_classes).to(device)
    
    # 初始化逐类别指标计算器
    class_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)
    class_auroc = AUROC(task="multiclass", num_classes=num_classes, average=None).to(device)
    class_average_precision = AveragePrecision(task="multiclass", num_classes=num_classes, average=None).to(device)
    class_f1_score = F1Score(task="multiclass", num_classes=num_classes, average=None).to(device)
    class_precision = Precision(task="multiclass", num_classes=num_classes, average=None).to(device)
    class_recall = Recall(task="multiclass", num_classes=num_classes, average=None).to(device)
    class_specificity = Specificity(task="multiclass", num_classes=num_classes, average=None).to(device)
    
    # 计算总体指标（标量）
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()
    
    # 计算逐类别指标
    class_acc_values: npt.NDArray[np.float64] = class_accuracy(y_pred_probs, y_true).cpu().numpy()
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
        'class_accuracy_values': class_acc_values,
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
        'class_accuracy': class_accuracy,
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


def plot_metrics_with_torchmetrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    使用TorchMetrics的plot方法可视化指标，包括总体指标和逐类别指标
    
    Args:
        metrics: 包含指标对象和计算值的字典
        save_dir: 保存目录路径
    """
    # 将所有标量特征放在一张图中可视化
    def plot_all_scalar_metrics() -> None:
        # 准备标量指标数据
        metric_names: List[str] = ['Accuracy', 'AUROC', 'Average Precision', 'F1 Score', 'Precision', 'Recall', 'Specificity']
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
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'all_scalar_metrics.png'))
        else: 
            plt.show()
        plt.close()
    
    # 绘制逐类别指标统计图
    def plot_classwise_metric(metric_name: str, metric: Metric, title: str, ylabel: str) -> None:
        """
        绘制单个指标的逐类别统计图，并为每个类别添加数据标签
        
        Args:
            metric_name: 指标名称（用于保存文件）
            metric: 指标对象
            title: 图表标题
            ylabel: y轴标签
        """
        fig: Figure
        ax: Axes
        fig, ax = metric.plot()

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        
        # 为每个点添加数据标签
        try:
            # 针对点图获取线和点元素
            for line in ax.lines:
                # 获取点的坐标数据
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                
                # 为每个点添加数值标签
                for i, (x, y) in enumerate(zip(x_data, y_data)):
                    # 在点的上方添加数值标签
                    ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
        except Exception as e:
            print(f"为{metric_name}添加数据标签时出错: {e}")
        
        fig.tight_layout()
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, f'classwise_{metric_name}.png'))
        else: 
            plt.show()
        plt.close(fig)
    
    # 绘制所有标量指标在一张图中
    plot_all_scalar_metrics()
    
    # 绘制逐类别指标统计图
    plot_classwise_metric('accuracy', metrics['class_accuracy'], 
                         'Class-wise Accuracy', 'Accuracy')
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
    
    # 使用TorchMetrics的plot方法绘制混淆矩阵 - 按照官方文档规范
    try:
        confmat: torchmetrics.classification.ConfusionMatrix = metrics['conf_matrix']
        fig: Figure
        ax: Axes
        fig, ax = confmat.plot()

        ax.set_title('Confusion Matrix')
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        else: 
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")
    
    try:
        # 对于多分类任务，ROC.plot()会自动为每个类别绘制曲线
        roc: torchmetrics.classification.ROC = metrics['roc']
        fig: Figure
        ax: Axes
        fig, ax = roc.plot(score=True)
        
        ax.set_title('ROC Curve')
        # ax.set_xlabel('False Positive Rate')
        # ax.set_ylabel('True Positive Rate')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Classes')  # 确保图例显示类别信息
        fig.tight_layout()
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'roc.png'))
        else: 
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")
    
    # 使用TorchMetrics的plot方法绘制PR曲线 - 按照官方文档规范
    try:
        # 对于多分类任务，PrecisionRecallCurve.plot()会自动为每个类别绘制曲线
        prc: torchmetrics.classification.PrecisionRecallCurve = metrics['prc']
        fig, ax = prc.plot(score=True)
        fig: Figure = fig
        ax: Axes = ax
        
        ax.set_title('Precision-Recall Curve')
        # ax.set_xlabel('Recall')
        # ax.set_ylabel('Precision')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Classes')  # 确保图例显示类别信息
        fig.tight_layout()
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'prc.png'))
        else: 
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"绘制PR曲线时出错: {e}")


def visualize_all_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    可视化所有指标
    
    Args:
        metrics: 包含指标对象和计算值的字典
        save_dir: 保存目录路径
    """
    # 如果指定了保存目录，确保它存在
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用TorchMetrics的plot方法绘制所有指标
    plot_metrics_with_torchmetrics(metrics, save_dir)
    
    print("所有指标可视化完成！")
    if save_dir:
        print(f"可视化结果已保存到：{save_dir}")


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    打印指标摘要，包括总体指标和逐类别指标
    
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
    
    print("\n类别准确率:")
    for i in range(num_classes):
        print(f"  类别 {i}: {metrics['class_accuracy_values'][i]:.4f}")
    
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
    
    print(f"生成 {args.samples} 个样本，{args.classes} 个类别的多分类数据，目标错误率为 {args.error_rate}")
    
    # 生成多分类数据
    y_true, y_pred_probs = generate_multiclass_data(args.samples, args.classes, args.error_rate)
    
    # 计算指标
    print("计算评估指标...")
    metrics = compute_metrics(y_true, y_pred_probs, args.classes)
    
    # 打印指标摘要
    print_metrics_summary(metrics)
    
    # 可视化所有指标
    visualize_all_metrics(metrics, args.save_dir)


if __name__ == "__main__":
    main()