#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TorchMetrics包装器模块

该模块提供了TorchMetrics库中各种分类指标的包装类，
增强了plot方法以提供额外的控制功能。
"""

import numpy as np
import torchmetrics
import torchmetrics.classification
from typing import Optional, Dict, Any, Union, List, Tuple, Literal, cast
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class BinaryStatScores(torchmetrics.classification.BinaryStatScores):
    """
    二分类统计指标的包装类。
    """


class MultiClassStatScores(torchmetrics.classification.MulticlassStatScores):
    """
    多分类统计指标的包装类。
    """


class BinaryAccuracy(torchmetrics.classification.BinaryAccuracy):
    """
    Accuracy指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制Accuracy指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Accuracy添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAccuracy(torchmetrics.classification.MulticlassAccuracy):
    """
    Accuracy指标的包装类，增强了plot方法以提供额外的控制功能。
    在多分类问题中，总是应当设置 average='micro' 以获取正确的 Accuracy 值，
    TorchMetrics 中的 MulticlassAccuracy 指定 average='none'或'macro' 时，
    视同 Recall。
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            top_k: int = 1,
            multidim_average: Literal["global", "samplewise"] = "global",
            ignore_index: Optional[int] = None,
            validate_args: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average='micro',
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制Accuracy指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Accuracy添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryAUROC(torchmetrics.classification.BinaryAUROC):
    """
    AUROC指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制AUROC指标图，并添加额外的控制选项。

        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸

        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为AUROC添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAUROC(torchmetrics.classification.MulticlassAUROC):
    """
    AUROC指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制AUROC指标图，并添加额外的控制选项。

        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸

        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为AveragePrecision添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryAveragePrecision(torchmetrics.classification.BinaryAveragePrecision):
    """
    二分类的AveragePrecision指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制二分类Average Precision指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Average Precision添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAveragePrecision(torchmetrics.classification.MulticlassAveragePrecision):
    """
    AveragePrecision指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制AveragePrecision指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为AveragePrecision添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryConfusionMatrix(torchmetrics.classification.BinaryConfusionMatrix):
    """
    ConfusionMatrix指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制混淆矩阵，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            figsize: 图表尺寸，如果为None则使用默认尺寸
            normalize: 归一化方法，可以是'row', 'column', 'all'或None
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class BinaryConfusionMatrix(torchmetrics.classification.BinaryConfusionMatrix):
    """
    二分类的ConfusionMatrix指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制二分类混淆矩阵，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class MulticlassConfusionMatrix(torchmetrics.classification.MulticlassConfusionMatrix):
    """
    ConfusionMatrix指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制混淆矩阵，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            figsize: 图表尺寸，如果为None则使用默认尺寸
            normalize: 归一化方法，可以是'row', 'column', 'all'或None
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class BinaryF1Score(torchmetrics.classification.BinaryF1Score):
    """
    F1Score指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制F1Score指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为F1Score添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryF1Score(torchmetrics.classification.BinaryF1Score):
    """
    二分类的F1Score指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制二分类F1 Score指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为F1 Score添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassF1Score(torchmetrics.classification.MulticlassF1Score):
    """
    F1Score指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制F1Score指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为F1Score添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryPrecision(torchmetrics.classification.BinaryPrecision):
    """
    Precision指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制Precision指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Precision添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryPrecision(torchmetrics.classification.BinaryPrecision):
    """
    二分类的Precision指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制二分类Precision指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Precision添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassPrecision(torchmetrics.classification.MulticlassPrecision):
    """
    Precision指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制Precision指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Precision添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryPrecisionRecallCurve(torchmetrics.classification.BinaryPrecisionRecallCurve):
    """
    二分类的PrecisionRecallCurve指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None
             ) -> Tuple[Figure, Axes]:
        """
        绘制二分类Precision-Recall曲线，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            xlabel: x轴标签，如果为None则使用默认标签
            ylabel: y轴标签，如果为None则使用默认标签
            score: 是否显示AP分数
            figsize: 图表尺寸，如果为None则使用默认尺寸
            grid_kwargs: 传递给ax.grid函数的参数字典，如果为None则使用默认值
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置x轴标签
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 设置网格线
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        fig.tight_layout()
        return fig, ax


class MulticlassPrecisionRecallCurve(torchmetrics.classification.MulticlassPrecisionRecallCurve):
    """
    PrecisionRecallCurve指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None, legend_title: Optional[str] = "Classes"
             ) -> Tuple[Figure, Axes]:
        """
        绘制Precision-Recall曲线，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            xlabel: x轴标签，如果为None则使用默认标签
            ylabel: y轴标签，如果为None则使用默认标签
            score: 是否显示AP分数
            figsize: 图表尺寸，如果为None则使用默认尺寸
            grid_kwargs: 传递给ax.grid函数的参数字典，如果为None则使用默认值
            legend_title: 图例标题
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置x轴标签
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 设置网格线
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # 设置图例标题
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


class BinaryRecall(torchmetrics.classification.BinaryRecall):
    """
    二分类的Recall指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制二分类Recall指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Recall添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassRecall(torchmetrics.classification.MulticlassRecall):
    """
    Recall指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制Recall指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Recall添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinarySpecificity(torchmetrics.classification.BinarySpecificity):
    """
    二分类的Specificity指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制二分类Specificity指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Specificity添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassSpecificity(torchmetrics.classification.MulticlassSpecificity):
    """
    Specificity指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        绘制Specificity指标图，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            ylabel: y轴标签，如果为None则使用默认标签
            add_data_labels: 是否为每个数据点添加数值标签
            figsize: 图表尺寸，如果为None则使用默认尺寸
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 为数据点添加数值标签
        if add_data_labels:
            try:
                # 针对点图获取线和点元素
                for line in ax.lines:
                    # 获取点的坐标数据
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # 为每个点添加数值标签
                    for x, y in zip(x_data, y_data):
                        # 在点的上方添加数值标签
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"为Specificity添加数据标签时出错: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryROC(torchmetrics.classification.BinaryROC):
    """
    二分类的ROC指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None
             ) -> Tuple[Figure, Axes]:
        """
        绘制二分类ROC曲线，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            xlabel: x轴标签，如果为None则使用默认标签
            ylabel: y轴标签，如果为None则使用默认标签
            score: 是否显示AUROC分数
            figsize: 图表尺寸，如果为None则使用默认尺寸
            grid_kwargs: 传递给ax.grid函数的参数字典，如果为None则使用默认值
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置x轴标签
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 设置网格线
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        fig.tight_layout()
        return fig, ax


class MulticlassROC(torchmetrics.classification.MulticlassROC):
    """
    ROC指标的包装类，增强了plot方法以提供额外的控制功能。
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None, legend_title: Optional[str] = "Classes") -> Tuple[
        Figure, Axes]:
        """
        绘制ROC曲线，并添加额外的控制选项。
        
        Args:
            ax: 可选的matplotlib轴对象，如果为None则创建新的
            title: 图表标题，如果为None则使用默认标题
            xlabel: x轴标签，如果为None则使用默认标签
            ylabel: y轴标签，如果为None则使用默认标签
            score: 是否显示AUROC分数
            figsize: 图表尺寸，如果为None则使用默认尺寸
            grid_kwargs: 传递给ax.grid函数的参数字典，如果为None则使用默认值
            legend_title: 图例标题
            
        Returns:
            Tuple[Figure, Axes]: 图表和轴对象
        """
        # 调用原始的plot方法
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # 设置图表尺寸
        if figsize is not None:
            fig.set_size_inches(figsize)

        # 设置标题
        if title is not None:
            ax.set_title(title)

        # 设置x轴标签
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # 设置y轴标签
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # 设置网格线
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # 设置图例标题
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


def get_metric_configurer() -> Dict[str, Dict[str, Any]]:
    """
    获取所有包装后的指标配置信息，包括二分类和多分类版本
    
    Returns:
        Dict[str, Dict[str, Any]]: 指标名称到指标类和配置的映射，包含binary和multiclass两个版本
    """
    return {
        'accuracy': {
            'binary': BinaryAccuracy,
            'multiclass': MulticlassAccuracy,
            'description': 'Accuracy指标，衡量正确预测的比例'
        },
        'auroc': {
            'binary': BinaryAUROC,
            'multiclass': MulticlassAUROC,
            'description': 'AUROC指标，衡量ROC曲线下的面积'
        },
        'average_precision': {
            'binary': BinaryAveragePrecision,
            'multiclass': MulticlassAveragePrecision,
            'description': 'Average Precision指标，衡量PR曲线下的面积'
        },
        'confusion_matrix': {
            'binary': BinaryConfusionMatrix,
            'multiclass': MulticlassConfusionMatrix,
            'description': '混淆矩阵，展示各类别预测的正确与错误情况'
        },
        'f1_score': {
            'binary': BinaryF1Score,
            'multiclass': MulticlassF1Score,
            'description': 'F1 Score指标，精确率和召回率的调和平均'
        },
        'precision': {
            'binary': BinaryPrecision,
            'multiclass': MulticlassPrecision,
            'description': '精确率指标，衡量预测为正例中实际为正例的比例'
        },
        'precision_recall_curve': {
            'binary': BinaryPrecisionRecallCurve,
            'multiclass': MulticlassPrecisionRecallCurve,
            'description': '精确率-召回率曲线，展示不同阈值下精确率和召回率的关系'
        },
        'recall': {
            'binary': BinaryRecall,
            'multiclass': MulticlassRecall,
            'description': '召回率指标，衡量实际为正例中被正确预测的比例'
        },
        'specificity': {
            'binary': BinarySpecificity,
            'multiclass': MulticlassSpecificity,
            'description': '特异度指标，衡量实际为负例中被正确预测的比例'
        },
        'roc': {
            'binary': BinaryROC,
            'multiclass': MulticlassROC,
            'description': 'ROC曲线，展示不同阈值下真正例率和假正例率的关系'
        }
    }


def create_metric(metric_name: str, task_type: Literal['binary', 'multiclass'] = 'multiclass',
                  **kwargs) -> torchmetrics.Metric:
    """
    创建指定的包装后指标实例
    
    Args:
        metric_name: 指标名称
        task_type: 任务类型，'binary' 或 'multiclass'
        **kwargs: 传递给指标构造函数的参数
        
    Returns:
        torchmetrics.Metric: 指标实例
        
    Raises:
        ValueError: 如果指标名称不存在或任务类型无效
    """
    configurer: Dict[str, Dict[str, Any]] = get_metric_configurer()

    if metric_name not in configurer:
        raise ValueError(f"未知的指标名称: {metric_name}. 可用的指标名称: {list(configurer.keys())}")

    if task_type not in ['binary', 'multiclass']:
        raise ValueError(f"无效的任务类型: {task_type}. 可用的任务类型: ['binary', 'multiclass']")

    metric_class = configurer[metric_name][task_type]
    return cast(torchmetrics.Metric, metric_class(**kwargs))
