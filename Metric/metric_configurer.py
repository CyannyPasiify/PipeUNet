# -*- coding: utf-8 -*-
"""
Metrics Configurers

This module provides wrapper classes for various metrics from the TorchMetrics and MONAI libraries,
enhancing the plot method to provide additional control functionality (for TorchMetrics).

TorchMetrics IO Requirements

Elements shall all be int type
Binary: pred=(N,...) gt(binary map)=(N,...) dtype=torch.int [0,1]
Multiclass: pred=(N,...) gt(label map)=(N,...) dtype=torch.int [0,num_classes-1]
Multilabel: pred=(N,C,...) gt(C-binary maps)=(N,C,...) dtype=torch.int [0,1]

MONAI metrics IO Requirements

To avoid unnecessary trouble, we require
all inputs shall be pred=(B,C,*Sp,...) gt=(B,C,*Sp,...)
Segmentation: dtype=torch.int [0,1]
Image-To-Image: dtype=torch.float
"""

import numpy as np
import torch
import torchmetrics
import torchmetrics.classification
from typing import Optional, Dict, Any, Union, List, Tuple, Literal, Sequence, Type, cast
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import monai.metrics as mm
from monai.config import TensorOrList
from monai.metrics.regression import KernelType
from monai.utils import MetricReduction, Weight

"""
TorchMetrics IO Requirements

Elements shall all be int type
Binary: pred=(N,...) gt(binary map)=(N,...) dtype=torch.int [0,1]
Multiclass: pred=(N,...) gt(label map)=(N,...) dtype=torch.int [0,num_classes-1]
Multilabel: pred=(N,C,...) gt(C-binary maps)=(N,C,...) dtype=torch.int [0,1]
"""


# region Torchmetrics
def assert_input_torchmetrics(
        task_type: Literal["binary", "multiclass", "multilabel"],
        y_pred: torch.Tensor,
        y_gt: torch.Tensor,
        num_classes: Optional[int] = None
):
    assert task_type in ['binary', 'multiclass', 'multilabel'], f'task_type={task_type} is not supported'
    # type = Tensor
    assert isinstance(y_pred, torch.Tensor), f'y_pred [type={type(y_pred)}] is not a torch.Tensor'
    assert isinstance(y_gt, torch.Tensor), f'y_pred [type={type(y_gt)}] is not a torch.Tensor'
    # dtype = int
    assert y_pred.dtype == torch.int, f'y_pred [dtype={y_pred.dtype}] shall be {torch.int}'
    assert y_gt.dtype == torch.int, f'y_gt [dtype={y_gt.dtype}] shall be {torch.int}'
    # size dim match
    assert y_pred.ndim == y_gt.ndim, f'y_pred [ndim={y_pred.ndim}] and y_gt [ndim={y_gt.ndim}] shall have the same ndim'
    # enough size dim
    if task_type in ['binary', 'multiclass']:
        assert y_pred.ndim > 0, f'y_pred [ndim={y_pred.ndim}] shall be at least 1 (N, ...)]'
        assert y_gt.ndim > 0, f'y_gt [ndim={y_gt.ndim}] shall be at least 1 (N, ...)]'
    else:  # multilabel
        assert y_pred.ndim > 1, f'y_pred [ndim={y_pred.ndim}] shall be at least 2 (N, C, ...)]'
        assert y_gt.ndim > 1, f'y_gt [ndim={y_gt.ndim}] shall be at least 2 (N, C, ...)]'

    sz_pred: Tuple[int, ...] = tuple(y_pred.size())
    sz_gt: Tuple[int, ...] = tuple(y_gt.size())
    # size match
    assert sz_pred == sz_gt, f'y_pred [size={sz_pred}] and y_gt [size={sz_gt}] shall have the same size'
    # size non-zero
    assert 0 not in sz_pred, f'y_pred [size={sz_pred}] shall have non-zero size in all dimensions'
    assert 0 not in sz_gt, f'y_pred [size={sz_gt}] shall have non-zero size in all dimensions'
    # value range
    if task_type in ['binary', 'multilabel']:
        allowed_labels: torch.Tensor = torch.tensor([0, 1], dtype=torch.int)
        assert torch.all(torch.isin(y_pred, allowed_labels)), \
            f'y_pred [unique={list(y_pred.unique())}] shall only contain [0, 1]'
        assert torch.all(torch.isin(y_gt, allowed_labels)), \
            f'y_gt [unique={list(y_gt.unique())}] shall only contain [0, 1]'
    elif task_type == "multiclass":
        assert num_classes is not None, 'you shall specify num_classes for multiclass assertion'
        label_list: List[int] = list(range(num_classes))
        allowed_labels: torch.Tensor = torch.tensor(label_list, dtype=torch.int)
        assert torch.all(torch.isin(y_pred, allowed_labels)), \
            f'y_pred [unique={list(y_pred.unique())}] shall only contain {label_list}'
        assert torch.all(torch.isin(y_gt, allowed_labels)), \
            f'y_gt [unique={list(y_gt.unique())}] shall only contain {label_list}'


class BinaryStatScores(torchmetrics.classification.BinaryStatScores):
    """
    Wrapper class for binary classification statistical metrics.
    """


class MulticlassStatScores(torchmetrics.classification.MulticlassStatScores):
    """
    Wrapper class for multiclass classification statistical metrics.
    """


class MultilabelStatScores(torchmetrics.classification.MultilabelStatScores):
    """
    Wrapper class for multiclass classification statistical metrics.
    """


class BinaryAccuracy(torchmetrics.classification.BinaryAccuracy):
    """
    Wrapper class for Accuracy metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Accuracy metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Accuracy: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAccuracy(torchmetrics.classification.MulticlassAccuracy):
    """
    Wrapper class for Accuracy metric, enhanced with additional plot control functionality.
    For multiclass problems, average='micro' should always be set to get the correct Accuracy value.
    When MulticlassAccuracy in TorchMetrics specifies average='none' or 'macro',
    it is equivalent to Recall.
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            top_k: int = 1,
            average: Optional[Union[Literal["micro"]]] = "micro",
            multidim_average: Literal["global", "samplewise"] = "global",
            ignore_index: Optional[int] = None,
            validate_args: bool = True,
            **kwargs: Any
    ) -> None:
        assert average in ["micro", None]
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Accuracy metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Accuracy: {e}")

        fig.tight_layout()
        return fig, ax


class MultilabelAccuracy(torchmetrics.classification.MultilabelAccuracy):
    """
    Wrapper class for Accuracy metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Accuracy metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Accuracy: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryAUROC(torchmetrics.classification.BinaryAUROC):
    """
    Wrapper class for AUROC metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot AUROC metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AUROC: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAUROC(torchmetrics.classification.MulticlassAUROC):
    """
    Wrapper class for AUROC metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot AUROC metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AveragePrecision: {e}")

        fig.tight_layout()
        return fig, ax


class MultilabelAUROC(torchmetrics.classification.MultilabelAUROC):
    """
    Wrapper class for AUROC metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot AUROC metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AveragePrecision: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryAveragePrecision(torchmetrics.classification.BinaryAveragePrecision):
    """
    Wrapper class for binary classification AveragePrecision metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Average Precision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Average Precision: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAveragePrecision(torchmetrics.classification.MulticlassAveragePrecision):
    """
    Wrapper class for AveragePrecision metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot AveragePrecision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AveragePrecision: {e}")

        fig.tight_layout()
        return fig, ax


class MultilabelAveragePrecision(torchmetrics.classification.MultilabelAveragePrecision):
    """
    Wrapper class for AveragePrecision metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot AveragePrecision metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AveragePrecision: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryConfusionMatrix(torchmetrics.classification.BinaryConfusionMatrix):
    """
    Wrapper class for ConfusionMatrix metric, enhanced with additional plot control functionality.
    """

    # type: ignore[override]
    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            figsize: Optional[Tuple[int, int]] = None,
            **kwargs: Any  # nonsense
    ) -> Tuple[Figure, Axes]:
        """
        Plot confusion matrix with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class MulticlassConfusionMatrix(torchmetrics.classification.MulticlassConfusionMatrix):
    """
    Wrapper class for ConfusionMatrix metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            figsize: Optional[Tuple[int, int]] = None,
            **kwargs: Any  # nonsense
    ) -> Tuple[Figure, Axes]:
        """
        Plot confusion matrix with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class MultilabelConfusionMatrix(torchmetrics.classification.MultilabelConfusionMatrix):
    """
    Wrapper class for ConfusionMatrix metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            figsize: Optional[Tuple[int, int]] = None,
            **kwargs: Any  # nonsense
    ) -> Tuple[Figure, List[Axes]]:
        """
        Plot confusion matrix with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, List[Axes]]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        axes: np.ndarray  # [Axes]
        fig, axes = super().plot(ax=ax)
        axes: List[Axes] = axes.tolist()

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        fig.suptitle(title)

        fig.tight_layout()
        return fig, axes


class BinaryF1Score(torchmetrics.classification.BinaryF1Score):
    """
    Wrapper class for F1Score metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot F1Score metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to F1Score: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassF1Score(torchmetrics.classification.MulticlassF1Score):
    """
    Wrapper class for F1Score metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot F1Score metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to F1Score: {e}")

        fig.tight_layout()
        return fig, ax


class MultilabelF1Score(torchmetrics.classification.MultilabelF1Score):
    """
    Wrapper class for F1Score metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot F1Score metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to F1Score: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryPrecision(torchmetrics.classification.BinaryPrecision):
    """
    Wrapper class for Precision metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Precision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Precision: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassPrecision(torchmetrics.classification.MulticlassPrecision):
    """
    Wrapper class for Precision metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Precision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Precision: {e}")

        fig.tight_layout()
        return fig, ax


class MultilabelPrecision(torchmetrics.classification.MultilabelPrecision):
    """
    Wrapper class for Precision metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Precision metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Precision: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryPrecisionRecallCurve(torchmetrics.classification.BinaryPrecisionRecallCurve):
    """
    Wrapper class for binary classification PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            score: bool = True,
            figsize: Optional[Tuple[int, int]] = None,
            grid_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Precision-Recall curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AP score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        fig.tight_layout()
        return fig, ax


class MulticlassPrecisionRecallCurve(torchmetrics.classification.MulticlassPrecisionRecallCurve):
    """
    Wrapper class for PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            score: bool = True,
            figsize: Optional[Tuple[int, int]] = None,
            grid_kwargs: Optional[Dict[str, Any]] = None,
            legend_title: Optional[str] = "Classes"
    ) -> Tuple[Figure, Axes]:
        """
        Plot Precision-Recall curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AP score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            legend_title: Legend title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # Set legend title
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


class MultilabelPrecisionRecallCurve(torchmetrics.classification.MultilabelPrecisionRecallCurve):
    """
    Wrapper class for PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            score: bool = True,
            figsize: Optional[Tuple[int, int]] = None,
            grid_kwargs: Optional[Dict[str, Any]] = None,
            legend_title: Optional[str] = "Classes"
    ) -> Tuple[Figure, Axes]:
        """
        Plot Precision-Recall curve with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AP score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            legend_title: Legend title

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # Set legend title
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


class BinaryRecall(torchmetrics.classification.BinaryRecall):
    """
    Wrapper class for binary classification Recall metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Recall metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Recall: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassRecall(torchmetrics.classification.MulticlassRecall):
    """
    Wrapper class for Recall metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Recall metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Recall: {e}")

        fig.tight_layout()
        return fig, ax


class MultilabelRecall(torchmetrics.classification.MultilabelRecall):
    """
    Wrapper class for Recall metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Recall metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Recall: {e}")

        fig.tight_layout()
        return fig, ax


class BinarySpecificity(torchmetrics.classification.BinarySpecificity):
    """
    Wrapper class for binary classification Specificity metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Specificity metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Specificity: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassSpecificity(torchmetrics.classification.MulticlassSpecificity):
    """
    Wrapper class for Specificity metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Specificity metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Specificity: {e}")

        fig.tight_layout()
        return fig, ax


class MultilabelSpecificity(torchmetrics.classification.MultilabelSpecificity):
    """
    Wrapper class for Specificity metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Specificity metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()  # noqa
                    y_data: np.ndarray = line.get_ydata()  # noqa

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Specificity: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryROC(torchmetrics.classification.BinaryROC):
    """
    Wrapper class for binary classification ROC metric, enhanced with additional plot control functionality.
    """

    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            score: bool = True,
            figsize: Optional[Tuple[int, int]] = None,
            grid_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot binary classification ROC curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AUROC score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        fig.tight_layout()
        return fig, ax


class MulticlassROC(torchmetrics.classification.MulticlassROC):
    """
    Wrapper class for ROC metric, enhanced with additional plot control functionality.
    """

    def plot(
            self, ax: Optional[Axes] = None,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            score: bool = True,
            figsize: Optional[Tuple[int, int]] = None,
            grid_kwargs: Optional[Dict[str, Any]] = None,
            legend_title: Optional[str] = "Classes"
    ) -> Tuple[Figure, Axes]:
        """
        Plot ROC curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AUROC score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            legend_title: Legend title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # Set legend title
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


class MultilabelROC(torchmetrics.classification.MultilabelROC):
    """
    Wrapper class for ROC metric, enhanced with additional plot control functionality.
    """

    def plot(
            self, ax: Optional[Axes] = None,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            score: bool = True,
            figsize: Optional[Tuple[int, int]] = None,
            grid_kwargs: Optional[Dict[str, Any]] = None,
            legend_title: Optional[str] = "Classes"
    ) -> Tuple[Figure, Axes]:
        """
        Plot ROC curve with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AUROC score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            legend_title: Legend title

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # Set legend title
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


# endregion

# region Torchmetrics Alias
BSS = BinaryStatScores
MCSS = MulticlassStatScores
MLSS = MultilabelStatScores
BACC = BinaryAccuracy
MCACC = MulticlassAccuracy
MLACC = MultilabelAccuracy
BAUROC = BinaryAUROC
MCAUC = MulticlassAUROC
MLAUC = MultilabelAUROC
BAP = BinaryAveragePrecision
MCAP = MulticlassAveragePrecision
MLAP = MultilabelAveragePrecision
BCM = BinaryConfusionMatrix
MCCM = MulticlassConfusionMatrix
MLCM = MultilabelConfusionMatrix
BF1 = BinaryF1Score
MCF1 = MulticlassF1Score
MLF1 = MultilabelF1Score
BPREC = BinaryPrecision
MCPREC = MulticlassPrecision
MLPREC = MultilabelPrecision
BPRC = BinaryPrecisionRecallCurve
MCPRC = MulticlassPrecisionRecallCurve
MLPRC = MultilabelPrecisionRecallCurve
BREC = BinaryRecall
MCREC = MulticlassRecall
MLREC = MultilabelRecall
BSPE = BinarySpecificity
MCSPE = MulticlassSpecificity
MLSPE = MultilabelSpecificity
BROC = BinaryROC
MCROC = MulticlassROC
MLROC = MultilabelROC
# endregion


"""
MONAI metrics IO Requirements

To avoid unnecessary trouble, we require
all inputs shall be pred=(B,C,*Sp,...) gt=(B,C,*Sp,...) 
Segmentation: dtype=torch.int [0,1]
Image-To-Image: dtype=torch.float
"""


# region MONAI
def assert_input_monaimetrics(
        task_type: Literal["segmentation", "img2img"],
        y_pred: torch.Tensor,
        y_gt: torch.Tensor
):
    assert task_type in ['segmentation', 'img2img'], f'task_type={task_type} is not supported'
    # type = Tensor
    assert isinstance(y_pred, torch.Tensor), f'y_pred [type={type(y_pred)}] is not a torch.Tensor'
    assert isinstance(y_gt, torch.Tensor), f'y_pred [type={type(y_gt)}] is not a torch.Tensor'
    # dtype = int (segmentation) |  float (img2img)
    valid_dtype: torch._C.dtype = torch.int if task_type == "segmentation" else torch.float
    assert y_pred.dtype == valid_dtype, f'y_pred [dtype={y_pred.dtype}] shall be {valid_dtype}'
    assert y_gt.dtype == valid_dtype, f'y_gt [dtype={y_gt.dtype}] shall be {valid_dtype}'
    # size dim match
    assert y_pred.ndim == y_gt.ndim, f'y_pred [ndim={y_pred.ndim}] and y_gt [ndim={y_gt.ndim}] shall have the same ndim'
    # enough size dim
    assert y_pred.ndim > 2, f'y_pred [ndim={y_pred.ndim}] shall be at least 3 (B, C, *Sp, ...)]'
    assert y_gt.ndim > 2, f'y_gt [ndim={y_gt.ndim}] shall be at least 3 (B, C, *Sp, ...)]'

    sz_pred: Tuple[int, ...] = tuple(y_pred.size())
    sz_gt: Tuple[int, ...] = tuple(y_gt.size())
    # size match
    assert sz_pred == sz_gt, f'y_pred [size={sz_pred}] and y_gt [size={sz_gt}] shall have the same size'
    # size non-zero
    assert 0 not in sz_pred, f'y_pred [size={sz_pred}] shall have non-zero size in all dimensions'
    assert 0 not in sz_gt, f'y_pred [size={sz_gt}] shall have non-zero size in all dimensions'
    # value range
    if task_type in ['segmentation']:
        allowed_labels: torch.Tensor = torch.tensor([0, 1], dtype=torch.int)
        assert torch.all(torch.isin(y_pred, allowed_labels)), \
            f'y_pred [unique={list(y_pred.unique())}] shall only contain [0, 1]'
        assert torch.all(torch.isin(y_gt, allowed_labels)), \
            f'y_gt [unique={list(y_gt.unique())}] shall only contain [0, 1]'


# region Segmentation Region Overlapping metrics
class DiceScore(mm.DiceMetric):
    """
    Wrapper class for Dice metric.
    """

    def __init__(
            self,
            include_background: bool = False,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False,
            ignore_empty: bool = True,
            num_classes: Optional[int] = None,
            return_with_label: Union[bool, List[str]] = False,
    ) -> None:
        """
        Computes Dice score for a set of pairs of prediction-groundtruth labels. It supports single-channel label maps
        or multi-channel images with class segmentations per channel. This allows the computation for both multi-class
        and multi-label tasks.

        If either prediction ``y_pred`` or ground truth ``y`` have shape BCHW[D], it is expected that these represent one-
        hot segmentations for C number of classes. If either shape is B1HW[D], it is expected that these are label maps
        and the number of classes must be specified by the ``num_classes`` parameter. In either case for either inputs,
        this metric applies no activations and so non-binary values will produce unexpected results if this metric is used
        for binary overlap measurement (ie. either was expected to be one-hot formatted). Soft labels are thus permitted by
        this metric. Typically this implies that raw predictions from a network must first be activated and possibly made
        into label maps, eg. for a multi-class prediction tensor softmax and then argmax should be applied over the channel
        dimensions to produce a label map.

        The ``include_background`` parameter can be set to `False` to exclude the first category (channel index 0) which
        is by convention assumed to be background. If the non-background segmentations are small compared to the total
        image size they can get overwhelmed by the signal from the background. This assumes the shape of both prediction
        and ground truth is BCHW[D].

        The typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

        Further information can be found in the official
        `MONAI Dice Overview <https://github.com/Project-MONAI/tutorials/blob/main/modules/dice_loss_metric_notes.ipynb>`.

        Args:
            include_background: whether to include Dice computation on the first channel/category of the prediction and
                ground truth. Defaults to ``False``, use ``False`` to exclude the background class.
            reduction: defines mode of reduction to the metrics, this will only apply reduction on `not-nan` values. The
                available reduction modes are enumerated by :py:class:`monai.utils.enums.MetricReduction`. If "none", is
                selected, the metric will not do reduction.
            get_not_nans: whether to return the `not_nans` count. If True, aggregate() returns `(metric, not_nans)` where
                `not_nans` counts the number of valid values in the result, and will have the same shape.
            ignore_empty: whether to ignore empty ground truth cases during calculation. If `True`, the `NaN` value will be
                set for an empty ground truth cases, otherwise 1 will be set if the predictions of empty ground truth cases
                are also empty.
            num_classes: number of input channels (always including the background). When this is ``None``,
                ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
                single-channel class indices and the number of classes is not automatically inferred from data.
            return_with_label: whether to return the metrics with label, only works when reduction is "mean_batch".
                If `True`, use "label_{index}" as the key corresponding to C channels; if ``include_background`` is True,
                the index begins at "0", otherwise at "1". It can also take a list of label names.
                The outcome will then be returned as a dictionary.
        """
        super().__init__(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
            ignore_empty=ignore_empty,
            num_classes=num_classes,
            return_with_label=return_with_label
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


class GeneralizedDiceScore(mm.GeneralizedDiceScore):
    """
    Wrapper class for GeneralizedDice metric.
    """

    def __init__(
            self,
            include_background: bool = False,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            weight_type: Union[Weight, str] = Weight.SQUARE
    ) -> None:
        """
        Compute the Generalized Dice Score metric between tensors.

        This metric is the complement of the Generalized Dice Loss defined in:
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

        The inputs `y_pred` and `y` are expected to be one-hot, binarized batch-first tensors, i.e., NCHW[D].

        Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

        Args:
            include_background: Whether to include the background class (assumed to be in channel 0) in the
                score computation. Defaults to False.
            reduction: Define mode of reduction to the metrics. Available reduction modes:
                {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
                Default value is changed from `MetricReduction.MEAN_BATCH` to `MetricReduction.MEAN` in v1.5.0.
                Old versions computed `mean` when `mean_batch` was provided due to bug in reduction.
            weight_type: {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to transform
                ground truth volume into a weight factor. Defaults to ``"square"``.

        Raises:
            ValueError: When the `reduction` is not one of MetricReduction enum.
        """
        super().__init__(
            include_background=include_background,
            reduction=reduction,
            weight_type=weight_type
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> torch.Tensor:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


class MeanIoU(mm.MeanIoU):
    """
    Wrapper class for IoU metric.
    """

    def __init__(
            self,
            include_background: bool = False,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False,
            ignore_empty: bool = True
    ) -> None:
        """
        Compute average Intersection over Union (IoU) score between two tensors.
        It supports both multi-classes and multi-labels tasks.
        Input `y_pred` is compared with ground truth `y`.
        `y_pred` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
        in ``monai.transforms.post`` first to achieve binarized values.
        The `include_background` parameter can be set to ``False`` to exclude
        the first category (channel index 0) which is by convention assumed to be background. If the non-background
        segmentations are small compared to the total image size they can get overwhelmed by the signal from the
        background.
        `y_pred` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

        Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

        Args:
            include_background: whether to include IoU computation on the first channel of
                the predicted output. Defaults to ``False``.
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
                Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
            ignore_empty: whether to ignore empty ground truth cases during calculation.
                If `True`, NaN value will be set for empty ground truth cases.
                If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

        """
        super().__init__(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
            ignore_empty=ignore_empty
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


# endregion

# region Segmentation Contour Dist. metrics
class HausdorffDistance(mm.HausdorffDistanceMetric):
    """
    Wrapper class for Hausdorff Distance metric.
    """

    def __init__(
            self,
            include_background: bool = False,
            distance_metric: str = "euclidean",
            percentile: Optional[float] = None,
            directed: bool = False,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        """
            Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
            It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
            parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
            `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
            You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
            `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
            The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

            Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

            Args:
                include_background: whether to include distance computation on the first channel of
                    the predicted output. Defaults to ``False``.
                distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
                    the metric used to compute surface distance. Defaults to ``"euclidean"``.
                percentile: an optional float number between 0 and 100. If specified, the corresponding
                    percentile of the Hausdorff Distance rather than the maximum result will be achieved.
                    Defaults to ``None``.
                directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
                reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                    available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                    ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
                get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
                    Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
            """
        super().__init__(
            include_background=include_background,
            distance_metric=distance_metric,
            percentile=percentile,
            directed=directed,
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


class SurfaceDistance(mm.SurfaceDistanceMetric):
    """
    Wrapper class for Surface Distance metric.
    """

    def __init__(
            self,
            include_background: bool = False,
            symmetric: bool = False,
            distance_metric: str = "euclidean",
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        """
        Compute Surface Distance between two tensors. It can support both multi-classes and multi-labels tasks.
        It supports both symmetric and asymmetric surface distance calculation.
        Input `y_pred` is compared with ground truth `y`.
        `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
        You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
        `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

        Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

        Args:
            include_background: whether to include distance computation on the first channel of
                the predicted output. Defaults to ``False``.
            symmetric: whether to calculate the symmetric average surface distance between
                `seg_pred` and `seg_gt`. Defaults to ``False``.
            distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
                the metric used to compute surface distance. Defaults to ``"euclidean"``.
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
                Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

        """
        super().__init__(
            include_background=include_background,
            symmetric=symmetric,
            distance_metric=distance_metric,
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


# SurfaceDiceMetric
class NormalizedSurfaceDiceScore(mm.SurfaceDiceMetric):
    """
    Wrapper class for Normalized Surface Dice (NSD) Score metric.
    """

    def __init__(
            self,
            class_thresholds: List[float],
            include_background: bool = False,
            distance_metric: Literal["chessboard", "chessboard", "taxicab"] = "euclidean",
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False,
            use_subvoxels: bool = False
    ) -> None:
        super().__init__(
            class_thresholds=class_thresholds,
            include_background=include_background,
            distance_metric=distance_metric,
            reduction=reduction,
            get_not_nans=get_not_nans,
            use_subvoxels=use_subvoxels
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


# endregion

# region Image-to-Image metrics
class MeanSquaredError(mm.MSEMetric):
    """
    Wrapper class for Mean Squared Error (MSE) metric.
    """

    def __init__(
            self,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        r"""Compute Mean Squared Error between two tensors using function:

        .. math::
            \operatorname {MSE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i} \right)^{2}.

        More info: https://en.wikipedia.org/wiki/Mean_squared_error

        Input `y_pred` is compared with ground truth `y`.
        Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

        Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

        Args:
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
        """
        super().__init__(
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


class MeanAbsoluteError(mm.MAEMetric):
    """
    Wrapper class for Mean Absolute Error (MAE) metric.
    """

    def __init__(
            self,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        r"""Compute Mean Absolute Error between two tensors using function:

            .. math::
                \operatorname {MAE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left|y_i-\hat{y_i}\right|.

            More info: https://en.wikipedia.org/wiki/Mean_absolute_error

            Input `y_pred` is compared with ground truth `y`.
            Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

            Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

            Args:
                reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                    available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                    ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
                get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

        """
        super().__init__(
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


class RootMeanSquaredError(mm.RMSEMetric):
    """
    Wrapper class for Root Mean Squared Error (RMSE) metric.
    """

    def __init__(
            self,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        r"""Compute Root Mean Squared Error between two tensors using function:

            .. math::
                \operatorname {RMSE}\left(Y, \hat{Y}\right) ={ \sqrt{ \frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i}\right)^2 } } \
                = \sqrt {\operatorname{MSE}\left(Y, \hat{Y}\right)}.

            More info: https://en.wikipedia.org/wiki/Root-mean-square_deviation

            Input `y_pred` is compared with ground truth `y`.
            Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

            Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

            Args:
                reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                    available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                    ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
                get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

        """
        super().__init__(
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


# PSNRMetric
class PeakSignalToNoiseRatio(mm.PSNRMetric):
    """
    Wrapper class for Peak Signal To Noise Ratio (PSNR) metric.
    """

    def __init__(
            self,
            max_val: Union[int, float],
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        r"""Compute Peak Signal To Noise Ratio between two tensors using function:

        .. math::
            \operatorname{PSNR}\left(Y, \hat{Y}\right) = 20 \cdot \log_{10} \left({\mathit{MAX}}_Y\right) \
            -10 \cdot \log_{10}\left(\operatorname{MSE\left(Y, \hat{Y}\right)}\right)

        More info: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Help taken from:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py line 4139

        Input `y_pred` is compared with ground truth `y`.
        Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

        Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

        Args:
            max_val: The dynamic range of the images/volumes (i.e., the difference between the
                maximum and the minimum allowed values e.g. 255 for a uint8 image).
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
        """
        super().__init__(
            max_val=max_val,
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


# SSIMMetric
class StructuralSimilarityIndexMeasure(mm.SSIMMetric):
    """
    Wrapper class for Structural Similarity Index Measure (SSIM) metric.
    """

    def __init__(
            self,
            spatial_dims: int,
            data_range: float = 1.0,
            kernel_type: Union[KernelType, str] = KernelType.GAUSSIAN,
            win_size: Union[int, Sequence[int]] = 11,
            kernel_sigma: Union[float, Sequence[float]] = 1.5,
            k1: float = 0.01,
            k2: float = 0.03,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        r"""
        Computes the Structural Similarity Index Measure (SSIM).

        .. math::
            \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                    \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

        For more info, visit
            https://vicuesoft.com/glossary/term/ssim-ms-ssim/

        SSIM reference paper:
            Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
            similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

        Args:
            spatial_dims: number of spatial dimensions of the input images.
            data_range: value range of input images. (usually 1.0 or 255)
            kernel_type: type of kernel, can be "gaussian" or "uniform".
            win_size: window size of kernel
            kernel_sigma: standard deviation for Gaussian kernel.
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
            get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
        """
        super().__init__(
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            win_size=win_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


# MultiScaleSSIMMetric
class MultiScaleStructuralSimilarityIndexMeasure(mm.MultiScaleSSIMMetric):
    """
    Wrapper class for Multi-Scale Structural Similarity Index Measure (MS-SSIM) metric.
    """

    def __init__(
            self,
            spatial_dims: int,
            data_range: float = 1.0,
            kernel_type: Union[KernelType, str] = KernelType.GAUSSIAN,
            kernel_size: Union[int, Sequence[int]] = 11,
            kernel_sigma: Union[float, Sequence[float]] = 1.5,
            k1: float = 0.01,
            k2: float = 0.03,
            weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False
    ) -> None:
        """
        Computes the Multi-Scale Structural Similarity Index Measure (MS-SSIM).

        MS-SSIM reference paper:
            Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. "Multiscale structural
            similarity for image quality assessment." In The Thirty-Seventh Asilomar Conference
            on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). IEEE

        Args:
            spatial_dims: number of spatial dimensions of the input images.
            data_range: value range of input images. (usually 1.0 or 255)
            kernel_type: type of kernel, can be "gaussian" or "uniform".
            kernel_size: size of kernel
            kernel_sigma: standard deviation for Gaussian kernel.
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            weights: parameters for image similarity and contrast sensitivity at different resolution scores.
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
            get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
        """
        super().__init__(
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            weights=weights,
            reduction=reduction,
            get_not_nans=get_not_nans
        )

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


# endregion

# endregion

# region MONAI Alias
Dice = DiceScore
GDice = GeneralizedDiceScore
IoU = MeanIoU
HD = HausdorffDistance
SD = SurfaceDistance
NSD = NormalizedSurfaceDiceScore
MSE = MeanSquaredError
MAE = MeanAbsoluteError
RMSE = RootMeanSquaredError
PSNR = PeakSignalToNoiseRatio
SSIM = StructuralSimilarityIndexMeasure
MSSSIM = MultiScaleStructuralSimilarityIndexMeasure
# endregion
