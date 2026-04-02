# -*- coding: utf-8 -*-
"""
Metrics Configurers

This module provides wrapper classes for various metrics from the TorchMetrics and MONAI libraries,
enhancing the plot method to provide additional control functionality (for TorchMetrics).
"""
import numpy as np
import torch
from torch import Tensor
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

Inputs:
    [General]
    | ------------- | [Pred,GT] Shape when 'multidim_average'=            | -------------------------------------------------------------------------------------------------- |
    | Metric Type ↓ | 'global'                 | 'samplewise'             | [Pred,GT] dtype                                                                                    |
    | Binary        | [(N,*),         (N,*)  ] | [(N,+),         (N,+)  ] | [int(01-label)|float(logits, may apply sigmoid if out [0,1] range, shall apply thresholding), int] |
    | Multiclass    | [(N,*)|(N,C,*), (N,*)  ] | [(N,+)|(N,C,+), (N,+)  ] | [int(multi-label)|float(logits, shall apply argmax/softmax),                                  int] |
    | Multilabel    | [(N,C,*),       (N,C,*)] | [(N,C,+),       (N,C,+)] | [int(01-label)|float(logits, may apply sigmoid if out [0,1] range, shall apply thresholding), int] |
    * means zero or more spatial dims, + means one or more spatial dims
    
Outputs: 
    [General]
    dtype=torch.float
    | ------------------- | 'average'=                                                    |
    | 'multidim_average'↓ | None/'none'                  | 'micro' | 'macro' | 'weighted' |
    | 'global'            | ()  for Binary, others (C)   | ()      | ()      | ()         |
    | 'samplewise'        | (N) for Binary, others (N,C) | (N)     | (N)     | (N)        |
    
    [Special: StatScores]
    dtype=torch.int
    | --------------------- | 'average'=                                                        |
    | 'multidim_average'↓   | None/'none'                      | 'micro' | 'macro' | 'weighted' |
    | 'global'              | (5)   for Binary, others (C,5)   | (5)     | (5)     | (5)        |
    | 'samplewise'          | (N,5) for Binary, others (N,C,5) | (N,5)   | (N,5)   | (N,5)      |
    The 5 in shape (*,5) stands for [tp, fp, tn, fn, sup], sup=tp+fn 
    
    [Special: ConfusionMatrix]
    dtype=torch.int if not using normalization, otherwise torch.float
    | Metric Type                | Shape   |
    | Binary                     | (2,2)   |
    | Multiclass                 | (C,C)   |
    | Multilabel                 | (C,2,2) |
    Structure for Confusion Matrix(ces)
    Binary: (2,2)
    [[ TN FP ]
     [ FN TP ]]
    Multiclass: (C,C)
    [[ 0→0     0→1      ⋯    0→(C-1)     ]
     [ 1→0     1→1      ⋯    1→(C-1)     ]
     [ ⋮       ⋮        ⋱    ⋮           ]
     [ (C-1)→0 (C-1)→1  ⋯    (C-1)→(C-1) ]]
    Multilabel: (C,2,2)
    ┌                                               ┐
    │ ┌           ┐ ┌           ┐     ┌           ┐ │
    │ │ Label 0   │ │ Label 1   │     │ Label C-1 │ │
    │ │ [ TN FP ] │ │ [ TN FP ] │     │ [ TN FP ] │ │
    │ │ [ FN TP ] │ │ [ FN TP ] │ ... │ [ FN TP ] │ │
    │ └           ┘ └           ┘     └           ┘ │
    └                                               ┘
    
    [Special: PrecisionRecallCurve]
    The example usage and doc description for PRCurve returned (precision, recall, thresholds) metrics are 
    consistent, indicating that (n_thresholds+1) for precision and recall while (n_thresholds) for thresholds!
    For convenience, we slightly modifies the implementation of compute():
    · Pad an extra 1.0 threshold for the end point (precision=1.0, recall=0.0), making all list length as TH=n_thresholds+1.
    · Reverse point sequence by flipping precision, recall, thresholds each.
    Statements after modification are as follows:
    dtype=torch.float
    | Metric Type | fpr (Inc↑)    | tpr (Inc↑)    | thresholds (Dec↓) |
    | Binary      | (TH)          | (TH)          | (TH)              |
    | Multiclass  | C*(TH)|(C,TH) | C*(TH)|(C,TH) | C*(TH)|(TH)       |
    | Multilabel  | C*(TH)|(C,TH) | C*(TH)|(C,TH) | C*(TH)|(TH)       |
    Binary
        (precision, recall, thresholds)
        precision: Tensor(TH) with precision values
        recall: Tensor(TH) with recall values
        thresholds: Tensor(TH) with decreasing threshold values
        Note: The implementation both supports calculating the metric in a non-binned but accurate version and a binned 
        version that is less accurate but more memory efficient. Setting the thresholds argument to None will activate 
        the non-binned version that uses memory of size O(n_samples) whereas setting the thresholds argument to either an
        integer, list or a 1d tensor will use a binned version that uses memory of size O(n_thresholds) (constant memory).
    Multiclass
        (precision, recall, thresholds)
        precision:
            thresholds=None: List[C]→Tensor(TH) with precision values (length may differ between classes).
            thresholds={specified}: Tensor(C,TH) with precision values.
        recall:
            thresholds=None: List[C]→Tensor(TH) with recall values (length may differ between classes).
            thresholds={specified}: Tensor(C,TH) with recall values.
        thresholds:
            thresholds=None: List[C]→Tensor(TH) with decreasing threshold values (length may differ between classes).
            thresholds={specified}: Tensor(TH) with shared threshold values for all classes.
    Multilabel
        (precision, recall, thresholds)
        precision:
            thresholds=None: List[C]→Tensor(TH) with precision values (length may differ between labels).
            thresholds={specified}: Tensor(C,TH) with precision values.
        recall:
            thresholds=None: List[C]→Tensor(TH) with recall values (length may differ between labels)
            thresholds={specified}: Tensor(C,TH) with recall values.
        thresholds:
            thresholds=None: List[C]→Tensor(TH) with decreasing threshold values (length may differ between labels).
            thresholds={specified}: Tensor(TH) with shared threshold values for all labels.
             
    [Special: ROC]
    The example usage and doc description for returned (fpr, tpr, thresholds) metrics are inconsistent.
    This is probably a BUG!
    We assume the example usage is correct, and Tensor dim for fpr, tpr and thresholds 
    share the same length (denoted as TH), not (n_thresholds+1) for fpr and tpr while (n_thresholds) for thresholds!
    You may assume TH=n_thresholds+1.
    dtype=torch.float
    | Metric Type | fpr (Inc↑)    | tpr (Inc↑)    | thresholds (Dec↓) |
    | Binary      | (TH)          | (TH)          | (TH)              |
    | Multiclass  | C*(TH)|(C,TH) | C*(TH)|(C,TH) | C*(TH)|(TH)       |
    | Multilabel  | C*(TH)|(C,TH) | C*(TH)|(C,TH) | C*(TH)|(TH)       |
    Binary
        (fpr, tpr, thresholds)
        fpr: Tensor(TH) with false positive rate values
        fpr: Tensor(TH) with true positive rate values
        thresholds: Tensor(TH) with decreasing threshold values
        Note: The implementation both supports calculating the metric in a non-binned but accurate version and a binned 
        version that is less accurate but more memory efficient. Setting the thresholds argument to None will activate 
        the non-binned version that uses memory of size O(n_samples) whereas setting the thresholds argument to either an 
        integer, list or a 1d tensor will use a binned version that uses memory of size O(n_thresholds) (constant memory).
        The outputted thresholds will be in reversed order to ensure that they correspond to both fpr and tpr which are 
        sorted in reversed order during their calculation, such that they are monotome increasing.
    Multiclass
        (fpr, tpr, thresholds)
        fpr:
            thresholds=None: List[C]→Tensor(TH) with false positive rate values (length may differ between classes).
            thresholds={specified}: Tensor(C,TH) with false positive rate values.
        tpr:
            thresholds=None: List[C]→Tensor(TH) with true positive rate values (length may differ between classes).
            thresholds={specified}: Tensor(C,TH) with true positive rate values.
        thresholds:
            thresholds=None: List[C]→Tensor(TH) with decreasing threshold values (length may differ between classes).
            thresholds={specified}: Tensor(TH) with shared threshold values for all classes.
    Multilabel
        (fpr, tpr, thresholds)
        fpr:
            thresholds=None: List[C]→Tensor(TH) with false positive rate values (length may differ between labels).
            thresholds={specified}: Tensor(C,TH) with false positive rate values.
        tpr:
            thresholds=None: List[C]→Tensor(TH) with true positive rate values (length may differ between labels)
            thresholds={specified}: Tensor(C,TH) with true positive rate values.
        thresholds:
            thresholds=None: List[C]→Tensor(TH) with decreasing threshold values (length may differ between labels).
            thresholds={specified}: Tensor(TH) with shared threshold values for all labels.
"""


# region Torchmetrics
def assert_input_torchmetrics(
        task_type: Literal["binary", "multiclass", "multilabel"],
        y_pred: Tensor,
        y_gt: Tensor,
        multidim_average: Optional[Literal["global", "samplewise"]] = None,
        num_classes: Optional[int] = None
):
    def is_integer(dtype: torch.dtype) -> bool:
        try:
            torch.iinfo(dtype)  # success when dtype is integer
            return True
        except TypeError:
            return False

    assert task_type in ['binary', 'multiclass', 'multilabel'], f'task_type={task_type} is not supported'
    # type = Tensor
    assert isinstance(y_pred, Tensor), f'y_pred [type={type(y_pred)}] is not a Tensor'
    assert isinstance(y_gt, Tensor), f'y_pred [type={type(y_gt)}] is not a Tensor'
    # gt.dtype = int
    assert is_integer(y_gt.dtype), f'y_gt [dtype={y_gt.dtype}] shall be integer type'
    # size dim match
    if task_type in ['binary', 'multilabel']:
        assert y_pred.ndim == y_gt.ndim, f'y_pred [ndim={y_pred.ndim}] and y_gt [ndim={y_gt.ndim}] shall have the same ndim'
    else:  # multiclass
        assert y_pred.ndim == y_gt.ndim or y_pred.ndim == y_gt.ndim + 1, f'y_pred [ndim={y_pred.ndim}] and y_gt [ndim={y_gt.ndim}] shall have the same ndim or y_pred could have only one extra channel dim'
    # enough size dim and compatible pred.dtype
    extra_spatial_dim: int = 1 if multidim_average == 'samplewise' else 0
    if task_type in ['binary', 'multiclass']:
        assert y_gt.ndim > extra_spatial_dim, f'y_gt [ndim={y_gt.ndim}] shall be at least {extra_spatial_dim + 1} (N, {"+spatial" if extra_spatial_dim > 0 else "*spatial"})]'
        if task_type == 'multiclass' and y_pred.ndim == y_gt.ndim + 1:  # pred shall be logits (N,C,...)
            assert y_pred.ndim > extra_spatial_dim + 1, f'y_pred [ndim={y_pred.ndim}] shall be at least {extra_spatial_dim + 2} (N, C, {"+spatial" if extra_spatial_dim > 0 else "*spatial"})]'
            assert torch.is_floating_point(
                y_pred), f'y_pred [dtype={y_pred.dtype}] shall floating point type in {task_type} task while representing as logits (N, C, {"+spatial" if extra_spatial_dim > 0 else "*spatial"})'
        else:  # pred shall be label (N,...)
            assert y_pred.ndim > extra_spatial_dim, f'y_pred [ndim={y_pred.ndim}] shall be at least {extra_spatial_dim + 1} (N, {"+spatial" if extra_spatial_dim > 0 else "*spatial"})]'
    else:  # multilabel
        assert y_pred.ndim > extra_spatial_dim, f'y_pred [ndim={y_pred.ndim}] shall be at least {extra_spatial_dim + 2} (N, C, {"+spatial" if extra_spatial_dim > 0 else "*spatial"})]'
        assert y_gt.ndim > extra_spatial_dim, f'y_gt [ndim={y_gt.ndim}] shall be at least {extra_spatial_dim + 2} (N, {"+spatial" if extra_spatial_dim > 0 else "*spatial"})]'

    sz_pred: Tuple[int, ...] = tuple(y_pred.size())
    sz_gt: Tuple[int, ...] = tuple(y_gt.size())
    # size non-zero
    assert 0 not in sz_pred, f'y_pred [size={sz_pred}] shall have non-zero size in all dimensions'
    assert 0 not in sz_gt, f'y_pred [size={sz_gt}] shall have non-zero size in all dimensions'
    # size match
    assert sz_pred[0] == sz_gt[0], f'y_pred [N={sz_pred[0]}] and y_gt [N={sz_gt[0]}] shall have the same size'
    if task_type in ['binary', 'multilabel'] or y_pred.ndim == y_gt.ndim:
        assert sz_pred == sz_gt, f'y_pred [size={sz_pred}] and y_gt [size={sz_gt}] shall have the same size'
    else:  # multiclass
        assert len(sz_pred) < 3 or sz_pred[2:] == sz_gt[1:], \
            f'y_pred [spatial_size={sz_pred[2:]}] and y_gt [spatial_size={sz_gt[1:]}] shall have the same size'

    # value range
    if task_type in ['binary', 'multilabel']:
        allowed_labels: Tensor = torch.tensor([0, 1], dtype=y_gt.dtype)
        assert torch.all(torch.isin(y_gt, allowed_labels)), \
            f'y_gt [unique={tuple(y_gt.unique())}] shall only contain [0, 1]'
        if is_integer(y_pred.dtype):
            assert torch.all(torch.isin(y_pred, allowed_labels)), \
                f'y_pred [unique={tuple(y_pred.unique())}] shall only contain [0, 1]'
    elif task_type == "multiclass":
        assert num_classes is not None, 'you shall specify num_classes for multiclass assertion'
        label_list: List[int] = list(range(num_classes))
        allowed_labels: Tensor = torch.tensor(label_list, dtype=torch.int)
        assert torch.all(torch.isin(y_gt, allowed_labels)), \
            f'y_gt [unique={tuple(y_gt.unique())}] shall only contain {label_list}'
        if is_integer(y_pred.dtype):
            assert torch.all(torch.isin(y_pred, allowed_labels)), \
                f'y_pred [unique={tuple(y_pred.unique())}] shall only contain {label_list}'


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
            average: Literal["micro"] = "micro",  # Always micro
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
    Confusion Matrix
    [[ TN FP ]
     [ FN TP ]]
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
    r"""
    Wrapper class for ConfusionMatrix metric, enhanced with additional plot control functionality.
    Confusion Matrix
    [[ 0→0     0→1      ⋯    0→(C-1)     ]
     [ 1→0     1→1      ⋯    1→(C-1)     ]
     [ ⋮       ⋮        ⋱    ⋮           ]
     [ (C-1)→0 (C-1)→1  ⋯    (C-1)→(C-1) ]]

    Compute the `confusion matrix`_ for multiclass tasks.

    The confusion matrix :math:`C` is constructed such that :math:`C_{i, j}` is equal to the number of observations
    known to be in class :math:`i` but predicted to be in class :math:`j`. Thus row indices of the confusion matrix
    correspond to the true class labels and column indices correspond to the predicted class labels.

    For multiclass tasks, the confusion matrix is a NxN matrix, where:

    - :math:`C_{i, i}` represents the number of true positives for class :math:`i`
    - :math:`\sum_{j=1, j\neq i}^N C_{i, j}` represents the number of false negatives for class :math:`i`
    - :math:`\sum_{j=1, j\neq i}^N C_{j, i}` represents the number of false positives for class :math:`i`
    - the sum of the remaining cells in the matrix represents the number of true negatives for class :math:`i`

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``confusion_matrix``: [num_classes, num_classes] matrix

    Args:
        num_classes: Integer specifying the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassConfusionMatrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassConfusionMatrix(num_classes=3)
        >>> metric(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassConfusionMatrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassConfusionMatrix(num_classes=3)
        >>> metric(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])
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
    num_labels=C Separated Confusion Matrices
    ┌                                               ┐
    │ ┌           ┐ ┌           ┐     ┌           ┐ │
    │ │ Label 0   │ │ Label 1   │     │ Label C-1 │ │
    │ │ [ TN FP ] │ │ [ TN FP ] │     │ [ TN FP ] │ │
    │ │ [ FN TP ] │ │ [ FN TP ] │ ... │ [ FN TP ] │ │
    │ └           ┘ └           ┘     └           ┘ │
    └                                               ┘
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

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        """Compute metric.
        This wrapped implementation could align thresholds size with precision and recall,
        ensuring they all have the same size (n_thresholds+1) and thresholds in decreasing order
        to meet PRCurve left-to-right (x=recall) nature.
        """
        precision, recall, thresholds = super().compute()
        # precision ↑ recall ↓ thresholds ↑
        if thresholds.size(0) != precision.size(0) and thresholds.size(0) + 1 != precision.size(0):
            raise ValueError('Can not manage size for thresholds')
        # Cat redundant ones to align size [*thresh, 1.0]
        thresholds = torch.cat([thresholds, torch.ones(1, dtype=thresholds.dtype, device=thresholds.device)])
        # Flip all metrics, now precision ↓ recall ↑ thresholds ↓
        precision = precision.flip(0)
        recall = recall.flip(0)
        thresholds = thresholds.flip(0)
        return precision, recall, thresholds

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

    def compute(self) -> Union[tuple[Tensor, Tensor, Tensor], tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric.
        This wrapped implementation could align thresholds size with precision and recall,
        ensuring they all have the same size (n_thresholds+1) and thresholds in decreasing order
        to meet PRCurve left-to-right (x=recall) nature.
        """
        # precision ↑ recall ↓ thresholds ↑
        precision: Union[Tensor, List[Tensor]]
        recall: Union[Tensor, List[Tensor]]
        thresholds: Union[Tensor, List[Tensor]]
        precision, recall, thresholds = super().compute()
        if isinstance(precision, torch.Tensor):  # assume they are all Tensor(C,n_thresholds)
            if thresholds.size(0) != precision.size(0) and thresholds.size(0) + 1 != recall.size(0):
                raise ValueError('Can not manage size for thresholds')
            # Cat redundant ones to align size [*thresh, 1.0]
            thresholds: Tensor = torch.cat(
                [thresholds, torch.ones(1, dtype=thresholds.dtype, device=thresholds.device)])
            # Flip all metrics, now precision ↓ recall ↑ thresholds ↓
            precision: Tensor = precision.flip(0)
            recall: Tensor = recall.flip(0)
            thresholds: Tensor = thresholds.flip(0)
        else:  # assume they are all [C]*Tensor(n_thresholds[+1])
            for idx in range(len(thresholds)):
                prec: Tensor
                rec: Tensor
                thresh: Tensor
                prec, rec, thresh = precision[idx], recall[idx], thresholds[idx]
                if thresh.size(0) != prec.size(0) and thresh.size(0) + 1 != prec.size(0):
                    raise ValueError('Can not manage size for thresholds')
                # Cat redundant ones to align size [*thresh, 1.0]
                thresh = torch.cat([thresh, torch.ones(1, dtype=thresh.dtype, device=thresh.device)])
                # Flip all metrics, now precision ↓ recall ↑ thresholds ↓
                precision[idx] = prec.flip(0)
                recall[idx] = rec.flip(0)
                thresholds[idx] = thresh.flip(0)
        return precision, recall, thresholds

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

    def compute(self) -> Union[tuple[Tensor, Tensor, Tensor], tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric.
        This wrapped implementation could align thresholds size with precision and recall,
        ensuring they all have the same size (n_thresholds+1) and thresholds in decreasing order
        to meet PRCurve left-to-right (x=recall) nature.
        """
        # precision ↑ recall ↓ thresholds ↑
        precision: Union[Tensor, List[Tensor]]
        recall: Union[Tensor, List[Tensor]]
        thresholds: Union[Tensor, List[Tensor]]
        precision, recall, thresholds = super().compute()
        if isinstance(precision, torch.Tensor):  # assume they are all Tensor(C,n_thresholds)
            if thresholds.size(0) != precision.size(0) and thresholds.size(0) + 1 != recall.size(0):
                raise ValueError('Can not manage size for thresholds')
            # Cat redundant ones to align size [*thresh, 1.0]
            thresholds: Tensor = torch.cat(
                [thresholds, torch.ones(1, dtype=thresholds.dtype, device=thresholds.device)])
            # Flip all metrics, now precision ↓ recall ↑ thresholds ↓
            precision: Tensor = precision.flip(0)
            recall: Tensor = recall.flip(0)
            thresholds: Tensor = thresholds.flip(0)
        else:  # assume they are all [C]*Tensor(n_thresholds[+1])
            for idx in range(len(thresholds)):
                prec: Tensor
                rec: Tensor
                thresh: Tensor
                prec, rec, thresh = precision[idx], recall[idx], thresholds[idx]
                if thresh.size(0) != prec.size(0) and thresh.size(0) + 1 != prec.size(0):
                    raise ValueError('Can not manage size for thresholds')
                # Cat redundant ones to align size [*thresh, 1.0]
                thresh = torch.cat([thresh, torch.ones(1, dtype=thresh.dtype, device=thresh.device)])
                # Flip all metrics, now precision ↓ recall ↑ thresholds ↓
                precision[idx] = prec.flip(0)
                recall[idx] = rec.flip(0)
                thresholds[idx] = thresh.flip(0)
        return precision, recall, thresholds

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
MCAUROC = MulticlassAUROC
MLAUROC = MultilabelAUROC
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
MCRECALL = MulticlassRecall
MLREC = MultilabelRecall
BSPE = BinarySpecificity
MCSPEC = MulticlassSpecificity
MLSPE = MultilabelSpecificity
BROC = BinaryROC
MCROC = MulticlassROC
MLROC = MultilabelROC
# endregion


"""
MONAI metrics IO Requirements

To avoid unnecessary trouble, we require all inputs shall be multi-channels

Inputs:
    pred=(B,C,*Spatial) gt=(B,C,*Spatial) 
    Segmentation: dtype=torch.int [0,1]
    Image-To-Image: dtype=torch.float
    
Outputs: 
    dtype=torch.float
    | Metric Type    | Reduction                                                                   |
    | ↓              | none  | mean  | sum   | mean_batch | sum_batch | mean_channel | sum_channel |
    | Segmentation   | (B,C) | ()    | ()    | (C)        | (C)       | (B)          | (B)         |
    | Image-To-Image | (B)   | ()    | ()    | ()         | ()        | (B)          | (B)         |
"""


# region MONAI
def assert_input_monaimetrics(
        task_type: Literal["segmentation", "img2img"],
        y_pred: Tensor,
        y_gt: Tensor
):
    def is_integer(dtype: torch.dtype) -> bool:
        try:
            torch.iinfo(dtype)  # success when dtype is integer
            return True
        except TypeError:
            return False

    assert task_type in ['segmentation', 'img2img'], f'task_type={task_type} is not supported'
    # type = Tensor
    assert isinstance(y_pred, Tensor), f'y_pred [type={type(y_pred)}] is not a Tensor'
    assert isinstance(y_gt, Tensor), f'y_pred [type={type(y_gt)}] is not a Tensor'
    # dtype = int (segmentation) |  float (img2img)
    if task_type == "segmentation":
        assert is_integer(y_pred.dtype), f'y_pred [dtype={y_pred.dtype}] shall be integer type'
        assert is_integer(y_gt.dtype), f'y_gt [dtype={y_gt.dtype}] shall be integer type'
    else:  # task_type == "img2img":
        assert torch.is_floating_point(y_pred), f'y_pred [dtype={y_pred.dtype}] shall be floating point type'
        assert torch.is_floating_point(y_gt), f'y_gt [dtype={y_gt.dtype}] shall be floating point type'
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
        allowed_labels: Tensor = torch.tensor([0, 1], dtype=torch.int)
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
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
    ) -> Tensor:
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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


class NormalizedSurfaceDiceScore(mm.SurfaceDiceMetric):
    """
    Wrapper class for Normalized Surface Dice (NSD) Score metric.
    """

    def __init__(
            self,
            class_thresholds: List[float],
            include_background: bool = False,
            distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
            get_not_nans: bool = False,
            use_subvoxels: bool = False
    ) -> None:
        """
        Computes the Normalized Surface Dice (NSD) for each batch sample and class of
        predicted segmentations `y_pred` and corresponding reference segmentations `y` according to equation :eq:`nsd`.
        This implementation is based on https://arxiv.org/abs/2111.05408 and supports 2D and 3D images.
        Be aware that by default (`use_subvoxels=False`), the computation of boundaries is different from DeepMind's
        implementation https://github.com/deepmind/surface-distance.
        In this implementation, the length/area of a segmentation boundary is
        interpreted as the number of its edge pixels. In DeepMind's implementation, the length of a segmentation boundary
        depends on the local neighborhood (cf. https://arxiv.org/abs/1809.04430).
        This issue is discussed here: https://github.com/Project-MONAI/MONAI/issues/4103.

        The class- and batch sample-wise NSD values can be aggregated with the function `aggregate`.

        Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

        Args:
            class_thresholds: List of class-specific thresholds.
                The thresholds relate to the acceptable amount of deviation in the segmentation boundary in pixels.
                Each threshold needs to be a finite, non-negative number.
            include_background: Whether to include NSD computation on the first channel of the predicted output.
                Defaults to ``False``.
            distance_metric: The metric used to compute surface distances.
                One of [``"euclidean"``, ``"chessboard"``, ``"taxicab"``].
                Defaults to ``"euclidean"``.
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            get_not_nans: whether to return the `not_nans` count.
                Defaults to ``False``.
                `not_nans` is the number of batch samples for which not all class-specific NSD values were nan values.
                If set to ``True``, the function `aggregate` will return both the aggregated NSD and the `not_nans` count.
                If set to ``False``, `aggregate` will only return the aggregated NSD.
            use_subvoxels: Whether to use subvoxel distances. Defaults to ``False``.
            """
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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        ret = super().__call__(y_pred=y_pred, y=y, **kwargs)
        ret = super().aggregate().squeeze()
        return ret


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
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
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

# region Efficiency metrics
import abc
from datetime import datetime, timedelta


class BaseEfficiencyMetric(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> Any:
        pass


class VoxelProcessingPerSecond(BaseEfficiencyMetric):
    def __init__(self, init_datetime: Optional[datetime] = None) -> None:
        self.time_checkpoint: Optional[datetime] = init_datetime

    def __call__(
            self,
            volume: Optional[Tensor] = None,
            time_point: Optional[datetime] = None  # If not specified, record now time
    ) -> Tensor:
        vps: float = 0.
        now_datetime = datetime.now() if time_point is None else time_point
        if volume is not None and self.time_checkpoint is not None:
            time_delta: timedelta = now_datetime - self.time_checkpoint
            # Calculation
            seconds: float = time_delta.total_seconds()
            vps: float = seconds / float(np.prod(volume.size()))
        self.time_checkpoint = now_datetime  # Record time point, maybe another start later
        return torch.tensor(vps, dtype=torch.float, device=volume.device if volume is not None else torch.device('cpu'))


# endregion

# region Efficiency Alias
VPS = VoxelProcessingPerSecond
# endregion
