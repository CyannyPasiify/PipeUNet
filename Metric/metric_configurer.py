# -*- coding: utf-8 -*-
"""
Metrics Configurers

This module provides wrapper classes for various metrics from the TorchMetrics and MONAI libraries,
enhancing the plot method to provide additional control functionality (for TorchMetrics).
"""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchmetrics
import torchmetrics.classification
from typing import TypeVar, Optional, Dict, Any, Union, List, Tuple, Literal, Sequence, Type, cast
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import monai.metrics as mm
from monai.config import TensorOrList
from monai.metrics.regression import KernelType
from monai.utils import MetricReduction, Weight
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing_extensions import override
from wandb import Config

T = TypeVar("T")
TLSeq = Union[List[T], Tuple[T, ...]]

SupportedMetric = Union[torchmetrics.Metric, mm.Metric]

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


@dataclass
class ConfigMetricBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "metric")

    def _assert_init_essentials(
            self,
            *args,
            **kwargs
    ) -> None:
        if self.is_ready(): return
        self.init_essentials(*args, **kwargs)

    @abstractmethod
    def init_essentials(
            self,
            *args,
            **kwargs
    ) -> 'ConfigMetricBase':
        self.metric: SupportedMetric = torchmetrics.MaxMetric()  # Just placeholder
        return self

    def __call__(
            self,
            *args,
            **kwargs
    ) -> torch.Tensor:
        self._assert_init_essentials()
        return self.metric(*args, **kwargs)

    def to(self, *args, **kwargs) -> 'ConfigMetricBase':
        self._assert_init_essentials()
        return self

    def get_metric_operator(self, *args, **kwargs) -> SupportedMetric:
        self._assert_init_essentials(*args, **kwargs)
        return self.metric


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


@dataclass
class ConfigMetricTorch(ConfigMetricBase, metaclass=ABCMeta):
    @override
    def __call__(
            self,
            preds: Tensor,
            target: Tensor,
            *args,
            **kwargs
    ) -> Tensor:
        self._assert_init_essentials()
        ret = self.metric(preds, target, *args, **kwargs)
        self.metric.reset()
        return ret

    def plot(
            self,
            *args,
            **kwargs
    ) -> Tuple[Figure, Axes]:
        self._assert_init_essentials()
        return self.metric.plot(*args, **kwargs)

    def to(self, *args, **kwargs) -> 'ConfigMetricBase':
        r"""Move and/or cast the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self

        Examples::

            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

            >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.3741+0.j,  0.2382+0.j],
                    [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
            >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
            tensor([[0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

        """
        self._assert_init_essentials()
        self.metric.to(*args, **kwargs)
        return self


@dataclass
class ConfigMetricTorchScalar(ConfigMetricTorch, metaclass=ABCMeta):
    @override
    def plot(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            add_data_labels: bool = True,
            figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        self._assert_init_essentials()
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = self.metric.plot(ax=ax)

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
                print(f"Error adding data labels to {self.__class__.__name__}: {e}")

        fig.tight_layout()
        return fig, ax


@dataclass
class ConfigMetricBinaryStatScores(ConfigMetricTorch):
    """
    Wrapper class for binary classification statistical metrics.
    """
    threshold: float = 0.5
    multidim_average: Literal["global", "samplewise"] = "global"
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryStatScores':
        self.metric = torchmetrics.classification.BinaryStatScores(
            threshold=self.threshold,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassStatScores(ConfigMetricTorch):
    """
    Wrapper class for multiclass classification statistical metrics.
    """
    num_classes: int = 2
    top_k: int = 1
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro"
    multidim_average: Literal["global", "samplewise"] = "global"
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassStatScores':
        self.metric = torchmetrics.classification.MulticlassStatScores(
            num_classes=self.num_classes,
            top_k=self.top_k,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelStatScores(ConfigMetricTorch):
    """
    Wrapper class for multilabel classification statistical metrics.
    """
    num_labels: int = 2
    threshold: float = 0.5
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro"
    multidim_average: Literal["global", "samplewise"] = "global"
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelStatScores':
        self.metric = torchmetrics.classification.MultilabelStatScores(
            num_labels=self.num_labels,
            threshold=self.threshold,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinaryAccuracy(ConfigMetricBinaryStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for binary Accuracy metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryAccuracy':
        self.metric = torchmetrics.classification.BinaryAccuracy(
            threshold=self.threshold,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs,
        )
        return self


@dataclass
class ConfigMetricMulticlassAccuracy(ConfigMetricMulticlassStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multiclass Accuracy metric, enhanced with additional plot control functionality.
    For multiclass problems, average='micro' should always be set to get the correct Accuracy value.
    When MulticlassAccuracy in TorchMetrics specifies average='none' or 'macro',
    it is equivalent to Recall.
    """
    average: Literal["micro"] = "micro"  # Always micro

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassAccuracy':
        self.metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes,
            top_k=self.top_k,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelAccuracy(ConfigMetricMultilabelStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multilabel Accuracy metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelAccuracy':
        self.metric = torchmetrics.classification.MultilabelAccuracy(
            num_labels=self.num_labels,
            threshold=self.threshold,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


class _BetterBinaryPrecisionRecallCurve(torchmetrics.classification.BinaryPrecisionRecallCurve):
    """
    Better implemented class for binary classification PrecisionRecallCurve metric, enhanced with additional plot control functionality.
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


class _BetterMulticlassPrecisionRecallCurve(torchmetrics.classification.MulticlassPrecisionRecallCurve):
    """
    Better implemented class for multiclass PrecisionRecallCurve metric, enhanced with additional plot control functionality.
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


class _BetterMultilabelPrecisionRecallCurve(torchmetrics.classification.MultilabelPrecisionRecallCurve):
    """
    Better implemented class for multilabel PrecisionRecallCurve metric, enhanced with additional plot control functionality.
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


class _BetterBinaryROC(torchmetrics.classification.BinaryROC):
    """
    Better implemented class for binary classification ROC metric, enhanced with additional plot control functionality.
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


class _BetterMulticlassROC(torchmetrics.classification.MulticlassROC):
    """
    Better implemented class for multiclass ROC metric, enhanced with additional plot control functionality.
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


class _BetterMultilabelROC(torchmetrics.classification.MultilabelROC):
    """
    Better implemented class for multilabel ROC metric, enhanced with additional plot control functionality.
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


@dataclass
class ConfigMetricBinaryPrecisionRecallCurve(ConfigMetricTorch):
    """
    Wrapper class for binary classification PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """
    thresholds: Optional[Union[int, List[float], Tensor]] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True
    normalization: Optional[Literal["sigmoid", "softmax"]] = "sigmoid"
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryPrecisionRecallCurve':
        self.metric = _BetterBinaryPrecisionRecallCurve(
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            normalization=self.normalization,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassPrecisionRecallCurve(ConfigMetricTorch):
    """
    Wrapper class for multiclass PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """
    num_classes: int = 2
    thresholds: Optional[Union[int, List[float], Tensor]] = None
    average: Optional[Literal["micro", "macro"]] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassPrecisionRecallCurve':
        self.metric = _BetterMulticlassPrecisionRecallCurve(
            num_classes=self.num_classes,
            thresholds=self.thresholds,
            average=self.average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelPrecisionRecallCurve(ConfigMetricTorch):
    """
    Wrapper class for multilabel PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """
    num_labels: int = 2
    thresholds: Optional[Union[int, List[float], Tensor]] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelPrecisionRecallCurve':
        self.metric = _BetterMultilabelPrecisionRecallCurve(
            num_labels=self.num_labels,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinaryROC(ConfigMetricTorch):
    """
    Wrapper class for binary classification ROC metric, enhanced with additional plot control functionality.
    """
    thresholds: Optional[Union[int, List[float], Tensor]] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True
    normalization: Optional[Literal["sigmoid", "softmax"]] = "sigmoid"
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryROC':
        self.metric = _BetterBinaryROC(
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            normalization=self.normalization,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassROC(ConfigMetricTorch):
    """
    Wrapper class for multiclass ROC metric, enhanced with additional plot control functionality.
    """
    num_classes: int = 2
    thresholds: Optional[Union[int, List[float], Tensor]] = None
    average: Optional[Literal["micro", "macro"]] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassROC':
        self.metric = _BetterMulticlassROC(
            num_classes=self.num_classes,
            thresholds=self.thresholds,
            average=self.average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelROC(ConfigMetricTorch):
    """
    Wrapper class for multilabel ROC metric, enhanced with additional plot control functionality.
    """
    num_labels: int = 2
    thresholds: Optional[Union[int, List[float], Tensor]] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelROC':
        self.metric = _BetterMultilabelROC(
            num_labels=self.num_labels,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinaryAUROC(ConfigMetricBinaryROC, ConfigMetricTorchScalar):
    """
    Wrapper class for binary AUROC metric, enhanced with additional plot control functionality.
    """
    max_fpr: Optional[float] = None
    thresholds: Optional[Union[int, list[float], Tensor]] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryAUROC':
        self.metric = torchmetrics.classification.BinaryAUROC(
            max_fpr=self.max_fpr,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassAUROC(ConfigMetricMulticlassROC, ConfigMetricTorchScalar):
    """
    Wrapper class for multiclass AUROC metric, enhanced with additional plot control functionality.
    """
    average: Optional[Literal["macro", "weighted", "none"]] = "macro"

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassAUROC':
        self.metric = torchmetrics.classification.MulticlassAUROC(
            num_classes=self.num_classes,
            average=self.average,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelAUROC(ConfigMetricMultilabelROC, ConfigMetricTorchScalar):
    """
    Wrapper class for multilabel AUROC metric, enhanced with additional plot control functionality.
    """
    average: Optional[Literal["macro", "weighted", "none"]] = "macro"

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelAUROC':
        self.metric = torchmetrics.classification.MultilabelAUROC(
            num_labels=self.num_labels,
            average=self.average,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinaryAveragePrecision(ConfigMetricBinaryPrecisionRecallCurve, ConfigMetricTorchScalar):
    """
    Wrapper class for binary AveragePrecision metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryAveragePrecision':
        self.metric = torchmetrics.classification.BinaryAveragePrecision(
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            normalization=self.normalization,
            **self.kwargs,
        )
        return self


@dataclass
class ConfigMetricMulticlassAveragePrecision(ConfigMetricMulticlassPrecisionRecallCurve, ConfigMetricTorchScalar):
    """
    Wrapper class for multiclass AveragePrecision metric, enhanced with additional plot control functionality.
    """
    average: Optional[Literal["macro", "weighted", "none"]] = "macro"

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassAveragePrecision':
        self.metric = torchmetrics.classification.MulticlassAveragePrecision(
            num_classes=self.num_classes,
            average=self.average,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelAveragePrecision(ConfigMetricMultilabelPrecisionRecallCurve, ConfigMetricTorchScalar):
    """
    Wrapper class for multilabel AveragePrecision metric, enhanced with additional plot control functionality.
    """
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro"

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelAveragePrecision':
        self.metric = torchmetrics.classification.MultilabelAveragePrecision(
            num_labels=self.num_labels,
            average=self.average,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


class _BetterBinaryConfusionMatrix(torchmetrics.classification.BinaryConfusionMatrix):
    """
    Wrapper class for binary ConfusionMatrix metric, enhanced with additional plot control functionality.
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


class _BetterMulticlassConfusionMatrix(torchmetrics.classification.MulticlassConfusionMatrix):
    r"""
    Wrapper class for multiclass ConfusionMatrix metric, enhanced with additional plot control functionality.
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

    Attributes:
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


class _BetterMultilabelConfusionMatrix(torchmetrics.classification.MultilabelConfusionMatrix):
    """
    Wrapper class for multilabel ConfusionMatrix metric, enhanced with additional plot control functionality.
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


@dataclass
class ConfigMetricBinaryConfusionMatrix(ConfigMetricTorch):
    """
    Wrapper class for binary ConfusionMatrix metric, enhanced with additional plot control functionality.
    Confusion Matrix
    [[ TN FP ]
     [ FN TP ]]
    """
    threshold: float = 0.5
    ignore_index: Optional[int] = None
    normalize: Optional[Literal["true", "pred", "all", "none"]] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryConfusionMatrix':
        self.metric = _BetterBinaryConfusionMatrix(
            threshold=self.threshold,
            ignore_index=self.ignore_index,
            normalize=self.normalize,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassConfusionMatrix(ConfigMetricTorch):
    r"""
    Wrapper class for multiclass ConfusionMatrix metric, enhanced with additional plot control functionality.
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

    Attributes:
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
    """
    num_classes: int = 2
    ignore_index: Optional[int] = None
    normalize: Optional[Literal["none", "true", "pred", "all"]] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassConfusionMatrix':
        self.metric = _BetterMulticlassConfusionMatrix(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            normalize=self.normalize,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelConfusionMatrix(ConfigMetricTorch):
    """
    Wrapper class for multilabel ConfusionMatrix metric, enhanced with additional plot control functionality.
    num_labels=C Separated Confusion Matrices
    ┌                                               ┐
    │ ┌           ┐ ┌           ┐     ┌           ┐ │
    │ │ Label 0   │ │ Label 1   │     │ Label C-1 │ │
    │ │ [ TN FP ] │ │ [ TN FP ] │     │ [ TN FP ] │ │
    │ │ [ FN TP ] │ │ [ FN TP ] │ ... │ [ FN TP ] │ │
    │ └           ┘ └           ┘     └           ┘ │
    └                                               ┘
    """
    num_labels: int = 2
    threshold: float = 0.5
    ignore_index: Optional[int] = None
    normalize: Optional[Literal["none", "true", "pred", "all"]] = None
    validate_args: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelConfusionMatrix':
        self.metric = _BetterMultilabelConfusionMatrix(
            num_labels=self.num_labels,
            threshold=self.threshold,
            ignore_index=self.ignore_index,
            normalize=self.normalize,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinaryF1Score(ConfigMetricBinaryStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for binary F1Score metric, enhanced with additional plot control functionality.
    """
    zero_division: float = 0

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryF1Score':
        self.metric = torchmetrics.classification.BinaryF1Score(
            threshold=self.threshold,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            zero_division=self.zero_division,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassF1Score(ConfigMetricMulticlassStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multiclass F1Score metric, enhanced with additional plot control functionality.
    """
    zero_division: float = 0

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassF1Score':
        self.metric = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.num_classes,
            top_k=self.top_k,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            zero_division=self.zero_division,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelF1Score(ConfigMetricMultilabelStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multilabel F1Score metric, enhanced with additional plot control functionality.
    """
    zero_division: float = 0

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelF1Score':
        self.metric = torchmetrics.classification.MultilabelF1Score(
            num_labels=self.num_labels,
            threshold=self.threshold,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            zero_division=self.zero_division,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinaryPrecision(ConfigMetricBinaryStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for binary Precision metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryPrecision':
        self.metric = torchmetrics.classification.BinaryPrecision(
            threshold=self.threshold,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassPrecision(ConfigMetricMulticlassStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multiclass Precision metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassPrecision':
        self.metric = torchmetrics.classification.MulticlassPrecision(
            num_classes=self.num_classes,
            top_k=self.top_k,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelPrecision(ConfigMetricMultilabelStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multilabel Precision metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelPrecision':
        self.metric = torchmetrics.classification.MultilabelPrecision(
            num_labels=self.num_labels,
            threshold=self.threshold,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinaryRecall(ConfigMetricBinaryStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for binary Recall metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricBinaryRecall':
        self.metric = torchmetrics.classification.BinaryRecall(
            threshold=self.threshold,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassRecall(ConfigMetricMulticlassStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multiclass Recall metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassRecall':
        self.metric = torchmetrics.classification.MulticlassRecall(
            num_classes=self.num_classes,
            top_k=self.top_k,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelRecall(ConfigMetricMultilabelStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multilabel Recall metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelRecall':
        self.metric = torchmetrics.classification.MultilabelRecall(
            num_labels=self.num_labels,
            threshold=self.threshold,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricBinarySpecificity(ConfigMetricBinaryStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for binary Specificity metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricBinarySpecificity':
        self.metric = torchmetrics.classification.BinarySpecificity(
            threshold=self.threshold,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMulticlassSpecificity(ConfigMetricMulticlassStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multiclass Specificity metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricMulticlassSpecificity':
        self.metric = torchmetrics.classification.MulticlassSpecificity(
            num_classes=self.num_classes,
            top_k=self.top_k,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


@dataclass
class ConfigMetricMultilabelSpecificity(ConfigMetricMultilabelStatScores, ConfigMetricTorchScalar):
    """
    Wrapper class for multilabel Specificity metric, enhanced with additional plot control functionality.
    """

    @override
    def init_essentials(self) -> 'ConfigMetricMultilabelSpecificity':
        self.metric = torchmetrics.classification.MultilabelSpecificity(
            num_labels=self.num_labels,
            threshold=self.threshold,
            average=self.average,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
            validate_args=self.validate_args,
            **self.kwargs
        )
        return self


# endregion

# region Torchmetrics Alias
BSS = ConfigMetricBinaryStatScores
MCSS = ConfigMetricMulticlassStatScores
MLSS = ConfigMetricMultilabelStatScores
BACC = ConfigMetricBinaryAccuracy
MCACC = ConfigMetricMulticlassAccuracy
MLACC = ConfigMetricMultilabelAccuracy
BPRC = ConfigMetricBinaryPrecisionRecallCurve
MCPRC = ConfigMetricMulticlassPrecisionRecallCurve
MLPRC = ConfigMetricMultilabelPrecisionRecallCurve
BROC = ConfigMetricBinaryROC
MCROC = ConfigMetricMulticlassROC
MLROC = ConfigMetricMultilabelROC
BAUROC = ConfigMetricBinaryAUROC
MCAUROC = ConfigMetricMulticlassAUROC
MLAUROC = ConfigMetricMultilabelAUROC
BAP = ConfigMetricBinaryAveragePrecision
MCAP = ConfigMetricMulticlassAveragePrecision
MLAP = ConfigMetricMultilabelAveragePrecision
BCM = ConfigMetricBinaryConfusionMatrix
MCCM = ConfigMetricMulticlassConfusionMatrix
MLCM = ConfigMetricMultilabelConfusionMatrix
BF1 = ConfigMetricBinaryF1Score
MCF1 = ConfigMetricMulticlassF1Score
MLF1 = ConfigMetricMultilabelF1Score
BPREC = ConfigMetricBinaryPrecision
MCPREC = ConfigMetricMulticlassPrecision
MLPREC = ConfigMetricMultilabelPrecision
BREC = ConfigMetricBinaryRecall
MCRECALL = ConfigMetricMulticlassRecall
MLREC = ConfigMetricMultilabelRecall
BSPE = ConfigMetricBinarySpecificity
MCSPEC = ConfigMetricMulticlassSpecificity
MLSPE = ConfigMetricMultilabelSpecificity
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


@dataclass
class ConfigMetricMonai(ConfigMetricBase, metaclass=ABCMeta):
    @override
    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        self._assert_init_essentials()
        ret = self.metric(y_pred=y_pred, y=y, **kwargs)
        ret = self.metric.aggregate().squeeze()
        self.metric.reset()
        return ret


# region Segmentation Region Overlapping metrics
@dataclass
class ConfigMetricDiceScore(ConfigMetricMonai):
    """
    Wrapper class for Dice metric.

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

    Attributes:
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
    include_background: bool = False
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False
    ignore_empty: bool = True
    num_classes: Optional[int] = None
    return_with_label: Union[bool, List[str]] = False

    @override
    def init_essentials(self) -> 'ConfigMetricDiceScore':
        self.metric: mm.DiceMetric = mm.DiceMetric(
            include_background=self.include_background,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans,
            ignore_empty=self.ignore_empty,
            num_classes=self.num_classes,
            return_with_label=self.return_with_label
        )
        return self


@dataclass
class ConfigMetricGeneralizedDiceScore(ConfigMetricMonai):
    """
    Wrapper class for GeneralizedDice metric.

    Compute the Generalized Dice Score metric between tensors.

    This metric is the complement of the Generalized Dice Loss defined in:
    Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
    loss function for highly unbalanced segmentations. DLMIA 2017.

    The inputs `y_pred` and `y` are expected to be one-hot, binarized batch-first tensors, i.e., NCHW[D].

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Attributes:
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
    include_background: bool = False
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    weight_type: Union[Weight, str] = Weight.SQUARE

    @override
    def init_essentials(self) -> 'ConfigMetricGeneralizedDiceScore':
        self.metric: mm.GeneralizedDiceScore = mm.GeneralizedDiceScore(
            include_background=self.include_background,
            reduction=self.reduction,
            weight_type=self.weight_type
        )
        return self

    def __call__(
            self,
            y_pred: TensorOrList,
            y: Optional[TensorOrList] = None,
            **kwargs: Any
    ) -> Tensor:
        return super().__call__(y_pred, y, **kwargs)


@dataclass
class ConfigMetricMeanIoU(ConfigMetricMonai):
    """
    Wrapper class for IoU metric.

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

    Attributes:
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
    include_background: bool = False
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False
    ignore_empty: bool = True

    @override
    def init_essentials(self) -> 'ConfigMetricMeanIoU':
        self.metric: mm.MeanIoU = mm.MeanIoU(
            include_background=self.include_background,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans,
            ignore_empty=self.ignore_empty
        )
        return self


# endregion

# region Segmentation Contour Dist. metrics
@dataclass
class ConfigMetricHausdorffDistance(ConfigMetricMonai):
    """
    Wrapper class for Hausdorff Distance metric.

    Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Attributes:
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
    include_background: bool = False
    distance_metric: str = "euclidean"
    percentile: Optional[float] = None
    directed: bool = False
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricHausdorffDistance':
        self.metric: mm.HausdorffDistanceMetric = mm.HausdorffDistanceMetric(
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


@dataclass
class ConfigMetricSurfaceDistance(ConfigMetricMonai):
    """
    Wrapper class for Surface Distance metric.

    Compute Surface Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both symmetric and asymmetric surface distance calculation.
    Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Attributes:
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
    include_background: bool = False
    symmetric: bool = False
    distance_metric: str = "euclidean"
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricSurfaceDistance':
        self.metric: mm.SurfaceDistanceMetric = mm.SurfaceDistanceMetric(
            include_background=self.include_background,
            symmetric=self.symmetric,
            distance_metric=self.distance_metric,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


@dataclass
class ConfigMetricNormalizedSurfaceDiceScore(ConfigMetricMonai):
    """
    Wrapper class for Normalized Surface Dice (NSD) Score metric.

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

    Attributes:
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
    class_thresholds: List[float]
    include_background: bool = False
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean"
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False
    use_subvoxels: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricNormalizedSurfaceDiceScore':
        self.metric: mm.SurfaceDiceMetric = mm.SurfaceDiceMetric(
            class_thresholds=self.class_thresholds,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans,
            use_subvoxels=self.use_subvoxels
        )
        return self


# endregion

# region Image-to-Image metrics
@dataclass
class ConfigMetricMeanSquaredError(ConfigMetricMonai):
    r"""
    Wrapper class for Mean Squared Error (MSE) metric.

    Compute Mean Squared Error between two tensors using function:

    .. math::
        \operatorname {MSE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i} \right)^{2}.

    More info: https://en.wikipedia.org/wiki/Mean_squared_error

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Attributes:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
    """
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricMeanSquaredError':
        self.metric: mm.MSEMetric = mm.MSEMetric(
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


@dataclass
class ConfigMetricMeanAbsoluteError(ConfigMetricMonai):
    r"""
    Wrapper class for Mean Absolute Error (MAE) metric.

    Compute Mean Absolute Error between two tensors using function:

    .. math::
        \operatorname {MAE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left|y_i-\hat{y_i}\right|.

    More info: https://en.wikipedia.org/wiki/Mean_absolute_error

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Attributes:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
    """
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricMeanAbsoluteError':
        self.metric: mm.MAEMetric = mm.MAEMetric(
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


@dataclass
class ConfigMetricRootMeanSquaredError(ConfigMetricMonai):
    r"""
    Wrapper class for Root Mean Squared Error (RMSE) metric.

    Compute Root Mean Squared Error between two tensors using function:

    .. math::
        \operatorname {RMSE}\left(Y, \hat{Y}\right) ={ \sqrt{ \frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i}\right)^2 } } \
        = \sqrt {\operatorname{MSE}\left(Y, \hat{Y}\right)}.

    More info: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Attributes:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
    """
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricRootMeanSquaredError':
        self.metric: mm.RMSEMetric = mm.RMSEMetric(
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


@dataclass
class ConfigMetricPeakSignalToNoiseRatio(ConfigMetricMonai):
    r"""
    Wrapper class for Peak Signal To Noise Ratio (PSNR) metric.

    Compute Peak Signal To Noise Ratio between two tensors using function:

    .. math::
        \operatorname{PSNR}\left(Y, \hat{Y}\right) = 20 \cdot \log_{10} \left({\mathit{MAX}}_Y\right) \
        -10 \cdot \log_{10}\left(\operatorname{MSE\left(Y, \hat{Y}\right)}\right)

    More info: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Help taken from:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py line 4139

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Attributes:
        max_val: The dynamic range of the images/volumes (i.e., the difference between the
            maximum and the minimum allowed values e.g. 255 for a uint8 image).
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
    """
    max_val: Union[int, float]
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricPeakSignalToNoiseRatio':
        self.metric: mm.PSNRMetric = mm.PSNRMetric(
            max_val=self.max_val,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


@dataclass
class ConfigMetricStructuralSimilarityIndexMeasure(ConfigMetricMonai):
    r"""
    Wrapper class for Structural Similarity Index Measure (SSIM) metric.

    Computes the Structural Similarity Index Measure (SSIM).

    .. math::
        \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

    Attributes:
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
    spatial_dims: int
    data_range: float = 1.0
    kernel_type: Union[KernelType, str] = KernelType.GAUSSIAN
    win_size: Union[int, TLSeq[int]] = 11
    kernel_sigma: Union[float, TLSeq[float]] = 1.5
    k1: float = 0.01
    k2: float = 0.03
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricStructuralSimilarityIndexMeasure':
        self.metric: mm.SSIMMetric = mm.SSIMMetric(
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            win_size=self.win_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


@dataclass
class ConfigMetricMultiScaleStructuralSimilarityIndexMeasure(ConfigMetricMonai):
    """
    Wrapper class for Multi-Scale Structural Similarity Index Measure (MS-SSIM) metric.

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
    spatial_dims: int
    data_range: float = 1.0
    kernel_type: Union[KernelType, str] = KernelType.GAUSSIAN
    kernel_size: Union[int, TLSeq[int]] = 11
    kernel_sigma: Union[float, TLSeq[float]] = 1.5
    k1: float = 0.01
    k2: float = 0.03
    weights: TLSeq[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    get_not_nans: bool = False

    @override
    def init_essentials(self) -> 'ConfigMetricMultiScaleStructuralSimilarityIndexMeasure':
        self.metric: mm.MultiScaleSSIMMetric = mm.MultiScaleSSIMMetric(
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
            weights=self.weights,
            reduction=self.reduction,
            get_not_nans=self.get_not_nans
        )
        return self


# endregion

# endregion

# region MONAI Alias
Dice = ConfigMetricDiceScore
GDice = ConfigMetricGeneralizedDiceScore
IoU = ConfigMetricMeanIoU
HD = ConfigMetricHausdorffDistance
SD = ConfigMetricSurfaceDistance
NSD = ConfigMetricNormalizedSurfaceDiceScore
MSE = ConfigMetricMeanSquaredError
MAE = ConfigMetricMeanAbsoluteError
RMSE = ConfigMetricRootMeanSquaredError
PSNR = ConfigMetricPeakSignalToNoiseRatio
SSIM = ConfigMetricStructuralSimilarityIndexMeasure
MSSSIM = ConfigMetricMultiScaleStructuralSimilarityIndexMeasure

# endregion

# region Efficiency metrics
from datetime import datetime, timedelta


@dataclass
class ConfigMetricEfficiency(ConfigMetricBase):
    """ Identical """


class _VoxelProcessingPerSecond:
    def __init__(self, init_datetime: Optional[datetime] = None) -> None:
        self.time_checkpoint: Optional[datetime] = init_datetime

    def __call__(
            self,
            volume: Optional[Tensor] = None,
            *args,
            time_point: Optional[datetime] = None  # If not specified, record now time
    ) -> Tensor:
        vps: float = 0.
        now_datetime = datetime.now() if time_point is None else time_point
        if volume is not None and self.time_checkpoint is not None:
            time_delta: timedelta = now_datetime - self.time_checkpoint
            # Calculation
            seconds: float = time_delta.total_seconds()
            vps: float = float(np.prod(volume.size())) / seconds
        self.time_checkpoint = now_datetime  # Record time point, maybe another start later
        return torch.tensor(vps, dtype=torch.float, device=volume.device if volume is not None else torch.device('cpu'))


@dataclass
class ConfigMetricVoxelProcessingPerSecond(ConfigMetricEfficiency):
    @override
    def init_essentials(
            self,
            init_datetime: Optional[datetime] = None
    ) -> 'ConfigMetricVoxelProcessingPerSecond':
        self.metric: _VoxelProcessingPerSecond = _VoxelProcessingPerSecond(init_datetime)
        return self

    @override
    def __call__(
            self,
            volume: Optional[Tensor] = None,
            *args,
            time_point: Optional[datetime] = None  # If not specified, record now time
    ) -> Tensor:
        self._assert_init_essentials()
        return self.metric(volume, *args, time_point=time_point)


# endregion

# region Efficiency Alias
VPS = ConfigMetricVoxelProcessingPerSecond
# endregion
