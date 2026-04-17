# -*- coding: utf-8 -*-
from functools import partial
import torch
from torch import Tensor
import monai.inferers as mI
from monai.utils.enums import BlendMode, PytorchPadMode
from typing import TypeVar, Optional, Union, Sequence, List, Callable, Any, Tuple, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import override

T = TypeVar("T")
TLSeq = Union[List[T], Tuple[T, ...]]


@dataclass
class ConfigInfererBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "inferer")

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
    ) -> 'ConfigInfererBase':
        self.inferer: mI.Inferer = mI.SimpleInferer()  # Just placeholder
        return self

    def __call__(
            self,
            inputs: torch.Tensor,
            network: Callable[..., Union[torch.Tensor, TLSeq[torch.Tensor], Dict[Any, torch.Tensor]]],
            *args: Any,
            **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
        return self.inferer(inputs, network, *args, **kwargs)

    def get_inferer_operator(self, *args, **kwargs) -> mI.Inferer:
        self._assert_init_essentials(*args, **kwargs)
        return self.inferer


@dataclass
class ConfigInfererSimple(ConfigInfererBase):
    """
        SimpleInferer is the normal inference method that run model forward() directly.
    """
    @override
    def init_essentials(self) -> 'ConfigInfererSimple':
        self.inferer: mI.Inferer = mI.SimpleInferer()
        return self


@dataclass
class ConfigInfererSlidingWindow(ConfigInfererBase):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().
    This class is a wrapper for SlidingWindowInfererAdapt, which extends SlidingWindowInferer to automatically switch to
    buffered and then to CPU stitching, when OOM on GPU. It also records a size of such large images to automatically
    try CPU stitching for the next large image of a similar size.  If the stitching 'device' input parameter is provided,
    automatic adaptation won't be attempted, please keep the default option device = None for adaptive behavior.
    Note: the output might be on CPU (even if the input was on GPU), if the GPU memory was not sufficient.

    Args:
        roi_size: the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a tqdm progress bar.
        cpu_thresh: when provided, dynamically switch to stitching on cpu (to save gpu memory)
            when input image volume is larger than this threshold (in pixels/voxels).
            Otherwise use ``"device"``. Thus, the output may end-up on either cpu or gpu.
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.

    Note:
        ``sw_batch_size`` denotes the max number of windows per network inference iteration,
        not the batch size of inputs.
    """
    roi_size: Union[TLSeq[int], int] = (128, 128, 128)
    sw_batch_size: int = 1
    overlap: Union[TLSeq[float], float] = 0.25
    mode: Union[BlendMode, str] = BlendMode.GAUSSIAN
    sigma_scale: Union[TLSeq[float], float] = 0.125
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT
    cval: float = 0.0
    sw_device: Optional[Union[torch.device, str]] = None
    device: Optional[Union[torch.device, str]] = None
    progress: bool = False
    cpu_thresh: Optional[int] = None
    buffer_steps: Optional[int] = None
    buffer_dim: int = -1

    @override
    def init_essentials(self) -> 'ConfigInfererSlidingWindow':
        self.inferer: mI.SlidingWindowInfererAdapt = mI.SlidingWindowInfererAdapt(
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            padding_mode=self.padding_mode,
            cval=self.cval,
            sw_device=self.sw_device,
            device=self.device,
            progress=self.progress,
            cpu_thresh=self.cpu_thresh,
            buffer_steps=self.buffer_steps,
            buffer_dim=self.buffer_dim
        )
        return self


class _MainWithAuxSlidingWindowInferer(mI.SlidingWindowInfererAdapt):
    def _wrapped_network_io(
            self,
            input: Tensor,
            *args: Any,
            network: Callable[..., Tuple[Tensor, Sequence[Tensor]]],
            **kwargs: Any
    ) -> Sequence[Tensor]:
        cls_logits: Tensor
        aux_cls_logits: List[Tensor]
        cls_logits, aux_cls_logits = network(input)
        return [cls_logits] + list(reversed(aux_cls_logits))  # Map from largest to smallest

    def __call__(
            self,
            inputs: Tensor,
            network: Callable[..., Tuple[Tensor, Sequence[Tensor]]],
            *args: Any,
            **kwargs: Any,
    ) -> Tuple[Tensor, Sequence[Tensor]]:
        """
        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
            condition (torch.Tensor, optional): If provided via `**kwargs`,
                this tensor must match the shape of `inputs` and will be sliced, patched, or windowed alongside the inputs.
                The resulting segments will be passed to the model together with the corresponding input segments.
        """
        full_logits: Tuple[Tensor, ...] = super().__call__(
            inputs,
            partial(self._wrapped_network_io, network=network),
            args, kwargs
        )
        return full_logits[0], list(reversed(full_logits[1:]))


@dataclass
class ConfigInfererMainWithAuxSlidingWindow(ConfigInfererBase):
    roi_size: Union[Sequence[int], int]
    sw_batch_size: int = 1
    overlap: Union[Sequence[float], float] = 0.25
    mode: Union[BlendMode, str] = BlendMode.GAUSSIAN
    sigma_scale: Union[Sequence[float], float] = 0.125
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT
    cval: float = 0.0
    sw_device: Optional[Union[torch.device, str]] = None
    device: Optional[Union[torch.device, str]] = None
    progress: bool = False
    cpu_thresh: Optional[int] = None
    buffer_steps: Optional[int] = None
    buffer_dim: int = -1

    @override
    def init_essentials(self) -> 'ConfigInfererMainWithAuxSlidingWindow':
        self.inferer: _MainWithAuxSlidingWindowInferer = _MainWithAuxSlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            padding_mode=self.padding_mode,
            cval=self.cval,
            sw_device=self.sw_device,
            device=self.device,
            progress=self.progress,
            cpu_thresh=self.cpu_thresh,
            buffer_steps=self.buffer_steps,
            buffer_dim=self.buffer_dim
        )
        return self
