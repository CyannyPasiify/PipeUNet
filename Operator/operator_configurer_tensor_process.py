import torch
from torch import Tensor
import monai.transforms as mT
from typing import Dict, Optional, Any, Union, List, Tuple, Hashable, Literal
from typing_extensions import override
from dataclasses import dataclass
from Operator.operator_configurer import ConfigOperatorBase
from abc import ABCMeta, abstractmethod


@dataclass
class ConfigOperatorTensorProcessBase(ConfigOperatorBase, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, content: Tensor) -> Tensor:
        pass


@dataclass
class ConfigOperatorTensorProcessIdentity(ConfigOperatorTensorProcessBase):
    @override
    def is_ready(self) -> bool:
        # Always ok
        return True

    @override
    def init_essentials(self) -> 'ConfigOperatorTensorProcessIdentity':
        return self

    @override
    def __call__(self, content: Tensor) -> Tensor:
        return content


@dataclass
class ConfigOperatorTensorProcessMonaiAsDiscrete(ConfigOperatorTensorProcessBase):
    argmax: bool = False
    to_onehot: Optional[int] = None
    threshold: Optional[float] = None
    rounding: Optional[str] = None
    dim: int = 0
    keepdim: bool = True
    dtype: torch.dtype = torch.float

    @override
    def is_ready(self) -> bool:
        return hasattr(self, "transform")

    @override
    def init_essentials(self) -> 'ConfigOperatorTensorProcessMonaiAsDiscrete':
        self.transform: mT.AsDiscrete = mT.AsDiscrete(
            self.argmax,
            self.to_onehot,
            self.threshold,
            self.rounding,
            dim=self.dim,
            keepdim=self.keepdim,
            dtype=self.dtype
        )
        return self

    @override
    def __call__(
            self,
            content: Tensor,
            argmax: Optional[bool] = None,
            to_onehot: Optional[int] = None,
            threshold: Optional[float] = None,
            rounding: Optional[str] = None,
    ) -> Tensor:
        self._assert_init_essentials()
        return self.transform(
            content,
            argmax,
            to_onehot,
            threshold,
            rounding
        )

    @override
    def get_operator(self) -> Any:
        self._assert_init_essentials()
        return self.transform


@dataclass
class ConfigOperatorTensorProcessTorchSoftmax(ConfigOperatorTensorProcessBase):
    dim: Optional[int] = None

    @override
    def is_ready(self) -> bool:
        return hasattr(self, "operator")

    @override
    def init_essentials(self) -> 'ConfigOperatorTensorProcessTorchSoftmax':
        self.operator: torch.nn.Softmax = torch.nn.Softmax(dim=self.dim)
        return self

    @override
    def __call__(self, content: Tensor) -> Tensor:
        self._assert_init_essentials()
        return self.operator(content)

    @override
    def get_operator(self) -> Any:
        self._assert_init_essentials()
        return self.operator
