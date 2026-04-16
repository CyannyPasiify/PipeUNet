import torch
from torch import Tensor
from monai.config import NdarrayOrTensor
import monai.transforms as mT
from abc import ABC, abstractmethod
from typing import TypeVar, Dict, Optional, Any, Union, List, Tuple

from typing_extensions import override
from dataclasses import dataclass

T = TypeVar("T")
TLSeq = Union[List[T], Tuple[T, ...]]


@dataclass
class OperatorBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "_is_ready")

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
    ) -> 'OperatorBase':
        # Do anything
        return self

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


@dataclass
class OperatorIdentity(OperatorBase):
    @override
    def is_ready(self) -> bool:
        # Always ok
        return True

    @override
    def init_essentials(self) -> 'OperatorIdentity':
        return self

    @override
    def __call__(self, content: Any) -> Any:
        return content


@dataclass
class OperatorDisplayDictKeys(OperatorBase):
    tags: TLSeq[str] = ()

    @override
    def is_ready(self) -> bool:
        # Always ok
        return True

    @override
    def init_essentials(self) -> 'OperatorDisplayDictKeys':
        return self

    @override
    def __call__(self, ret_dict: Dict[str, Any]):
        desc: str = ''
        for tag in self.tags:
            desc += f'[{tag}]'
        desc += f'{ret_dict.keys()}'
        print(desc)


@dataclass
class OperatorDisplayConfMat(OperatorBase):
    phase: str  # Such as train, val, test, predict
    conf_mat_desc: str  # Such as ConfMat
    dim_desc: Tuple[Tuple[int, str], Tuple[int, str]]  # Length must be 2, Such as ((dim=0, gt), (dim=1, pred))

    @override
    def init_essentials(self) -> 'OperatorDisplayConfMat':
        assert len(self.dim_desc) == 2, \
            f'You shall specify dim_desc in format such as ((0, gt), (1, pred)), elem:=(dim, dim_name)'
        self._is_ready = True
        return self

    @override
    def __call__(self, mat: Tensor) -> Dict[str, Tensor]:
        self._assert_init_essentials()
        dim0, dim0_name = self.dim_desc[0]
        dim1, dim1_name = self.dim_desc[1]
        return {
            f'{self.phase}/{self.conf_mat_desc}[{dim0_name}-{i},{dim1_name}-{j}]': mat[i, j]
            for i in range(mat.size(dim0))
            for j in range(mat.size(dim1))
        }


@dataclass
class OperatorMonaiAsDiscrete(OperatorBase):
    argmax: bool = False
    to_onehot: Optional[int] = None
    threshold: Optional[float] = None
    rounding: Optional[str] = None
    dim: int = 0
    keepdim: bool = True
    dtype: torch.dtype = torch.float

    @override
    def is_ready(self) -> bool:
        # Always ok
        return hasattr(self, "transform")

    @override
    def init_essentials(self) -> 'OperatorMonaiAsDiscrete':
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
            img: NdarrayOrTensor,
            argmax: Optional[bool] = None,
            to_onehot: Optional[int] = None,
            threshold: Optional[float] = None,
            rounding: Optional[str] = None,
    ) -> NdarrayOrTensor:
        self._assert_init_essentials()
        return self.transform(
            img,
            argmax,
            to_onehot,
            threshold,
            rounding
        )


@dataclass
class OperatorTorchSoftmax(OperatorBase):
    dim: Optional[int] = None

    @override
    def is_ready(self) -> bool:
        # Always ok
        return hasattr(self, "operator")

    @override
    def init_essentials(self) -> 'OperatorTorchSoftmax':
        self.operator: torch.nn.Softmax = torch.nn.Softmax(dim=self.dim)
        return self

    @override
    def __call__(self, input: Tensor) -> Tensor:
        self._assert_init_essentials()
        return self.operator(input)
