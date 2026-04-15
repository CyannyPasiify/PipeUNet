import torch
from monai.config import NdarrayOrTensor
from torch import Tensor
import monai.transforms as mT
from typing import Dict, Optional, Type, Any, Union, List, Sequence, Tuple
from dataclasses import dataclass


@dataclass
class OperatorIdentity:
    def __call__(self, content: Any) -> Any:
        return content


@dataclass
class OperatorDisplayDictKeys:
    tags: Tuple[str, ...] = ()

    def __call__(self, ret_dict: Dict[str, Any]):
        desc: str = ''
        for tag in self.tags:
            desc += f'[{tag}]'
        desc += f'{ret_dict.keys()}'
        print(desc)


@dataclass
class OperatorDisplayConfMat:
    phase: str  # Such as train, val, test, predict
    conf_mat_desc: str  # Such as ConfMat
    dim_desc: Tuple[Tuple[int, str], Tuple[int, str]]  # Length must be 2, Such as ((dim=0, gt), (dim=1, pred))

    def __post_init__(self) -> None:
        assert len(self.dim_desc) == 2, \
            f'You shall specify dim_desc in format such as ((0, gt), (1, pred)), elem:=(dim, dim_name)'

    def __call__(self, mat: Tensor) -> Dict[str, Tensor]:
        dim0, dim0_name = self.dim_desc[0]
        dim1, dim1_name = self.dim_desc[1]
        return {
            f'{self.phase}/{self.conf_mat_desc}[{dim0_name}-{i},{dim1_name}-{j}]': mat[i, j]
            for i in range(mat.size(dim0))
            for j in range(mat.size(dim1))
        }


@dataclass
class OperatorMonaiAsDiscrete:
    argmax: bool = False
    to_onehot: Optional[int] = None
    threshold: Optional[float] = None
    rounding: Optional[str] = None
    dim: int = 0
    keepdim: bool = True
    dtype: torch.dtype = torch.float

    def __call__(
            self,
            img: NdarrayOrTensor,
            argmax: Optional[bool] = None,
            to_onehot: Optional[int] = None,
            threshold: Optional[float] = None,
            rounding: Optional[str] = None,
    ) -> NdarrayOrTensor:
        tf_as_discrete: mT.AsDiscrete = mT.AsDiscrete(
            self.argmax,
            self.to_onehot,
            self.threshold,
            self.rounding,
            dim=self.dim,
            keepdim=self.keepdim,
            dtype=self.dtype
        )
        return tf_as_discrete(
            img,
            argmax,
            to_onehot,
            threshold,
            rounding
        )


@dataclass
class OperatorTorchSoftmax:
    dim: Optional[int] = None

    def __call__(self, input: Tensor) -> Tensor:
        op: torch.nn.Softmax = torch.nn.Softmax(dim=self.dim)
        return op(input)
