import torch
from torch import Tensor
import monai.transforms as mT
from typing import Dict, Optional, Type, Any, Union, List, Sequence, Tuple


class OperatorIdentity:
    def __call__(self, content: Any) -> Any:
        return content


class OperatorDisplayDictKeys:
    def __init__(self, *tags: str) -> None:
        # Always reserve all params, do not modify type, do not modify name!
        self.tags: Tuple[str, ...] = tags

    def __call__(self, ret_dict: Dict[str, Any]):
        desc: str = ''
        for tag in self.tags:
            desc += f'[{tag}]'
        desc += f'{ret_dict.keys()}'
        print(desc)

    def __reduce__(self):
        return self.__class__, (self.tags,)


class OperatorDisplayConfMat:
    def __init__(
            self,
            phase: str,  # Such as train, val, test, predict
            conf_mat_desc: str,  # Such as ConfMat
            dim_desc: Sequence[Tuple[int, str]]  # Length must be 2, Such as [(dim=0, gt), (dim=1, pred)]
    ) -> None:
        assert len(dim_desc) == 2, \
            f'You shall specify dim_desc in format such as [(0, gt), (1, pred)], elem:=(dim, dim_name)'

        # Always reserve all params, do not modify type, do not modify name!
        self.phase: str = phase
        self.conf_mat_desc: str = conf_mat_desc
        self.dim_desc: Sequence[Tuple[int, str]] = dim_desc

    def __call__(self, mat: Tensor) -> Dict[str, Tensor]:
        dim0, dim0_name = self.dim_desc[0]
        dim1, dim1_name = self.dim_desc[1]
        return {
            f'{self.phase}/{self.conf_mat_desc}[{dim0_name}-{i},{dim1_name}-{j}]': mat[i, j]
            for i in range(mat.size(dim0))
            for j in range(mat.size(dim1))
        }

    def __reduce__(self):
        return self.__class__, (self.phase, self.conf_mat_desc, self.dim_desc)


class OperatorMonaiAsDiscrete(mT.AsDiscrete):
    def __reduce__(self):
        return (
            self.__class__,
            (
                self.argmax,
                self.to_onehot,
                self.threshold,
                self.rounding
            ),
            {"kwargs": self.kwargs},
        )

class OperatorTorchSoftmax(torch.nn.Softmax):
    def __reduce__(self):
        return (
            self.__class__,
            (self.dim,)
        )