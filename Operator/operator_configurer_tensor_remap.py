from torch import Tensor
from typing import Dict, Optional, Any, Union, List, Tuple, Hashable, Literal
from typing_extensions import override
from dataclasses import dataclass
from Operator.operator_configurer import ConfigOperatorBase
from abc import ABCMeta, abstractmethod


@dataclass
class ConfigOperatorTensorRemapBase(ConfigOperatorBase, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, content: Tensor) -> Dict[str, Tensor]:
        pass


@dataclass
class ConfigOperatorTensorRemapConfMat(ConfigOperatorTensorRemapBase):
    phase: str  # Such as train, val, test, predict
    conf_mat_desc: str  # Such as ConfMat
    dim_desc: Tuple[Tuple[int, str], Tuple[int, str]]  # Length must be 2, Such as ((dim=0, gt), (dim=1, pred))

    @override
    def init_essentials(self) -> 'ConfigOperatorTensorRemapConfMat':
        assert len(self.dim_desc) == 2, \
            f'You shall specify dim_desc in format such as ((0, gt), (1, pred)), elem:=(dim, dim_name)'
        self._is_ready = True
        return self

    @override
    def __call__(self, content: Tensor) -> Dict[str, Tensor]:
        # content is a ConfMat with size (C, C)
        self._assert_init_essentials()
        dim0, dim0_name = self.dim_desc[0]
        dim1, dim1_name = self.dim_desc[1]
        return {
            f'{self.phase}/{self.conf_mat_desc}[{dim0_name}-{i},{dim1_name}-{j}]': content[i, j].float()
            for i in range(content.size(dim0))
            for j in range(content.size(dim1))
        }


@dataclass
class ConfigOperatorTensorRemapClassWise(ConfigOperatorTensorRemapBase):
    phase: str  # Such as train, val, test, predict
    metric_desc: str
    include_background: bool

    @override
    def is_ready(self) -> bool:
        # Always ok
        return True

    @override
    def init_essentials(self, *args, **kwargs) -> 'ConfigOperatorTensorRemapClassWise':
        return self

    @override
    def __call__(self, content: Tensor) -> Dict[str, Tensor]:
        """
        content: Shape as (C)
        """
        self._assert_init_essentials()
        num_classes: int = content.size(0)
        addend: int = 0 if self.include_background else 1
        return {
            f'{self.phase}/{self.metric_desc}[{c + addend}]': content[c].float()
            for c in range(num_classes)
        }
