# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any, Type
from abc import ABC, abstractmethod

NormLayerType = Type[Union[
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LocalResponseNorm,
    nn.RMSNorm
]]

NLActType = Type[Union[
    nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ReLU6, nn.RReLU,
    nn.ELU, nn.CELU, nn.SELU, nn.GELU,
    nn.Sigmoid, nn.LogSigmoid, nn.SiLU, nn.Hardsigmoid,
    nn.Tanh, nn.Mish, nn.Hardtanh,
    nn.Softplus, nn.Softsign,
    nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.Softmin,
    nn.Softshrink, nn.Tanhshrink, nn.Hardshrink,
    nn.Hardswish,
    nn.Threshold,
    nn.MultiheadAttention,
]]


class IODescriptive(ABC):
    @abstractmethod
    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        return ''


# Conv-Norm-Act
# Template Module: Convolution followed by Normalization and Activation
# Pinout Diagram: [Valid]
#                ┌───────────────┐
#    input_feat ─│ Conv-Norm-Act │─ output_feat
#           (*)  │               │  (*)
#                └───────────────┘
# Expanded Diagram:
#                ┌───────┐                              ┌───────────────┐                              ┌────────────┐
#    input_feat ─│ Conv  │──────────────────────────────│ Normalization │──────────────────────────────│ Activation │─ output_feat
#           (*)  │ Layer │ {output_feat}   {input_feat} │ Layer         │ {output_feat}   {input_feat} │ Layer      │  (*)
#                └───────┘ (*)                          └───────────────┘ (*)                          └────────────┘
class ConvNormAct(nn.Module, IODescriptive):
    def __init__(
            self,
            conv_layer: Type[Union[nn.Conv3d, nn.ConvTranspose3d]],
            norm_layer: NormLayerType,
            act_layer: NLActType,
            conv_layer_kwargs: Dict[str, Any],
            norm_layer_kwargs: Dict[str, Any],
            act_layer_kwargs: Dict[str, Any],
            reserve_io: bool = True
    ):
        super(ConvNormAct, self).__init__()
        self.conv_layer: nn.Module = conv_layer(**conv_layer_kwargs)
        self.norm_layer: nn.Module = norm_layer(**norm_layer_kwargs)
        self.act_layer: nn.Module = act_layer(**act_layer_kwargs)

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        c_feat = self.conv_layer(input_feat)
        n_feat = self.norm_layer(c_feat)
        a_feat = self.act_layer(n_feat)
        output_feat = a_feat

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (hasattr(self, 'input_feat') and hasattr(self, 'output_feat'))
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


SizeLike = Union[int, Tuple[int, int, int]]


# Conv-BN-ReLU
# Standard Convolution followed by Batch Normalization and ReLU Activation
# Pinout Diagram: [Valid]
#                ┌──────────────┐
#    input_feat ─│ Conv-BN-ReLU │─ output_feat
# (B,Cin,X,Y,Z)  │              │  (B,Cout,X,Y,Z)
#                └──────────────┘
# Expanded Diagram:
#                ┌────────┐                              ┌───────────────┐                              ┌────────────┐
#    input_feat ─│ Conv3d │──────────────────────────────│ Batch         │──────────────────────────────│ ReLU       │─ output_feat
# (B,Cin,X,Y,Z)  │        │ {output_feat}   {input_feat} │ Normalization │ {output_feat}   {input_feat} │ Activation │  (B,Cout,X,Y,Z)
#                └────────┘ (B,Cout,X,Y,Z)               └───────────────┘ (B,Cout,X,Y,Z)               └────────────┘
class ConvBNReLU(nn.Module, IODescriptive):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: SizeLike,
            stride: SizeLike = 1,
            padding: Union[str, SizeLike] = 0,
            dilation: SizeLike = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            inplace: bool = False,
            reserve_io: bool = True
    ):
        super(ConvBNReLU, self).__init__()
        self.conv: nn.Conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        self.bn: nn.BatchNorm3d = nn.BatchNorm3d(
            num_features=out_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self.relu: nn.ReLU = nn.ReLU(inplace=inplace)

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        c_feat = self.conv(input_feat)
        n_feat = self.bn(c_feat)
        a_feat = self.relu(n_feat)
        output_feat = a_feat

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (hasattr(self, 'input_feat') and hasattr(self, 'output_feat'))
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# Concat
# Concat two Tensors at specified dimensions
# Pinout Diagram: [Valid]
#              ┌────────┐
#          x1 ─│ Concat │─ cat_feat
# (...,C1,...) │        │  (...,C1+C2,...)
#          x2 ─│        │
# (...,C2,...) │        │
#              └────────┘
class Concat(nn.Module, IODescriptive):
    def __init__(
            self,
            dim: int,
            reserve_io: bool = True
    ):
        super(Concat, self).__init__()
        self.dim: int = dim

        self.reserve_io: bool = reserve_io

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cat_feat = torch.cat([x1, x2], dim=self.dim)

        if self.reserve_io:
            setattr(self, 'x1', x1.cpu())
            setattr(self, 'x2', x2.cpu())
            setattr(self, 'cat_feat', cat_feat.cpu())
        return cat_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'x1')
                        and hasattr(self, 'x2')
                        and hasattr(self, 'cat_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    x1: {tuple(self.x1.size())}\n"
                     f"{prefix}    x2: {tuple(self.x2.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    cat_feat: {tuple(self.cat_feat.size())}\n")
        return desc


if __name__ == "__main__":
    pass
