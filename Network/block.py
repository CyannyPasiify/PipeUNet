# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any, Type

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


class ConvNormAct(nn.Module):
    def __init__(
            self,
            conv_layer: Type[nn.Conv3d, nn.ConvTranspose3d],
            norm_layer: NormLayerType,
            act_layer: NLActType,
            conv_layer_kwargs: Dict[str, Any],
            norm_layer_kwargs: Dict[str, Any],
            act_layer_kwargs: Dict[str, Any],
    ):
        super(ConvNormAct, self).__init__()
        self.conv_layer: nn.Module = conv_layer(**conv_layer_kwargs)
        self.norm_layer: nn.Module = norm_layer(**norm_layer_kwargs)
        self.act_layer: nn.Module = act_layer(**act_layer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.act_layer(x)
        return x


SizeLike = Union[int, Tuple[int, int, int]]


class ConvBNReLU(nn.Module):
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
            inplace: bool = False
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Concat(nn.Module):
    def __init__(
            self,
            dim: int
    ):
        super(Concat, self).__init__()
        self.dim: int = dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=self.dim)
        return x


if __name__ == "__main__":
    pass
