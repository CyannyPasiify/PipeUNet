# -*- coding: utf-8 -*-
"""
Basic Neural Network Building Blocks Module

This module provides fundamental building blocks for neural networks, including
convolution blocks with normalization and activation, and utility modules for
tensor operations.

Classes:
    IODescriptive: Abstract base class for modules with I/O description capability
    ConvNormAct: Generic convolution-normalization-activation block
    ConvBNReLU: Standard Conv3d-BatchNorm3d-ReLU block
    Concat: Tensor concatenation module

Type Aliases:
    NormLayerType: Union of available normalization layer types
    NLActType: Union of available activation layer types
    SizeLike: Integer or 3-tuple of integers for spatial dimensions
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any, Type
from abc import ABC, abstractmethod

# Type alias for normalization layers
NormLayerType = Type[Union[
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LocalResponseNorm,
    nn.RMSNorm
]]

# Type alias for activation layers
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
    """
    Abstract base class for modules with I/O description capability
    
    Provides interface for generating hierarchical I/O shape descriptions
    of neural network modules, useful for debugging and visualization.
    """
    @abstractmethod
    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        """
        Generate I/O description string
        
        Args:
            max_level: Maximum recursion level for nested modules
            indent: Current indentation level
            indent_placeholder: String used for indentation
            target_level: Target level for this description
            
        Returns:
            Formatted I/O description string
        """
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
    """
    Generic Convolution-Normalization-Activation block
    
    A flexible building block that applies convolution, normalization,
    and activation in sequence. Supports various layer types through
    dependency injection.
    
    Architecture:
        Input → Conv → Norm → Activation → Output
    
    Attributes:
        conv_layer: Convolution layer instance
        norm_layer: Normalization layer instance
        act_layer: Activation layer instance
        reserve_io: Whether to store I/O tensors for debugging
    """
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
        """
        Initialize ConvNormAct block
        
        Args:
            conv_layer: Convolution layer class (Conv3d or ConvTranspose3d)
            norm_layer: Normalization layer class
            act_layer: Activation layer class
            conv_layer_kwargs: Keyword arguments for convolution layer
            norm_layer_kwargs: Keyword arguments for normalization layer
            act_layer_kwargs: Keyword arguments for activation layer
            reserve_io: If True, store I/O tensors for debugging
        """
        super(ConvNormAct, self).__init__()
        self.conv_layer: nn.Module = conv_layer(**conv_layer_kwargs)
        self.norm_layer: nn.Module = norm_layer(**norm_layer_kwargs)
        self.act_layer: nn.Module = act_layer(**act_layer_kwargs)

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conv-Norm-Act sequence
        
        Args:
            input_feat: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor after Conv-Norm-Act operations
        """
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
        """
        Generate I/O shape description
        
        Args:
            max_level: Maximum recursion level
            indent: Current indentation level
            indent_placeholder: Indentation string
            target_level: Target description level
            
        Returns:
            Formatted I/O description string
        """
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


# Type alias for kernel size, stride, padding, and dilation
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
    """
    Standard 3D Convolution-BatchNorm-ReLU block
    
    A commonly used building block in 3D CNNs that applies:
    1. 3D convolution
    2. Batch normalization
    3. ReLU activation
    
    Architecture:
        Input → Conv3d → BatchNorm3d → ReLU → Output
    
    Attributes:
        conv: 3D convolution layer
        bn: Batch normalization layer
        relu: ReLU activation layer
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: SizeLike,
            stride: SizeLike = 1,
            padding: Union[str, SizeLike] = 0,
            dilation: SizeLike = 1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = "zeros",
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            inplace: bool = False,
            reserve_io: bool = True
    ):
        """
        Initialize ConvBNReLU block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to all three sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output channels
            bias: If True, adds a learnable bias to the output
            padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular')
            eps: Value added to the denominator for batch norm numerical stability
            momentum: Value used for running mean/variance computation
            affine: If True, batch norm has learnable affine parameters
            track_running_stats: If True, tracks running mean/variance
            inplace: If True, performs ReLU in-place
            reserve_io: If True, store I/O tensors for debugging
        """
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
        """
        Forward pass through Conv-BN-ReLU sequence
        
        Args:
            input_feat: Input tensor of shape (B, C_in, D, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, D', H', W')
        """
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
        """
        Generate I/O shape description
        
        Args:
            max_level: Maximum recursion level
            indent: Current indentation level
            indent_placeholder: Indentation string
            target_level: Target description level
            
        Returns:
            Formatted I/O description string
        """
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
    """
    Tensor concatenation module
    
    Concatenates two tensors along a specified dimension.
    Commonly used in U-Net architectures for skip connections.
    
    Architecture:
        x1 ───┐
              ├──[Concat at dim]──→ cat_feat
        x2 ───┘
    
    Attributes:
        dim: Dimension along which to concatenate
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            dim: int,
            reserve_io: bool = True
    ):
        """
        Initialize Concat module
        
        Args:
            dim: Dimension along which to concatenate tensors
            reserve_io: If True, store I/O tensors for debugging
        """
        super(Concat, self).__init__()
        self.dim: int = dim

        self.reserve_io: bool = reserve_io

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Concatenate two tensors along the specified dimension
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Concatenated tensor
            
        Note:
            The tensors must have the same shape except at the concatenation dimension
        """
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
        """
        Generate I/O shape description
        
        Args:
            max_level: Maximum recursion level
            indent: Current indentation level
            indent_placeholder: Indentation string
            target_level: Target description level
            
        Returns:
            Formatted I/O description string
        """
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
