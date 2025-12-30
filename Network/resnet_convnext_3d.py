#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D ResNet 和 ConvNeXt 实现

功能概述：
    提供支持3D多通道图像输入的ResNet和ConvNeXt实现
    用于医学图像分类等3D多通道图像任务

核心功能：
    1. ResNet3D: 基于ResNet的3D图像分类网络
    2. ConvNeXt3D: 基于ConvNeXt的3D图像分类网络
    3. 支持多通道3D图像输入和多分类任务
    4. 可配置的模型参数和尺寸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any, Type


class ChannelFirstLayerNorm(nn.Module):
    """
    通道优先的层归一化模块
    
    适用于通道维度在前的输入（如[batch_size, channels, depth, height, width]）
    自动处理通道维度的置换和恢复
    
    Args:
        normalized_shape: 需要归一化的形状（通道维度的大小）
        eps: 避免除零的小值
        elementwise_affine: 是否使用可学习的仿射参数
    """
    # 类型注解
    layer_norm: nn.LayerNorm
    
    def __init__(self, 
                 normalized_shape: Union[int, List[int], Tuple[int, ...]],
                 eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        # 初始化内部的LayerNorm
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图张量，形状为 [batch_size, channels, *spatial_dims]
                适用于2D ([batch_size, channels, height, width])
                或3D ([batch_size, channels, depth, height, width]) 输入
        
        Returns:
            输出特征图张量，形状与输入相同 [batch_size, channels, *spatial_dims]
        """
        # 保存原始形状和维度
        original_shape = x.shape
        ndim = x.ndim
        
        # 对于3D输入 ([batch_size, channels, depth, height, width])
        if ndim == 5:
            # 维度置换: 将通道维度移至最后
            # [batch_size, channels, depth, height, width] -> [batch_size, depth, height, width, channels]
            x = x.permute(0, 2, 3, 4, 1)
            
            # 应用层归一化
            x = self.layer_norm(x)
            
            # 维度置换回原始格式
            # [batch_size, depth, height, width, channels] -> [batch_size, channels, depth, height, width]
            x = x.permute(0, 4, 1, 2, 3)
        
        # 对于2D输入 ([batch_size, channels, height, width])
        elif ndim == 4:
            # 维度置换: 将通道维度移至最后
            # [batch_size, channels, height, width] -> [batch_size, height, width, channels]
            x = x.permute(0, 2, 3, 1)
            
            # 应用层归一化
            x = self.layer_norm(x)
            
            # 维度置换回原始格式
            # [batch_size, height, width, channels] -> [batch_size, channels, height, width]
            x = x.permute(0, 3, 1, 2)
        
        # 对于1D或其他维度输入
        else:
            raise ValueError(f"ChannelFirstLayerNorm只支持4D(2D数据)或5D(3D数据)输入，当前输入维度: {ndim}")
        
        # 确保输出形状与输入一致
        assert x.shape == original_shape, f"输出形状 {x.shape} 与输入形状 {original_shape} 不匹配"
        
        return x


class BasicBlock3D(nn.Module):
    """
    ResNet3D的基础块
    用于ResNet-18和ResNet-34
    """
    # 类型注解
    expansion: int
    conv1: nn.Conv3d
    bn1: nn.BatchNorm3d
    relu: nn.ReLU
    conv2: nn.Conv3d
    bn2: nn.BatchNorm3d
    downsample: Optional[nn.Module]
    stride: int
    
    expansion = 1
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1):
        super().__init__()
        # 检查参数
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock3D only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock3D")
        
        # 构建卷积层
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图张量，形状为 [batch_size, in_channels, depth, height, width]
            
        Returns:
            输出特征图张量，形状为 [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        """
        identity: torch.Tensor = x  # [batch_size, in_channels, depth, height, width]
        
        # 第一个卷积层
        out = self.conv1(x)  # [batch_size, out_channels, depth//stride, height//stride, width//stride]
        out = self.bn1(out)  # [batch_size, out_channels, depth//stride, height//stride, width//stride]
        out = self.relu(out)  # [batch_size, out_channels, depth//stride, height//stride, width//stride]
        
        # 第二个卷积层
        out = self.conv2(out)  # [batch_size, out_channels, depth//stride, height//stride, width//stride]
        out = self.bn2(out)  # [batch_size, out_channels, depth//stride, height//stride, width//stride]
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        
        out += identity  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        out = self.relu(out)  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        
        return out


class Bottleneck3D(nn.Module):
    """
    ResNet3D的瓶颈块
    用于ResNet-50, ResNet-101和ResNet-152
    """
    # 类型注解
    expansion: int
    conv1: nn.Conv3d
    bn1: nn.BatchNorm3d
    conv2: nn.Conv3d
    bn2: nn.BatchNorm3d
    conv3: nn.Conv3d
    bn3: nn.BatchNorm3d
    relu: nn.ReLU
    downsample: Optional[nn.Module]
    stride: int
    
    expansion = 4
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1):
        super().__init__()
        # 计算中间通道数
        width = int(out_channels * (base_width / 64.)) * groups
        
        # 构建卷积层
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(
            in_channels=width,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图张量，形状为 [batch_size, in_channels, depth, height, width]
            
        Returns:
            输出特征图张量，形状为 [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        """
        identity: torch.Tensor = x  # [batch_size, in_channels, depth, height, width]
        
        # 第一个卷积层 (1x1x1)
        out = self.conv1(x)  # [batch_size, width, depth, height, width]
        out = self.bn1(out)  # [batch_size, width, depth, height, width]
        out = self.relu(out)  # [batch_size, width, depth, height, width]
        
        # 第二个卷积层 (3x3x3)
        out = self.conv2(out)  # [batch_size, width, depth//stride, height//stride, width//stride]
        out = self.bn2(out)  # [batch_size, width, depth//stride, height//stride, width//stride]
        out = self.relu(out)  # [batch_size, width, depth//stride, height//stride, width//stride]
        
        # 第三个卷积层 (1x1x1)
        out = self.conv3(out)  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        out = self.bn3(out)  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        
        out += identity  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        out = self.relu(out)  # [batch_size, out_channels * expansion, depth//stride, height//stride, width//stride]
        
        return out


class ResNet3D(nn.Module):
    """
    3D ResNet 模型
    支持3D多通道图像输入的ResNet实现
    """
    # 类型注解
    in_channels: int
    dilation: int
    groups: int
    base_width: int
    conv1: nn.Conv3d
    bn1: nn.BatchNorm3d
    relu: nn.ReLU
    maxpool: nn.MaxPool3d
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential
    layer4: nn.Sequential
    avgpool: nn.AdaptiveAvgPool3d
    fc: nn.Linear
    
    def __init__(self, 
                 block: Type[Union[BasicBlock3D, Bottleneck3D]],
                 layers: List[int],
                 in_channels: int = 1,
                 num_classes: int = 2,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None):
        super().__init__()
        
        # 初始化参数
        self.in_channels = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element list")
        
        self.groups = groups
        self.base_width = width_per_group
        
        # 初始卷积层
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 构建ResNet层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                      dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                      dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                      dilate=replace_stride_with_dilation[2])
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(self, 
                    block: Type[Union[BasicBlock3D, Bottleneck3D]],
                    out_channels: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        """
        创建ResNet层
        
        Args:
            block: 块类型，BasicBlock3D或Bottleneck3D
            out_channels: 输出通道数
            blocks: 块的数量
            stride: 步长
            dilate: 是否使用膨胀卷积
            
        Returns:
            Sequential: 层的序列模块
        """
        downsample: Optional[nn.Sequential] = None
        previous_dilation: int = self.dilation
        
        # 如果需要降采样或通道数不匹配，创建downsample层
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.in_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
        
        layers: List[nn.Module] = []
        layers.append(block(
            in_channels=self.in_channels,
            out_channels=out_channels,
            stride=stride,
            downsample=downsample,
            groups=self.groups,
            base_width=self.base_width,
            dilation=previous_dilation
        ))
        
        self.in_channels = out_channels * block.expansion
        
        # 添加剩余的块
        for _ in range(1, blocks):
            layers.append(block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        
        # 零初始化残差块的最后一个BN层
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock3D):
                    nn.init.zeros_(m.bn2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入3D图像张量，形状为 [batch_size, in_channels, depth, height, width]
            
        Returns:
            分类预测张量，形状为 [batch_size, num_classes]
        """
        # x: [batch_size, in_channels, depth, height, width]
        
        # 初始卷积和池化
        x = self.conv1(x)  # [batch_size, 64, depth//2, height//2, width//2]
        x = self.bn1(x)  # [batch_size, 64, depth//2, height//2, width//2]
        x = self.relu(x)  # [batch_size, 64, depth//2, height//2, width//2]
        x = self.maxpool(x)  # [batch_size, 64, depth//4, height//4, width//4]
        
        # 通过ResNet层
        x = self.layer1(x)  # [batch_size, 64*block.expansion, depth//4, height//4, width//4]
        x = self.layer2(x)  # [batch_size, 128*block.expansion, depth//8, height//8, width//8]
        x = self.layer3(x)  # [batch_size, 256*block.expansion, depth//16, height//16, width//16]
        x = self.layer4(x)  # [batch_size, 512*block.expansion, depth//32, height//32, width//32]
        
        # 全局平均池化
        x = self.avgpool(x)  # [batch_size, 512*block.expansion, 1, 1, 1]
        x = torch.flatten(x, 1)  # [batch_size, 512*block.expansion]
        
        # 分类输出
        x = self.fc(x)  # [batch_size, num_classes]
        
        return x


class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt3D的基本块
    """
    # 类型注解
    dwconv: nn.Conv3d
    norm: nn.LayerNorm
    pwconv1: nn.Linear
    act: nn.GELU
    pwconv2: nn.Linear
    gamma: Optional[nn.Parameter]
    drop_path: nn.Module
    
    def __init__(self, 
                 dim: int,
                 kernel_size: int = 7,
                 drop_path_rate: float = 0.0,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        # 深度可分离卷积
        self.dwconv = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 全连接层
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 层缩放
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        
        # 随机失活路径
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图张量，形状为 [batch_size, dim, depth, height, width]
            
        Returns:
            输出特征图张量，形状为 [batch_size, dim, depth, height, width]
        """
        identity: torch.Tensor = x  # [batch_size, dim, depth, height, width]
        
        # 深度可分离卷积
        x = self.dwconv(x)  # [batch_size, dim, depth, height, width]
        x = x.permute(0, 2, 3, 4, 1)  # [batch_size, d, h, w, dim]
        
        # 层归一化和全连接层
        x = self.norm(x)  # [batch_size, depth, height, width, dim]
        x = self.pwconv1(x)  # [batch_size, depth, height, width, 4*dim]
        x = self.act(x)  # [batch_size, depth, height, width, 4*dim]
        x = self.pwconv2(x)  # [batch_size, depth, height, width, dim]
        
        # 层缩放
        if self.gamma is not None:
            x = self.gamma * x  # [batch_size, depth, height, width, dim]
        
        x = x.permute(0, 4, 1, 2, 3)  # [batch_size, dim, depth, height, width]
        
        # 残差连接和随机失活路径
        x = identity + self.drop_path(x)  # [batch_size, dim, depth, height, width]
        
        return x


class DropPath(nn.Module):
    """
    随机失活路径模块
    用于正则化
    """
    # 类型注解
    drop_prob: float
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为任意
            
        Returns:
            输出张量，形状与输入相同
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # 计算keep_prob
        keep_prob = 1 - self.drop_prob
        
        # 生成随机掩码
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        
        # 应用掩码
        output = x.div(keep_prob) * random_tensor
        
        return output


class ConvNeXt3D(nn.Module):
    """
    3D ConvNeXt 模型
    支持3D多通道图像输入的ConvNeXt实现
    """
    # 类型注解
    stem: nn.Sequential
    stages: nn.ModuleList
    norm: nn.LayerNorm
    head: nn.Linear
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 2,
                 depths: List[int] = [3, 3, 9, 3],
                 dims: List[int] = [96, 192, 384, 768],
                 kernel_size: int = 7,
                 drop_path_rate: float = 0.0,
                 layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.0):
        super().__init__()
        
        # 初始stem层 - 使用通道优先的LayerNorm
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=dims[0],
                kernel_size=4,
                stride=4,
                padding=0
            ),
            ChannelFirstLayerNorm(dims[0], eps=1e-6)
        )
        
        # 计算drop_path率
        dp_rates: List[float] = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur: int = 0
        
        # 构建ConvNeXt层
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock3D(
                    dim=dims[i],
                    kernel_size=kernel_size,
                    drop_path_rate=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            
            # 添加下采样层（除了最后一个stage）
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    ChannelFirstLayerNorm(dims[i], eps=1e-6),
                    nn.Conv3d(
                        in_channels=dims[i],
                        out_channels=dims[i + 1],
                        kernel_size=2,
                        stride=2
                    )
                )
                self.stages.append(downsample)
        
        # 分类头 - 全局平均池化后使用标准LayerNorm
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # 初始化权重
        self._initialize_weights(head_init_scale)
    
    def _initialize_weights(self, head_init_scale: float) -> None:
        """
        初始化模型权重
        
        Args:
            head_init_scale: 分类头权重缩放因子
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 缩放分类头权重
        self.head.weight.data.mul_(head_init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入3D图像张量，形状为 [batch_size, in_channels, depth, height, width]
            
        Returns:
            分类预测张量，形状为 [batch_size, num_classes]
        """
        # x: [batch_size, in_channels, depth, height, width]
        
        # 通过stem层
        x = self.stem(x)  # [batch_size, dims[0], depth//4, height//4, width//4]
        
        # 通过ConvNeXt层
        for stage in self.stages:
            x = stage(x)  # [batch_size, dim, depth//4/(2^i), height//4/(2^i), width//4/(2^i)]
        
        # 全局平均池化
        x = x.mean(dim=(2, 3, 4))  # [batch_size, dims[-1]]
        
        # 分类输出
        x = self.norm(x)  # [batch_size, dims[-1]]
        x = self.head(x)  # [batch_size, num_classes]
        
        return x


def create_resnet3d_model(in_channels: int = 1,
                         img_size: Tuple[int, int, int] = (64, 64, 64),
                         num_classes: int = 2,
                         model_size: str = 'resnet50') -> ResNet3D:
    """
    创建3D ResNet模型
    
    Args:
        in_channels: 输入通道数，图像数据的通道维度
        img_size: 输入图像尺寸 (depth, height, width)，表示3D图像的深度、高度和宽度
        num_classes: 分类类别数，模型输出的类别数量
        model_size: 模型尺寸，可选 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    
    Returns:
        ResNet3D: 3D ResNet模型实例，输入形状为 [batch_size, in_channels, depth, height, width]，
                输出形状为 [batch_size, num_classes]
    """
    model_configs = {
        'resnet18': {
            'block': BasicBlock3D,
            'layers': [2, 2, 2, 2]
        },
        'resnet34': {
            'block': BasicBlock3D,
            'layers': [3, 4, 6, 3]
        },
        'resnet50': {
            'block': Bottleneck3D,
            'layers': [3, 4, 6, 3]
        },
        'resnet101': {
            'block': Bottleneck3D,
            'layers': [3, 4, 23, 3]
        },
        'resnet152': {
            'block': Bottleneck3D,
            'layers': [3, 8, 36, 3]
        }
    }
    
    config = model_configs[model_size]
    return ResNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        **config
    )


def create_convnext3d_model(in_channels: int = 1,
                           img_size: Tuple[int, int, int] = (64, 64, 64),
                           num_classes: int = 2,
                           model_size: str = 'base') -> ConvNeXt3D:
    """
    创建3D ConvNeXt模型
    
    Args:
        in_channels: 输入通道数，图像数据的通道维度
        img_size: 输入图像尺寸 (depth, height, width)，表示3D图像的深度、高度和宽度
        num_classes: 分类类别数，模型输出的类别数量
        model_size: 模型尺寸，可选 'tiny', 'small', 'base', 'large'
    
    Returns:
        ConvNeXt3D: 3D ConvNeXt模型实例，输入形状为 [batch_size, in_channels, depth, height, width]，
                   输出形状为 [batch_size, num_classes]
    """
    model_configs = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'dims': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'dims': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'dims': [192, 384, 768, 1536]
        }
    }
    
    config = model_configs[model_size]
    return ConvNeXt3D(
        in_channels=in_channels,
        num_classes=num_classes,
        **config
    )


def create_resnext3d_model(in_channels: int = 1,
                          img_size: Tuple[int, int, int] = (64, 64, 64),
                          num_classes: int = 2,
                          model_size: str = 'resnext50_32x4d') -> ResNet3D:
    """
    创建3D ResNeXt模型
    
    Args:
        in_channels: 输入通道数，图像数据的通道维度
        img_size: 输入图像尺寸 (depth, height, width)，表示3D图像的深度、高度和宽度
        num_classes: 分类类别数，模型输出的类别数量
        model_size: 模型尺寸，可选 'resnext50_32x4d', 'resnext101_32x8d'
    
    Returns:
        ResNet3D: 3D ResNeXt模型实例，输入形状为 [batch_size, in_channels, depth, height, width]，
                输出形状为 [batch_size, num_classes]
    """
    model_configs = {
        'resnext50_32x4d': {
            'block': Bottleneck3D,
            'layers': [3, 4, 6, 3],
            'groups': 32,
            'width_per_group': 4
        },
        'resnext101_32x8d': {
            'block': Bottleneck3D,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 8
        }
    }
    
    config = model_configs[model_size]
    return ResNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        **config
    )


if __name__ == "__main__":
    # 测试模型
    batch_size = 2
    in_channels = 3
    img_size = (64, 64, 64)
    num_classes = 2
    
    # 测试ResNet3D
    resnet_model = create_resnet3d_model(in_channels=in_channels, img_size=img_size, num_classes=num_classes, model_size='resnet50')
    x = torch.randn(batch_size, in_channels, *img_size)
    y = resnet_model(x)
    print(f"ResNet3D 输入形状: {x.shape}")
    print(f"ResNet3D 输出形状: {y.shape}")
    
    # 测试ConvNeXt3D
    convnext_model = create_convnext3d_model(in_channels=in_channels, img_size=img_size, num_classes=num_classes, model_size='base')
    x = torch.randn(batch_size, in_channels, *img_size)
    y = convnext_model(x)
    print(f"ConvNeXt3D 输入形状: {x.shape}")
    print(f"ConvNeXt3D 输出形状: {y.shape}")