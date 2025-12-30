#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于HuggingFace Transformers的3D ResNet实现

功能概述：
    提供支持3D多通道图像输入的ResNet实现，基于HuggingFace Transformers
    用于医学图像分类等3D多通道图像任务

核心功能：
    1. HFResNet3D: 基于HuggingFace的3D ResNet图像分类网络
    2. 支持通过AutoModelForImageClassification创建预训练模型
    3. 支持多通道3D图像输入和多分类任务
    4. 可配置的模型参数和尺寸
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any
from transformers import AutoModelForImageClassification, AutoConfig


class HFResNet3D:
    """
    基于HuggingFace Transformers的3D ResNet模型
    使用AutoModelForImageClassification加载预训练模型并适配3D输入
    """
    # 类型注解
    model: nn.Module

    def __new__(
            cls,
            model_name='nwirandx/medicalnet-resnet3d10-23datasets'
    ):
        # use pretrained model
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        return model


class HFResNet3DCustom:
    """
    基于HuggingFace Transformers的3D ResNet模型
    使用AutoModelForImageClassification加载预训练模型并适配3D输入
    支持输入包装和分类器头数修改
    """
    # 类型注解
    model: nn.Module

    def __new__(
            cls,
            model_name: str = 'nwirandx/medicalnet-resnet3d10-23datasets',
            num_classes: int = 2,
            pre_trained: bool = True,
    ):
        return HFResNet3DWrapped(model_name, num_classes, pre_trained)


class HFResNet3DWrapped(nn.Module):
    def __init__(
            self,
            model_name: str = 'nwirandx/medicalnet-resnet3d10-23datasets',
            num_classes: int = 2,
            pre_trained: bool = True
    ) -> None:
        super().__init__()

        if pre_trained:
            # use pretrained model
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            # use a model from scratch
            self.model = AutoModelForImageClassification.from_config(
                config,
                trust_remote_code=True
            )
        self.head = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_classes, bias=True)
        self.model.classifier[1] = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            前向传播

            Args:
                x: 输入3D图像张量，形状为 [batch_size, in_channels, depth, height, width]

            Returns:
                分类预测张量，形状为 [batch_size, num_classes]
        """
        # x: [batch_size, in_channels, depth, height, width]
        N, C, D, H, W = x.shape
        batch_x = x.view(N * C, 1, D, H, W)
        feat: torch.Tensor = self.model(batch_x).logits
        feat = feat.view(N, C, -1)

        # 合并特征
        x = feat.sum(dim=1)  # [batch_size, classfier_in_features]

        # 分类输出
        x = self.head(x)  # [batch_size, num_classes]

        return x


if __name__ == "__main__":
    # 测试模型
    # batch_size = 2
    # in_channels = 1
    # img_size = (64, 64, 64)
    # num_classes = 2
    #
    # print("测试基本HFResNet3D模型:")
    # # 测试基本HFResNet3D模型
    # hf_resnet_model = HFResNet3D(
    #     'nwirandx/medicalnet-resnet3d101'
    # )
    # print(hf_resnet_model)
    # print(hf_resnet_model.classifier[1].in_features)
    # x = torch.randn(batch_size, in_channels, *img_size)
    # y = hf_resnet_model(x)
    # print(f"输入形状: {x.shape}")
    # print(f"输出: {y}")
    # print(f"输出形状: {y['logits'].shape}")
    # print(f"模型类型: {type(hf_resnet_model).__name__}")

    batch_size = 2
    in_channels = 4
    img_size = (64, 64, 64)
    num_classes = 3
    print("测试HFResNet3DCustom模型:")
    # 测试HFResNet3DCustom模型
    hf_resnet_model = HFResNet3DCustom(
        'nwirandx/medicalnet-resnet3d10-23datasets', num_classes
    )
    print(hf_resnet_model)
    x = torch.randn(batch_size, in_channels, *img_size)
    y = hf_resnet_model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出: {y}")
    print(f"输出形状: {y.shape}")
    print(f"模型类型: {type(hf_resnet_model).__name__}")

    print("\n模型测试完成！")
