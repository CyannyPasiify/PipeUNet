#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D模型中间层特征规格测试脚本

功能：
    测试ResNet3D、ConvNeXt3D和ResNeXt3D模型的中间层特征输出规格
    使用(2,4,128,128,128)规格的输入
"""

import torch
import torch.nn as nn
from resnet_convnext_3d import (
    create_resnet3d_model,
    create_convnext3d_model,
    create_resnext3d_model,
    ResNet3D,
    ConvNeXt3D
)


def test_resnet3d_features():
    """
    测试ResNet3D模型的中间层特征规格
    """
    print("===== 测试 ResNet3D 中间层特征规格 =====")
    
    # 创建ResNet3D模型
    model = create_resnet3d_model(
        in_channels=4,  # 使用4通道输入
        img_size=(128, 128, 128),
        num_classes=3,  # 多分类任务
        model_size='resnet50'
    )
    
    # 创建输入张量 [batch_size, channels, depth, height, width]
    x = torch.randn(2, 4, 128, 128, 128)
    print(f"输入形状: {x.shape}")
    
    # 注册钩子来捕获中间层输出
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # 为ResNet3D的关键层注册钩子
    model.conv1.register_forward_hook(get_features('conv1'))
    model.bn1.register_forward_hook(get_features('bn1'))
    model.maxpool.register_forward_hook(get_features('maxpool'))
    model.layer1.register_forward_hook(get_features('layer1'))
    model.layer2.register_forward_hook(get_features('layer2'))
    model.layer3.register_forward_hook(get_features('layer3'))
    model.layer4.register_forward_hook(get_features('layer4'))
    model.avgpool.register_forward_hook(get_features('avgpool'))
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 打印中间层特征规格
    print(f"conv1 输出形状: {features['conv1'].shape}")
    print(f"bn1 输出形状: {features['bn1'].shape}")
    print(f"maxpool 输出形状: {features['maxpool'].shape}")
    print(f"layer1 输出形状: {features['layer1'].shape}")
    print(f"layer2 输出形状: {features['layer2'].shape}")
    print(f"layer3 输出形状: {features['layer3'].shape}")
    print(f"layer4 输出形状: {features['layer4'].shape}")
    print(f"avgpool 输出形状: {features['avgpool'].shape}")
    print(f"最终输出形状: {output.shape}")
    print()


def test_convnext3d_features():
    """
    测试ConvNeXt3D模型的中间层特征规格
    """
    print("===== 测试 ConvNeXt3D 中间层特征规格 =====")
    
    # 创建ConvNeXt3D模型
    model = create_convnext3d_model(
        in_channels=4,  # 使用4通道输入
        img_size=(128, 128, 128),
        num_classes=3,  # 多分类任务
        model_size='base'
    )
    
    # 创建输入张量
    x = torch.randn(2, 4, 128, 128, 128)
    print(f"输入形状: {x.shape}")
    
    # 注册钩子来捕获中间层输出
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # 为stem层注册钩子
    model.stem.register_forward_hook(get_features('stem'))
    
    # 为每个stage注册钩子
    for i, stage in enumerate(model.stages):
        stage.register_forward_hook(get_features(f'stage_{i}'))
    
    # 为norm层注册钩子
    model.norm.register_forward_hook(get_features('norm'))
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 打印中间层特征规格
    print(f"stem 输出形状: {features['stem'].shape}")
    for i in range(len(model.stages)):
        if f'stage_{i}' in features:
            print(f"stage_{i} 输出形状: {features[f'stage_{i}'].shape}")
    print(f"norm 输出形状: {features['norm'].shape}")
    print(f"最终输出形状: {output.shape}")
    print()


def test_resnext3d_features():
    """
    测试ResNeXt3D模型的中间层特征规格
    """
    print("===== 测试 ResNeXt3D 中间层特征规格 =====")
    
    # 创建ResNeXt3D模型
    model = create_resnext3d_model(
        in_channels=4,  # 使用4通道输入
        img_size=(128, 128, 128),
        num_classes=3,  # 多分类任务
        model_size='resnext50_32x4d'
    )
    
    # 创建输入张量
    x = torch.randn(2, 4, 128, 128, 128)
    print(f"输入形状: {x.shape}")
    
    # 注册钩子来捕获中间层输出
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # 为ResNeXt3D的关键层注册钩子
    model.conv1.register_forward_hook(get_features('conv1'))
    model.bn1.register_forward_hook(get_features('bn1'))
    model.maxpool.register_forward_hook(get_features('maxpool'))
    model.layer1.register_forward_hook(get_features('layer1'))
    model.layer2.register_forward_hook(get_features('layer2'))
    model.layer3.register_forward_hook(get_features('layer3'))
    model.layer4.register_forward_hook(get_features('layer4'))
    model.avgpool.register_forward_hook(get_features('avgpool'))
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 打印中间层特征规格
    print(f"conv1 输出形状: {features['conv1'].shape}")
    print(f"bn1 输出形状: {features['bn1'].shape}")
    print(f"maxpool 输出形状: {features['maxpool'].shape}")
    print(f"layer1 输出形状: {features['layer1'].shape}")
    print(f"layer2 输出形状: {features['layer2'].shape}")
    print(f"layer3 输出形状: {features['layer3'].shape}")
    print(f"layer4 输出形状: {features['layer4'].shape}")
    print(f"avgpool 输出形状: {features['avgpool'].shape}")
    print(f"最终输出形状: {output.shape}")
    print()


def main():
    """
    主函数，测试所有模型的中间层特征规格
    """
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 测试不同模型的中间层特征规格
    test_resnet3d_features()
    test_convnext3d_features()
    test_resnext3d_features()
    
    print("所有模型中间层特征规格测试完成!")


if __name__ == "__main__":
    main()