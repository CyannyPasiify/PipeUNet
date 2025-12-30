#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失函数管道类

功能概述：
    提供两个自定义损失函数类，专为医学图像分割任务优化：
    1. CrossEntropyLoss：支持类别权重的多分类交叉熵损失函数，基于PyTorch内置实现并增强
    2. FocalLoss：基于论文实现的焦点损失函数，有效解决类别不平衡问题
    
核心功能：
    1. 多分类交叉熵：支持设置类别权重，处理2D/3D多分类医学图像分割任务
    2. Focal Loss：通过alpha类别平衡和gamma聚焦参数，有效处理极度不平衡的医学图像分割场景

参数说明：
    - CrossEntropyLoss：
      * weight: 类别权重张量，形状为[C]，用于平衡不同类别的重要性
      * reduction: 损失聚合方式（'mean':平均值，'sum':总和，'none':原始值）
    
    - FocalLoss：
      * alpha: 类别平衡参数，张量形状为[C]，用于处理类别频率不平衡
      * gamma: 聚焦参数（≥0），调节难易样本权重，越大越关注难分样本
      * reduction: 损失聚合方式（'mean':平均值，'sum':总和，'none':原始值）

使用示例：
    # 创建多分类交叉熵损失实例
    cross_entropy_loss = CrossEntropyLoss(
        weight=torch.tensor([1.0, 2.0, 3.0]),  # 为每个类别设置不同权重
        reduction='mean'
    )
    
    # 创建Focal Loss实例
    focal_loss = FocalLoss(
        alpha=torch.tensor([0.25, 0.75]),  # 类别平衡权重
        gamma=2.0,                         # 聚焦参数，默认2.0
        reduction='mean'
    )
    
    # 计算损失
    outputs = model(inputs)
    loss = focal_loss(outputs, targets)
    loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List


class CrossEntropyLoss(nn.Module):
    """
    支持类别权重的多分类交叉熵损失函数
    基于PyTorch的nn.CrossEntropyLoss实现，增加了输入校验和更友好的错误提示
    """
    
    # 类属性类型注解
    weight: Optional[torch.Tensor]
    reduction: str
    cross_entropy: nn.CrossEntropyLoss
    
    def __init__(self, 
                 weight: Optional[torch.Tensor] = None, 
                 reduction: str = 'mean'):
        """
        初始化多分类交叉熵损失函数
        
        Args:
            weight: 类别权重张量，形状为[C]，其中C是类别数量
            reduction: 损失计算方式，可选值为'mean', 'sum', 'none'
                'mean': 返回损失的平均值
                'sum': 返回损失的总和
                'none': 返回每个样本的损失值
                
        Raises:
            ValueError: 当reduction参数无效时抛出
            TypeError: 当weight参数类型不正确时抛出
        """
        super(CrossEntropyLoss, self).__init__()
        
        # 验证weight参数类型
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"weight参数必须是torch.Tensor类型，当前类型: {type(weight)}")
            if weight.dim() != 1:
                raise ValueError(f"weight参数必须是一维张量，当前维度: {weight.dim()}")
            if torch.any(weight <= 0):
                raise ValueError("weight参数中的所有元素必须大于0")
        self.weight: Optional[torch.Tensor] = weight
        
        # 验证reduction参数
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"不支持的reduction参数: {reduction}，请使用'mean', 'sum'或'none'")
        self.reduction: str = reduction
        
        # 初始化PyTorch内置的交叉熵损失函数
        self.cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        
        Args:
            input: 模型预测的原始分数，形状为[N, C]或[N, C, d1, d2, ...]
                   其中N是批量大小，C是类别数量
            target: 目标标签，形状为[N]或[N, d1, d2, ...]（类别索引）或[N, C]（one-hot编码）
                    
        Returns:
            torch.Tensor: 计算得到的损失值
        """
        # 使用PyTorch内置的交叉熵损失函数计算损失
        return self.cross_entropy(input, target)


class FocalLoss(nn.Module):
    """
    Focal Loss损失函数，用于处理类别不平衡问题
    实现了论文"Focal Loss for Dense Object Detection"中的公式，并增加了输入校验
    """
    
    # 类属性类型注解
    alpha: Optional[torch.Tensor]
    gamma: float
    reduction: str
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        初始化Focal Loss损失函数
        
        Args:
            alpha: 类别平衡参数，张量形状为[C]，表示每个类别的权重
            gamma: 聚焦参数，用于调节难易样本的权重，通常取2.0
                   gamma越大，对困难样本的关注越多
            reduction: 损失计算方式，可选值为'mean', 'sum', 'none'
                'mean': 返回损失的平均值
                'sum': 返回损失的总和
                'none': 返回每个样本的损失值
                
        Raises:
            ValueError: 当参数值无效时抛出
            TypeError: 当参数类型不正确时抛出
        """
        super(FocalLoss, self).__init__()

        # 验证alpha参数
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                if alpha.dim() != 1:
                    raise ValueError(f"alpha参数必须是一维张量，当前维度: {alpha.dim()}")
                if torch.any(alpha < 0):
                    raise ValueError("alpha参数中的所有元素不能小于0")
            else:
                raise TypeError(f"alpha参数必须是数字或torch.Tensor类型，当前类型: {type(alpha)}")
        self.alpha: Optional[torch.Tensor] = alpha
        
        # 验证gamma参数
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"gamma参数必须是数字类型，当前类型: {type(gamma)}")
        if gamma < 0:
            raise ValueError(f"gamma参数必须大于等于0，当前值: {gamma}")
        self.gamma: float = gamma
        
        # 验证reduction参数
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"不支持的reduction参数: {reduction}，请使用'mean', 'sum'或'none'")
        self.reduction: str = reduction
    
    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算Focal Loss
        
        Args:
            input: 模型预测的原始分数，形状为[N, C]或[N, C, d1, d2, ...]
                   其中N是批量大小，C是类别数量
            target: 目标标签，支持以下格式：
                    1. 类别索引：形状为[N]或[N, d1, d2, ...]
                    2. one-hot编码/概率向量：形状为[N, C]或[N, C, d1, d2, ...]
                    
        Returns:
            torch.Tensor: 计算得到的Focal Loss值
            
        Raises:
            TypeError: 当输入类型不正确时
            ValueError: 当输入形状不匹配或无效时
        """
        # 类型校验
        if not isinstance(input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError(f"input和target必须是torch.Tensor类型，当前类型: input={type(input)}, target={type(target)}")
        
        # 维度校验
        if input.dim() < 2:
            raise ValueError(f"input必须至少是2维张量 [N, C, ...]，当前维度: {input.dim()}")
        
        # 批量大小校验
        if input.size(0) != target.size(0):
            raise ValueError(f"输入批量大小 {input.size(0)} 与目标批量大小 {target.size(0)} 不匹配")
        
        # 记录原始形状信息
        input_shape: Tuple[int, ...] = input.shape  # [N, C]; [N, C, d1, d2, ...]
        target_shape: Tuple[int, ...] = target.shape  # [N], [N, d1, d2, ...]; [N, C], [N, C, d1, d2, ...]
        input_dim: int = len(input_shape)
        target_dim: int = len(target_shape)
        
        # 检查target维度是否与input兼容
        # 情况1: target是类别索引，维度应比input少1
        # 情况2: target是one-hot编码或概率向量，维度应与input相同
        if not ((target_dim == input_dim - 1) or (target_dim == input_dim)):
            raise ValueError(f"target维度 {target_dim} 与input维度 {input_dim} 不兼容。target维度应为input.dim()-1（类别索引）或input.dim()（one-hot/概率向量）")
        
        # 如果target是one-hot编码或概率向量，检查类别维度是否匹配
        if target_dim == input_dim and target_shape[1] != input_shape[1]:
            raise ValueError(f"target类别维度 {target_shape[1]} 与input类别维度 {input_shape[1]} 不匹配")
        
        # 如果target是类别索引，确保其值在有效范围内
        if target_dim == input_dim - 1:
            if target.min() < 0 or target.max() >= input_shape[1]:
                raise ValueError(f"类别索引值超出有效范围 [0, {input_shape[1]-1}]，实际范围 [{target.min()}, {target.max()}]")
        
        # 处理不同维度的输入
        if input_dim > 2:
            # 多维度输入（如图像分割），需要将空间维度合并
            input = input.view(input_shape[0], input_shape[1], -1)  # [N, C, D=prod(d1, d2, ...)]
            input = input.transpose(1, 2)  # [N, D, C]
            input = input.contiguous().view(-1, input_shape[1])  # [N*D, C]
            # 对应地调整target形状
            if target_dim == input_dim - 1:
                # target是类别索引 [N, d1, d2, ...] -> [N*D]
                target = target.view(-1)
            else:
                # target是one-hot编码或概率向量 [N, C, d1, d2, ...] -> [N*D, C]
                target = target.view(target_shape[0], target_shape[1], -1)  # [N, C, D=prod(d1, d2, ...)]
                target = target.transpose(1, 2)  # [N, D, C]
                target = target.contiguous().view(-1, target_shape[1])  # [N*D, C]

        # 如果target是类别索引，将其转换为one-hot编码
        if target_dim == input_dim - 1:
            # target [N]; [N*D] -> [N, C]; [N*D, C]
            target = F.one_hot(target, num_classes=input_shape[1])
        
        # target强制float，校验one-hot/soft label合法性
        target = target.float()
        if torch.any(target < 0):
            raise ValueError("target为one-hot/soft label时，所有元素必须非负")

        # 计算softmax概率
        # prob = F.softmax(input, dim=1)  # [N, C]; [N*D, C]
        # log_prob = torch.log(prob)  # [N, C]; [N*D, C]
        # 比以上更稳定的方法，可处理异常值
        log_prob: torch.Tensor = F.log_softmax(input, dim=1)  # [N, C]; [N*D, C]
        prob: torch.Tensor = torch.exp(log_prob)  # [N, C]; [N*D, C]
        
        # alpha设备对齐
        alpha: Optional[torch.Tensor] = self.alpha.to(input.device) if self.alpha is not None else None

        # 计算focal loss [N]; [N*D]
        focal_loss: torch.Tensor
        if alpha is not None:
            focal_loss = - (target * alpha.unsqueeze(0) * ((1 - prob) ** self.gamma) * log_prob).sum(dim=1)  # [N]; [N*D]
        else:
            focal_loss = - (target * ((1 - prob) ** self.gamma) * log_prob).sum(dim=1)  # [N]; [N*D]
        
        # 根据reduction参数进行聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 对于reduction='none'，需要将focal_loss恢复为与输入相同的形状
            # 对于多维度输入，需要恢复空间维度
            if input_dim > 2:
                # 计算原始的空间维度形状
                spatial_dims: Tuple[int, ...] = input_shape[2:]  # 获取除[N, C]外的所有维度
                # 恢复形状 [N*D] -> [N, d1, d2, ...]
                return focal_loss.view(input_shape[0], *spatial_dims)
            else:
                # 对于2D[N, C]输入，直接返回[N]形状
                return focal_loss


# 示例用法
def example_usage() -> None:
    """
    演示损失函数的使用方法，包括不同的输入形式和参数配置
    """
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    print("\n===== CrossEntropyLoss 使用示例 =====\n")
    
    # 创建随机输入张量 [batch_size, num_classes]
    # 示例1: 简单的2D输入 (批量大小=4, 类别数=3)
    print("示例1: 基础分类任务 (2D输入)")
    input_2d: torch.Tensor = torch.randn(4, 3)  # [4, 3]
    print(f"输入形状: {input_2d.shape}")
    print(f"输入数据: {input_2d}\n")
    
    # 示例1.1: 使用类别索引形式的target
    target_indices: torch.Tensor = torch.randint(0, 3, (4,))  # [4]
    print(f"类别索引形式的target: {target_indices}")
    
    # 创建不同reduction模式的损失函数实例
    cross_entropy_mean: CrossEntropyLoss = CrossEntropyLoss(reduction='mean')
    cross_entropy_sum: CrossEntropyLoss = CrossEntropyLoss(reduction='sum')
    cross_entropy_none: CrossEntropyLoss = CrossEntropyLoss(reduction='none')
    
    # 计算损失
    loss_mean_indices: torch.Tensor = cross_entropy_mean(input_2d, target_indices)
    loss_sum_indices: torch.Tensor = cross_entropy_sum(input_2d, target_indices)
    loss_none_indices: torch.Tensor = cross_entropy_none(input_2d, target_indices)
    
    print(f"mean reduction 损失: {loss_mean_indices:.4f}")
    print(f"sum reduction 损失: {loss_sum_indices:.4f}")
    print(f"none reduction 损失: {loss_none_indices}\n")
    
    # 示例1.1.1: 使用one-hot编码形式的target
    print("示例1.1.1: 使用one-hot编码形式的target")
    # 将类别索引转换为one-hot编码
    num_classes: int = 3
    target_onehot: torch.Tensor = torch.nn.functional.one_hot(target_indices, num_classes=num_classes).float()
    print(f"one-hot编码形式的target形状: {target_onehot.shape}")
    print(f"one-hot编码形式的target:\n{target_onehot}")
    
    loss_onehot: torch.Tensor = cross_entropy_mean(input_2d, target_onehot)
    print(f"直接使用one-hot编码的损失: {loss_onehot:.4f}")
    print(f"验证：与直接使用类别索引的损失是否相同: {torch.isclose(loss_mean_indices, loss_onehot)}")
    
    # 示例1.2: 使用带权重的CrossEntropyLoss
    print("示例1.2: 使用类别权重")
    class_weights: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
    weighted_cross_entropy: CrossEntropyLoss = CrossEntropyLoss(weight=class_weights, reduction='mean')
    loss_weighted: torch.Tensor = weighted_cross_entropy(input_2d, target_indices)
    print(f"类别权重: {class_weights}")
    print(f"带权重的损失: {loss_weighted:.4f}\n")
    
    # 示例2: 多维度输入 (如图像分割任务)
    print("示例2: 图像分割任务 (多维度输入)")
    input_4d: torch.Tensor = torch.randn(2, 3, 4, 4)  # [batch_size, channels, height, width]
    target_3d: torch.Tensor = torch.randint(0, 3, (2, 4, 4))  # [batch_size, height, width]
    print(f"输入形状: {input_4d.shape}")
    print(f"目标形状: {target_3d.shape}")
    
    loss_4d_mean: torch.Tensor = cross_entropy_mean(input_4d, target_3d)
    print(f"多维度输入的损失 (mean reduction): {loss_4d_mean:.4f}\n")
    
    print("\n===== FocalLoss 使用示例 =====\n")
    
    # 示例3: FocalLoss基本用法
    print("示例3: FocalLoss 基础用法")
    # 使用相同的输入和target
    focal_loss_mean: FocalLoss = FocalLoss(gamma=2.0, reduction='mean')
    focal_loss_sum: FocalLoss = FocalLoss(gamma=2.0, reduction='sum')
    focal_loss_none: FocalLoss = FocalLoss(gamma=2.0, reduction='none')
    
    fl_mean_indices: torch.Tensor = focal_loss_mean(input_2d, target_indices)
    fl_sum_indices: torch.Tensor = focal_loss_sum(input_2d, target_indices)
    fl_none_indices: torch.Tensor = focal_loss_none(input_2d, target_indices)
    
    print(f"FocalLoss (gamma=2.0, mean reduction): {fl_mean_indices:.4f}")
    print(f"FocalLoss (gamma=2.0, sum reduction): {fl_sum_indices:.4f}")
    print(f"FocalLoss (gamma=2.0, none reduction): {fl_none_indices}\n")
    
    # 示例4: 不同gamma值的FocalLoss
    print("示例4: 不同gamma值的影响")
    focal_gamma_0: FocalLoss = FocalLoss(gamma=0.0, reduction='mean')  # 相当于标准交叉熵
    focal_gamma_1: FocalLoss = FocalLoss(gamma=1.0, reduction='mean')
    focal_gamma_3: FocalLoss = FocalLoss(gamma=3.0, reduction='mean')
    
    fl_gamma_0: torch.Tensor = focal_gamma_0(input_2d, target_indices)
    fl_gamma_1: torch.Tensor = focal_gamma_1(input_2d, target_indices)
    fl_gamma_3: torch.Tensor = focal_gamma_3(input_2d, target_indices)
    
    print(f"FocalLoss (gamma=0.0): {fl_gamma_0:.4f} (相当于标准交叉熵)")
    print(f"FocalLoss (gamma=1.0): {fl_gamma_1:.4f}")
    print(f"FocalLoss (gamma=3.0): {fl_gamma_3:.4f}")
    print("注: gamma越大，对困难样本的关注越多\n")
    
    # 示例5: 使用alpha参数的FocalLoss
    print("示例5: 使用alpha参数的FocalLoss")
    # 多分类情况下的alpha权重
    alpha_weights: torch.Tensor = torch.tensor([0.25, 0.25, 0.5])
    focal_alpha: FocalLoss = FocalLoss(gamma=2.0, alpha=alpha_weights, reduction='mean')
    fl_alpha: torch.Tensor = focal_alpha(input_2d, target_indices)
    
    print(f"类别权重alpha: {alpha_weights}")
    print(f"带alpha权重的FocalLoss: {fl_alpha:.4f}\n")
    
    # 示例6: 二分类情况下的FocalLoss
    print("示例6: 二分类任务中的FocalLoss")
    # 创建二分类输入
    input_binary: torch.Tensor = torch.randn(4, 2)
    target_binary: torch.Tensor = torch.randint(0, 2, (4,))
    
    # 使用标量alpha（二分类场景）
    alpha_weights: torch.Tensor = torch.tensor([0.25, 0.75])
    focal_binary: FocalLoss = FocalLoss(gamma=2.0, alpha=alpha_weights, reduction='mean')
    fl_binary: torch.Tensor = focal_binary(input_binary, target_binary)
    
    print(f"二分类输入形状: {input_binary.shape}")
    print(f"二分类target: {target_binary}")
    print(f"二分类FocalLoss (alpha={alpha_weights}): {fl_binary:.4f}\n")
    
    # 示例7: 多维度输入的FocalLoss
    print("示例7: 分割任务中的FocalLoss")
    fl_4d_mean: torch.Tensor = focal_loss_mean(input_4d, target_3d)
    print(f"多维度输入的FocalLoss: {fl_4d_mean:.4f}\n")
    
    # 示例8: 梯度计算演示
    print("示例8: 梯度计算演示")
    # 创建可训练的参数
    params: torch.nn.Parameter = torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))
    input_grad: torch.Tensor = torch.nn.functional.linear(torch.randn(4, 3), params)
    
    # 计算损失
    loss_for_grad: torch.Tensor = focal_loss_mean(input_grad, target_indices)
    # 反向传播
    loss_for_grad.backward()
    
    print(f"损失值: {loss_for_grad:.4f}")
    print(f"参数梯度存在: {params.grad is not None}")
    print(f"梯度形状: {params.grad.shape if params.grad is not None else 'None'}\n")
    
    print("\n===== 损失函数对比 =====\n")
    
    # 对比CrossEntropyLoss和FocalLoss
    print("CrossEntropyLoss vs FocalLoss:")
    print(f"CrossEntropyLoss: {loss_mean_indices:.4f}")
    print(f"FocalLoss (gamma=2.0): {fl_mean_indices:.4f}")
    print("注: FocalLoss通常比CrossEntropyLoss值更小，因为它聚焦于困难样本")
    
    print("\n===== 使用建议 =====\n")
    print("1. 当类别分布均衡时，可以使用标准的CrossEntropyLoss")
    print("2. 当类别分布不均衡时，考虑以下选项：")
    print("   - 使用带weight参数的CrossEntropyLoss")
    print("   - 结合使用FocalLoss的alpha参数进行类别平衡")
    print("   - 使用FocalLoss，调整gamma参数（通常取2.0）")
    print("3. 对于分割任务，损失函数会自动处理多维度输入")
    print("4. 选择reduction模式：")
    print("   - 'mean': 训练时常用，对批量大小不敏感")
    print("   - 'sum': 某些特定场景下使用")
    print("   - 'none': 需要对每个样本的损失进行单独处理时使用")


if __name__ == "__main__":
    # 运行示例用法
    example_usage()
