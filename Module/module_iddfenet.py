#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDDFENet Lightning 模块

功能概述：
    提供基于 PyTorch Lightning 的 IDDFENet 模块实现
    集成模型、优化器、损失函数和学习率调度器
    兼容 EsophagusDataModule 数据格式
    支持多种性能指标的记录和可视化

核心功能：
    1. 支持 IDDFENet 模型的训练和评估
    2. 集成优化器配置器、损失函数配置器和学习率调度器配置器
    3. 提供标准化的训练、验证和测试流程
    4. 支持自动日志记录和检查点保存
    5. 兼容 EsophagusDataModule 的数据输入格式
    6. 记录多种性能指标（准确率、精确率、召回率、F1分数、AUROC、混淆矩阵等）
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, List
import lightning as L

# 导入模型
from Network.IDDFENet.IDDFENet import IDDFENet

# 导入优化器配置器
from Optimizer.optimizer_configurer import AdamW

# 导入损失函数配置器
from Loss.loss_configurer import CrossEntropyLoss, FocalLoss

# 导入学习率调度器配置器
from Scheduler.schedule_configurer import CosineAnnealingLR

# 导入性能指标
from Metric.metric_configurer import (
    BinaryStatScores,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinarySpecificity,
    BinaryROC,
    BinaryPrecisionRecallCurve
)


class IDDFENetLightningModule(L.LightningModule):
    """
    IDDFENet Lightning 模块类
    兼容 EsophagusDataModule 的数据输入格式
    记录多种性能指标
    """

    def __init__(self,
                 model_config: Dict[str, Any] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 T_max: int = 100,
                 eta_min: float = 1e-6,
                 loss_type: str = 'cross_entropy',
                 loss_config: Optional[Dict[str, Any]] = None,
                 soft_label_ratio: float = 0.0):
        """
        初始化 IDDFENet Lightning 模块
        
        Args:
            model_config: 模型配置参数
            learning_rate: 初始学习率
            weight_decay: 权重衰减系数
            T_max: CosineAnnealingLR 的最大迭代次数
            eta_min: CosineAnnealingLR 的最小学习率
            class_weights: 类别权重张量，用于损失函数
            loss_type: 损失函数类型，可选值: 'cross_entropy', 'focal'
            loss_config: 损失函数配置参数
            soft_label_ratio: 标签软化程度，0-无软化，1-完全软化（所有标签均相等）
        """
        super().__init__()

        # 保存超参数
        self.save_hyperparameters()

        # 设置默认模型配置
        if model_config is None:
            model_config = {}

        # 创建模型
        self.model = self._create_model(model_config)

        # 设置默认损失函数配置
        if loss_config is None:
            loss_config = {}

        # 初始化损失函数
        if loss_type == 'cross_entropy':
            if 'weight' in loss_config and loss_config['weight'] is not None:
                loss_config['weight'] = torch.Tensor(loss_config['weight']).float()
            self.loss_fn = CrossEntropyLoss(**loss_config)
        elif loss_type == 'focal':
            if 'alpha' in loss_config and loss_config['alpha'] is not None:
                loss_config['alpha'] = torch.Tensor(loss_config['alpha']).float()
            self.loss_fn = FocalLoss(**loss_config)
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")

        # 初始化指标
        self.train_stat_scores = BinaryStatScores()
        self.train_acc = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.train_auroc = BinaryAUROC()
        self.train_ap = BinaryAveragePrecision()
        self.train_specificity = BinarySpecificity()

        self.val_stat_scores = BinaryStatScores()
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()
        self.val_specificity = BinarySpecificity()
        # self.val_confusion_matrix = BinaryConfusionMatrix()
        # self.val_roc = BinaryROC()
        # self.val_prc = BinaryPrecisionRecallCurve()

        self.test_stat_scores = BinaryStatScores()
        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()
        self.test_ap = BinaryAveragePrecision()
        self.test_specificity = BinarySpecificity()
        # self.test_confusion_matrix = BinaryConfusionMatrix()
        # self.test_roc = BinaryROC()
        # self.test_prc = BinaryPrecisionRecallCurve()

    def _create_model(self, model_config: Dict[str, Any]) -> IDDFENet:
        """
        创建模型实例
        
        Args:
            model_config: 模型配置参数
            
        Returns:
            模型实例
        """
        return IDDFENet(**model_config)

    def forward(self, x: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, in_channels, depth, height, width]
            
        Returns:
            输出张量，形状为 [batch_size, num_classes]
        """
        return self.model(x, radiomics)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        
        Args:
            batch: 训练批次数据，兼容 EsophagusDataModule 格式，包含 'pre_img', 'post_img' 和 'label' 键
            batch_idx: 批次索引
            
        Returns:
            训练损失
        """
        # 从批次中提取数据（兼容 EsophagusDataModule 格式）
        # 合并前图像和后图像作为模型输入
        pre_img = batch['pre_img']
        post_img = batch['post_img']
        pre_mask = batch['pre_mask']
        post_mask = batch['post_mask']

        pre_masked_img = pre_mask * pre_img
        post_masked_img = post_mask * post_img

        radiomics = batch['radiomics']

        # 合并通道维度
        x = torch.cat([pre_img, pre_mask, post_img, post_mask], dim=1)

        # 获取标签
        y = batch['label']
        maybe_soft_y = y
        # 软化标签
        if self.hparams.soft_label_ratio > 0:
            class_count: int = y.size(1)
            soft_ratio: float = min(self.hparams.soft_label_ratio, 1.0)
            maybe_soft_y = y * (1.0 - soft_ratio) + soft_ratio / class_count

        # 前向传播
        logits = self(x, radiomics)

        # 计算损失
        loss = self.loss_fn(logits, maybe_soft_y)

        # 计算预测
        binary_probs = F.softmax(logits, dim=1)[:, 1]  # 使用二分类分析方法，只取正类预测值（越接近于1，越可能为正类）
        binary_preds = torch.argmax(logits, dim=1)  # 取预测类别（0或1）
        binary_y = y[:, 1].int()  # 将多维标签转换为二分类标签（1为正类，0为负类）

        # 更新和记录训练指标
        train_stat_scores = self.train_stat_scores(binary_preds, binary_y).float()
        tp, fp, tn, fn, sup = train_stat_scores[0], train_stat_scores[1], train_stat_scores[2], train_stat_scores[3], \
            train_stat_scores[4]
        self.train_acc.update(binary_preds, binary_y)
        self.train_precision.update(binary_preds, binary_y)
        self.train_recall.update(binary_preds, binary_y)
        self.train_f1.update(binary_preds, binary_y)
        self.train_auroc.update(binary_probs, binary_y)
        self.train_ap.update(binary_probs, binary_y)
        self.train_specificity.update(binary_preds, binary_y)

        # 记录训练指标
        self.log('train/idx', batch['idx'].float().mean(), on_step=True, prog_bar=False)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/tp', tp, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('train/fp', fp, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('train/tn', tn, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('train/fn', fn, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('train/sup', sup, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train/recall', self.train_recall, on_step=False, on_epoch=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True)
        self.log('train/auroc', self.train_auroc, on_step=False, on_epoch=True)
        self.log('train/ap', self.train_ap, on_step=False, on_epoch=True)
        self.log('train/specificity', self.train_specificity, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        验证步骤
        
        Args:
            batch: 验证批次数据，兼容 EsophagusDataModule 格式，包含 'pre_img', 'post_img' 和 'label' 键
            batch_idx: 批次索引
            
        Returns:
            包含验证损失和预测结果的字典
        """
        # 从批次中提取数据（兼容 EsophagusDataModule 格式）
        # 合并前图像和后图像作为模型输入
        pre_img = batch['pre_img']
        post_img = batch['post_img']
        pre_mask = batch['pre_mask']
        post_mask = batch['post_mask']

        pre_masked_img = pre_mask * pre_img
        post_masked_img = post_mask * post_img

        radiomics = batch['radiomics']

        # 合并通道维度
        x = torch.cat([pre_img, pre_mask, post_img, post_mask], dim=1)

        # 获取标签
        y = batch['label']

        # 前向传播
        logits = self(x, radiomics)

        # 计算损失
        loss = self.loss_fn(logits, y)

        # 计算预测
        binary_probs = F.softmax(logits, dim=1)[:, 1]  # 使用二分类分析方法，只取正类预测值（越接近于1，越可能为正类）
        binary_preds = torch.argmax(logits, dim=1)  # 取预测类别（0或1）
        binary_y = y[:, 1].int()  # 将多维标签转换为二分类标签（1为正类，0为负类）

        # 更新和记录验证指标
        val_stat_scores = self.val_stat_scores(binary_preds, binary_y).float()
        tp, fp, tn, fn, sup = val_stat_scores[0], val_stat_scores[1], val_stat_scores[2], val_stat_scores[3], \
            val_stat_scores[4]
        self.val_acc.update(binary_preds, binary_y)
        self.val_precision.update(binary_preds, binary_y)
        self.val_recall.update(binary_preds, binary_y)
        self.val_f1.update(binary_preds, binary_y)
        self.val_auroc.update(binary_probs, binary_y)
        self.val_ap.update(binary_probs, binary_y)
        self.val_specificity.update(binary_preds, binary_y)
        # self.val_confusion_matrix.update(binary_probs, binary_y)
        # self.val_roc.update(binary_probs, binary_y)
        # self.val_prc.update(binary_preds, binary_y)

        # 记录验证指标
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/tp', tp, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('val/fp', fp, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('val/tn', tn, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('val/fn', fn, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('val/sup', sup, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True)
        self.log('val/ap', self.val_ap, on_step=False, on_epoch=True)
        self.log('val/specificity', self.val_specificity, on_step=False, on_epoch=True)

        return {'val/loss': loss, 'preds': binary_preds, 'probs': binary_probs, 'labels': binary_y}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        测试步骤
        
        Args:
            batch: 测试批次数据，兼容 EsophagusDataModule 格式，包含 'pre_img', 'post_img' 和 'label' 键
            batch_idx: 批次索引
            
        Returns:
            包含测试损失和预测结果的字典
        """
        # 从批次中提取数据（兼容 EsophagusDataModule 格式）
        # 合并前图像和后图像作为模型输入
        pre_img = batch['pre_img']
        post_img = batch['post_img']
        pre_mask = batch['pre_mask']
        post_mask = batch['post_mask']

        pre_masked_img = pre_mask * pre_img
        post_masked_img = post_mask * post_img

        radiomics = batch['radiomics']

        # 合并通道维度
        x = torch.cat([pre_img, pre_mask, post_img, post_mask], dim=1)

        # 获取标签
        y = batch['label']

        # 前向传播
        logits = self(x, radiomics)

        # 计算损失
        loss = self.loss_fn(logits, y)

        # 计算预测
        binary_probs = F.softmax(logits, dim=1)[:, 1]  # 使用二分类分析方法，只取正类预测值（越接近于1，越可能为正类）
        binary_preds = torch.argmax(logits, dim=1)  # 取预测类别（0或1）
        binary_y = y[:, 1].int()  # 将多维标签转换为二分类标签（1为正类，0为负类）

        # 更新和记录测试指标
        test_stat_scores = self.test_stat_scores(binary_preds, binary_y).float()
        tp, fp, tn, fn, sup = test_stat_scores[0], test_stat_scores[1], test_stat_scores[2], test_stat_scores[3], \
            test_stat_scores[4]
        self.test_acc.update(binary_preds, binary_y)
        self.test_precision.update(binary_preds, binary_y)
        self.test_recall.update(binary_preds, binary_y)
        self.test_f1.update(binary_preds, binary_y)
        self.test_auroc.update(binary_probs, binary_y)
        self.test_ap.update(binary_probs, binary_y)
        self.test_specificity.update(binary_preds, binary_y)
        # self.test_confusion_matrix.update(binary_preds, binary_y)

        # 记录测试指标
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/tp', tp, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('test/fp', fp, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('test/tn', tn, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('test/fn', fn, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('test/sup', sup, on_step=False, on_epoch=True, prog_bar=True, reduce_fx='sum')
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test/auroc', self.test_auroc, on_step=False, on_epoch=True)
        self.log('test/ap', self.test_ap, on_step=False, on_epoch=True)
        self.log('test/specificity', self.test_specificity, on_step=False, on_epoch=True)

        # 存储混淆矩阵（在测试结束后可以访问）
        # self.test_confusion_matrix 会在测试结束后更新

        return {'test/loss': loss, 'preds': binary_preds, 'probs': binary_probs, 'labels': binary_y}

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        配置优化器和学习率调度器
        
        Returns:
            包含优化器和调度器配置的字典
        """
        # 使用导入的 AdamW 优化器
        self.optimizer = AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate
        )

        # 使用导入的 CosineAnnealingLR 调度器
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.hparams.T_max,
            eta_min=self.hparams.eta_min
        )

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        预测步骤
        
        Args:
            batch: 预测批次数据，兼容 EsophagusDataModule 格式，包含 'pre_img', 'post_img' 键
            batch_idx: 批次索引
            
        Returns:
            包含预测结果和概率的字典
        """
        # 从批次中提取数据（兼容 EsophagusDataModule 格式）
        # 合并前图像和后图像作为模型输入
        pre_img = batch['pre_img']
        post_img = batch['post_img']
        pre_mask = batch['pre_mask']
        post_mask = batch['post_mask']

        pre_masked_img = pre_mask * pre_img
        post_masked_img = post_mask * post_img

        radiomics = batch['radiomics']

        # 合并通道维度
        x = torch.cat([pre_img, pre_mask, post_img, post_mask], dim=1)

        # 前向传播
        logits = self(x, radiomics)

        # 计算概率和预测类别（二分类）
        binary_probs = F.softmax(logits, dim=1)[:, 1]  # 取正类概率
        binary_preds = torch.argmax(logits, dim=1)  # 取预测类别（0或1）

        # 构建结果字典
        result = {
            'logits': logits,  # 原始 logits，用于二分类分析，对每个样本预测一个二维向量，表示[负类几率值, 正类几率值] (未归一化的)
            'probs': binary_probs,
            'preds': binary_preds
        }

        # 如果批次中包含标签（可选），也一并返回
        if 'label' in batch:
            result['labels'] = batch['label']

        # 如果批次中包含其他信息，也一并返回
        if 'idx' in batch:
            result['idx'] = batch['idx']

        return result


def create_iddfenet_module(
        model_config: Optional[Dict[str, Any]] = None,
        soft_label_ratio: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        T_max: int = 200,
        eta_min: float = 1e-6,
        loss_type: str = 'cross_entropy',
        loss_config: Optional[Dict[str, Any]] = None) -> IDDFENetLightningModule:
    """
    创建 IDDFENet Lightning 模块的工厂函数
    
    Args:
        model_config: 模型配置参数
        soft_label_ratio: 标签软化程度，0-无软化，1-完全软化（所有标签均相等）
        learning_rate: 初始学习率
        weight_decay: 权重衰减系数
        T_max: 余弦退火调度器的最大迭代次数
        eta_min: 余弦退火调度器的最小学习率
        class_weights: 类别权重
        loss_type: 损失函数类型
        loss_config: 损失函数配置参数
        
    Returns:
        IDDFENetLightningModule 实例
    """
    # 确保模型配置中设置了正确的输入通道数（兼容 EsophagusDataModule，通常为 4）
    if model_config is None:
        model_config = {}

    return IDDFENetLightningModule(
        model_config=model_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        T_max=T_max,
        eta_min=eta_min,
        loss_type=loss_type,
        loss_config=loss_config,
        soft_label_ratio=soft_label_ratio
    )


if __name__ == "__main__":
    # 示例用法
    print("创建 IDDFENet Lightning 模块示例（兼容 EsophagusDataModule）")

    # 创建 ResNet-18 3D 模型配置
    iddfenet_config = {
        'block': 'basic',
        'layers': [2, 2, 2, 2],
        'branch_in_channels': 2,
        # 兼容 EsophagusDataModule，使用 4 = 2 + 2 通道（pre_img, pre_masked_img 和 post_img, post_masked_img）
        'num_classes': 2
    }

    # 创建 Lightning 模块
    module = create_iddfenet_module(
        model_config=iddfenet_config,
        learning_rate=1e-4,
        T_max=200
    )

    # 创建示例输入（模拟 EsophagusDataModule 的输出格式）
    batch_size = 2
    example_idx = torch.randint(0, batch_size, (batch_size,))  # 随机序号
    example_pre_img = torch.randn(batch_size, 1, 128, 128, 128)  # 前图像
    example_post_img = torch.randn(batch_size, 1, 128, 128, 128)  # 后图像
    example_pre_mask = torch.randn(batch_size, 1, 128, 128, 128)  # 前蒙版
    example_post_mask = torch.randn(batch_size, 1, 128, 128, 128)  # 后蒙版
    example_label = torch.randint(0, 2, (batch_size,))  # 随机标签
    import torch.nn.functional as F

    # One-hot encoded
    example_label = F.one_hot(example_label, 2).float()

    # 创建批次字典
    example_batch = {
        'idx': example_idx,
        'pre_img': example_pre_img,
        'pre_mask': example_pre_mask,
        'post_img': example_post_img,
        'post_mask': example_post_mask,
        'label': example_label
    }

    # 测试训练步骤
    print("测试训练步骤:")
    loss = module.training_step(example_batch, 0)
    print(f"训练损失: {loss.item()}")

    # 测试验证步骤
    print("\n测试验证步骤:")
    val_result = module.validation_step(example_batch, 0)
    print(f"验证损失: {val_result['val/loss'].item()}")
    print(f"预测形状: {val_result['preds'].shape}")

    # 测试预测步骤
    print("\n测试预测步骤:")
    pred_result = module.predict_step(example_batch, 0)
    print(f"预测类别形状: {pred_result['preds'].shape}")
    print(f"预测概率形状: {pred_result['probs'].shape}")

    # 打印模型摘要
    print("\n模型结构摘要:")
    print(module.model)
