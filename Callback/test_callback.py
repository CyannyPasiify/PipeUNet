#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回调函数测试脚本

功能概述：
    创建一个简单的神经网络，支持通过命令行参数--callback指定要测试的回调函数类型
    依次测试各种PyTorch Lightning回调函数的效果
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from typing import Dict, Any, List, Optional

# 导入我们的回调函数配置模块
from Callback.callback_configurer import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichModelSummary,
    RichProgressBar,
    TQDMProgressBar
)


class SimpleModel(pl.LightningModule):
    """简单的神经网络模型用于测试回调函数"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # 记录训练指标
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # 记录验证指标
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1),
            'name': 'learning_rate'
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


def create_datasets(input_dim: int = 20, num_samples: int = 1000, batch_size: int = 32):
    """创建简单的数据集用于测试"""
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    
    # 创建随机数据
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 2, (num_samples,))
    
    # 分割数据集
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_x, val_x = torch.split(x, [train_size, val_size])
    train_y, val_y = torch.split(y, [train_size, val_size])
    
    # 创建数据集
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def create_callback(callback_type: str) -> Optional[pl.Callback]:
    """根据指定的类型创建回调函数实例"""
    callback_type = callback_type.lower()
    
    if callback_type == 'devicestatsmonitor':
        print("创建 DeviceStatsMonitor 回调")
        return DeviceStatsMonitor(cpu_stats=True)
    
    elif callback_type == 'earlystopping':
        print("创建 EarlyStopping 回调")
        return EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=True)
    
    elif callback_type == 'learningratemonitor':
        print("创建 LearningRateMonitor 回调")
        return LearningRateMonitor(logging_interval='step', log_weight_decay=True)
    
    elif callback_type == 'modelcheckpoint':
        print("创建 ModelCheckpoint 回调")
        return ModelCheckpoint(
            dirpath='checkpoints',
            filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
            monitor='val_acc',
            save_top_k=3,
            mode='max',
            save_last=True
        )
    
    elif callback_type == 'modelsummary':
        print("创建 ModelSummary 回调")
        return ModelSummary(max_depth=2)
    
    elif callback_type == 'richmodelsummary':
        print("创建 RichModelSummary 回调")
        try:
            return RichModelSummary(max_depth=2)
        except ModuleNotFoundError:
            print("警告: 'rich' 包未安装，请使用 'pip install rich' 安装")
            return None
    
    elif callback_type == 'richprogressbar':
        print("创建 RichProgressBar 回调")
        try:
            return RichProgressBar(refresh_rate=2, leave=True)
        except ModuleNotFoundError:
            print("警告: 'rich' 包未安装，请使用 'pip install rich' 安装")
            return None
    
    elif callback_type == 'tqdmprogressbar':
        print("创建 TQDMProgressBar 回调")
        return TQDMProgressBar(refresh_rate=2, leave=True)
    
    else:
        print(f"未知的回调类型: {callback_type}")
        return None


def run_test(callback_type: str):
    """运行指定回调函数的测试"""
    print(f"\n===== 测试回调函数: {callback_type} =====")
    
    # 创建回调函数
    callback = create_callback(callback_type)
    if callback is None:
        return
    
    # 创建数据集
    train_loader, val_loader = create_datasets()
    
    # 创建模型
    model = SimpleModel()
    
    # 创建Trainer，使用指定的回调
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[callback],
        accelerator='auto',
        devices=1,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        logger=pl.loggers.CSVLogger(save_dir='logs', name=callback_type)
    )
    
    # 训练模型
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n===== 回调函数 {callback_type} 测试完成 =====\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试PyTorch Lightning回调函数')
    parser.add_argument('--callback', type=str, default='all', 
                        help='要测试的回调函数类型，可选值: devicestatsmonitor, earlystopping, learningratemonitor, ' \
                             'modelcheckpoint, modelsummary, richmodelsummary, richprogressbar, tqdmprogressbar, all')
    
    args = parser.parse_args()
    
    # 定义所有可用的回调类型
    all_callbacks = [
        'devicestatsmonitor',
        'earlystopping', 
        'learningratemonitor',
        'modelcheckpoint',
        'modelsummary',
        'richmodelsummary',
        'richprogressbar',
        'tqdmprogressbar'
    ]
    
    # 运行测试
    if args.callback == 'all':
        print("测试所有回调函数")
        for callback_type in all_callbacks:
            run_test(callback_type)
    else:
        run_test(args.callback)


if __name__ == '__main__':
    main()