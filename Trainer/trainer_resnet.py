#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器配置模块

功能概述：
    提供基于PyTorch Lightning的Trainer配置
    集成多种日志记录器（CSVLogger、TensorBoardLogger、WandbLogger）
    配置模型检查点，为每种验证指标保留top-3模型
    使用Rich进行模型参数查看和进度条显示
    支持实验名称和版本号配置

核心功能：
    1. 多日志记录器配置：同时使用CSV、TensorBoard和Wandb
    2. 模型检查点配置：为每种验证指标保留最优的top-3模型
    3. Rich可视化：模型参数查看（depth=5）和进度条
    4. 实验配置：设置实验名称为ESO-2025.11.8_ResNet18
    5. 日志保存：将实验日志保存在相应目录结构中
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Literal, Mapping
import torch
import lightning as L
from lightning import Trainer
from Logger.logger_configurer import CSVLogger, TensorBoardLogger, WandbLogger
from Callback.callback_configurer import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
    EarlyStopping,
    TQDMProgressBar
)
from lightning.pytorch.strategies import Strategy

SupportedPrecision = Optional[Union[Literal[64, 32, 16,
"transformer-engine", "transformer-engine-float16",
"16-true", "16-mixed", "bf16-true", "bf16-mixed",
"32-true", "64-true", "64", "32", "16", "bf16"]]]

# 实验配置
EXPERIMENT_NAME = "ESO-2025.11.8_ResNet"
EXPERIMENT_VERSION = "20251117"


class TrainerSegmentationDefault:
    """
    食管项目训练器类
    配置和管理PyTorch Lightning Trainer实例
    集成多种日志记录器和回调函数
    """

    def __init__(
            self,
            max_epochs: int = 200,
            accelerator: str = "gpu" if torch.cuda.is_available() else "cpu",
            devices: Union[int, List[int], str] = 1,
            precision: SupportedPrecision = 32,
            experiment_name: str = EXPERIMENT_NAME,
            experiment_version: str = EXPERIMENT_VERSION,
            log_dir: str = 'logs',
            wandb_project: str = "ESO-pCR-Prediction",
            enable_ddp: bool = False
    ):
        """
        初始化训练器配置
        
        Args:
            max_epochs: 最大训练轮数
            accelerator: 加速设备，可选值: 'gpu', 'cpu', 'tpu'
            devices: 使用的设备数量或列表
            precision: 精度，可选值: 32, 16, 'bf16'
            experiment_name: 实验名称
            experiment_version: 实验版本号
            log_dir: 日志保存目录，默认为项目根目录下的logs
            wandb_project: Weights & Biases项目名称
            enable_ddp: 是否启用分布式数据并行
        """
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.experiment_name = experiment_name
        self.experiment_version = experiment_version

        # 设置日志目录
        self.log_dir = log_dir

        self.wandb_project = wandb_project
        self.enable_ddp = enable_ddp

        # 创建必要的目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 创建Python控制台日志器
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.logger.info("Logging initialized")

        # 初始化Trainer实例
        self.trainer = None

    def _get_loggers(self) -> List[Union[CSVLogger, TensorBoardLogger, WandbLogger]]:
        """
        获取配置好的日志记录器列表
        
        Returns:
            日志记录器列表
        """
        loggers = []

        # CSVLogger
        csv_log_dir = os.path.join(self.log_dir, self.experiment_name)
        csv_logger = CSVLogger(
            save_dir=csv_log_dir,
            name=None,
            version=self.experiment_version,
            flush_logs_every_n_steps=10
        )
        loggers.append(csv_logger)

        # TensorBoardLogger
        tb_log_dir = self.log_dir
        tb_logger = TensorBoardLogger(
            save_dir=tb_log_dir,
            name=self.experiment_name,
            version=self.experiment_version,
            sub_dir='tensorboard'
        )
        loggers.append(tb_logger)

        # WandbLogger
        wb_log_dir = os.path.join(self.log_dir, self.experiment_name, self.experiment_version)
        wandb_logger = WandbLogger(
            project=self.wandb_project,
            name=f"{self.experiment_name}_{self.experiment_version}",
            save_dir=wb_log_dir,
            log_model=False  # 模型通过ModelCheckpoint保存
        )
        loggers.append(wandb_logger)

        self.logger.info(f"已配置日志记录器: CSVLogger, TensorBoardLogger, WandbLogger")
        return loggers

    def _get_model_checkpoints(self) -> List[ModelCheckpoint]:
        """
        获取配置好的模型检查点回调列表
        为每种验证指标保留top-3模型
        
        Returns:
            模型检查点回调列表
        """
        checkpoints = []

        # 验证指标列表
        val_metrics: List[Tuple[str, Literal['min', 'max']]] = [
            ("val/loss", "min"),
            ("val/acc", "max"),
            ("val/precision", "max"),
            ("val/recall", "max"),
            ("val/f1", "max"),
            ("val/auroc", "max"),
            ("val/ap", "max"),
            ("val/specificity", "max")
        ]

        # 为每个指标创建检查点
        for metric_name, mode in val_metrics:
            checkpoint_dir = os.path.join(
                self.log_dir,
                self.experiment_name,
                self.experiment_version,
                "checkpoints",
                metric_name
            )

            checkpoint = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f"{{epoch:03d}}-{metric_name.replace('/', '_')}={{{metric_name}:4f}}",
                monitor=metric_name,
                mode=mode,
                save_top_k=3,
                save_last=False
            )
            checkpoints.append(checkpoint)

        # 保存里程碑检查点
        milestone_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(
                self.log_dir,
                self.experiment_name,
                self.experiment_version,
                "checkpoints",
                "milestone"
            ),
            filename="{epoch:03d}-val_loss={val/loss:4f}",
            monitor="epoch",
            save_top_k=-1,
            save_last=False,
            every_n_epochs=10
        )
        checkpoints.append(milestone_checkpoint)

        self.logger.info(f"已配置模型检查点: 为{len(val_metrics)}种验证指标各保留top-3模型")
        return checkpoints

    def _get_callbacks(self) -> List[Any]:
        """
        获取配置好的回调函数列表
        
        Returns:
            回调函数列表
        """
        callbacks = []

        # 添加Rich进度条
        rich_progress_bar = RichProgressBar(
            refresh_rate=1,
            leave=True
        )
        callbacks.append(rich_progress_bar)

        # 添加Rich模型摘要（depth=5）
        rich_model_summary = RichModelSummary(
            max_depth=5
        )
        callbacks.append(rich_model_summary)

        # 添加学习率监控器
        lr_monitor = LearningRateMonitor(
            logging_interval="epoch",
            log_momentum=True,
            log_weight_decay=True
        )
        callbacks.append(lr_monitor)

        # 添加早停策略
        early_stopping = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=100,
            verbose=True
        )
        callbacks.append(early_stopping)

        # 添加模型检查点
        model_checkpoints = self._get_model_checkpoints()
        callbacks.extend(model_checkpoints)

        self.logger.info(f"已配置回调函数: Rich进度条, Rich模型摘要, 学习率监控, 早停, 模型检查点")
        return callbacks

    def _get_strategy(self) -> Union[Strategy, str]:
        """
        获取训练策略
        
        Returns:
            训练策略或None
        """
        if self.enable_ddp and self.accelerator == "gpu":
            strategy = 'ddp'
            self.logger.info(f"已启用DDP策略")
            return strategy
        return 'auto'

    def get_trainer(self) -> Trainer:
        """
        获取配置好的Trainer实例
        
        Returns:
            Trainer实例
        """
        if self.trainer is None:
            # 获取日志记录器
            loggers = self._get_loggers()

            # 获取回调函数
            callbacks = self._get_callbacks()

            # 获取策略
            strategy = self._get_strategy()

            # 创建Trainer实例
            self.trainer = Trainer(
                # 基础配置
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                precision=self.precision,
                strategy=strategy,

                # 日志和回调
                logger=loggers,
                callbacks=callbacks,

                # 训练配置
                deterministic='warn',
                accumulate_grad_batches=1,

                # 梯度裁剪
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",

                # 日志记录
                log_every_n_steps=1,
                enable_progress_bar=True,
                enable_model_summary=True,

                # 分布式训练
                sync_batchnorm=True if self.enable_ddp else False,

                # 其他
                num_sanity_val_steps=2,
                check_val_every_n_epoch=1,
                val_check_interval=1.0,

                # 调试
                fast_dev_run=False,
                overfit_batches=0.0
            )

            self.logger.info(f"已创建Trainer实例: {self.experiment_name} v{self.experiment_version}")

        return self.trainer

    def fit(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            model: LightningModule实例
            datamodule: LightningDataModule实例

        Returns:
            训练结果字典
        """
        trainer = self.get_trainer()

        self.logger.info(f"开始训练: {self.experiment_name} {self.experiment_version}")
        self.logger.info(f"使用设备: {self.accelerator}, 设备数量: {self.devices}")

        # 开始训练
        start_time = datetime.now()
        trainer.fit(model=model, datamodule=datamodule)
        end_time = datetime.now()

        training_time = end_time - start_time
        self.logger.info(f"训练完成! 耗时: {training_time}")

        # 获取训练结果
        results = {
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "max_epochs": self.max_epochs,
            "training_time": str(training_time),
            "best_val_loss": trainer.logged_metrics.get("val/loss", None),
            "best_val_acc": trainer.logged_metrics.get("val/acc", None)
        }

        return results

    def test(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[str] = None
    ) -> List[Mapping[str, float]]:
        """
        测试模型

        Args:
            model: LightningModule实例
            datamodule: LightningDataModule实例
            ckpt_path: 检查点路径，默认为None（使用当前模型权重）

        Returns:
            测试结果列表
        """
        trainer = self.get_trainer()

        self.logger.info(f"开始测试: {self.experiment_name} {self.experiment_version}")
        if ckpt_path:
            self.logger.info(f"使用检查点: {ckpt_path}")

        # 开始测试
        start_time = datetime.now()
        test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        end_time = datetime.now()

        test_time = end_time - start_time
        self.logger.info(f"测试完成! 耗时: {test_time}")

        return test_results

    def predict(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[str] = None
    ) -> List[Any]:
        """
        模型预测

        Args:
            model: LightningModule实例
            datamodule: LightningDataModule实例
            ckpt_path: 检查点路径，默认为None（使用当前模型权重）

        Returns:
            预测结果列表
        """
        trainer = self.get_trainer()

        self.logger.info(f"开始预测: {self.experiment_name} {self.experiment_version}")
        if ckpt_path:
            self.logger.info(f"使用检查点: {ckpt_path}")

        # 开始预测
        start_time = datetime.now()
        predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        end_time = datetime.now()

        predict_time = end_time - start_time
        self.logger.info(f"预测完成! 耗时: {predict_time}")

        return predictions


if __name__ == "__main__":
    pass
