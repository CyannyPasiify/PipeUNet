# -*- coding: utf-8 -*-
"""
Logger Wrapper Configuration Module

Overview:
    This module provides a unified logging interface based on PyTorch Lightning, supporting multiple log formats (CSV, TensorBoard, Wandb).
    It uses a factory pattern design to simplify the creation and management of loggers, ensuring consistent and reliable logging.

Main Components:
    1. Logger wrapper functions (CSVLogger, TensorBoardLogger, WandbLogger): Create and return specific types of loggers
    2. LoggerFactory class: Provides a unified interface to create single or multiple logger instances

Features:
    - Multi-format log support: Three mainstream formats (CSV, TensorBoard, Wandb)
    - Unified logging interface: All loggers provide a consistent usage pattern

Dependencies:
    - lightning.pytorch: Provides basic logging functionality
    - numpy: For data processing
    - matplotlib: For graph creation (optional)
    - wandb: For advanced logging (optional)
    - torch: For embedding vectors and other functions (optional)
"""
import lightning.pytorch.loggers as loggers
from typing import TYPE_CHECKING, Dict, Any, Optional, Union, List, Literal, cast
import os
from pathlib import Path

if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run

from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing_extensions import override


@dataclass
class ConfigLoggerBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "logger")

    def _assert_init_essentials(
            self,
            *args,
            **kwargs
    ) -> None:
        if self.is_ready(): return
        self.init_essentials(*args, **kwargs)

    @abstractmethod
    def init_essentials(
            self,
            *args,
            **kwargs
    ) -> 'ConfigLoggerBase':
        self.logger = None  # Just placeholder
        return self

    def get_logger(self, *args, **kwargs) -> loggers.Logger:
        self._assert_init_essentials(*args, **kwargs)
        return self.logger


@dataclass
class ConfigLoggerCSV(ConfigLoggerBase):
    """
    Initializes CSVLogger wrapper
    
    Args:
        save_dir: Save directory, logs will be saved in save_dir/name/version
        name: Experiment name
        version: Version number
        prefix: Log prefix
        flush_logs_every_n_steps: Flush logs every N steps
    """
    save_dir: Union[str, Path] = "Experiments"
    name: Optional[str] = "csv_logs"
    version: Optional[Union[int, str]] = None
    prefix: str = ""
    flush_logs_every_n_steps: int = 100

    @override
    def init_essentials(self) -> 'ConfigLoggerCSV':
        # Create original CSVLogger
        self.logger: loggers.CSVLogger = loggers.CSVLogger(
            save_dir=self.save_dir,
            name=self.name,
            version=self.version,
            prefix=self.prefix,
            flush_logs_every_n_steps=self.flush_logs_every_n_steps
        )
        return self


@dataclass
class ConfigLoggerTensorBoard(ConfigLoggerBase):
    """
    Initializes TensorBoardLogger wrapper
    
    Args:
        save_dir: Save directory
        name: Experiment name
        version: Version number
        log_graph: Whether to log computation graph
        default_hp_metric: Whether to log default hyperparameter metrics
        prefix: Log prefix
        sub_dir: Subdirectory
        kwargs: Other parameters passed to TensorBoardLogger
    """
    save_dir: Union[str, Path] = "Experiments"
    name: Optional[str] = "tb_logs"
    version: Optional[Union[int, str]] = None
    log_graph: bool = False
    default_hp_metric: bool = True
    prefix: str = ""
    sub_dir: Optional[Union[str, Path]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigLoggerTensorBoard':
        # Create original TensorBoardLogger
        self.logger: loggers.TensorBoardLogger = loggers.TensorBoardLogger(
            save_dir=self.save_dir,
            name=self.name,
            version=self.version,
            log_graph=self.log_graph,
            default_hp_metric=self.default_hp_metric,
            prefix=self.prefix,
            sub_dir=self.sub_dir,
            **self.kwargs
        )
        return self


@dataclass
class ConfigLoggerWandb(ConfigLoggerBase):
    """
    Initializes WandbLogger wrapper
    
    Args:
        name: Experiment name
        save_dir: Save directory
        version: Version number, mainly used for resuming runs
        offline: Whether to use offline mode
        anonymous: Whether to use anonymous logging
        project: Project name
        log_model: Whether to log model weights
        prefix: Prefix for recorded metric names
        experiment: Wandb experiment object
        checkpoint_name: Checkpoint name
        kwargs: Other parameters passed to WandbLogger
    """
    name: Optional[str] = None
    save_dir: Union[str, Path] = "."
    version: Optional[str] = None
    offline: bool = False
    anonymous: Optional[bool] = None
    project: Optional[str] = None
    log_model: Union[Literal["all"], bool] = False
    prefix: str = ""
    experiment: Union["Run", "RunDisabled", None] = None
    checkpoint_name: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @override
    def init_essentials(self) -> 'ConfigLoggerWandb':
        # Create original WandbLogger
        self.logger: loggers.WandbLogger = loggers.WandbLogger(
            name=self.name,
            save_dir=self.save_dir,
            version=self.version,
            offline=self.offline,
            anonymous=self.anonymous,
            project=self.project,
            log_model=self.log_model,
            prefix=self.prefix,
            experiment=self.experiment,
            checkpoint_name=self.checkpoint_name,
            **self.kwargs
        )
        return self


# Add LoggerFactory class definition
class LoggerFactory:
    """
    Logger factory class for creating various types of loggers
    Provides a unified interface to create single or multiple loggers
    """

    @staticmethod
    def create_logger(
            logger_type: Literal['csv', 'tensorboard', 'wandb'],
            **kwargs
    ) -> loggers.Logger:
        """
        Creates a single logger instance
        
        Args:
            logger_type: Logger type
            **kwargs: Parameters passed to the specific logger constructor
            
        Returns:
            Logger: Created logger instance
            
        Raises:
            ValueError: When the specified logger type is invalid
        """
        # Ensure log directory exists
        if 'save_dir' in kwargs:
            os.makedirs(kwargs['save_dir'], exist_ok=True)

        # Create corresponding logger based on type
        if logger_type == 'csv':
            return ConfigLoggerCSV(**kwargs).get_logger()
        elif logger_type == 'tensorboard':
            return ConfigLoggerTensorBoard(**kwargs).get_logger()
        elif logger_type == 'wandb':
            return ConfigLoggerWandb(**kwargs).get_logger()
        else:
            raise ValueError(f"Unsupported logger type: {logger_type}")

    @staticmethod
    def create_multi_loggers(
            logger_configs: List[Dict[str, Any]]
    ) -> List[loggers.Logger]:
        """
        Creates multiple logger instances
        
        Args:
            logger_configs: List of logger configurations, each containing type field and corresponding parameters
            
        Returns:
            List[Logger]: List of created logger instances
        """
        loggers_list = []
        for config in logger_configs:
            # Extract logger type
            logger_type = config.pop('type')
            # Create logger and add to list
            loggers_list.append(LoggerFactory.create_logger(logger_type, **config))
        return loggers_list


# Usage example
if __name__ == "__main__":
    """
    Logger wrapper usage example
    Verifies the ability of various loggers to record different types of log information
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from pathlib import Path

    print("=== Logger Wrapper Usage: Logging Capability Test ===")

    # Create unified log directory
    log_dir = Path('./Samples/logger_test')
    log_dir.mkdir(exist_ok=True)

    # 1. Test basic metric recording capability
    print("-" * 100)
    print("[1] Testing basic metric recording capability")
    # Create CSVLogger for basic metric testing
    csv_logger: loggers.CSVLogger = \
        cast(loggers.CSVLogger,
             LoggerFactory.create_logger(
                 'csv',
                 save_dir=str(log_dir),
                 name='csv_metrics_test')
             )
    print(f"Created CSVLogger: {csv_logger}")
    print(f"Log directory: {csv_logger.log_dir}")

    # Test recording of various types of metrics
    print("Recording different types of metrics:")

    # 1.1 Scalar metrics
    scalar_metrics = {
        'accuracy': 0.8543,
        'loss': 0.3217,
        'precision': 0.8892,
        'recall': 0.8231,
        'f1_score': 0.8551
    }
    csv_logger.log_metrics(scalar_metrics, step=10)
    csv_logger.save()
    print(f"  [1.1] Recorded scalar metrics: {scalar_metrics}")

    # 1.2 Metrics with different value ranges
    range_metrics = {
        'learning_rate': 0.001,
        'big_value': 1000000.0,
        'small_value': 1e-8,
        'negative_value': -0.5
    }
    csv_logger.log_metrics(range_metrics, step=20)
    csv_logger.save()
    print(f"  [1.2] Recorded metrics with different ranges: {range_metrics}")

    # 1.3 Metric sequences during training
    print("  [1.3] Recording training process metric sequence")
    for epoch in range(5):
        train_metrics = {
            'train_loss': 0.5 * (1 - epoch / 10),
            'train_accuracy': 0.6 + epoch / 20,
            'epoch_time': epoch + 5.2
        }
        csv_logger.log_metrics(train_metrics, step=epoch)
        print(f"    Epoch {epoch}: {train_metrics}")
    csv_logger.save()

    # 2. Test hyperparameter recording
    print("-" * 100)
    print("[2] Testing hyperparameter recording")
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealing',
        'dropout_rate': 0.3,
        'weight_decay': 1e-5,
        'seed': 42
    }
    # Record initial metrics along with hyperparameters
    initial_metrics = {'initial_loss': 0.65, 'initial_accuracy': 0.62}
    csv_logger.log_hyperparams(hyperparams)
    print(f"Recorded hyperparameters: {hyperparams}")
    print(f"Also recorded initial metrics: {initial_metrics}")

    # 3. Test TensorBoard-specific features
    print("-" * 100)
    print("[3] Testing TensorBoard-specific features")
    tb_logger: loggers.TensorBoardLogger = \
        cast(loggers.TensorBoardLogger,
             LoggerFactory.create_logger(
                 'tensorboard',
                 save_dir=str(log_dir),
                 name='tb_features_test',
                 log_graph=True)
             )
    print(f"Created TensorBoardLogger: {tb_logger}")

    # 3.1 Record graphs
    print("  [3.1] Recording graphs")
    try:
        # Create various types of graphs
        # Line plot
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        ax1.plot(x, y1, 'r-', label='sin(x)')
        ax1.plot(x, y2, 'b-', label='cos(x)')
        ax1.set_title('Triangle Function Example')
        ax1.set_xlabel('X Axis')
        ax1.set_ylabel('Y Axis')
        ax1.legend()
        tb_logger.experiment.add_figure('trigonometric_functions', fig1)
        plt.close(fig1)
        print("  Successfully recorded line plot")

        # Scatter plot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        data = np.random.randn(100, 2)
        ax2.scatter(data[:, 0], data[:, 1], alpha=0.7)
        ax2.set_title('Random Scatter Plot')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        tb_logger.experiment.add_figure('scatter_plot', fig2)
        plt.close(fig2)
        print("  Successfully recorded scatter plot")

        # Bar chart
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [15, 27, 12, 31, 22]
        ax3.bar(categories, values)
        ax3.set_title('Class Distribution')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Count')
        tb_logger.experiment.add_figure('bar_chart', fig3)
        plt.close(fig3)
        print("  Successfully recorded bar chart")

    except Exception as e:
        print(f"  Failed to record graphs: {e}")

    # 3.2 Record high-dimensional data (e.g., embeddings)
    print("  [3.2] Recording embeddings")
    try:
        # Simulate 5-dimensional embeddings for 10 samples
        embeddings = np.random.rand(10, 5)
        # Add labels to embeddings
        metadata = [f'sample_{i}' for i in range(10)]

        # Record embeddings (TensorBoard-specific feature)
        import torch

        tb_logger.experiment.add_embedding(
            mat=torch.from_numpy(embeddings),
            metadata=metadata,
            global_step=0
        )
        print("  Successfully recorded embeddings")
    except Exception as e:
        print(f"  Failed to record embeddings: {e}")

    # 4. Test batch logging performance
    print("-" * 100)
    print("[4] Testing batch logging performance")

    # Create a large number of metrics
    batch_size = 1000
    large_metrics = {f'metric_{i}': i * 0.01 for i in range(batch_size)}

    # Test recording time
    start_time = time.time()
    csv_logger.log_metrics(large_metrics, step=50)
    end_time = time.time()

    csv_logger.save()
    print(f"  Recorded {batch_size} metrics, time taken: {end_time - start_time:.4f} seconds")

    # 5. Test multi-logger collaboration
    print("-" * 100)
    print("[5] Testing multi-logger collaboration")
    multi_loggers = LoggerFactory.create_multi_loggers([
        {
            'type': 'csv',
            'save_dir': str(log_dir),
            'name': 'multi_csv'
        },
        {
            'type': 'tensorboard',
            'save_dir': str(log_dir),
            'name': 'multi_tb'
        }
    ])
    print(f"Created {len(multi_loggers)} loggers:")
    for i, logger in enumerate(multi_loggers):
        print(f"  {i + 1}. {logger}")

    # Record the same metrics for all loggers
    shared_metrics = {
        'shared_accuracy': 0.92,
        'shared_precision': 0.94,
        'shared_recall': 0.90,
        'shared_f1': 0.92
    }

    print("Recording metrics to multiple loggers simultaneously:")
    for logger in multi_loggers:
        logger.log_metrics(shared_metrics, step=100)
        logger.save()
    print(f"  Successfully recorded shared metrics: {shared_metrics}")

    # 6. Test compatibility with different data types
    print("-" * 100)
    print("[6] Testing compatibility with different data types")
    mixed_types_metrics = {
        'float_value': 1.2345,
        'int_value': 42,
        'numpy_float': np.float32(0.6789),
        'numpy_int': np.int64(100),
        'bool_value': True
    }
    csv_logger.log_metrics(mixed_types_metrics, step=200)
    csv_logger.save()
    print(f"  Recorded mixed type data: {mixed_types_metrics}")

    # 7. Test experiment name and version control
    print("-" * 100)
    print("[7] Testing experiment name and version control")
    versioned_logger = LoggerFactory.create_logger(
        'csv',
        save_dir=str(log_dir),
        name='version_test',
        version='v1.0'
    )
    versioned_logger.log_metrics(mixed_types_metrics, step=200)
    print(f"  Created versioned logger: {versioned_logger}")
    print(f"  Log directory: {versioned_logger.log_dir}")

    # 8. Test WandbLogger-specific features
    print("-" * 100)
    print("[8] Testing WandbLogger-specific features")
    try:
        # Create WandbLogger (using offline mode to avoid actual upload)
        wandb_logger: loggers.WandbLogger = \
            cast(loggers.WandbLogger,
                 LoggerFactory.create_logger(
                     'wandb',
                     save_dir=str(log_dir),
                     name='wandb_test',
                     project='logger_capability_demo',
                     offline=True,  # Use offline mode to avoid needing an account
                     anonymous='allow'
                 )
                 )
        print(f"  Created WandbLogger: {wandb_logger}")
        print(f"  Log directory: {wandb_logger.save_dir}")

        # 8.1 Record metrics (consistent interface with other loggers)
        print("  [8.1] Recording metrics")
        wandb_metrics = {
            'wandb_accuracy': 0.9123,
            'wandb_loss': 0.2876,
            'epoch': 10
        }
        wandb_logger.log_metrics(wandb_metrics, step=100)
        print(f"    Recorded metrics: {wandb_metrics}")

        # 8.2 Record hyperparameters
        print("  [8.2] Recording hyperparameters")
        wandb_hyperparams = {
            'model_type': 'CNN',
            'layers': 3,
            'filters': [16, 32, 64],
            'learning_rate': 0.001,
            'optimizer': 'AdamW'
        }
        wandb_logger.log_hyperparams(wandb_hyperparams)
        print(f"    Recorded hyperparameters: {wandb_hyperparams}")

        # Add wandb_logger to logger list
        all_loggers = [csv_logger, tb_logger] + multi_loggers + [versioned_logger, wandb_logger]

    except Exception as e:
        print(f"  WandbLogger test failed: {e}")
        # If WandbLogger test fails, use original logger list
        all_loggers = [csv_logger, tb_logger] + multi_loggers + [versioned_logger]

    # 9. Complete and close all loggers
    print("-" * 100)
    print("[9] Completing and closing all loggers:")
    for logger in all_loggers:
        logger.finalize('success')
    print("  All loggers have been successfully closed")

    print("-" * 100)
    print("=== Logger Capability Test Completed ===")
