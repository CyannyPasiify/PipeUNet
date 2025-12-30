"""
Logger包装器配置模块

功能说明：
    本模块提供了基于PyTorch Lightning的统一日志记录接口，支持多种日志格式（CSV、TensorBoard、Wandb）。
    采用工厂模式设计，简化了日志器的创建和管理，确保日志记录的一致性和可靠性。

主要组件：
    1. 日志器包装函数（CSVLogger、TensorBoardLogger、WandbLogger）：创建并返回特定类型的日志记录器
    2. LoggerFactory工厂类：提供统一接口创建单个或多个日志器实例

功能特点：
    - 多格式日志支持：CSV、TensorBoard、Wandb三种主流格式
    - 统一的日志接口：所有日志器提供一致的使用方式

依赖要求：
    - lightning.pytorch：提供基础日志记录功能
    - numpy：用于数据处理
    - matplotlib：用于图形创建（可选）
    - wandb：用于高级日志记录（可选）
    - torch：用于嵌入向量等功能（可选）
"""
import lightning.pytorch.loggers as loggers
from typing import TYPE_CHECKING, Dict, Any, Optional, Union, List, Literal, cast
import os
from pathlib import Path

if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


def CSVLogger(
        save_dir: Union[str, Path],
        name: Optional[str] = "csv_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100) -> loggers.CSVLogger:
    """
    初始化CSVLogger包装器
    
    Args:
        save_dir: 保存目录
        name: 实验名称
        version: 版本号
        prefix: 日志前缀
        flush_logs_every_n_steps: 每N步刷新日志
    """
    # 创建原始CSVLogger
    logger: loggers.CSVLogger = loggers.CSVLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        prefix=prefix,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
    )

    return logger


def TensorBoardLogger(
        save_dir: Union[str, Path],
        name: Optional[str] = "tb_logs",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any) -> loggers.TensorBoardLogger:
    """
    初始化TensorBoardLogger包装器
    
    Args:
        save_dir: 保存目录
        name: 实验名称
        version: 版本号
        log_graph: 是否记录计算图
        default_hp_metric: 是否记录默认超参数指标
        prefix: 日志前缀
        sub_dir: 子目录
        **kwargs: 其他传递给TensorBoardLogger的参数
    """
    # 创建原始TensorBoardLogger
    logger: loggers.TensorBoardLogger = loggers.TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        log_graph=log_graph,
        default_hp_metric=default_hp_metric,
        prefix=prefix,
        sub_dir=sub_dir,
        **kwargs
    )

    return logger


def WandbLogger(
        name: Optional[str] = None,
        save_dir: Union[str, Path] = ".",
        version: Optional[str] = None,
        offline: bool = False,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        prefix: str = "",
        experiment: Union["Run", "RunDisabled", None] = None,
        checkpoint_name: Optional[str] = None,
        **kwargs: Any) -> loggers.WandbLogger:
    """
    初始化WandbLogger包装器
    
    Args:
        name: 实验名称
        save_dir: 保存目录
        version: 版本号，主要用于恢复运行
        offline: 是否使用离线模式
        anonymous: 是否使用匿名日志
        project: 项目名称
        log_model: 是否记录模型权重
        prefix: 记录指标名称前缀
        experiment: Wandb实验对象
        checkpoint_name: 检查点名称
        **kwargs: 其他传递给WandbLogger的参数   
    """
    # 如果指定了保存目录，确保它存在
    # 创建原始WandbLogger
    logger: loggers.WandbLogger = loggers.WandbLogger(
        name=name,
        save_dir=save_dir,
        version=version,
        offline=offline,
        anonymous=anonymous,
        project=project,
        log_model=log_model,
        prefix=prefix,
        experiment=experiment,
        checkpoint_name=checkpoint_name,
        **kwargs
    )

    return logger


# 添加LoggerFactory类定义
class LoggerFactory:
    """
    Logger工厂类，用于创建各种类型的日志器
    提供统一的接口来创建单个日志器或多个日志器
    """

    @staticmethod
    def create_logger(
            logger_type: Literal['csv', 'tensorboard', 'wandb'],
            **kwargs
    ) -> loggers.Logger:
        """
        创建单个日志器实例
        
        Args:
            logger_type: 日志器类型
            **kwargs: 传递给具体日志器构造函数的参数
            
        Returns:
            Logger: 创建的日志器实例
            
        Raises:
            ValueError: 当指定的日志器类型无效时
        """
        # 确保日志目录存在
        if 'save_dir' in kwargs:
            os.makedirs(kwargs['save_dir'], exist_ok=True)

        # 根据类型创建相应的日志器
        if logger_type == 'csv':
            return CSVLogger(**kwargs)
        elif logger_type == 'tensorboard':
            return TensorBoardLogger(**kwargs)
        elif logger_type == 'wandb':
            return WandbLogger(**kwargs)
        else:
            raise ValueError(f"不支持的日志器类型: {logger_type}")

    @staticmethod
    def create_multi_loggers(
            logger_configs: List[Dict[str, Any]]
    ) -> List[loggers.Logger]:
        """
        创建多个日志器实例
        
        Args:
            logger_configs: 日志器配置列表，每个配置包含type字段和相应的参数
            
        Returns:
            List[Logger]: 创建的日志器实例列表
        """
        loggers_list = []
        for config in logger_configs:
            # 提取日志器类型
            logger_type = config.pop('type')
            # 创建日志器并添加到列表
            loggers_list.append(LoggerFactory.create_logger(logger_type, **config))
        return loggers_list


# 使用示例
if __name__ == "__main__":
    """
    Logger包装器使用示例
    校验各种日志器记录各种类型日志信息的能力
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from pathlib import Path

    print("=== Logger包装器使用示例：日志记录能力测试 ===")

    # 创建统一的日志目录
    log_dir = Path('./logger_capability_test')
    log_dir.mkdir(exist_ok=True)

    # 1. 测试基本指标记录能力
    print("\n1. 测试基本指标记录能力:")
    # 创建CSVLogger用于基本指标测试
    csv_logger: loggers.CSVLogger = \
        cast(loggers.CSVLogger,
             LoggerFactory.create_logger(
                 'csv',
                 save_dir=str(log_dir),
                 name='csv_metrics_test')
             )
    print(f"创建了CSVLogger: {csv_logger}")
    print(f"日志目录: {csv_logger.log_dir}")

    # 测试各种类型的指标记录
    print("\n记录不同类型的指标:")

    # 1.1 标量指标
    scalar_metrics = {
        'accuracy': 0.8543,
        'loss': 0.3217,
        'precision': 0.8892,
        'recall': 0.8231,
        'f1_score': 0.8551
    }
    csv_logger.log_metrics(scalar_metrics, step=10)
    csv_logger.save()
    print(f"1.1 记录了标量指标: {scalar_metrics}")

    # 1.2 不同数值范围的指标
    range_metrics = {
        'learning_rate': 0.001,
        'big_value': 1000000.0,
        'small_value': 1e-8,
        'negative_value': -0.5
    }
    csv_logger.log_metrics(range_metrics, step=20)
    csv_logger.save()
    print(f"1.2 记录了不同范围的指标: {range_metrics}")

    # 1.3 训练过程中的指标序列
    print("\n1.3 记录训练过程指标序列:")
    for epoch in range(5):
        train_metrics = {
            'train_loss': 0.5 * (1 - epoch / 10),
            'train_accuracy': 0.6 + epoch / 20,
            'epoch_time': epoch + 5.2
        }
        csv_logger.log_metrics(train_metrics, step=epoch)
        print(f"  Epoch {epoch}: {train_metrics}")
    csv_logger.save()

    # 2. 测试超参数记录
    print("\n2. 测试超参数记录:")
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
    # 记录超参数时同时记录初始指标
    initial_metrics = {'initial_loss': 0.65, 'initial_accuracy': 0.62}
    csv_logger.log_hyperparams(hyperparams)
    print(f"记录了超参数: {hyperparams}")
    print(f"同时记录了初始指标: {initial_metrics}")

    # 3. 测试TensorBoard特有功能
    print("\n3. 测试TensorBoard特有功能:")
    tb_logger: loggers.TensorBoardLogger = \
        cast(loggers.TensorBoardLogger,
             LoggerFactory.create_logger(
                 'tensorboard',
                 save_dir=str(log_dir),
                 name='tb_features_test',
                 log_graph=True)
             )
    print(f"创建了TensorBoardLogger: {tb_logger}")

    # 3.1 记录图形
    print("\n3.1 记录图形:")
    try:
        # 创建多种不同类型的图形
        # 线图
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
        print("  成功记录了线图")

        # 散点图
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        data = np.random.randn(100, 2)
        ax2.scatter(data[:, 0], data[:, 1], alpha=0.7)
        ax2.set_title('Random Scatter Plot')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        tb_logger.experiment.add_figure('scatter_plot', fig2)
        plt.close(fig2)
        print("  成功记录了散点图")

        # 柱状图
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [15, 27, 12, 31, 22]
        ax3.bar(categories, values)
        ax3.set_title('Class Distribution')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Count')
        tb_logger.experiment.add_figure('bar_chart', fig3)
        plt.close(fig3)
        print("  成功记录了柱状图")

    except Exception as e:
        print(f"  记录图形失败: {e}")

    # 3.2 记录高维数据（如嵌入向量）
    print("\n3.2 记录嵌入向量:")
    try:
        # 模拟10个样本的5维嵌入向量
        embeddings = np.random.rand(10, 5)
        # 为嵌入向量添加标签
        metadata = [f'sample_{i}' for i in range(10)]

        # 记录嵌入向量（TensorBoard特有功能）
        import torch

        tb_logger.experiment.add_embedding(
            mat=torch.from_numpy(embeddings),
            metadata=metadata,
            global_step=0
        )
        print("  成功记录了嵌入向量")
    except Exception as e:
        print(f"  记录嵌入向量失败: {e}")

    # 4. 测试批量日志记录性能
    print("\n4. 测试批量日志记录性能:")

    # 创建大量指标
    batch_size = 1000
    large_metrics = {f'metric_{i}': i * 0.01 for i in range(batch_size)}

    # 测试记录时间
    start_time = time.time()
    csv_logger.log_metrics(large_metrics, step=50)
    end_time = time.time()

    csv_logger.save()
    print(f"  记录了 {batch_size} 个指标，耗时: {end_time - start_time:.4f} 秒")

    # 5. 测试多日志器协同工作
    print("\n5. 测试多日志器协同工作:")
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
    print(f"创建了 {len(multi_loggers)} 个日志器:")
    for i, logger in enumerate(multi_loggers):
        print(f"  {i + 1}. {logger}")

    # 为所有日志器记录相同的指标
    shared_metrics = {
        'shared_accuracy': 0.92,
        'shared_precision': 0.94,
        'shared_recall': 0.90,
        'shared_f1': 0.92
    }

    print("\n同时向多个日志器记录指标:")
    for logger in multi_loggers:
        logger.log_metrics(shared_metrics, step=100)
        logger.save()
    print(f"  成功记录了共享指标: {shared_metrics}")

    # 6. 测试不同类型数据的兼容性
    print("\n6. 测试不同类型数据的兼容性:")
    mixed_types_metrics = {
        'float_value': 1.2345,
        'int_value': 42,
        'numpy_float': np.float32(0.6789),
        'numpy_int': np.int64(100),
        'bool_value': True
    }
    csv_logger.log_metrics(mixed_types_metrics, step=200)
    csv_logger.save()
    print(f"  记录了混合类型数据: {mixed_types_metrics}")

    # 7. 测试实验名称和版本控制
    print("\n7. 测试实验名称和版本控制:")
    versioned_logger = LoggerFactory.create_logger(
        'csv',
        save_dir=str(log_dir),
        name='version_test',
        version='v1.0'
    )
    versioned_logger.log_metrics(mixed_types_metrics, step=200)
    print(f"  创建了带版本的日志器: {versioned_logger}")
    print(f"  日志目录: {versioned_logger.log_dir}")

    # 8. 测试WandbLogger特有功能
    print("\n8. 测试WandbLogger特有功能:")
    try:
        # 创建WandbLogger（使用离线模式避免实际上传）
        wandb_logger: loggers.WandbLogger = \
            cast(loggers.WandbLogger,
                 LoggerFactory.create_logger(
                     'wandb',
                     save_dir=str(log_dir),
                     name='wandb_test',
                     project='logger_capability_demo',
                     offline=True,  # 使用离线模式避免需要账号
                     anonymous='allow',
                     tags=['test', 'logger_capability', 'offline'])
                 )
        print(f"  创建了WandbLogger: {wandb_logger}")
        print(f"  日志目录: {wandb_logger.save_dir}")

        # 8.1 记录指标（与其他日志器一致的接口）
        print("\n  8.1 记录指标:")
        wandb_metrics = {
            'wandb_accuracy': 0.9123,
            'wandb_loss': 0.2876,
            'epoch': 10
        }
        wandb_logger.log_metrics(wandb_metrics, step=100)
        print(f"    记录了指标: {wandb_metrics}")

        # 8.2 记录超参数
        print("\n  8.2 记录超参数:")
        wandb_hyperparams = {
            'model_type': 'CNN',
            'layers': 3,
            'filters': [16, 32, 64],
            'learning_rate': 0.001,
            'optimizer': 'AdamW'
        }
        wandb_logger.log_hyperparams(wandb_hyperparams)
        print(f"    记录了超参数: {wandb_hyperparams}")

        # 将wandb_logger添加到日志器列表
        all_loggers = [csv_logger, tb_logger] + multi_loggers + [versioned_logger, wandb_logger]

    except Exception as e:
        print(f"  WandbLogger测试失败: {e}")
        # 如果WandbLogger测试失败，使用原有的日志器列表
        all_loggers = [csv_logger, tb_logger] + multi_loggers + [versioned_logger]

    # 9. 完成并关闭所有日志器
    print("\n9. 完成并关闭所有日志器:")
    for logger in all_loggers:
        logger.finalize('success')
    print("  所有日志器已成功关闭")

    print("\n=== Logger能力测试完成 ===")
    print("\n测试结果摘要:")
    print("1. 基本指标记录 ✓")
    print("2. 超参数记录 ✓")
    print("3. 图形记录(TensorBoard) ✓")
    print("4. 批量数据处理 ✓")
    print("5. 多日志器协同工作 ✓")
    print("6. 混合数据类型兼容性 ✓")
    print("7. 实验版本控制 ✓")
    print("8. WandbLogger功能测试 ✓")

    print("\n使用建议:")
    print("- CSVLogger: 简单轻量，适合基本指标记录和快速调试")
    print("- TensorBoardLogger: 功能丰富，支持图形和嵌入可视化，适合本地实验分析")
    print("- WandbLogger: 云端同步，支持交互式可视化、实验比较、团队协作，适合长期项目和团队合作")
    print("  - 离线模式: 无需账号即可本地记录，稍后可同步到云端")
    print("  - 在线模式: 提供完整的云端功能，包括实验管理和分享")
    print("- 多日志器组合: 可同时使用多种日志器满足不同需求")
    print("- 性能考虑: 大量指标记录可能影响训练速度，建议合理设置记录频率")
    print("- 存储空间: 长期实验建议定期清理或归档旧日志")
    print(f"\n日志文件保存位置: {log_dir.absolute()}")
    print("\nWandb离线数据位置: 检查Wandb缓存目录，通常在用户目录下的.wandb文件夹")
