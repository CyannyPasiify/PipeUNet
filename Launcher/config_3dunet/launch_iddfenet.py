#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet 模型训练与测试启动器

功能概述：
    提供命令行界面来启动IDDFENet模型的训练和测试流程
    支持配置模型参数、训练参数和数据加载参数
    集成PyTorch Lightning Trainer进行训练和评估
    支持多种日志记录器和检查点保存策略
    支持通过YAML配置文件加载参数
"""

import os
import argparse
import torch
import lightning as L
import monai
import yaml
import json

# 导入必要的模块
from Module.module_iddfenet import IDDFENetLightningModule
from DataModule.dm_pipe import EsophagusDataModule
from Trainer.trainer_resnet import EsophagusTrainer


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间
    """
    parser = argparse.ArgumentParser(description='ResNet 模型训练与测试启动器')
    
    # 配置文件参数 - 所有参数都从配置文件读取
    parser.add_argument('--config', type=str, required=True,
                        help='YAML配置文件路径')
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    从YAML文件加载配置
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def merge_args_with_config(args: argparse.Namespace, config: dict) -> dict:
    """
    处理配置文件内容，确保所有必要的参数都存在
    
    Args:
        args: 命令行参数（仅包含config路径）
        config: 配置文件内容
    
    Returns:
        dict: 处理后的配置
    """
    # 深拷贝配置字典以避免修改原始配置
    merged_config = json.loads(json.dumps(config))
    
    # 确保必要的顶层参数存在
    required_top_level = ['mode', 'save_dir', 'checkpoint_path', 'random_seed']
    for param in required_top_level:
        if param not in merged_config:
            # random_seed默认值设为42
            merged_config[param] = 42 if param == 'random_seed' else None
    
    return merged_config


def set_seed(seed: int) -> None:
    """
    设置随机种子以确保实验可复现
    
    Args:
        seed: 随机种子值
    """
    L.seed_everything(seed)
    monai.utils.misc.set_determinism(seed)
    torch.cuda.manual_seed_all(seed)


def configure_trainer(config: dict) -> EsophagusTrainer:
    """
    配置 PyTorch Lightning Trainer
    
    Args:
        config: 配置字典
    
    Returns:
        EsophagusTrainer: 配置好的Trainer实例
    """
    # 获取trainer配置
    trainer_config = config['trainer']
    
    # 配置加速设备 CPU/GPUs
    accelerator = trainer_config['accelerator']
    devices = trainer_config['devices']

    if accelerator == 'gpu' and not torch.cuda.is_available():
        accelerator = 'cpu'
    if accelerator == 'cpu' and isinstance(devices, list):
        devices = devices[0]

    # 配置Trainer
    trainer = EsophagusTrainer(
        max_epochs=trainer_config['max_epochs'],
        accelerator=accelerator,
        devices=devices,
        precision=trainer_config['precision'],
        experiment_name=trainer_config['experiment_name'],
        experiment_version=trainer_config['version'],
        log_dir=trainer_config['log_dir'],
        wandb_project=trainer_config['wandb_project'],
        enable_ddp=trainer_config['enable_ddp'],
    )

    return trainer


def create_model(config: dict) -> L.LightningModule:
    """
    创建ResNet模型实例
    
    Args:
        config: 配置字典
    
    Returns:
        L.LightningModule: ResNet模型实例
    """
    # 获取model配置
    model_config = config['model']
    
    # 创建模型
    model = IDDFENetLightningModule(
        model_config=model_config['model_config'],
        soft_label_ratio=model_config.get('soft_label_ratio', 0.0),
        learning_rate=model_config['learning_rate'],
        T_max=model_config['T_max'],
        eta_min=model_config['eta_min'],
        loss_type=model_config.get('loss_type', 'cross_entropy'),
        loss_config=model_config.get('loss_config', {})
    )

    return model


def create_datamodule(config: dict) -> EsophagusDataModule:
    """
    创建数据模块实例
    
    Args:
        config: 配置字典
    
    Returns:
        EsophagusDataModule: 数据模块实例
    """
    # 获取data配置
    data_config = config['data']
    
    # 从顶层配置获取random_seed，不再从data中获取
    random_seed = config['random_seed']
    
    datamodule = EsophagusDataModule(
        root_dir_train=data_config['root_dir_train'],
        root_dir_val=data_config['root_dir_val'],
        root_dir_test=data_config['root_dir_test'],
        root_dir_predict=data_config['root_dir_predict'],
        manifest_file_train=data_config['manifest_file_train'],
        manifest_file_val=data_config['manifest_file_val'],
        manifest_file_test=data_config['manifest_file_test'],
        manifest_file_predict=data_config['manifest_file_predict'],
        n_radiomics=data_config['n_radiomics'],
        batch_size_train=data_config['batch_size_train'],
        batch_size_val=data_config['batch_size_val'],
        batch_size_test=data_config['batch_size_test'],
        batch_size_predict=data_config['batch_size_predict'],
        sample_weight_train=data_config['sample_weight_train'],
        num_workers=data_config['num_workers'],
        random_seed=random_seed
    )

    return datamodule


def train_model(config: dict) -> None:
    """
    训练模型
    
    Args:
        config: 配置字典
    """
    trainer_config = config['trainer']
    print(f"开始训练模型: {trainer_config['experiment_name']}, 版本: {trainer_config['version']}")

    # 设置随机种子
    set_seed(config['random_seed'])

    # 创建数据模块
    datamodule = create_datamodule(config)

    # 创建模型
    model = create_model(config)

    # 配置Trainer
    trainer = configure_trainer(config)

    # 开始训练
    trainer.fit(model, datamodule=datamodule)

    print("训练完成！")


def test_model(config: dict) -> None:
    """
    测试模型
    
    Args:
        config: 配置字典
    """
    if 'checkpoint_path' not in config or not config['checkpoint_path']:
        raise ValueError("测试模式需要指定checkpoint_path参数")

    print(f"开始测试模型，使用检查点: {config['checkpoint_path']}")

    # 设置随机种子
    set_seed(config['random_seed'])

    # 创建数据模块
    datamodule = create_datamodule(config)

    # 从检查点加载模型
    model = DualResNetLightningModule.load_from_checkpoint(config['checkpoint_path'])
    model.eval()

    # 配置Trainer
    trainer = configure_trainer(config)

    # 开始测试
    trainer.test(model, datamodule=datamodule)

    print("测试完成！")


def predict_with_model(config: dict) -> None:
    """
    使用模型进行预测
    
    Args:
        config: 配置字典
    """
    if 'checkpoint_path' not in config or not config['checkpoint_path']:
        raise ValueError("预测模式需要指定checkpoint_path参数")

    if 'save_dir' not in config or not config['save_dir']:
        raise ValueError("预测模式需要指定save_dir参数")

    print(f"开始预测，使用检查点: {config['checkpoint_path']}")
    print(f"预测结果将保存到: {config['save_dir']}")

    # 确保保存目录存在
    os.makedirs(config['save_dir'], exist_ok=True)

    # 设置随机种子
    set_seed(config['random_seed'])

    # 创建数据模块（预测模式）
    datamodule = create_datamodule(config)

    # 从检查点加载模型
    model = DualResNetLightningModule.load_from_checkpoint(config['checkpoint_path'])
    model.eval()

    # 配置Trainer
    trainer = configure_trainer(config)

    # 进行预测
    predictions = trainer.predict(model, datamodule=datamodule)

    # 保存预测结果
    all_preds = []
    all_probs = []
    all_indices = []

    for pred_batch in predictions:
        all_preds.append(pred_batch['preds'])
        all_probs.append(pred_batch['probs'])
        if 'idx' in pred_batch:
            all_indices.append(pred_batch['idx'])

    # 合并所有批次的预测结果
    all_preds_tensor = torch.cat(all_preds)
    all_probs_tensor = torch.cat(all_probs)

    # 准备保存为CSV格式的数据
    import pandas as pd
    
    # 确保所有批次的预测结果已经处理完毕
    results = []
    
    # 从数据模块获取预测数据集
    predict_dataset = datamodule.datasets['predict']
    
    # 遍历所有预测结果和对应的索引
    for i in range(len(all_preds_tensor)):
        # 获取预测结果
        pred = all_preds_tensor[i].item()
        prob = all_probs_tensor[i].item()
        
        # 获取样本索引
        idx = torch.cat(all_indices)[i].item() if all_indices else i
        
        # 从数据集中获取额外信息
        try:
            sample = predict_dataset.manifest.iloc[idx]
            # 提取所需信息，如果不存在则设为None
            pid = sample.get('pid', None)
            pCR = sample.get('pCR', None)
            center = sample.get('center', None)
            pre_img = sample.get('pre_img', None)
            pre_mask = sample.get('pre_mask', None)
            post_img = sample.get('post_img', None)
            post_mask = sample.get('post_mask', None)
            
            # 将路径对象转换为字符串（如果需要）
            if isinstance(pre_img, object) and hasattr(pre_img, '__str__'):
                pre_img = str(pre_img)
            if isinstance(pre_mask, object) and hasattr(pre_mask, '__str__'):
                pre_mask = str(pre_mask)
            if isinstance(post_img, object) and hasattr(post_img, '__str__'):
                post_img = str(post_img)
            if isinstance(post_mask, object) and hasattr(post_mask, '__str__'):
                post_mask = str(post_mask)
                
        except Exception as e:
            print(f"获取样本信息时出错 (索引 {idx}): {e}")
            pid, pCR, center, pre_img, pre_mask, post_img, post_mask = None, None, None, None, None, None, None
        
        # 添加到结果列表
        results.append({
            'idx': idx,
            'prediction': pred,
            'probability': prob,
            'pid': pid,
            'pCR': pCR,
            'center': center,
            'pre_img': pre_img,
            'pre_mask': pre_mask,
            'post_img': post_img,
            'post_mask': post_mask
        })
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(results)
    predictions_file = os.path.join(config['save_dir'], 'predictions.csv')
    df.to_csv(predictions_file, index=False, encoding='utf-8')

    print(f"预测完成！结果已保存到 {predictions_file}")


def main() -> None:
    """
    主函数
    """
    # 解析命令行参数（仅获取配置文件路径）
    args = parse_arguments()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 处理配置文件，确保必要参数存在
    processed_config = merge_args_with_config(args, config)
    
    print(f"配置信息: {json.dumps(processed_config, indent=2, ensure_ascii=False)}")

    # 验证必要的顶层参数
    required_params = {'mode', 'random_seed'}
    for param in required_params:
        if param not in processed_config or processed_config[param] is None:
            raise ValueError(f"配置中缺少必要的参数: {param}")
    
    # 根据模式执行不同的功能
    if processed_config['mode'] == 'train':
        train_model(processed_config)
    elif processed_config['mode'] == 'test':
        test_model(processed_config)
    elif processed_config['mode'] == 'predict':
        predict_with_model(processed_config)
    else:
        raise ValueError(f"不支持的模式: {processed_config['mode']}")


if __name__ == "__main__":
    main()
