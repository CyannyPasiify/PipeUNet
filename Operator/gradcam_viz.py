#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM 可视化工具

功能概述：
    用于医学图像的Grad-CAM可视化，支持加载3D模型和NIfTI格式图像数据
    实现基于梯度的类激活图计算，帮助理解模型决策过程

命令行参数：
    -pri/--pre_image: 预处理图像文件路径 (nii.gz)
    -prm/--pre_mask: 预处理掩码文件路径 (nii.gz)
    -pti/--post_image: 后处理图像文件路径 (nii.gz)
    -ptm/--post_mask: 后处理掩码文件路径 (nii.gz)
    -ckpt/--checkpoint: 模型权重文件路径
    -t/--target_layers: 目标可视化层列表（可多个）
    -o/--output_cam: 输出目录
"""

import argparse
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional
import monai.transforms as transforms


class GradCAM:
    """Grad-CAM 类用于计算和可视化特征图"""

    def __init__(self, model: torch.nn.Module, target_layers: List[str]):
        """
        初始化 Grad-CAM
        
        Args:
            model: PyTorch 模型
            target_layers: 目标可视化层列表
        """
        self.model = model
        self.model.eval()  # 设置模型为评估模式
        self.target_layers = target_layers
        self.activations = {}  # 存储特征图激活值
        self.gradients = {}  # 存储梯度值

        # 注册钩子函数
        self._register_hooks()

    def _register_hooks(self):
        """为目标层注册前向和反向钩子"""

        def get_target_layer(layer_path: str) -> torch.nn.Module:
            """根据路径获取模型层"""
            parts = layer_path.split('.')
            layer = self
            for part in parts:
                if hasattr(layer, part):
                    layer = getattr(layer, part)
                else:
                    try:
                        part = int(part)  # 如果是索引
                        layer = layer[part]
                    except (ValueError, IndexError):
                        raise ValueError(f"无法找到层: {layer_path}")
            return layer

        for layer_path in self.target_layers:
            target_layer = get_target_layer(layer_path)

            def save_activation(name):
                def hook(module, input, output):
                    self.activations[name] = output.detach()

                return hook

            def save_gradient(name):
                def hook(module, grad_input, grad_output):
                    self.gradients[name] = grad_output[0].detach()

                return hook

            target_layer.register_forward_hook(save_activation(layer_path))
            target_layer.register_backward_hook(save_gradient(layer_path))

    def compute_cam(self, input_tensor: Dict[str, torch.Tensor], target_class: Optional[int] = None):
        """
        计算 Grad-CAM
        
        Args:
            input_tensor: 输入数据字典
            target_class: 目标类别索引
            
        Returns:
            cam_dict: 各层的 CAM 结果字典
        """
        # 提取数据（兼容 EsophagusDataModule 格式）
        # 合并前图像和后图像作为模型输入
        pre_img = input_tensor['pre_img']
        post_img = input_tensor['post_img']
        pre_mask = input_tensor['pre_mask']
        post_mask = input_tensor['post_mask']

        pre_masked_img = pre_mask * pre_img
        post_masked_img = post_mask * post_img

        # 合并通道维度
        # x = torch.cat([pre_img, pre_mask, post_img, post_mask], dim=1) # 20251224
        # x = torch.cat([pre_masked_img, pre_mask, post_masked_img, post_mask], dim=1) # 20251225
        x = torch.cat([pre_img, pre_mask, post_img, post_mask], dim=1) # 20251225_2

        # 前向传播
        self.model.zero_grad()
        outputs = self.model(x)

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # 反向传播
        loss = outputs[0, target_class]
        loss.backward()

        cam_dict = {}

        for layer_name in self.target_layers:
            if layer_name not in self.activations or layer_name not in self.gradients:
                continue

            # 获取特征图和梯度
            activations = self.activations[layer_name]
            gradients = self.gradients[layer_name]

            # 计算梯度权重
            weights = torch.mean(gradients, dim=(2, 3, 4))  # 对空间维度平均

            # 计算 CAM
            cam = torch.zeros(activations.shape[2:], dtype=activations.dtype)
            for i, weight in enumerate(weights[0]):
                cam += weight * activations[0, i]

            # 激活和归一化
            # cam = cam.clamp_min(0)  # ReLU 激活
            # cam = cam / (cam.max() + 1e-8)  # 归一化
            cam = torch.sigmoid(cam) # Sigmoid 激活
            centered_abs_cam = torch.abs(cam - 0.5)
            max_abs_cam = torch.max(centered_abs_cam)
            cam = (cam - 0.5) / max_abs_cam * 0.5 + 0.5

            cam_dict[layer_name] = cam.detach()

        return cam_dict


def main():
    """主函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Grad-CAM 可视化工具")
    parser.add_argument("-pri", "--pre_image", type=str, help="预处理图像路径 (nii.gz)")
    parser.add_argument("-prm", "--pre_mask", type=str, help="预处理掩码路径 (nii.gz)")
    parser.add_argument("-pti", "--post_image", type=str, help="后处理图像路径 (nii.gz)")
    parser.add_argument("-ptm", "--post_mask", type=str, help="后处理掩码路径 (nii.gz)")
    parser.add_argument("-ckpt", "--checkpoint", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("-t", "--target_layers", type=str, nargs="+", required=True, help="目标可视化层列表")
    parser.add_argument("-o", "--output_cam", type=str, required=True, help="输出目录")
    parser.add_argument("-r", "--ref_volume", type=str, help="参照图像路径 (nii.gz)，用于对齐规格")
    parser.add_argument("-tc", "--target_class", type=int, help="指定当前样本的所属类别，用于Grad-CAM计算")
    parser.add_argument("-p", "--prefix", type=str, choices=['pre', 'post'], default=None)
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_cam, exist_ok=True)

    print("加载模型...")
    # 加载模型 使用 DualResNetLightningModule 从检查点加载
    from Module.module_share_weight_dual_branch_resnet import DualResNetLightningModule

    # 从检查点加载整个模型
    model = DualResNetLightningModule.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))
    model = model.to(torch.device('cpu'))
    model.eval()
    print("模型加载完成")
    print(model)

    print("加载输入数据...")
    # 加载数据（参考 ds_pipe.py 的加载方式）
    keys = ['pre_img', 'pre_mask', 'post_img', 'post_mask']
    if args.ref_volume:
        keys.append('ref_volume')

    loader = transforms.LoadImaged(
        keys=keys,
        image_only=False,
        ensure_channel_first=True
    )

    data_dict = {}
    if args.pre_image:
        data_dict['pre_img'] = args.pre_image
    if args.pre_mask:
        data_dict['pre_mask'] = args.pre_mask
    if args.post_image:
        data_dict['post_img'] = args.post_image
    if args.post_mask:
        data_dict['post_mask'] = args.post_mask
    if args.ref_volume:
        data_dict['ref_volume'] = args.ref_volume

    loaded_data = loader(data_dict)

    # 确定参照图像
    ref_image = None
    if 'ref_volume' in loaded_data:
        ref_image = loaded_data['ref_volume']
    elif 'pre_img' in loaded_data:
        ref_image = loaded_data['pre_img']

    # 准备输入张量
    process_data = {}
    for key in loaded_data:
        if isinstance(loaded_data[key], torch.Tensor):
            process_data[key] = loaded_data[key].unsqueeze(0).to(torch.device('cpu'))  # 添加 batch 维度
    print("数据加载完成")

    print("初始化 Grad-CAM...")
    # 初始化 Grad-CAM
    # 从 ResNetLightningModule 加载的模型需要访问其内部的模型
    # ResNetLightningModule 使用 self.model 来存储实际的网络
    target_model = model.model  # 这是实际的 ResNet/ConvNeXt 模型
    gradcam = GradCAM(target_model, args.target_layers)

    print("计算 Grad-CAM...")
    # 计算 CAM
    cam_dict = gradcam.compute_cam(process_data, args.target_class)

    print("保存激活图...")
    # 初始化 MONAI 的 SaveImage 转换
    save_image = transforms.SaveImage(
        output_dir=args.output_cam,
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False
    )

    # 获取参照图像规格
    ref_affine = None
    ref_shape = None
    if ref_image is not None:
        if hasattr(ref_image, 'affine'):
            ref_affine = ref_image.affine
        if hasattr(ref_image, 'shape'):
            ref_shape = ref_image.shape
            if len(ref_shape) == 4:  # 去除通道维度
                ref_shape = ref_shape[1:]
        elif isinstance(ref_image, torch.Tensor):
            ref_shape = ref_image.shape[1:]  # 去除通道维度

    idx = 0
    # 保存激活图为 NIfTI 文件
    for layer_name, cam in cam_dict.items():
        idx += 1
        # 生成文件名（替换点和空格）
        filename = f'{idx:03d}_' + layer_name.replace('.', '_').replace('/', '_')
        
        # 转换为 MONAI MetaTensor 格式
        cam_tensor = cam.unsqueeze(0)  # 添加通道维度
        
        # 保存图像
        # save_image(cam_tensor, filename=os.path.join(args.output_cam, filename))
        # print(f"保存到: {os.path.join(args.output_cam, filename + '.nii.gz')}")

        # 如果需要重采样对齐
        if ref_affine is not None and ref_shape is not None and cam.shape != ref_shape:
            import monai.transforms as mt

            print(f"重采样 {layer_name} 激活图以匹配参照图像规格...")

            # 创建 MONAI 重采样转换
            resampler = mt.Resize(spatial_size=ref_shape, mode='bilinear', align_corners=True)
            print(ref_shape)

            # 重采样
            resampled_cam = resampler(cam_tensor)
            # if args.prefix is not None:
            #     resampled_cam = resampled_cam * process_data[f'{args.prefix}_mask'].squeeze(0)  # 蒙版掩蔽
            resampled_cam.affine = ref_affine
            # 保存对齐版本
            aligned_filename = f'{idx:03d}_' + layer_name.replace('.', '_').replace('/', '_') + '_aligned'
            save_image(resampled_cam, filename=os.path.join(args.output_cam, aligned_filename))
            print(f"保存对齐版本到: {os.path.join(args.output_cam, aligned_filename + '.nii.gz')}")

    print("所有任务完成！")


if __name__ == "__main__":
    main()