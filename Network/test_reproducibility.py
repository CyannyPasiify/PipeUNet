#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D模型参数可重复性测试脚本

功能：
    测试相同配置下实例化的ResNet3D、ConvNeXt3D和ResNeXt3D模型参数是否一致
    验证模型初始化的可重复性
"""

import torch
import torch.nn as nn
from resnet_convnext_3d import (
    create_resnet3d_model,
    create_convnext3d_model,
    create_resnext3d_model
)


def test_model_reproducibility(create_model_func, model_config):
    """
    测试模型初始化的可重复性
    
    参数:
        create_model_func: 模型创建函数
        model_config: 模型配置参数字典
    
    返回:
        bool: 参数是否一致
    """
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
    
    # 第一次实例化模型
    model1 = create_model_func(**model_config)
    
    # 重置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
    
    # 第二次实例化模型
    model2 = create_model_func(**model_config)
    
    # 比较参数
    params1 = list(model1.state_dict().values())
    params2 = list(model2.state_dict().values())
    
    # 检查参数数量是否相同
    if len(params1) != len(params2):
        print(f"参数数量不同: {len(params1)} vs {len(params2)}")
        return False
    
    # 逐个比较参数值
    all_match = True
    for i, (p1, p2) in enumerate(zip(params1, params2)):
        if not torch.allclose(p1, p2):
            # 计算参数差异
            max_diff = torch.max(torch.abs(p1 - p2)).item()
            mean_diff = torch.mean(torch.abs(p1 - p2)).item()
            print(f"参数 {i} 不匹配: 最大差异={max_diff:.6f}, 平均差异={mean_diff:.6f}")
            all_match = False
    
    return all_match


def test_resnet3d_reproducibility():
    """
    测试ResNet3D模型的可重复性
    """
    print("===== 测试 ResNet3D 模型可重复性 =====")
    
    # 定义模型配置
    model_config = {
        'in_channels': 4,
        'img_size': (128, 128, 128),
        'num_classes': 3,
        'model_size': 'resnet50'
    }
    
    # 测试可重复性
    is_reproducible = test_model_reproducibility(create_resnet3d_model, model_config)
    
    if is_reproducible:
        print("[PASS] ResNet3D 模型参数一致，可重复性验证通过！")
    else:
        print("[FAIL] ResNet3D 模型参数不一致，可重复性验证失败！")
    print()
    
    return is_reproducible


def test_convnext3d_reproducibility():
    """
    测试ConvNeXt3D模型的可重复性
    """
    print("===== 测试 ConvNeXt3D 模型可重复性 =====")
    
    # 定义模型配置
    model_config = {
        'in_channels': 4,
        'img_size': (128, 128, 128),
        'num_classes': 3,
        'model_size': 'base'
    }
    
    # 测试可重复性
    is_reproducible = test_model_reproducibility(create_convnext3d_model, model_config)
    
    if is_reproducible:
        print("[PASS] ConvNeXt3D 模型参数一致，可重复性验证通过！")
    else:
        print("[FAIL] ConvNeXt3D 模型参数不一致，可重复性验证失败！")
    print()
    
    return is_reproducible


def test_resnext3d_reproducibility():
    """
    测试ResNeXt3D模型的可重复性
    """
    print("===== 测试 ResNeXt3D 模型可重复性 =====")
    
    # 定义模型配置
    model_config = {
        'in_channels': 4,
        'img_size': (128, 128, 128),
        'num_classes': 3,
        'model_size': 'resnext50_32x4d'
    }
    
    # 测试可重复性
    is_reproducible = test_model_reproducibility(create_resnext3d_model, model_config)
    
    if is_reproducible:
        print("[PASS] ResNeXt3D 模型参数一致，可重复性验证通过！")
    else:
        print("[FAIL] ResNeXt3D 模型参数不一致，可重复性验证失败！")
    print()
    
    return is_reproducible


def test_different_seeds_effect():
    """
    测试不同随机种子对模型参数的影响
    """
    print("===== 测试不同随机种子对 ResNet3D 模型参数的影响 =====")
    
    # 定义模型配置
    model_config = {
        'in_channels': 4,
        'img_size': (128, 128, 128),
        'num_classes': 3,
        'model_size': 'resnet50'
    }
    
    # 使用种子42实例化模型
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
    model_seed42 = create_resnet3d_model(**model_config)
    
    # 使用种子123实例化模型
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123) if torch.cuda.is_available() else None
    model_seed123 = create_resnet3d_model(**model_config)
    
    # 比较参数
    params_seed42 = list(model_seed42.state_dict().values())
    params_seed123 = list(model_seed123.state_dict().values())
    
    # 检查参数是否不同
    all_same = True
    for i, (p42, p123) in enumerate(zip(params_seed42, params_seed123)):
        if not torch.allclose(p42, p123):
            all_same = False
            # 计算参数差异
            max_diff = torch.max(torch.abs(p42 - p123)).item()
            mean_diff = torch.mean(torch.abs(p42 - p123)).item()
            print(f"不同种子参数 {i} 差异: 最大差异={max_diff:.6f}, 平均差异={mean_diff:.6f}")
            # 只打印前5个不同的参数，避免输出过多
            if i >= 4:
                break
    
    if all_same:
        print("[FAIL] 警告：不同随机种子生成的模型参数相同！")
    else:
        print("[PASS] 不同随机种子生成的模型参数不同，符合预期！")
    print()


def main():
    """
    主函数，测试所有模型的可重复性
    """
    print("开始模型参数可重复性测试...")
    
    # 测试相同种子下的模型可重复性
    resnet_reproducible = test_resnet3d_reproducibility()
    convnext_reproducible = test_convnext3d_reproducibility()
    resnext_reproducible = test_resnext3d_reproducibility()
    
    # 测试不同种子的影响
    test_different_seeds_effect()
    
    # 总结
    print("===== 测试结果总结 =====")
    print(f"ResNet3D 可重复性: {'通过' if resnet_reproducible else '失败'}")
    print(f"ConvNeXt3D 可重复性: {'通过' if convnext_reproducible else '失败'}")
    print(f"ResNeXt3D 可重复性: {'通过' if resnext_reproducible else '失败'}")
    
    all_passed = resnet_reproducible and convnext_reproducible and resnext_reproducible
    if all_passed:
        print("[PASS] 所有模型可重复性验证通过！")
    else:
        print("[FAIL] 部分模型可重复性验证失败！")


if __name__ == "__main__":
    main()