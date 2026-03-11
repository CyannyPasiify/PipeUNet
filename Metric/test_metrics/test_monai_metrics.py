#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONAI Metrics Shape Test Tool

This script tests the output shapes of MONAI metrics from metric_configurer.py
under different reduction modes: none, mean, sum, mean_batch, sum_batch, mean_channel, sum_channel.
It prints the output shape in tuple format for each metric and reduction mode.
"""

import torch
from typing import List, Tuple, Dict, Any, Callable, Union, Optional, Literal

# Import the metrics and assert function
from Metric.metric_configurer import Dice, GDice, IoU, HD, SD, NSD, MSE, MAE, RMSE, PSNR, SSIM, MSSSIM, \
    assert_input_monaimetrics

# Define reduction modes to test
REDUCTION_MODES: List[str] = [
    "none",
    "mean",
    "sum",
    "mean_batch",
    "sum_batch",
    "mean_channel",
    "sum_channel"
]

# Test data configuration
BATCH_SIZE: int = 4
NUM_CHANNELS: int = 7
VOLUME_X: int = 96
VOLUME_Y: int = 128
VOLUME_Z: int = 160


def generate_segmentation_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate test data for segmentation metrics
    Returns: (y_pred, y_gt) where both are binary tensors with shape (B, C, H, W, D)
    """
    y_pred: torch.Tensor = torch.randint(0, 2, (BATCH_SIZE, NUM_CHANNELS, VOLUME_X, VOLUME_Y, VOLUME_Z),
                                         dtype=torch.int)
    y_gt: torch.Tensor = torch.randint(0, 2, (BATCH_SIZE, NUM_CHANNELS, VOLUME_X, VOLUME_Y, VOLUME_Z), dtype=torch.int)
    return y_pred, y_gt


def generate_image_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate test data for image-to-image metrics
    Returns: (y_pred, y_gt) where both are float tensors with shape (B, C, X, Y, Z)
    """
    y_pred: torch.Tensor = torch.randn(BATCH_SIZE, NUM_CHANNELS, VOLUME_X, VOLUME_Y, VOLUME_Z, dtype=torch.float)
    y_gt: torch.Tensor = torch.randn(BATCH_SIZE, NUM_CHANNELS, VOLUME_X, VOLUME_Y, VOLUME_Z, dtype=torch.float)
    return y_pred, y_gt


def test_metric_shape(
        metric_name: str,
        metric_class: Any,
        data_generator: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        **metric_kwargs: Dict[str, Any]
) -> None:
    """
    Test a metric's output shape under different reduction modes
    
    Args:
        metric_name: Name of the metric
        metric_class: Metric class to test
        data_generator: Function to generate test data
        **metric_kwargs: Additional keyword arguments for the metric
    """
    y_pred, y_gt = data_generator()

    # Determine task type based on data generator
    task_type: Literal["segmentation", "img2img"] = \
        "segmentation" if data_generator == generate_segmentation_data else "img2img"

    # Validate input data using assert_input_monaimetrics
    assert_input_monaimetrics(
        task_type=task_type,
        y_pred=y_pred,
        y_gt=y_gt
    )

    print(f"{metric_name}:")
    for reduction in REDUCTION_MODES:
        try:
            # Create metric instance with current reduction mode
            metric: Any = metric_class(reduction=reduction, **metric_kwargs)

            # Compute metric
            result: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = metric(y_pred, y_gt)

            # Get shape
            if isinstance(result, tuple):
                # Some metrics return (metric, not_nans) when get_not_nans=True
                shape: torch.Size = result[0].shape
            else:
                shape: torch.Size = result.shape

            # Print shape
            print(f"  reduction={reduction}: {tuple(shape)} {result}")
        except Exception as e:
            print(f"  reduction={reduction}: ERROR - {e}")


def main() -> None:
    """
    Main function to run all metric shape tests
    """
    print("Testing MONAI Metrics Output Shapes")
    print("=" * 60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of channels: {NUM_CHANNELS}")
    print(f"Image size: {VOLUME_X}x{VOLUME_Y}x{VOLUME_Z}")
    print("=" * 60)

    # Test segmentation metrics
    segmentation_metrics: List[Union[
        Tuple[str, Any, Callable[[], Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[str, Any, Callable[[], Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]
    ]] = [
        ("Dice", Dice, generate_segmentation_data),
        ("GDice", GDice, generate_segmentation_data),
        ("IoU", IoU, generate_segmentation_data),
        ("HD", HD, generate_segmentation_data),
        ("SD", SD, generate_segmentation_data),
        ("NSD", NSD, generate_segmentation_data, {"class_thresholds": [0.5] * (NUM_CHANNELS - 1)}),
        # exclude background=0
    ]

    # Test image-to-image metrics
    image_metrics: List[Union[Tuple[str, Any, Callable[[], Tuple[torch.Tensor, torch.Tensor]]], Tuple[
        str, Any, Callable[[], Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]]] = [
        ("MSE", MSE, generate_image_data),
        ("MAE", MAE, generate_image_data),
        ("RMSE", RMSE, generate_image_data),
        ("PSNR", PSNR, generate_image_data, {"max_val": 1.0}),
        ("SSIM", SSIM, generate_image_data, {"spatial_dims": 3}),
        ("MSSSIM", MSSSIM, generate_image_data, {"spatial_dims": 3, "weights": (0.0448, 0.2856, 0.3001)}),
    ]

    # Run all tests
    for idx, metric_info in enumerate(segmentation_metrics + image_metrics):
        idx: int
        metric_info: Union[Tuple[str, Any, Callable[[], Tuple[torch.Tensor, torch.Tensor]]], Tuple[
            str, Any, Callable[[], Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]]

        if len(metric_info) == 4:
            name: str
            cls: Any
            data_gen: Callable[[], Tuple[torch.Tensor, torch.Tensor]]
            kwargs: Dict[str, Any]
            name, cls, data_gen, kwargs = metric_info
        else:
            name: str
            cls: Any
            data_gen: Callable[[], Tuple[torch.Tensor, torch.Tensor]]
            name, cls, data_gen = metric_info
            kwargs: Dict[str, Any] = {}

        print(f"[{idx}] Testing {name}: {cls.__name__}")
        print("-" * 60)
        test_metric_shape(name, cls, data_gen, **kwargs)
        print()


if __name__ == "__main__":
    print("Starting test script...")
    try:
        main()
        print("Test script completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
