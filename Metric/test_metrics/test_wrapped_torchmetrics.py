#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiclass Metrics Test and Visualization Tool - Wrapped Class Version

This script generates simulated multiclass data, calculates various evaluation metrics using wrapped classes from metric_configurer.py,
and generates visualization charts. Supports custom sample count, class count, and error rate distribution.
Also detects consistency between wrapped classes and original torchmetrics classes.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchmetrics
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy.typing as npt

# Import wrapped classes from metric_configurer
from Metric.metric_configurer import (
    # Multiclass metrics
    MetricMulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision, MulticlassConfusionMatrix,
    MulticlassF1Score, MulticlassPrecision, MetricMulticlassPrecisionRecallCurve, MulticlassRecall,
    MulticlassSpecificity, MetricMulticlassROC,
    # Binary metrics
    MetricBinaryAccuracy, MetricBinaryAUROC, BinaryAveragePrecision, BinaryConfusionMatrix,
    BinaryF1Score, BinaryPrecision, MetricBinaryPrecisionRecallCurve, BinaryRecall,
    BinarySpecificity, MetricBinaryROC,
    # Multilabel metrics
    MetricMultilabelAccuracy, MultilabelAUROC, MultilabelAveragePrecision, MultilabelConfusionMatrix,
    MultilabelF1Score, MultilabelPrecision, _BetterMultilabelPrecisionRecallCurve, MultilabelRecall,
    MultilabelSpecificity, MetricMultilabelROC,
    # Assert functions
    assert_input_torchmetrics
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Namespace containing all command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Classification Metrics Test and Visualization Tool - Wrapped Class Version')
    parser.add_argument('-s', '--samples', type=int, default=1000,
                        help='Number of samples, default is 1000')
    parser.add_argument('-c', '--classes', type=int, default=3,
                        help='Number of classes, default is 3')
    parser.add_argument('-p', '--error_rate', type=float, default=0.2,
                        help='Expected error rate, default is 0.2')
    parser.add_argument('-o', '--save_dir', type=str, default=None,
                        help='Directory path to save visualization results, default is not to save')
    parser.add_argument('-x', '--compare', action='store_true', default=False,
                        help='Compare consistency between wrapped classes and original classes')
    parser.add_argument('-b', '--binary', action='store_true', default=False,
                        help='Enable binary classification test mode')
    parser.add_argument('-mc', '--multiclass', action='store_true', default=False,
                        help='Enable multiclass classification test mode')
    parser.add_argument('-ml', '--multilabel', action='store_true', default=False,
                        help='Enable multilabel classification test mode')
    return parser.parse_args()


def generate_binary_data(num_samples: int, error_rate: float) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Generate binary classification data, simulating the distribution with specified error rate
    Ensure predictions have error_rate probability of not matching the label
    
    Args:
        num_samples: Number of samples
        error_rate: Target error rate
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - y_true: True labels (num_samples,)
            - y_pred_probs: Prediction probabilities (num_samples, 1) or (num_samples,), depending on need
    """
    # Set random seed to ensure reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Generate true labels (binary classification, 0 or 1)
    y_true: npt.NDArray[np.int64] = np.random.randint(0, 2, size=num_samples)

    # Initialize prediction probabilities
    y_pred_probs: npt.NDArray[np.float64] = np.zeros(num_samples)

    # Generate prediction probabilities for each sample
    for i in range(num_samples):
        true_class = y_true[i]

        # Randomly determine if current sample is correct or incorrect
        if np.random.random() < error_rate:
            # Incorrect case: prediction probability favors the wrong class
            # If true class is 1, prediction probability is between 0-0.45
            # If true class is 0, prediction probability is between 0.55-1.0
            if true_class == 1:
                y_pred_probs[i] = np.random.uniform(0, 0.45)
            else:
                y_pred_probs[i] = np.random.uniform(0.55, 1.0)
        else:
            # Correct case: prediction probability favors the correct class
            # If true class is 1, prediction probability is between 0.55-1.0
            # If true class is 0, prediction probability is between 0-0.45
            if true_class == 1:
                y_pred_probs[i] = np.random.uniform(0.55, 1.0)
            else:
                y_pred_probs[i] = np.random.uniform(0, 0.45)

    # Add small amount of noise to increase randomness
    noise = np.random.normal(0, 0.05, size=num_samples)
    y_pred_probs += noise

    # Ensure probabilities are within 0-1 range
    y_pred_probs = np.clip(y_pred_probs, 0, 1)

    return torch.tensor(y_true, dtype=torch.int64), torch.tensor(y_pred_probs, dtype=torch.float32)


def generate_multiclass_data(num_samples: int, num_classes: int, error_rate: float) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Generate multiclass data, simulating the distribution with specified error rate
    Ensure predictions have error_rate probability of not matching the label (not the maximum probability among all classes)
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        error_rate: Target error rate
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - y_true: True labels (num_samples,)
            - y_pred_probs: Prediction probabilities (num_samples, num_classes)
    """
    # Set random seed to ensure reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Generate true labels (uniform distribution)
    y_true: npt.NDArray[np.int64] = np.random.randint(0, num_classes, size=num_samples)

    # Initialize prediction probabilities
    y_pred_probs: npt.NDArray[np.float64] = np.zeros((num_samples, num_classes))

    # Generate prediction probabilities for each sample
    for i in range(num_samples):
        true_class = y_true[i]

        # Randomly determine if current sample is correct or incorrect
        if np.random.random() < error_rate:
            # Incorrect case: ensure true class is not the one with maximum probability
            # Handle special case when there's only one class
            if num_classes <= 1:
                # Cannot generate incorrect prediction with only one class, set probability to 1.0
                y_pred_probs[i, 0] = 1.0
                continue

            # Generate indices of all classes except the true class
            other_classes = np.setdiff1d(np.arange(num_classes), [true_class])

            # Ensure other_classes is not empty
            if len(other_classes) == 0:
                # If only one class, set probability to 1.0
                y_pred_probs[i, true_class] = 1.0
                continue

            # Randomly select an incorrect class as the one with maximum probability
            max_class = np.random.choice(other_classes)

            # Randomly generate maximum probability value (range: 0.55-0.95) to ensure it's significantly larger than others
            max_prob = np.random.uniform(0.55, 0.95)
            # Set maximum probability to the incorrect class
            y_pred_probs[i, max_class] = max_prob

            # Distribute remaining probability to other classes, including the true class
            remaining_prob = 1.0 - max_prob

            # Distribute probability to other classes
            if num_classes > 2:
                # Generate random weights for all non-max_class classes
                non_max_classes = np.setdiff1d(np.arange(num_classes), [max_class])
                other_probs = np.random.random(len(non_max_classes))

                # Normalize and distribute probabilities
                other_probs_normalized = other_probs / np.sum(other_probs)
                for j, cls in enumerate(non_max_classes):
                    y_pred_probs[i, cls] = remaining_prob * other_probs_normalized[j]
            else:
                # Binary classification case, directly assign remaining probability to true class
                y_pred_probs[i, true_class] = remaining_prob
        else:
            # Correct case: ensure true class is the one with maximum probability
            # Handle special case when there's only one class
            if num_classes <= 1:
                y_pred_probs[i, 0] = 1.0
                continue

            # Randomly generate maximum probability value (range: 0.55-0.95)
            max_prob = np.random.uniform(0.55, 0.95)
            # Set maximum probability to the true class
            y_pred_probs[i, true_class] = max_prob

            # Randomly distribute remaining probability to other classes
            remaining_prob = 1.0 - max_prob
            if num_classes > 1:
                # Generate random weights for other classes
                other_classes = np.setdiff1d(np.arange(num_classes), [true_class])
                if len(other_classes) > 0:
                    other_probs = np.random.random(len(other_classes))
                    other_probs_normalized = other_probs / np.sum(other_probs)
                    for j, cls in enumerate(other_classes):
                        y_pred_probs[i, cls] = remaining_prob * other_probs_normalized[j]

    # Add small amount of noise to increase randomness
    noise = np.random.normal(0, 0.05, size=(num_samples, num_classes))
    y_pred_probs += noise

    # Ensure probability normalization
    y_pred_probs = np.clip(y_pred_probs, 0, None)  # Avoid negative probabilities
    y_pred_probs = y_pred_probs / y_pred_probs.sum(axis=1, keepdims=True)

    return torch.tensor(y_true, dtype=torch.int64), torch.tensor(y_pred_probs, dtype=torch.float32)


def generate_multilabel_data(num_samples: int, num_labels: int, error_rate: float) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Generate multilabel classification data, simulating the distribution with specified error rate
    
    Args:
        num_samples: Number of samples
        num_labels: Number of labels
        error_rate: Target error rate
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - y_true: True labels (num_samples, num_labels), binary indicators
            - y_pred_probs: Prediction probabilities (num_samples, num_labels)
    """
    # Set random seed to ensure reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Generate true labels (each sample can have multiple classes)
    y_true: npt.NDArray[np.int64] = np.random.randint(0, 2, size=(num_samples, num_labels))

    # Ensure at least one class per sample
    for i in range(num_samples):
        if np.sum(y_true[i]) == 0:
            y_true[i, np.random.randint(0, num_labels)] = 1

    # Initialize prediction probabilities
    y_pred_probs: npt.NDArray[np.float64] = np.zeros((num_samples, num_labels))

    # Generate prediction probabilities for each sample and class
    for i in range(num_samples):
        for j in range(num_labels):
            true_label = y_true[i, j]

            # Randomly determine if current class prediction is correct or incorrect
            if np.random.random() < error_rate:
                # Incorrect case
                if true_label == 1:
                    # Predict probability close to 0
                    y_pred_probs[i, j] = np.random.uniform(0, 0.45)
                else:
                    # Predict probability close to 1
                    y_pred_probs[i, j] = np.random.uniform(0.55, 1.0)
            else:
                # Correct case
                if true_label == 1:
                    # Predict probability close to 1
                    y_pred_probs[i, j] = np.random.uniform(0.55, 1.0)
                else:
                    # Predict probability close to 0
                    y_pred_probs[i, j] = np.random.uniform(0, 0.45)

    # Add small amount of noise to increase randomness
    noise = np.random.normal(0, 0.05, size=(num_samples, num_labels))
    y_pred_probs += noise

    # Ensure probabilities are within 0-1 range
    y_pred_probs = np.clip(y_pred_probs, 0, 1)

    return torch.tensor(y_true, dtype=torch.int64), torch.tensor(y_pred_probs, dtype=torch.float32)


def compute_binary_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor) -> Dict[str, Any]:
    """
    Calculate various binary classification evaluation metrics using wrapped TorchMetrics
    
    Args:
        y_true: True labels
        y_pred_probs: Prediction probabilities
    
    Returns:
        Dict[str, Any]: Dictionary containing metric objects and calculated values
    """
    # Ensure running on appropriate device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # Validate input data using assert_input_torchmetrics
    assert_input_torchmetrics(
        task_type="binary",
        y_pred=y_pred_probs,
        y_gt=y_true,
        multidim_average="global"
    )

    # Get predicted classes
    y_pred: torch.Tensor = (y_pred_probs >= 0.5).long()

    # Initialize binary classification metric calculators
    accuracy = MetricBinaryAccuracy().to(device)
    auroc = MetricBinaryAUROC().to(device)
    average_precision = BinaryAveragePrecision().to(device)
    conf_matrix = BinaryConfusionMatrix().to(device)
    f1_score = BinaryF1Score().to(device)
    prc = MetricBinaryPrecisionRecallCurve().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    roc = MetricBinaryROC().to(device)
    specificity = BinarySpecificity().to(device)

    # Calculate metric values
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # Update confusion matrix, PRC curve, ROC curve
    conf_matrix(y_pred_probs, y_true)
    prc(y_pred_probs, y_true)
    roc(y_pred_probs, y_true)

    metrics = {
        # Metric values
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # Metric objects (for visualization)
        'accuracy': accuracy,
        'auroc': auroc,
        'average_precision': average_precision,
        'conf_matrix': conf_matrix,
        'f1_score': f1_score,
        'precision': precision,
        'prc': prc,
        'recall': recall,
        'roc': roc,
        'specificity': specificity,
        # Data
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred,
        'num_classes': 2
    }

    return metrics


def compute_multiclass_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, num_classes: int) -> Dict[str, Any]:
    """
    Calculate various evaluation metrics using wrapped TorchMetrics, including overall metrics and per-class metrics
    
    Args:
        y_true: True labels
        y_pred_probs: Prediction probabilities
        num_classes: Number of classes
    
    Returns:
        Dict[str, Any]: Dictionary containing metric objects and calculated values
    """
    # Ensure running on appropriate device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # Validate input data using assert_input_torchmetrics
    assert_input_torchmetrics(
        task_type="multiclass",
        y_pred=y_pred_probs,
        y_gt=y_true,
        multidim_average="global",
        num_classes=num_classes
    )

    # Get predicted classes
    y_pred: torch.Tensor = y_pred_probs.argmax(dim=1)

    # Initialize metric calculators - overall metrics (macro average but for Acc)
    accuracy = MetricMulticlassAccuracy(num_classes=num_classes).to(device)
    auroc = MulticlassAUROC(num_classes=num_classes).to(device)
    average_precision = MulticlassAveragePrecision(num_classes=num_classes).to(device)
    conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    f1_score = MulticlassF1Score(num_classes=num_classes).to(device)
    prc = MetricMulticlassPrecisionRecallCurve(num_classes=num_classes).to(device)
    precision = MulticlassPrecision(num_classes=num_classes).to(device)
    recall = MulticlassRecall(num_classes=num_classes).to(device)
    roc = MetricMulticlassROC(num_classes=num_classes).to(device)
    specificity = MulticlassSpecificity(num_classes=num_classes).to(device)

    # Initialize per-class metric calculators
    class_auroc = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
    class_average_precision = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    class_f1_score = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    class_precision = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    class_recall = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    class_specificity = MulticlassSpecificity(num_classes=num_classes, average=None).to(device)

    # Calculate overall metrics (scalars)
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # Calculate per-class metrics
    class_auroc_values: npt.NDArray[np.float64] = class_auroc(y_pred_probs, y_true).cpu().numpy()
    class_ap_values: npt.NDArray[np.float64] = class_average_precision(y_pred_probs, y_true).cpu().numpy()
    class_f1_values: npt.NDArray[np.float64] = class_f1_score(y_pred_probs, y_true).cpu().numpy()
    class_precision_values: npt.NDArray[np.float64] = class_precision(y_pred_probs, y_true).cpu().numpy()
    class_recall_values: npt.NDArray[np.float64] = class_recall(y_pred_probs, y_true).cpu().numpy()
    class_specificity_values: npt.NDArray[np.float64] = class_specificity(y_pred_probs, y_true).cpu().numpy()

    # Update confusion matrix, PRC curve, ROC curve
    conf_matrix(y_pred_probs, y_true)
    prc(y_pred_probs, y_true)
    roc(y_pred_probs, y_true)

    metrics = {
        # Overall metric values
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # Per-class metric values
        'class_auroc_values': class_auroc_values,
        'class_ap_values': class_ap_values,
        'class_f1_values': class_f1_values,
        'class_precision_values': class_precision_values,
        'class_recall_values': class_recall_values,
        'class_specificity_values': class_specificity_values,
        # Metric objects (for visualization)
        'accuracy': accuracy,
        'auroc': auroc,
        'average_precision': average_precision,
        'conf_matrix': conf_matrix,
        'f1_score': f1_score,
        'precision': precision,
        'prc': prc,
        'recall': recall,
        'roc': roc,
        'specificity': specificity,
        # Per-class metric objects (for visualization)
        'class_auroc': class_auroc,
        'class_average_precision': class_average_precision,
        'class_f1_score': class_f1_score,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_specificity': class_specificity,
        # Data
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred,
        'num_classes': num_classes
    }

    return metrics


def compute_multilabel_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, num_labels: int) -> Dict[str, Any]:
    """
    Calculate various evaluation metrics using wrapped TorchMetrics for multilabel classification
    
    Args:
        y_true: True labels (num_samples, num_labels)
        y_pred_probs: Prediction probabilities (num_samples, num_labels)
        num_labels: Number of labels
    
    Returns:
        Dict[str, Any]: Dictionary containing metric objects and calculated values
    """
    # Ensure running on appropriate device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # Validate input data using assert_input_torchmetrics
    assert_input_torchmetrics(
        task_type="multilabel",
        y_pred=y_pred_probs,
        y_gt=y_true,
        multidim_average="global"
    )

    # Get predicted classes
    y_pred: torch.Tensor = (y_pred_probs >= 0.5).float()

    # Initialize metric calculators - overall metrics (macro average)
    accuracy = MetricMultilabelAccuracy(num_labels=num_labels).to(device)
    auroc = MultilabelAUROC(num_labels=num_labels).to(device)
    average_precision = MultilabelAveragePrecision(num_labels=num_labels).to(device)
    conf_matrix = MultilabelConfusionMatrix(num_labels=num_labels).to(device)
    f1_score = MultilabelF1Score(num_labels=num_labels).to(device)
    prc = _BetterMultilabelPrecisionRecallCurve(num_labels=num_labels).to(device)
    precision = MultilabelPrecision(num_labels=num_labels).to(device)
    recall = MultilabelRecall(num_labels=num_labels).to(device)
    roc = MetricMultilabelROC(num_labels=num_labels).to(device)
    specificity = MultilabelSpecificity(num_labels=num_labels).to(device)

    # Initialize per-class metric calculators
    class_accuracy = MetricMultilabelAccuracy(num_labels=num_labels, average=None).to(device)
    class_auroc = MultilabelAUROC(num_labels=num_labels, average=None).to(device)
    class_average_precision = MultilabelAveragePrecision(num_labels=num_labels, average=None).to(device)
    class_f1_score = MultilabelF1Score(num_labels=num_labels, average=None).to(device)
    class_precision = MultilabelPrecision(num_labels=num_labels, average=None).to(device)
    class_recall = MultilabelRecall(num_labels=num_labels, average=None).to(device)
    class_specificity = MultilabelSpecificity(num_labels=num_labels, average=None).to(device)

    # Calculate overall metrics (scalars)
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # Calculate per-class metrics
    class_accuracy_values: npt.NDArray[np.float64] = class_accuracy(y_pred_probs, y_true).cpu().numpy()
    class_auroc_values: npt.NDArray[np.float64] = class_auroc(y_pred_probs, y_true).cpu().numpy()
    class_ap_values: npt.NDArray[np.float64] = class_average_precision(y_pred_probs, y_true).cpu().numpy()
    class_f1_values: npt.NDArray[np.float64] = class_f1_score(y_pred_probs, y_true).cpu().numpy()
    class_precision_values: npt.NDArray[np.float64] = class_precision(y_pred_probs, y_true).cpu().numpy()
    class_recall_values: npt.NDArray[np.float64] = class_recall(y_pred_probs, y_true).cpu().numpy()
    class_specificity_values: npt.NDArray[np.float64] = class_specificity(y_pred_probs, y_true).cpu().numpy()

    # Update confusion matrix, PRC curve, ROC curve
    conf_matrix(y_pred_probs, y_true)
    prc(y_pred_probs, y_true)
    roc(y_pred_probs, y_true)

    metrics = {
        # Overall metric values
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # Per-class metric values
        'class_accuracy_values': class_accuracy_values,
        'class_auroc_values': class_auroc_values,
        'class_ap_values': class_ap_values,
        'class_f1_values': class_f1_values,
        'class_precision_values': class_precision_values,
        'class_recall_values': class_recall_values,
        'class_specificity_values': class_specificity_values,
        # Metric objects (for visualization)
        'accuracy': accuracy,
        'auroc': auroc,
        'average_precision': average_precision,
        'conf_matrix': conf_matrix,
        'f1_score': f1_score,
        'precision': precision,
        'prc': prc,
        'recall': recall,
        'roc': roc,
        'specificity': specificity,
        # Per-class metric objects (for visualization)
        'class_accuracy': class_accuracy,
        'class_auroc': class_auroc,
        'class_average_precision': class_average_precision,
        'class_f1_score': class_f1_score,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_specificity': class_specificity,
        # Data
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred,
        'num_labels': num_labels
    }

    return metrics


def compute_original_binary_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor) -> Dict[str, Any]:
    """
    Calculate various binary classification evaluation metrics using original TorchMetrics, for comparison with wrapped classes
    
    Args:
        y_true: True labels
        y_pred_probs: Prediction probabilities
    
    Returns:
        Dict[str, Any]: Dictionary containing metric objects and calculated values
    """
    # Ensure running on appropriate device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # Get original TorchMetrics binary classification metric classes
    from torchmetrics.classification import (
        BinaryAccuracy as OriginalBinaryAccuracy,
        BinaryAUROC as OriginalBinaryAUROC,
        BinaryAveragePrecision as OriginalBinaryAveragePrecision,
        BinaryF1Score as OriginalBinaryF1Score,
        BinaryPrecision as OriginalBinaryPrecision,
        BinaryRecall as OriginalBinaryRecall,
        BinarySpecificity as OriginalBinarySpecificity
    )

    # Initialize binary classification metric calculators
    accuracy = OriginalBinaryAccuracy().to(device)
    auroc = OriginalBinaryAUROC().to(device)
    average_precision = OriginalBinaryAveragePrecision().to(device)
    f1_score = OriginalBinaryF1Score().to(device)
    precision = OriginalBinaryPrecision().to(device)
    recall = OriginalBinaryRecall().to(device)
    specificity = OriginalBinarySpecificity().to(device)

    # Calculate metric values
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    metrics = {
        # Metric values
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        'num_classes': 2
    }

    return metrics


def compute_original_multiclass_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, num_classes: int) -> Dict[
    str, Any]:
    """
    Calculate various evaluation metrics using original TorchMetrics, for comparison with wrapped classes
    
    Args:
        y_true: True labels
        y_pred_probs: Prediction probabilities
        num_classes: Number of classes
    
    Returns:
        Dict[str, Any]: Dictionary containing metric objects and calculated values
    """
    # Ensure running on appropriate device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # Get original TorchMetrics classes
    from torchmetrics.classification import (
        MulticlassAccuracy as OriginalAccuracy,
        MulticlassAUROC as OriginalAUROC,
        MulticlassAveragePrecision as OriginalAveragePrecision,
        MulticlassF1Score as OriginalF1Score,
        MulticlassPrecision as OriginalPrecision,
        MulticlassRecall as OriginalRecall,
        MulticlassSpecificity as OriginalSpecificity
    )

    # Initialize metric calculators - overall metrics (macro average but for Acc)
    accuracy = OriginalAccuracy(num_classes=num_classes, average='micro').to(device)
    auroc = OriginalAUROC(num_classes=num_classes).to(device)
    average_precision = OriginalAveragePrecision(num_classes=num_classes).to(device)
    f1_score = OriginalF1Score(num_classes=num_classes).to(device)
    precision = OriginalPrecision(num_classes=num_classes).to(device)
    recall = OriginalRecall(num_classes=num_classes).to(device)
    specificity = OriginalSpecificity(num_classes=num_classes).to(device)

    # Initialize per-class metric calculators
    class_auroc = OriginalAUROC(num_classes=num_classes, average=None).to(device)
    class_average_precision = OriginalAveragePrecision(num_classes=num_classes, average=None).to(device)
    class_f1_score = OriginalF1Score(num_classes=num_classes, average=None).to(device)
    class_precision = OriginalPrecision(num_classes=num_classes, average=None).to(device)
    class_recall = OriginalRecall(num_classes=num_classes, average=None).to(device)
    class_specificity = OriginalSpecificity(num_classes=num_classes, average=None).to(device)

    # Calculate overall metrics (scalars)
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # Calculate per-class metrics
    class_auroc_values: npt.NDArray[np.float64] = class_auroc(y_pred_probs, y_true).cpu().numpy()
    class_ap_values: npt.NDArray[np.float64] = class_average_precision(y_pred_probs, y_true).cpu().numpy()
    class_f1_values: npt.NDArray[np.float64] = class_f1_score(y_pred_probs, y_true).cpu().numpy()
    class_precision_values: npt.NDArray[np.float64] = class_precision(y_pred_probs, y_true).cpu().numpy()
    class_recall_values: npt.NDArray[np.float64] = class_recall(y_pred_probs, y_true).cpu().numpy()
    class_specificity_values: npt.NDArray[np.float64] = class_specificity(y_pred_probs, y_true).cpu().numpy()

    metrics = {
        # Overall metric values
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # Per-class metric values
        'class_auroc_values': class_auroc_values,
        'class_ap_values': class_ap_values,
        'class_f1_values': class_f1_values,
        'class_precision_values': class_precision_values,
        'class_recall_values': class_recall_values,
        'class_specificity_values': class_specificity_values,
        'num_classes': num_classes
    }

    return metrics


def compute_original_multilabel_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, num_labels: int) -> Dict[
    str, Any]:
    """
    Calculate various evaluation metrics using original TorchMetrics for multilabel classification, for comparison with wrapped classes
    
    Args:
        y_true: True labels (num_samples, num_labels)
        y_pred_probs: Prediction probabilities (num_samples, num_labels)
        num_labels: Number of labels
    
    Returns:
        Dict[str, Any]: Dictionary containing metric objects and calculated values
    """
    # Ensure running on appropriate device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = y_true.to(device)
    y_pred_probs = y_pred_probs.to(device)

    # Get original TorchMetrics classes
    from torchmetrics.classification import (
        MultilabelAccuracy as OriginalAccuracy,
        MultilabelAUROC as OriginalAUROC,
        MultilabelAveragePrecision as OriginalAveragePrecision,
        MultilabelF1Score as OriginalF1Score,
        MultilabelPrecision as OriginalPrecision,
        MultilabelRecall as OriginalRecall,
        MultilabelSpecificity as OriginalSpecificity
    )

    # Initialize metric calculators - overall metrics (macro average)
    accuracy = OriginalAccuracy(num_labels=num_labels).to(device)
    auroc = OriginalAUROC(num_labels=num_labels).to(device)
    average_precision = OriginalAveragePrecision(num_labels=num_labels).to(device)
    f1_score = OriginalF1Score(num_labels=num_labels).to(device)
    precision = OriginalPrecision(num_labels=num_labels).to(device)
    recall = OriginalRecall(num_labels=num_labels).to(device)
    specificity = OriginalSpecificity(num_labels=num_labels).to(device)

    # Initialize per-class metric calculators
    class_accuracy = OriginalAccuracy(num_labels=num_labels, average=None).to(device)
    class_auroc = OriginalAUROC(num_labels=num_labels, average=None).to(device)
    class_average_precision = OriginalAveragePrecision(num_labels=num_labels, average=None).to(device)
    class_f1_score = OriginalF1Score(num_labels=num_labels, average=None).to(device)
    class_precision = OriginalPrecision(num_labels=num_labels, average=None).to(device)
    class_recall = OriginalRecall(num_labels=num_labels, average=None).to(device)
    class_specificity = OriginalSpecificity(num_labels=num_labels, average=None).to(device)

    # Calculate overall metrics (scalars)
    acc_value: float = accuracy(y_pred_probs, y_true).item()
    auroc_value: float = auroc(y_pred_probs, y_true).item()
    ap_value: float = average_precision(y_pred_probs, y_true).item()
    f1_value: float = f1_score(y_pred_probs, y_true).item()
    precision_value: float = precision(y_pred_probs, y_true).item()
    recall_value: float = recall(y_pred_probs, y_true).item()
    specificity_value: float = specificity(y_pred_probs, y_true).item()

    # Calculate per-class metrics
    class_accuracy_values: npt.NDArray[np.float64] = class_accuracy(y_pred_probs, y_true).cpu().numpy()
    class_auroc_values: npt.NDArray[np.float64] = class_auroc(y_pred_probs, y_true).cpu().numpy()
    class_ap_values: npt.NDArray[np.float64] = class_average_precision(y_pred_probs, y_true).cpu().numpy()
    class_f1_values: npt.NDArray[np.float64] = class_f1_score(y_pred_probs, y_true).cpu().numpy()
    class_precision_values: npt.NDArray[np.float64] = class_precision(y_pred_probs, y_true).cpu().numpy()
    class_recall_values: npt.NDArray[np.float64] = class_recall(y_pred_probs, y_true).cpu().numpy()
    class_specificity_values: npt.NDArray[np.float64] = class_specificity(y_pred_probs, y_true).cpu().numpy()

    metrics = {
        # Overall metric values
        'accuracy_value': acc_value,
        'auroc_value': auroc_value,
        'average_precision_value': ap_value,
        'f1_score_value': f1_value,
        'precision_value': precision_value,
        'recall_value': recall_value,
        'specificity_value': specificity_value,
        # Per-class metric values
        'class_accuracy_values': class_accuracy_values,
        'class_auroc_values': class_auroc_values,
        'class_ap_values': class_ap_values,
        'class_f1_values': class_f1_values,
        'class_precision_values': class_precision_values,
        'class_recall_values': class_recall_values,
        'class_specificity_values': class_specificity_values,
        'num_labels': num_labels
    }

    return metrics


def compare_binary_metrics(wrapped_metrics: Dict[str, Any], original_metrics: Dict[str, Any],
                           tolerance: float = 1e-6) -> bool:
    """
    Compare if binary classification metrics calculated by wrapped classes match those from original classes
    
    Args:
        wrapped_metrics: Metrics calculated using wrapped classes
        original_metrics: Metrics calculated using original classes
        tolerance: Tolerance for float comparison
    
    Returns:
        bool: Whether all metrics match
    """
    print("\n=== Binary Classification Wrapped vs Original Metrics Consistency Check ===")

    # Scalar metrics comparison
    scalar_metrics = ['accuracy_value', 'auroc_value', 'average_precision_value',
                      'f1_score_value', 'precision_value', 'recall_value', 'specificity_value']

    all_consistent = True

    for metric_name in scalar_metrics:
        wrapped_val = wrapped_metrics[metric_name]
        original_val = original_metrics[metric_name]
        is_consistent = abs(wrapped_val - original_val) < tolerance

        print(f"{metric_name}: Wrapped={wrapped_val:.6f}, Original={original_val:.6f}, Consistent: {is_consistent}")

        if not is_consistent:
            all_consistent = False

    if all_consistent:
        print("\n[PASS] All binary classification metrics match!")
    else:
        print("\n[FAIL] Found inconsistent binary classification metrics!")

    print("=" * 60 + "\n")
    return all_consistent


def compare_multiclass_metrics(wrapped_metrics: Dict[str, Any], original_metrics: Dict[str, Any],
                               tolerance: float = 1e-6) -> bool:
    """
    Compare if multiclass metrics calculated by wrapped classes match those from original classes
    
    Args:
        wrapped_metrics: Metrics calculated using wrapped classes
        original_metrics: Metrics calculated using original classes
        tolerance: Tolerance for float comparison
    
    Returns:
        bool: Whether all metrics match
    """
    print("\n=== Wrapped vs Original Metrics Consistency Check ===")

    # Scalar metrics comparison
    scalar_metrics = ['accuracy_value', 'auroc_value', 'average_precision_value',
                      'f1_score_value', 'precision_value', 'recall_value', 'specificity_value']

    all_consistent = True

    for metric_name in scalar_metrics:
        wrapped_val = wrapped_metrics[metric_name]
        original_val = original_metrics[metric_name]
        is_consistent = abs(wrapped_val - original_val) < tolerance

        print(f"{metric_name}: Wrapped={wrapped_val:.6f}, Original={original_val:.6f}, Consistent: {is_consistent}")

        if not is_consistent:
            all_consistent = False

    # Per-class metrics comparison
    class_metrics = ['class_auroc_values', 'class_ap_values', 'class_f1_values', 'class_precision_values',
                     'class_recall_values', 'class_specificity_values']

    for metric_name in class_metrics:
        wrapped_vals = wrapped_metrics[metric_name]
        original_vals = original_metrics[metric_name]

        # Check if array lengths match
        if len(wrapped_vals) != len(original_vals):
            print(f"{metric_name}: Length mismatch! Wrapped={len(wrapped_vals)}, Original={len(original_vals)}")
            all_consistent = False
            continue

        # Element-wise comparison
        is_consistent = np.allclose(wrapped_vals, original_vals, atol=tolerance)

        print(f"{metric_name}: Consistent: {is_consistent}")
        if not is_consistent:
            # Find inconsistent positions
            diff_indices = np.where(np.abs(wrapped_vals - original_vals) >= tolerance)[0]
            for idx in diff_indices:
                print(f"  Class {idx}: Wrapped={wrapped_vals[idx]:.6f}, Original={original_vals[idx]:.6f}")
            all_consistent = False

    if all_consistent:
        print("\n[PASS] All metrics match!")
    else:
        print("\n[FAIL] Found inconsistent metrics!")

    print("=" * 60 + "\n")
    return all_consistent


def compare_multilabel_metrics(wrapped_metrics: Dict[str, Any], original_metrics: Dict[str, Any],
                               tolerance: float = 1e-6) -> bool:
    """
    Compare if multilabel metrics calculated by wrapped classes match those from original classes
    
    Args:
        wrapped_metrics: Metrics calculated using wrapped classes
        original_metrics: Metrics calculated using original classes
        tolerance: Tolerance for float comparison
    
    Returns:
        bool: Whether all metrics match
    """
    print("\n=== Multilabel Wrapped vs Original Metrics Consistency Check ===")

    # Scalar metrics comparison
    scalar_metrics = ['accuracy_value', 'auroc_value', 'average_precision_value',
                      'f1_score_value', 'precision_value', 'recall_value', 'specificity_value']

    all_consistent = True

    for metric_name in scalar_metrics:
        wrapped_val = wrapped_metrics[metric_name]
        original_val = original_metrics[metric_name]
        is_consistent = abs(wrapped_val - original_val) < tolerance

        print(f"{metric_name}: Wrapped={wrapped_val:.6f}, Original={original_val:.6f}, Consistent: {is_consistent}")

        if not is_consistent:
            all_consistent = False

    # Per-class metrics comparison
    class_metrics = ['class_accuracy_values', 'class_auroc_values', 'class_ap_values', 'class_f1_values',
                     'class_precision_values', 'class_recall_values', 'class_specificity_values']

    for metric_name in class_metrics:
        wrapped_vals = wrapped_metrics[metric_name]
        original_vals = original_metrics[metric_name]

        # Check if array lengths match
        if len(wrapped_vals) != len(original_vals):
            print(f"{metric_name}: Length mismatch! Wrapped={len(wrapped_vals)}, Original={len(original_vals)}")
            all_consistent = False
            continue

        # Element-wise comparison
        is_consistent = np.allclose(wrapped_vals, original_vals, atol=tolerance)

        print(f"{metric_name}: Consistent: {is_consistent}")
        if not is_consistent:
            # Find inconsistent positions
            diff_indices = np.where(np.abs(wrapped_vals - original_vals) >= tolerance)[0]
            for idx in diff_indices:
                print(f"  Class {idx}: Wrapped={wrapped_vals[idx]:.6f}, Original={original_vals[idx]:.6f}")
            all_consistent = False

    if all_consistent:
        print("\n[PASS] All multilabel metrics match!")
    else:
        print("\n[FAIL] Found inconsistent multilabel metrics!")

    print("=" * 60 + "\n")
    return all_consistent


def plot_binary_metrics_with_wrapped_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    Visualize binary classification metrics using plot methods from wrapped TorchMetrics
    Utilize enhanced functionality of wrapped classes for more flexible control
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
        save_dir: Directory path to save visualizations
    """

    # Visualize all scalar metrics in a single plot
    def plot_all_scalar_metrics() -> None:
        # Prepare scalar metric data
        metric_names: List[str] = ['Accuracy', 'AUROC', 'Average Precision', 'F1 Score',
                                   'Precision', 'Recall', 'Specificity']
        metric_values: List[float] = [
            metrics['accuracy_value'],
            metrics['auroc_value'],
            metrics['average_precision_value'],
            metrics['f1_score_value'],
            metrics['precision_value'],
            metrics['recall_value'],
            metrics['specificity_value']
        ]

        # Use different colors
        colors = ['skyblue', 'yellow', 'orange', 'purple', 'red', 'brown', 'lightgreen']

        # Create chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylim(0, 1)
        plt.ylabel('Value')
        plt.title('Binary Classification Metrics')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'binary_all_scalar_metrics.png'))
        else:
            plt.show()
        plt.close()

    # Plot all scalar metrics in one chart
    plot_all_scalar_metrics()

    # Use wrapped class plot method to draw confusion matrix
    try:
        conf_matrix: BinaryConfusionMatrix = metrics['conf_matrix']
        fig, ax = conf_matrix.plot(
            title='Binary Confusion Matrix',
            figsize=(10, 8)
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'binary_confusion_matrix.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing confusion matrix: {e}")

    # Use wrapped class plot method to draw ROC curve
    try:
        roc: MetricBinaryROC = metrics['roc']
        fig, ax = roc.plot(
            score=True,
            title='Binary ROC Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': '--', 'alpha': 0.3}
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'binary_roc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing ROC curve: {e}")

    # Use wrapped class plot method to draw PR curve
    try:
        prc: MetricBinaryPrecisionRecallCurve = metrics['prc']
        fig, ax = prc.plot(
            score=True,
            title='Binary Precision-Recall Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': ':', 'alpha': 0.7}
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'binary_prc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing PR curve: {e}")


def plot_multiclass_metrics_with_wrapped_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    Visualize metrics using plot methods from wrapped TorchMetrics, including overall metrics and per-class metrics
    Utilize enhanced functionality of wrapped classes for more flexible control
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
        save_dir: Directory path to save visualizations
    """

    # Visualize all scalar metrics in a single plot
    def plot_all_scalar_metrics() -> None:
        # Prepare scalar metric data
        metric_names: List[str] = ['Accuracy', 'Macro-AUROC', 'Macro-Average Precision', 'Macro-F1 Score',
                                   'Macro-Precision', 'Macro-Recall', 'Macro-Specificity']
        metric_values: List[float] = [
            metrics['accuracy_value'],
            metrics['auroc_value'],
            metrics['average_precision_value'],
            metrics['f1_score_value'],
            metrics['precision_value'],
            metrics['recall_value'],
            metrics['specificity_value']
        ]

        # Use different colors
        colors = ['skyblue', 'yellow', 'orange', 'purple', 'red', 'brown', 'lightgreen']

        # Create chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylim(0, 1)
        plt.ylabel('Value')
        plt.title('Classification Metrics')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'all_scalar_metrics.png'))
        else:
            plt.show()
        plt.close()

    # Plot per-class metric charts using wrapped class enhanced plot functionality
    def plot_classwise_metric(metric_name: str, metric: torchmetrics.Metric, title: str, ylabel: str) -> None:
        """
        Plot per-class chart for a single metric using wrapped class plot method
        Utilize additional control options provided by wrapped classes
        
        Args:
            metric_name: Metric name (for saving file)
            metric: Wrapped metric object
            title: Chart title
            ylabel: Y-axis label
        """
        try:
            # Use wrapped class plot method, utilizing its enhanced functionality
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = metric.plot(
                title=title,
                ylabel=ylabel,
                add_data_labels=True,
                figsize=(10, 6)
            )

            if save_dir:
                fig.savefig(os.path.join(save_dir, f'classwise_{metric_name}.png'))
            else:
                plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"Error drawing {metric_name}: {e}")

    # Plot all scalar metrics in one chart
    plot_all_scalar_metrics()

    # Plot per-class metric charts using wrapped class enhanced functionality
    plot_classwise_metric('auroc', metrics['class_auroc'],
                          'Class-wise AUROC', 'AUROC')
    plot_classwise_metric('average_precision', metrics['class_average_precision'],
                          'Class-wise Average Precision', 'Average Precision')
    plot_classwise_metric('f1_score', metrics['class_f1_score'],
                          'Class-wise F1 Score', 'F1 Score')
    plot_classwise_metric('precision', metrics['class_precision'],
                          'Class-wise Precision', 'Precision')
    plot_classwise_metric('recall', metrics['class_recall'],
                          'Class-wise Recall', 'Recall')
    plot_classwise_metric('specificity', metrics['class_specificity'],
                          'Class-wise Specificity', 'Specificity')

    # Use wrapped class plot method to draw confusion matrix
    try:
        conf_matrix: MulticlassConfusionMatrix = metrics['conf_matrix']
        fig, ax = conf_matrix.plot(
            title='Confusion Matrix',
            figsize=(10, 8)
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing confusion matrix: {e}")

    # Use wrapped class plot method to draw ROC curve, utilizing grid_kwargs parameter
    try:
        roc: MetricMulticlassROC = metrics['roc']
        fig, ax = roc.plot(
            score=True,
            title='ROC Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': '--', 'alpha': 0.3},
            legend_title='Classes'
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'roc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing ROC curve: {e}")

    # Use wrapped class plot method to draw PR curve, utilizing grid_kwargs parameter
    try:
        prc: MetricMulticlassPrecisionRecallCurve = metrics['prc']
        fig, ax = prc.plot(
            score=True,
            title='Precision-Recall Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': ':', 'alpha': 0.7},
            legend_title='Classes'
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'prc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing PR curve: {e}")


def plot_multilabel_metrics_with_wrapped_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    Visualize multilabel metrics using plot methods from wrapped TorchMetrics
    Utilize enhanced functionality of wrapped classes for more flexible control
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
        save_dir: Directory path to save visualizations
    """

    # Visualize all scalar metrics in a single plot
    def plot_all_scalar_metrics() -> None:
        # Prepare scalar metric data
        metric_names: List[str] = ['Accuracy', 'Macro-AUROC', 'Macro-Average Precision', 'Macro-F1 Score',
                                   'Macro-Precision', 'Macro-Recall', 'Macro-Specificity']
        metric_values: List[float] = [
            metrics['accuracy_value'],
            metrics['auroc_value'],
            metrics['average_precision_value'],
            metrics['f1_score_value'],
            metrics['precision_value'],
            metrics['recall_value'],
            metrics['specificity_value']
        ]

        # Use different colors
        colors = ['skyblue', 'yellow', 'orange', 'purple', 'red', 'brown', 'lightgreen']

        # Create chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylim(0, 1)
        plt.ylabel('Value')
        plt.title('Multilabel Classification Metrics')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'multilabel_all_scalar_metrics.png'))
        else:
            plt.show()
        plt.close()

    # Plot class-wise metrics using wrapped class enhanced functionality
    def plot_classwise_metric(metric_name: str, metric_obj: Any, title: str, ylabel: str) -> None:
        try:
            fig, ax = metric_obj.plot(
                title=title,
                ylabel=ylabel,
                add_data_labels=True,
                figsize=(10, 6)
            )

            if save_dir:
                fig.savefig(os.path.join(save_dir, f'multilabel_classwise_{metric_name}.png'))
            else:
                plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"Error drawing {metric_name}: {e}")

    # Plot all scalar metrics in one chart
    plot_all_scalar_metrics()

    # Plot per-class metric charts using wrapped class enhanced functionality
    plot_classwise_metric('accuracy', metrics['class_accuracy'],
                          'Class-wise Accuracy', 'Accuracy')
    plot_classwise_metric('auroc', metrics['class_auroc'],
                          'Class-wise AUROC', 'AUROC')
    plot_classwise_metric('average_precision', metrics['class_average_precision'],
                          'Class-wise Average Precision', 'Average Precision')
    plot_classwise_metric('f1_score', metrics['class_f1_score'],
                          'Class-wise F1 Score', 'F1 Score')
    plot_classwise_metric('precision', metrics['class_precision'],
                          'Class-wise Precision', 'Precision')
    plot_classwise_metric('recall', metrics['class_recall'],
                          'Class-wise Recall', 'Recall')
    plot_classwise_metric('specificity', metrics['class_specificity'],
                          'Class-wise Specificity', 'Specificity')

    # Use wrapped class plot method to draw confusion matrix
    try:
        conf_matrix: MultilabelConfusionMatrix = metrics['conf_matrix']
        fig, ax = conf_matrix.plot(
            title='Multilabel Confusion Matrix',
            figsize=(10, 8)
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'multilabel_confusion_matrix.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing confusion matrix: {e}")

    # Use wrapped class plot method to draw ROC curve, utilizing grid_kwargs parameter
    try:
        roc: MetricMultilabelROC = metrics['roc']
        fig, ax = roc.plot(
            score=True,
            title='Multilabel ROC Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': '--', 'alpha': 0.3},
            legend_title='Classes'
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'multilabel_roc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing ROC curve: {e}")

    # Use wrapped class plot method to draw PR curve, utilizing grid_kwargs parameter
    try:
        prc: _BetterMultilabelPrecisionRecallCurve = metrics['prc']
        fig, ax = prc.plot(
            score=True,
            title='Multilabel Precision-Recall Curve',
            figsize=(10, 8),
            grid_kwargs={'visible': True, 'linestyle': ':', 'alpha': 0.7},
            legend_title='Classes'
        )

        if save_dir:
            fig.savefig(os.path.join(save_dir, 'multilabel_prc.png'))
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error drawing PR curve: {e}")


def visualize_binary_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    Visualize all binary classification metrics using wrapped classes
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
        save_dir: Directory path to save visualizations
    """
    # If save directory is specified, ensure it exists
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use wrapped class plot methods to draw all binary classification metrics
    plot_binary_metrics_with_wrapped_metrics(metrics, save_dir)

    print("All binary classification metrics visualization completed!")
    if save_dir:
        print(f"Visualization results saved to: {save_dir}")


def visualize_multiclass_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    Visualize all metrics using wrapped classes
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
        save_dir: Directory path to save visualizations
    """
    # If save directory is specified, ensure it exists
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use wrapped class plot methods to draw all metrics
    plot_multiclass_metrics_with_wrapped_metrics(metrics, save_dir)

    print("All metrics visualization completed!")
    if save_dir:
        print(f"Visualization results saved to: {save_dir}")


def visualize_multilabel_metrics(metrics: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    """
    Visualize all multilabel classification metrics using wrapped classes
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
        save_dir: Directory path to save visualizations
    """
    # If save directory is specified, ensure it exists
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use wrapped class plot methods to draw all multilabel classification metrics
    plot_multilabel_metrics_with_wrapped_metrics(metrics, save_dir)

    print("All multilabel classification metrics visualization completed!")
    if save_dir:
        print(f"Visualization results saved to: {save_dir}")


def print_binary_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print binary classification metrics summary
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
    """
    print("\n=== Binary Classification Metrics Summary ===")
    print(f"Accuracy: {metrics['accuracy_value']:.4f}")
    print(f"AUROC: {metrics['auroc_value']:.4f}")
    print(f"Average Precision: {metrics['average_precision_value']:.4f}")
    print(f"F1 Score: {metrics['f1_score_value']:.4f}")
    print(f"Precision: {metrics['precision_value']:.4f}")
    print(f"Recall: {metrics['recall_value']:.4f}")
    print(f"Specificity: {metrics['specificity_value']:.4f}")
    print("====================\n")


def print_multiclass_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print multiclass metrics summary, including overall metrics and per-class metrics
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
    """
    print("\n=== Multiclass Metrics Summary ===")
    print(f"Accuracy: {metrics['accuracy_value']:.4f}")
    print(f"AUROC: {metrics['auroc_value']:.4f}")
    print(f"Average Precision: {metrics['average_precision_value']:.4f}")
    print(f"F1 Score: {metrics['f1_score_value']:.4f}")
    print(f"Precision: {metrics['precision_value']:.4f}")
    print(f"Recall: {metrics['recall_value']:.4f}")
    print(f"Specificity: {metrics['specificity_value']:.4f}")

    # Print per-class metrics
    print("\n=== Per-Class Metrics ===")
    num_classes = metrics['num_classes']

    print("\nClass AUROC:")
    for i in range(num_classes):
        print(f"  Class {i}: {metrics['class_auroc_values'][i]:.4f}")

    print("\nClass Average Precision:")
    for i in range(num_classes):
        print(f"  Class {i}: {metrics['class_ap_values'][i]:.4f}")

    print("\nClass F1 Score:")
    for i in range(num_classes):
        print(f"  Class {i}: {metrics['class_f1_values'][i]:.4f}")

    print("\nClass Precision:")
    for i in range(num_classes):
        print(f"  Class {i}: {metrics['class_precision_values'][i]:.4f}")

    print("\nClass Recall:")
    for i in range(num_classes):
        print(f"  Class {i}: {metrics['class_recall_values'][i]:.4f}")

    print("\nClass Specificity:")
    for i in range(num_classes):
        print(f"  Class {i}: {metrics['class_specificity_values'][i]:.4f}")

    print("====================\n")


def print_multilabel_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print multilabel metrics summary, including overall metrics and per-class metrics
    
    Args:
        metrics: Dictionary containing metric objects and calculated values
    """
    print("\n=== Multilabel Metrics Summary ===")
    print(f"Accuracy: {metrics['accuracy_value']:.4f}")
    print(f"AUROC: {metrics['auroc_value']:.4f}")
    print(f"Average Precision: {metrics['average_precision_value']:.4f}")
    print(f"F1 Score: {metrics['f1_score_value']:.4f}")
    print(f"Precision: {metrics['precision_value']:.4f}")
    print(f"Recall: {metrics['recall_value']:.4f}")
    print(f"Specificity: {metrics['specificity_value']:.4f}")

    # Print per-class metrics
    print("\n=== Per-Class Metrics ===")
    num_labels = metrics['num_labels']

    print("\nClass Accuracy:")
    for i in range(num_labels):
        print(f"  Class {i}: {metrics['class_accuracy_values'][i]:.4f}")

    print("\nClass AUROC:")
    for i in range(num_labels):
        print(f"  Class {i}: {metrics['class_auroc_values'][i]:.4f}")

    print("\nClass Average Precision:")
    for i in range(num_labels):
        print(f"  Class {i}: {metrics['class_ap_values'][i]:.4f}")

    print("\nClass F1 Score:")
    for i in range(num_labels):
        print(f"  Class {i}: {metrics['class_f1_values'][i]:.4f}")

    print("\nClass Precision:")
    for i in range(num_labels):
        print(f"  Class {i}: {metrics['class_precision_values'][i]:.4f}")

    print("\nClass Recall:")
    for i in range(num_labels):
        print(f"  Class {i}: {metrics['class_recall_values'][i]:.4f}")

    print("\nClass Specificity:")
    for i in range(num_labels):
        print(f"  Class {i}: {metrics['class_specificity_values'][i]:.4f}")

    print("====================\n")


def main() -> None:
    """
    Main function
    """
    # Parse command line arguments
    args = parse_arguments()

    # Binary test mode
    if args.binary:
        print(f"Generating {args.samples} binary classification samples with error rate {args.error_rate}...")

        # Generate binary classification data
        y_true, y_pred_probs = generate_binary_data(args.samples, args.error_rate)

        print("\nCalculating binary classification metrics using wrapped TorchMetrics...")
        # Calculate binary classification metrics using wrapped classes
        wrapped_metrics = compute_binary_metrics(y_true, y_pred_probs)

        # Print metrics summary
        print_binary_metrics_summary(wrapped_metrics)

        # Compare consistency between wrapped and original classes
        if args.compare:
            print("Calculating binary classification metrics using original TorchMetrics for comparison...")
            original_metrics = compute_original_binary_metrics(y_true, y_pred_probs)
            compare_binary_metrics(wrapped_metrics, original_metrics)

        # Visualize all binary classification metrics
        visualize_binary_metrics(wrapped_metrics, args.save_dir)

    # Multiclass test mode
    if args.multiclass:
        print(f"Generating {args.samples} samples with {args.classes} classes and error rate {args.error_rate}...")

        # Generate multiclass data
        y_true, y_pred_probs = generate_multiclass_data(args.samples, args.classes, args.error_rate)

        print("\nCalculating multiclass metrics using wrapped TorchMetrics...")
        # Calculate metrics using wrapped classes
        wrapped_metrics = compute_multiclass_metrics(y_true, y_pred_probs, args.classes)

        # Print metrics summary
        print_multiclass_metrics_summary(wrapped_metrics)

        # Compare consistency between wrapped and original classes
        if args.compare:
            print("Calculating metrics using original TorchMetrics for comparison...")
            original_metrics = compute_original_multiclass_metrics(y_true, y_pred_probs, args.classes)
            compare_multiclass_metrics(wrapped_metrics, original_metrics)

        # Visualize all metrics
        visualize_multiclass_metrics(wrapped_metrics, args.save_dir)

    # Multilabel test mode
    if args.multilabel:
        print(f"Generating {args.samples} samples with {args.classes} classes and error rate {args.error_rate}...")

        # Generate multilabel classification data
        y_true, y_pred_probs = generate_multilabel_data(args.samples, args.classes, args.error_rate)

        print("\nCalculating multilabel classification metrics using wrapped TorchMetrics...")
        # Calculate multilabel classification metrics using wrapped classes
        wrapped_metrics = compute_multilabel_metrics(y_true, y_pred_probs, args.classes)

        # Print metrics summary
        print_multilabel_metrics_summary(wrapped_metrics)

        # Compare consistency between wrapped and original classes
        if args.compare:
            print("Calculating multilabel classification metrics using original TorchMetrics for comparison...")
            original_metrics = compute_original_multilabel_metrics(y_true, y_pred_probs, args.classes)
            compare_multilabel_metrics(wrapped_metrics, original_metrics)

        # Visualize all multilabel classification metrics
        visualize_multilabel_metrics(wrapped_metrics, args.save_dir)

    print("\nTest completed!")


if __name__ == "__main__":
    main()
