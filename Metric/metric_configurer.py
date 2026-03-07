# -*- coding: utf-8 -*-
"""
Metrics Configurers

This module provides wrapper classes for various metrics from the TorchMetrics and MONAI libraries,
enhancing the plot method to provide additional control functionality (for TorchMetrics).
"""

import numpy as np
import torchmetrics
import torchmetrics.classification
from typing import Optional, Dict, Any, Union, List, Tuple, Literal, cast
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class BinaryStatScores(torchmetrics.classification.BinaryStatScores):
    """
    Wrapper class for binary classification statistical metrics.
    """


class MultiClassStatScores(torchmetrics.classification.MulticlassStatScores):
    """
    Wrapper class for multiclass classification statistical metrics.
    """


class BinaryAccuracy(torchmetrics.classification.BinaryAccuracy):
    """
    Wrapper class for Accuracy metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot Accuracy metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Accuracy: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAccuracy(torchmetrics.classification.MulticlassAccuracy):
    """
    Wrapper class for Accuracy metric, enhanced with additional plot control functionality.
    For multiclass problems, average='micro' should always be set to get the correct Accuracy value.
    When MulticlassAccuracy in TorchMetrics specifies average='none' or 'macro',
    it is equivalent to Recall.
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            top_k: int = 1,
            multidim_average: Literal["global", "samplewise"] = "global",
            ignore_index: Optional[int] = None,
            validate_args: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average='micro',
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot Accuracy metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Accuracy: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryAUROC(torchmetrics.classification.BinaryAUROC):
    """
    Wrapper class for AUROC metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot AUROC metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AUROC: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAUROC(torchmetrics.classification.MulticlassAUROC):
    """
    Wrapper class for AUROC metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot AUROC metric with additional control options.

        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AveragePrecision: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryAveragePrecision(torchmetrics.classification.BinaryAveragePrecision):
    """
    Wrapper class for binary classification AveragePrecision metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Average Precision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Average Precision: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassAveragePrecision(torchmetrics.classification.MulticlassAveragePrecision):
    """
    Wrapper class for AveragePrecision metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot AveragePrecision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to AveragePrecision: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryConfusionMatrix(torchmetrics.classification.BinaryConfusionMatrix):
    """
    Wrapper class for ConfusionMatrix metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot confusion matrix with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class MulticlassConfusionMatrix(torchmetrics.classification.MulticlassConfusionMatrix):
    """
    Wrapper class for ConfusionMatrix metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot confusion matrix with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax


class BinaryF1Score(torchmetrics.classification.BinaryF1Score):
    """
    Wrapper class for F1Score metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot F1Score metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to F1Score: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassF1Score(torchmetrics.classification.MulticlassF1Score):
    """
    Wrapper class for F1Score metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot F1Score metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to F1Score: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryPrecision(torchmetrics.classification.BinaryPrecision):
    """
    Wrapper class for Precision metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot Precision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Precision: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassPrecision(torchmetrics.classification.MulticlassPrecision):
    """
    Wrapper class for Precision metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot Precision metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Precision: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryPrecisionRecallCurve(torchmetrics.classification.BinaryPrecisionRecallCurve):
    """
    Wrapper class for binary classification PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None
             ) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Precision-Recall curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AP score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        fig.tight_layout()
        return fig, ax


class MulticlassPrecisionRecallCurve(torchmetrics.classification.MulticlassPrecisionRecallCurve):
    """
    Wrapper class for PrecisionRecallCurve metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None, legend_title: Optional[str] = "Classes"
             ) -> Tuple[Figure, Axes]:
        """
        Plot Precision-Recall curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AP score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            legend_title: Legend title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # Set legend title
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


class BinaryRecall(torchmetrics.classification.BinaryRecall):
    """
    Wrapper class for binary classification Recall metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Recall metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Recall: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassRecall(torchmetrics.classification.MulticlassRecall):
    """
    Wrapper class for Recall metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot Recall metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Recall: {e}")

        fig.tight_layout()
        return fig, ax


class BinarySpecificity(torchmetrics.classification.BinarySpecificity):
    """
    Wrapper class for binary classification Specificity metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot binary classification Specificity metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Specificity: {e}")

        fig.tight_layout()
        return fig, ax


class MulticlassSpecificity(torchmetrics.classification.MulticlassSpecificity):
    """
    Wrapper class for Specificity metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             ylabel: Optional[str] = None, add_data_labels: bool = True,
             figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Plot Specificity metric with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            ylabel: Y-axis label, uses default if None
            add_data_labels: Whether to add value labels to each data point
            figsize: Chart size, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Add data labels
        if add_data_labels:
            try:
                # Get lines and points for scatter plot
                for line in ax.lines:
                    # Get point coordinate data
                    x_data: np.ndarray = line.get_xdata()
                    y_data: np.ndarray = line.get_ydata()

                    # Add value labels to each point
                    for x, y in zip(x_data, y_data):
                        # Add value label above the point
                        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
            except Exception as e:
                print(f"Error adding data labels to Specificity: {e}")

        fig.tight_layout()
        return fig, ax


class BinaryROC(torchmetrics.classification.BinaryROC):
    """
    Wrapper class for binary classification ROC metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None
             ) -> Tuple[Figure, Axes]:
        """
        Plot binary classification ROC curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AUROC score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        fig.tight_layout()
        return fig, ax


class MulticlassROC(torchmetrics.classification.MulticlassROC):
    """
    Wrapper class for ROC metric, enhanced with additional plot control functionality.
    """

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             score: bool = True, figsize: Optional[Tuple[int, int]] = None,
             grid_kwargs: Optional[Dict[str, Any]] = None, legend_title: Optional[str] = "Classes") -> Tuple[
        Figure, Axes]:
        """
        Plot ROC curve with additional control options.
        
        Args:
            ax: Optional matplotlib axes object, creates new if None
            title: Chart title, uses default if None
            xlabel: X-axis label, uses default if None
            ylabel: Y-axis label, uses default if None
            score: Whether to display AUROC score
            figsize: Chart size, uses default if None
            grid_kwargs: Dictionary of parameters passed to ax.grid function, uses default if None
            legend_title: Legend title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects
        """
        # Call original plot method
        fig: Figure
        ax: Axes
        fig, ax = super().plot(ax=ax, score=score)

        # Set figure size
        if figsize is not None:
            fig.set_size_inches(figsize)

        # Set title
        if title is not None:
            ax.set_title(title)

        # Set x-axis label
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # Set y-axis label
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set grid lines
        if grid_kwargs is None:
            grid_kwargs = {'visible': True, 'linestyle': '--', 'alpha': 0.7}
        ax.grid(**grid_kwargs)

        # Set legend title
        if legend_title is not None and ax.get_legend():
            ax.legend(title=legend_title)

        fig.tight_layout()
        return fig, ax


def get_metric_configurer() -> Dict[str, Dict[str, Any]]:
    """
    Get configuration information for all wrapped metrics, including binary and multiclass versions.
    
    Returns:
        Dict[str, Dict[str, Any]]: Mapping from metric names to metric classes and configurations,
        containing both binary and multiclass versions
    """
    return {
        'accuracy': {
            'binary': BinaryAccuracy,
            'multiclass': MulticlassAccuracy,
            'description': 'Accuracy metric, measures the proportion of correct predictions'
        },
        'auroc': {
            'binary': BinaryAUROC,
            'multiclass': MulticlassAUROC,
            'description': 'AUROC metric, measures the area under the ROC curve'
        },
        'average_precision': {
            'binary': BinaryAveragePrecision,
            'multiclass': MulticlassAveragePrecision,
            'description': 'Average Precision metric, measures the area under the PR curve'
        },
        'confusion_matrix': {
            'binary': BinaryConfusionMatrix,
            'multiclass': MulticlassConfusionMatrix,
            'description': 'Confusion matrix, shows correct and incorrect predictions for each class'
        },
        'f1_score': {
            'binary': BinaryF1Score,
            'multiclass': MulticlassF1Score,
            'description': 'F1 Score metric, harmonic mean of precision and recall'
        },
        'precision': {
            'binary': BinaryPrecision,
            'multiclass': MulticlassPrecision,
            'description': 'Precision metric, measures the proportion of actual positives among predicted positives'
        },
        'precision_recall_curve': {
            'binary': BinaryPrecisionRecallCurve,
            'multiclass': MulticlassPrecisionRecallCurve,
            'description': 'Precision-Recall curve, shows the relationship between precision and recall at different thresholds'
        },
        'recall': {
            'binary': BinaryRecall,
            'multiclass': MulticlassRecall,
            'description': 'Recall metric, measures the proportion of actual positives correctly predicted'
        },
        'specificity': {
            'binary': BinarySpecificity,
            'multiclass': MulticlassSpecificity,
            'description': 'Specificity metric, measures the proportion of actual negatives correctly predicted'
        },
        'roc': {
            'binary': BinaryROC,
            'multiclass': MulticlassROC,
            'description': 'ROC curve, shows the relationship between true positive rate and false positive rate at different thresholds'
        }
    }


def create_metric(metric_name: str, task_type: Literal['binary', 'multiclass'] = 'multiclass',
                  **kwargs) -> torchmetrics.Metric:
    """
    Create a specified wrapped metric instance.
    
    Args:
        metric_name: Metric name
        task_type: Task type, 'binary' or 'multiclass'
        **kwargs: Parameters passed to the metric constructor
        
    Returns:
        torchmetrics.Metric: Metric instance
        
    Raises:
        ValueError: If the metric name does not exist or the task type is invalid
    """
    configurer: Dict[str, Dict[str, Any]] = get_metric_configurer()

    if metric_name not in configurer:
        raise ValueError(f"Unknown metric name: {metric_name}. Available metric names: {list(configurer.keys())}")

    if task_type not in ['binary', 'multiclass']:
        raise ValueError(f"Invalid task type: {task_type}. Available task types: ['binary', 'multiclass']")

    metric_class = configurer[metric_name][task_type]
    return cast(torchmetrics.Metric, metric_class(**kwargs))
