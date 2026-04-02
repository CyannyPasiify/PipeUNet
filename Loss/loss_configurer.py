# -*- coding: utf-8 -*-
"""
Loss Configurers
"""

import torch
import torch.nn as nn
import monai.losses as mL
from typing import Optional, Union, Tuple, List, Dict, Callable, Sequence, Literal

from monai.losses.perceptual import PercetualNetworkType
from monai.metrics.regression import KernelType
from monai.utils import LossReduction, Weight


# region Semantic Segmentation (Point-wise Classification) Loss Category
class FocalLoss(mL.FocalLoss):
    """
    A Wrapped FocalLoss from MONAI (now they are identical).

    FocalLoss is an extension of BCEWithLogitsLoss that down-weights loss from
    high confidence correct predictions.

    Reimplementation of the Focal Loss described in:

        - ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), T. Lin et al., ICCV 2017
        - "AnatomyNet: Deep learning for fast and fully automated whole-volume segmentation of head and neck anatomy",
          Zhu et al., Medical Physics 2018
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            gamma: float = 2.0,
            alpha: Optional[float] = None,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            use_softmax: bool = False
    ):
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the loss calculation.
                If False, `alpha` is invalid when using softmax.
            to_onehot_y: whether to convert the label `y` into the one-hot format. Defaults to False.
            gamma: value of the exponent gamma in the definition of the Focal loss. Defaults to 2.
            alpha: value of the alpha in the definition of the alpha-balanced Focal loss.
                The value should be in [0, 1]. Defaults to None.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            use_softmax: whether to use softmax to transform the original logits into probabilities.
                If True, softmax is used. If False, sigmoid is used. Defaults to False.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            gamma=gamma,
            alpha=alpha,
            weight=weight,
            reduction=reduction,
            use_softmax=use_softmax
        )


class TverskyLoss(mL.TverskyLoss):
    """
    A Wrapped TverskyLoss from MONAI.

    Compute the Tversky loss defined in:

        Sadegh et al. (2017) Tversky loss function for image segmentation
        using 3D fully convolutional deep networks. (https://arxiv.org/abs/1706.05721)

        Wang, Z. et. al. (2023) Dice Semimetric Losses: Optimizing the Dice Score with
        Soft Labels. MICCAI 2023.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L631
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            alpha: float = 0.5,
            beta: float = 0.5,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            soft_label: bool = False
    ):
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            alpha: weight of false positives
            beta: weight of false negatives
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            soft_label: whether the target contains non-binary values (soft labels) or not.
                If True a soft label formulation of the loss will be used.

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``].
                Incompatible values.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            alpha=alpha,
            beta=beta,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            soft_label=soft_label
        )


class ContrastiveLoss(mL.ContrastiveLoss):
    """
    A Wrapped ContrastiveLoss from MONAI.

    Compute the Contrastive loss defined in:

        Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International
        conference on machine learning. PMLR, 2020. (http://proceedings.mlr.press/v119/chen20j.html)

    Adapted from:
        https://github.com/Sara-Ahmed/SiT/blob/1aacd6adcd39b71efc903d16b4e9095b97dda76f/losses.py#L5
    """

    def __init__(
            self,
            temperature: float = 0.5,
            batch_size: int = -1
    ) -> None:
        """
        Args:
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.

        Raises:
            ValueError: When an input of dimension length > 2 is passed
            ValueError: When input and target are of different shapes
        """
        super().__init__(
            temperature=temperature,
            batch_size=batch_size
        )


class DiceLoss(mL.DiceLoss):
    """
    A Wrapped DiceLoss from MONAI.

    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`. And the same axis of `target` can be 1 or N (one-hot format).
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            soft_label: bool = False
    ):
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            soft_label: whether the target contains non-binary values (soft labels) or not.
                If True a soft label formulation of the loss will be used.

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``].
                Incompatible values.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            jaccard=jaccard,
            reduction=reduction,
            batch=batch,
            weight=weight,
            soft_label=soft_label
        )


class MaskedDiceLoss(mL.MaskedDiceLoss):
    """
    A Wrapped MaskedDiceLoss from MONAI.

    Add an additional `masking` process before `DiceLoss`, accept a binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1` will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target` to normal `DiceLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss computation and
    hence gradient calculation.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            soft_label: bool = False
    ):
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            soft_label: whether the target contains non-binary values (soft labels) or not.
                If True a soft label formulation of the loss will be used.

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``].
                Incompatible values.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            jaccard=jaccard,
            reduction=reduction,
            batch=batch,
            weight=weight,
            soft_label=soft_label
        )


class DeepSupervisionDiceLoss(nn.Module):
    """
    A deep supervision compatible DiceLoss from MONAI.

    For each stage:
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`. And the same axis of `target` can be 1 or N (one-hot format).
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            soft_label: bool = False,
            ds_weight_mode: str = 'exp',
            ds_weights: Optional[List[float]] = None
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            soft_label: whether the target contains non-binary values (soft labels) or not.
                If True a soft label formulation of the loss will be used.
            ds_weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level. Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2: 1, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
            ds_weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
                regardless of the weight_mode
        """
        super(DeepSupervisionDiceLoss, self).__init__()
        self._base_loss: mL.DiceLoss = mL.DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            jaccard=jaccard,
            reduction=reduction,
            batch=batch,
            weight=weight,
            soft_label=soft_label
        )
        self.ds_loss: mL.DeepSupervisionLoss = mL.DeepSupervisionLoss(
            loss=self._base_loss,
            weight_mode=ds_weight_mode,
            weights=ds_weights
        )

    def forward(self, input: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD] or list of BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
        """
        return self.ds_loss(input=input, target=target)


class GeneralizedDiceLoss(mL.GeneralizedDiceLoss):
    """
    A Wrapped GeneralizedDiceLoss from MONAI.

    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            w_type: Union[Weight, str] = Weight.SQUARE,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            soft_label: bool = False
    ):
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, intersection over union is computed from each item in the batch.
                If True, the class-weighted intersection and union areas are first summed across the batches.
            soft_label: whether the target contains non-binary values (soft labels) or not.
                If True a soft label formulation of the loss will be used.

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``].
                Incompatible values.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            w_type=w_type,
            reduction=reduction,
            batch=batch,
            soft_label=soft_label
        )


class DeepSupervisionGeneralizedDiceLoss(nn.Module):
    """
    A deep supervision compatible GeneralizedDiceLoss from MONAI.

    For each stage:
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            w_type: Union[Weight, str] = Weight.SQUARE,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            soft_label: bool = False,
            ds_weight_mode: str = 'exp',
            ds_weights: Optional[List[float]] = None
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, intersection over union is computed from each item in the batch.
                If True, the class-weighted intersection and union areas are first summed across the batches.
            soft_label: whether the target contains non-binary values (soft labels) or not.
                If True a soft label formulation of the loss will be used.
            ds_weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level. Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2: 1, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
            ds_weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
                regardless of the weight_mode

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``].
                Incompatible values.
        """
        super(DeepSupervisionGeneralizedDiceLoss, self).__init__()
        self._base_loss: mL.GeneralizedDiceLoss = mL.GeneralizedDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            w_type=w_type,
            reduction=reduction,
            batch=batch,
            soft_label=soft_label
        )
        self.ds_loss: mL.DeepSupervisionLoss = mL.DeepSupervisionLoss(
            loss=self._base_loss,
            weight_mode=ds_weight_mode,
            weights=ds_weights
        )

    def forward(self, input: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD] or list of BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
        """
        return self.ds_loss(input=input, target=target)


class DiceCELoss(mL.DiceCELoss):
    """
    A Wrapped DiceCELoss from MONAI.

    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss`` and ``torch.nn.BCEWithLogitsLoss()``.
    In this implementation, two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            jaccard: bool = False,
            reduction: Literal['mean', 'sum'] = "mean",
            batch: bool = False,
            weight: Optional[torch.Tensor] = None,
            lambda_dice: float = 1.0,
            lambda_ce: float = 1.0,
            label_smoothing: float = 0.0
    ):
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: a rescaling weight given to each class for cross entropy loss for `CrossEntropyLoss`.
                or a weight of positive examples to be broadcasted with target used as `pos_weight` for `BCEWithLogitsLoss`.
                See ``torch.nn.CrossEntropyLoss()`` or ``torch.nn.BCEWithLogitsLoss()`` for more information.
                The weight is also used in `DiceLoss`.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.
            label_smoothing: a value in [0, 1] range. If > 0, the labels are smoothed
                by the given factor to reduce overfitting.
                Defaults to 0.0.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            jaccard=jaccard,
            reduction=reduction,
            batch=batch,
            weight=weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            label_smoothing=label_smoothing
        )


class DeepSupervisionDiceCELoss(nn.Module):
    """
    A deep supervision compatible DiceCELoss from MONAI.

    For each stage:
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss`` and ``torch.nn.BCEWithLogitsLoss()``.
    In this implementation, two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            jaccard: bool = False,
            reduction: Literal['mean', 'sum'] = "mean",
            batch: bool = False,
            weight: Optional[torch.Tensor] = None,
            lambda_dice: float = 1.0,
            lambda_ce: float = 1.0,
            label_smoothing: float = 0.0,
            ds_weight_mode: str = 'exp',
            ds_weights: Optional[List[float]] = None
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: a rescaling weight given to each class for cross entropy loss for `CrossEntropyLoss`.
                or a weight of positive examples to be broadcasted with target used as `pos_weight` for `BCEWithLogitsLoss`.
                See ``torch.nn.CrossEntropyLoss()`` or ``torch.nn.BCEWithLogitsLoss()`` for more information.
                The weight is also used in `DiceLoss`.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.
            label_smoothing: a value in [0, 1] range. If > 0, the labels are smoothed
                by the given factor to reduce overfitting.
                Defaults to 0.0.
            ds_weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level. Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2: 1, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
            ds_weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
                regardless of the weight_mode
        """
        super(DeepSupervisionDiceCELoss, self).__init__()
        self._base_loss: mL.DiceCELoss = mL.DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            jaccard=jaccard,
            reduction=reduction,
            batch=batch,
            weight=weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            label_smoothing=label_smoothing
        )
        self.ds_loss: mL.DeepSupervisionLoss = mL.DeepSupervisionLoss(
            loss=self._base_loss,
            weight_mode=ds_weight_mode,
            weights=ds_weights
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
        """
        return self.ds_loss(input=input, target=target)


class DiceFocalLoss(mL.DiceFocalLoss):
    """
    A Wrapped DiceFocalLoss from MONAI.

    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background``, ``weight``, ``reduction``, and ``alpha`` are used for both losses,
    and other parameters are only used for dice loss.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            jaccard: bool = False,
            reduction: Literal['mean', 'sum'] = "mean",
            batch: bool = False,
            gamma: float = 2.0,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            lambda_dice: float = 1.0,
            lambda_focal: float = 1.0,
            alpha: Optional[float] = None
    ):
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.
            alpha: value of the alpha in the definition of the alpha-balanced Focal loss. The value should be in
                [0, 1]. Defaults to None.

        Raises:
            ValueError: if either `lambda_dice` or `lambda_focal` is less than 0.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            jaccard=jaccard,
            reduction=reduction,
            batch=batch,
            gamma=gamma,
            weight=weight,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            alpha=alpha
        )


class DeepSupervisionDiceFocalLoss(nn.Module):
    """
    A deep supervision compatible DiceFocalLoss from MONAI.

    For each stage:
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background``, ``weight``, ``reduction``, and ``alpha`` are used for both losses,
    and other parameters are only used for dice loss.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            jaccard: bool = False,
            reduction: Literal['mean', 'sum'] = "mean",
            batch: bool = False,
            gamma: float = 2.0,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            lambda_dice: float = 1.0,
            lambda_focal: float = 1.0,
            alpha: Optional[float] = None,
            ds_weight_mode: str = 'exp',
            ds_weights: Optional[List[float]] = None
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.
            alpha: value of the alpha in the definition of the alpha-balanced Focal loss. The value should be in
                [0, 1]. Defaults to None.
            ds_weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level. Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2: 1, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
            ds_weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
                regardless of the weight_mode

        Raises:
            ValueError: if either `lambda_dice` or `lambda_focal` is less than 0.
        """
        super(DeepSupervisionDiceFocalLoss, self).__init__()
        self._base_loss: mL.DiceFocalLoss = mL.DiceFocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            jaccard=jaccard,
            reduction=reduction,
            batch=batch,
            gamma=gamma,
            weight=weight,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            alpha=alpha
        )
        self.ds_loss: mL.DeepSupervisionLoss = mL.DeepSupervisionLoss(
            loss=self._base_loss,
            weight_mode=ds_weight_mode,
            weights=ds_weights
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
        """
        return self.ds_loss(input=input, target=target)


class GeneralizedDiceFocalLoss(mL.GeneralizedDiceFocalLoss):
    """
    A Wrapped GeneralizedDiceFocalLoss from MONAI.

    Compute both Generalized Dice Loss and Focal Loss, and return their weighted average. The details of Generalized Dice Loss
    and Focal Loss are available at ``monai.losses.GeneralizedDiceLoss`` and ``monai.losses.FocalLoss``.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            w_type: Union[Weight] = Weight.SQUARE,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            gamma: float = 2.0,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            lambda_gdl: float = 1.0,
            lambda_focal: float = 1.0
    ):
        """
        Args:
            include_background (bool, optional): if False channel index 0 (background category) is excluded from the calculation.
                Defaults to True.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid (bool, optional): if True, apply a sigmoid function to the prediction. Defaults to False.
            softmax (bool, optional): if True, apply a softmax function to the prediction. Defaults to False.
            w_type (Union[Weight, str], optional): {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to transform
                ground-truth volume to a weight factor. Defaults to ``"square"``.
            reduction (Union[LossReduction, str], optional): {``"none"``, ``"mean"``, ``"sum"``}. Specified the reduction to
                apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch (bool, optional): whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, i.e., the areas are computed for each item in the batch.
            gamma (float, optional): value of the exponent gamma in the definition of the Focal loss. Defaults to 2.0.
            weight (Optional[Union[Sequence[float], float, int, torch.Tensor]], optional): weights to apply to
                the voxels of each class. If None no weights are applied. The input can be a single value
                (same weight for all classes), a sequence of values (the length of the sequence hould be the same as
                the number of classes). Defaults to None.
            lambda_gdl (float, optional): the trade-off weight value for Generalized Dice Loss. The value should be
                no less than 0.0. Defaults to 1.0.
            lambda_focal (float, optional): the trade-off weight value for Focal Loss. The value should be no less
                than 0.0. Defaults to 1.0.

        Raises:
            ValueError: if either `lambda_gdl` or `lambda_focal` is less than 0.
        """
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            w_type=w_type,
            reduction=reduction,
            batch=batch,
            gamma=gamma,
            weight=weight,
            lambda_gdl=lambda_gdl,
            lambda_focal=lambda_focal
        )


class DeepSupervisionGeneralizedDiceFocalLoss(nn.Module):
    """
    A deep supervision compatible GeneralizedDiceFocalLoss from MONAI.

    For each stage:
    Compute both Generalized Dice Loss and Focal Loss, and return their weighted average. The details of Generalized Dice Loss
    and Focal Loss are available at ``monai.losses.GeneralizedDiceLoss`` and ``monai.losses.FocalLoss``.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            w_type: Union[Weight] = Weight.SQUARE,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            gamma: float = 2.0,
            weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
            lambda_gdl: float = 1.0,
            lambda_focal: float = 1.0,
            ds_weight_mode: str = 'exp',
            ds_weights: Optional[List[float]] = None
    ) -> None:
        """
        Args:
            include_background (bool, optional): if False channel index 0 (background category) is excluded from the calculation.
                Defaults to True.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid (bool, optional): if True, apply a sigmoid function to the prediction. Defaults to False.
            softmax (bool, optional): if True, apply a softmax function to the prediction. Defaults to False.
            w_type (Union[Weight, str], optional): {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to transform
                ground-truth volume to a weight factor. Defaults to ``"square"``.
            reduction (Union[LossReduction, str], optional): {``"none"``, ``"mean"``, ``"sum"``}. Specified the reduction to
                apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch (bool, optional): whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, i.e., the areas are computed for each item in the batch.
            gamma (float, optional): value of the exponent gamma in the definition of the Focal loss. Defaults to 2.0.
            weight (Optional[Union[Sequence[float], float, int, torch.Tensor]], optional): weights to apply to
                the voxels of each class. If None no weights are applied. The input can be a single value
                (same weight for all classes), a sequence of values (the length of the sequence hould be the same as
                the number of classes). Defaults to None.
            lambda_gdl (float, optional): the trade-off weight value for Generalized Dice Loss. The value should be
                no less than 0.0. Defaults to 1.0.
            lambda_focal (float, optional): the trade-off weight value for Focal Loss. The value should be no less
                than 0.0. Defaults to 1.0.
            ds_weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level. Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2: 1, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
            ds_weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
                regardless of the weight_mode

        Raises:
            ValueError: if either `lambda_gdl` or `lambda_focal` is less than 0.
        """
        super(DeepSupervisionGeneralizedDiceFocalLoss, self).__init__()
        self._base_loss: mL.GeneralizedDiceFocalLoss = mL.GeneralizedDiceFocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            w_type=w_type,
            reduction=reduction,
            batch=batch,
            gamma=gamma,
            weight=weight,
            lambda_gdl=lambda_gdl,
            lambda_focal=lambda_focal
        )
        self.ds_loss: mL.DeepSupervisionLoss = mL.DeepSupervisionLoss(
            loss=self._base_loss,
            weight_mode=ds_weight_mode,
            weights=ds_weights
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
        """
        return self.ds_loss(input=input, target=target)


class HausdorffDTLoss(mL.HausdorffDTLoss):
    """
    A Wrapped HausdorffDTLoss from MONAI.

    Compute channel-wise binary Hausdorff loss based on distance transform. It can support both multi-classes and
    multi-labels tasks. The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target`
    (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The original paper: Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with
    Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
    """

    def __init__(
            self,
            alpha: float = 2.0,
            include_background: bool = False,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
    ):
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``].
                Incompatible values.

        """
        super().__init__(
            alpha=alpha,
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            reduction=reduction,
            batch=batch
        )


# endregion

# region Reconstruction Loss Category
class SSIMLoss(mL.SSIMLoss):
    """
    A Wrapped SSIMLoss from MONAI (now they are identical).

    Compute the loss function based on the Structural Similarity Index Measure (SSIM) Metric.

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
    """

    def __init__(
            self,
            spatial_dims: int,
            data_range: float = 1.0,
            kernel_type: Union[KernelType, str] = KernelType.GAUSSIAN,
            win_size: Union[int, Sequence[int]] = 11,
            kernel_sigma: Union[float, Sequence[float]] = 1.5,
            k1: float = 0.01,
            k2: float = 0.03,
            reduction: Union[LossReduction, str] = LossReduction.MEAN
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions of the input images.
            data_range: value range of input images. (usually 1.0 or 255)
            kernel_type: type of kernel, can be "gaussian" or "uniform".
            win_size: window size of kernel
            kernel_sigma: standard deviation for Gaussian kernel.
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            win_size=win_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            reduction=reduction
        )


class PerceptualLoss(mL.PerceptualLoss):
    """
    A Wrapped PerceptualLoss from MONAI (now they are identical).

    Perceptual loss using features from pretrained deep neural networks trained. The function supports networks
    pretrained on: ImageNet that use the LPIPS approach from Zhang, et al. "The unreasonable effectiveness of deep
    features as a perceptual metric." https://arxiv.org/abs/1801.03924 ; RadImagenet from Mei, et al. "RadImageNet: An
    Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"
    https://pubs.rsna.org/doi/full/10.1148/ryai.210315 ; MedicalNet from Chen et al. "Med3D: Transfer Learning for
    3D Medical Image Analysis" https://arxiv.org/abs/1904.00625 ;
    and ResNet50 from Torchvision: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html .

    The fake 3D implementation is based on a 2.5D approach where we calculate the 2D perceptual loss on slices from all
    three axes and average. The full 3D approach uses a 3D network to calculate the perceptual loss.
    MedicalNet networks are only compatible with 3D inputs and support channel-wise loss.
    """

    def __init__(
            self,
            spatial_dims: int,
            network_type: str = PercetualNetworkType.alex,
            is_fake_3d: bool = True,
            fake_3d_ratio: float = 0.5,
            cache_dir: Optional[str] = None,
            pretrained: bool = True,
            pretrained_path: Optional[str] = None,
            pretrained_state_dict_key: Optional[str] = None,
            channel_wise: bool = False
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            network_type: {``"alex"``, ``"vgg"``, ``"squeeze"``, ``"radimagenet_resnet50"``,
            ``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``, ``"resnet50"``}
                Specifies the network architecture to use. Defaults to ``"alex"``.
            is_fake_3d: if True use 2.5D approach for a 3D perceptual loss.
            fake_3d_ratio: ratio of how many slices per axis are used in the 2.5D approach.
            cache_dir: path to cache directory to save the pretrained network weights.
            pretrained: whether to load pretrained weights. This argument only works when using networks from
                LIPIS or Torchvision. Defaults to ``"True"``.
            pretrained_path: if `pretrained` is `True`, users can specify a weights file to be loaded
                via using this argument. This argument only works when ``"network_type"`` is "resnet50".
                Defaults to `None`.
            pretrained_state_dict_key: if `pretrained_path` is not `None`, this argument is used to
                extract the expected state dict. This argument only works when ``"network_type"`` is "resnet50".
                Defaults to `None`.
            channel_wise: if True, the loss is returned per channel. Otherwise the loss is averaged over the channels.
                    Defaults to ``False``.
    """
        super().__init__(
            spatial_dims=spatial_dims,
            network_type=network_type,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
            cache_dir=cache_dir,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            pretrained_state_dict_key=pretrained_state_dict_key,
            channel_wise=channel_wise
        )


# endregion


# region Test Functions
class _LossFunctionTester:
    """
    Loss Function Tester Class
    
    Provides test functions to verify input/output specifications of all loss functions.
    Each test function validates the shape and type of inputs and outputs.
    """

    @staticmethod
    def _generate_test_tensors(
            batch_size: int = 2,
            num_classes: int = 3,
            spatial_shape: Tuple[int, ...] = (16, 16, 16),
            device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate test input and target tensors
        
        Args:
            batch_size: Batch size
            num_classes: Number of classes
            spatial_shape: Spatial dimensions (H, W) or (D, H, W)
            device: Device to place tensors on
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        input_shape = (batch_size, num_classes, *spatial_shape)
        target_shape = (batch_size, num_classes, *spatial_shape)

        # Generate random logits/probabilities
        input_tensor = torch.randn(input_shape, device=device)
        # Generate random one-hot or class indices
        target_tensor = torch.randint(0, num_classes, (batch_size, 1, *spatial_shape), device=device).float()

        return input_tensor, target_tensor

    @staticmethod
    def _generate_deep_supervision_tensors(
            batch_size: int = 2,
            num_classes: int = 3,
            spatial_shape: Tuple[int, ...] = (16, 16, 16),
            num_levels: int = 3,
            device: str = 'cpu'
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Generate test tensors for deep supervision losses
        
        Args:
            batch_size: Batch size
            num_classes: Number of classes
            spatial_shape: Spatial dimensions at full resolution
            num_levels: Number of deep supervision levels
            device: Device to place tensors on
            
        Returns:
            Tuple of (input_tensors_list, target_tensor)
        """
        input_tensors = []
        current_shape = list(spatial_shape)

        for level in range(num_levels):
            input_shape = (batch_size, num_classes, *current_shape)
            input_tensor = torch.randn(input_shape, device=device)
            input_tensors.append(input_tensor)
            # Downsample for next level
            current_shape = [s // 2 for s in current_shape]

        # Target at full resolution
        target_shape = (batch_size, num_classes, *spatial_shape)
        target_tensor = torch.randint(0, num_classes, (batch_size, 1, *spatial_shape), device=device).float()

        return input_tensors, target_tensor

    @staticmethod
    def _test_basic_loss(
            loss_fn: nn.Module,
            loss_name: str,
            batch_size: int = 2,
            num_classes: int = 3,
            spatial_shape: Tuple[int, ...] = (16, 16, 16),
            device: str = 'cpu'
    ) -> bool:
        """
        Test basic loss function (input: tensor, target: tensor -> output: scalar tensor)
        
        Args:
            loss_fn: Loss function instance
            loss_name: Name of the loss function
            batch_size: Batch size for test
            num_classes: Number of classes
            spatial_shape: Spatial dimensions
            device: Device to run test on
            
        Returns:
            True if test passes, False otherwise
        """
        try:
            input_tensor, target_tensor = _LossFunctionTester._generate_test_tensors(
                batch_size, num_classes, spatial_shape, device
            )

            # Forward pass
            loss_value = loss_fn(input_tensor, target_tensor)

            # Validate output
            assert isinstance(loss_value, torch.Tensor), f"{loss_name}: Output should be a torch.Tensor"
            assert loss_value.dim() == 0, f"{loss_name}: Output should be a scalar, got shape {loss_value.shape}"
            assert not torch.isnan(loss_value), f"{loss_name}: Output is NaN"
            assert not torch.isinf(loss_value), f"{loss_name}: Output is Inf"

            print(
                f"✓ {loss_name}: PASSED (input: {input_tensor.shape}, target: {target_tensor.shape}, output: {loss_value.item():.6f})")
            return True

        except Exception as e:
            print(f"✗ {loss_name}: FAILED - {str(e)}")
            return False

    @staticmethod
    def _test_deep_supervision_loss(
            loss_fn: nn.Module,
            loss_name: str,
            batch_size: int = 2,
            num_classes: int = 3,
            spatial_shape: Tuple[int, ...] = (16, 16, 16),
            num_levels: int = 3,
            device: str = 'cpu'
    ) -> bool:
        """
        Test deep supervision loss function (input: list of tensors, target: tensor -> output: scalar tensor)
        
        Args:
            loss_fn: Loss function instance
            loss_name: Name of the loss function
            batch_size: Batch size for test
            num_classes: Number of classes
            spatial_shape: Spatial dimensions at full resolution
            num_levels: Number of deep supervision levels
            device: Device to run test on
            
        Returns:
            True if test passes, False otherwise
        """
        try:
            input_tensors, target_tensor = _LossFunctionTester._generate_deep_supervision_tensors(
                batch_size, num_classes, spatial_shape, num_levels, device
            )

            # Forward pass
            loss_value = loss_fn(input_tensors, target_tensor)

            # Validate output
            assert isinstance(loss_value, torch.Tensor), f"{loss_name}: Output should be a torch.Tensor"
            assert loss_value.dim() == 0, f"{loss_name}: Output should be a scalar, got shape {loss_value.shape}"
            assert not torch.isnan(loss_value), f"{loss_name}: Output is NaN"
            assert not torch.isinf(loss_value), f"{loss_name}: Output is Inf"

            input_shapes = [t.shape for t in input_tensors]
            print(
                f"✓ {loss_name}: PASSED (inputs: {input_shapes}, target: {target_tensor.shape}, output: {loss_value.item():.6f})")
            return True

        except Exception as e:
            print(f"✗ {loss_name}: FAILED - {str(e)}")
            return False

    @staticmethod
    def _test_ssim_loss(device: str = 'cpu') -> bool:
        """Test SSIMLoss"""
        try:
            loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0)

            # SSIM expects input and target with same shape (B, C, H, W, D)
            batch_size, channels = 2, 1
            spatial_shape = (16, 16, 16)

            input_tensor = torch.rand(batch_size, channels, *spatial_shape, device=device)
            target_tensor = torch.rand(batch_size, channels, *spatial_shape, device=device)

            loss_value = loss_fn(input_tensor, target_tensor)

            assert isinstance(loss_value, torch.Tensor), "SSIMLoss: Output should be a torch.Tensor"
            assert loss_value.dim() == 0, f"SSIMLoss: Output should be a scalar, got shape {loss_value.shape}"
            assert not torch.isnan(loss_value), "SSIMLoss: Output is NaN"

            print(
                f"✓ SSIMLoss: PASSED (input: {input_tensor.shape}, target: {target_tensor.shape}, output: {loss_value.item():.6f})")
            return True

        except Exception as e:
            print(f"✗ SSIMLoss: FAILED - {str(e)}")
            return False

    @staticmethod
    def _test_perceptual_loss(device: str = 'cpu') -> bool:
        """Test PerceptualLoss"""
        try:
            # Perceptual loss typically expects 1 or 3 channel images
            loss_fn = PerceptualLoss(spatial_dims=3, network_type='resnet50', is_fake_3d=True, pretrained=False)

            batch_size, channels = 2, 1
            spatial_shape = (32, 32, 32)  # Larger size for perceptual loss

            input_tensor = torch.rand(batch_size, channels, *spatial_shape, device=device)
            target_tensor = torch.rand(batch_size, channels, *spatial_shape, device=device)

            loss_value = loss_fn(input_tensor, target_tensor)

            assert isinstance(loss_value, torch.Tensor), "PerceptualLoss: Output should be a torch.Tensor"
            assert loss_value.dim() == 0, f"PerceptualLoss: Output should be a scalar, got shape {loss_value.shape}"

            print(
                f"✓ PerceptualLoss: PASSED (input: {input_tensor.shape}, target: {target_tensor.shape}, output: {loss_value.item():.6f})")
            return True

        except Exception as e:
            print(f"✗ PerceptualLoss: FAILED - {str(e)}")
            return False

    @staticmethod
    def _test_contrastive_loss(device: str = 'cpu') -> bool:
        """Test ContrastiveLoss"""
        try:
            loss_fn = ContrastiveLoss(temperature=0.5)

            # Contrastive loss expects (batch_size, feature_dim) inputs
            batch_size, feature_dim = 8, 128

            input_tensor = torch.randn(batch_size, feature_dim, device=device)
            target_tensor = torch.randn(batch_size, feature_dim, device=device)

            loss_value = loss_fn(input_tensor, target_tensor)

            assert isinstance(loss_value, torch.Tensor), "ContrastiveLoss: Output should be a torch.Tensor"
            assert loss_value.dim() == 0, f"ContrastiveLoss: Output should be a scalar, got shape {loss_value.shape}"
            assert not torch.isnan(loss_value), "ContrastiveLoss: Output is NaN"

            print(
                f"✓ ContrastiveLoss: PASSED (input: {input_tensor.shape}, target: {target_tensor.shape}, output: {loss_value.item():.6f})")
            return True

        except Exception as e:
            print(f"✗ ContrastiveLoss: FAILED - {str(e)}")
            return False

    @staticmethod
    def run_all_tests(device: str = 'cpu') -> Dict[str, bool]:
        """
        Run all loss function tests
        
        Args:
            device: Device to run tests on ('cpu' or 'cuda')
            
        Returns:
            Dictionary mapping loss names to test results
        """
        print("=" * 80)
        print("Loss Function Specification Tests")
        print("=" * 80)
        print(f"Device: {device}")
        print("-" * 80)

        results = {}

        # Test semantic segmentation losses
        print("\n[Semantic Segmentation Losses]")
        print("-" * 40)

        # FocalLoss
        results['FocalLoss'] = _LossFunctionTester._test_basic_loss(
            FocalLoss(use_softmax=True, to_onehot_y=True),
            'FocalLoss',
            device=device
        )

        # TverskyLoss
        results['TverskyLoss'] = _LossFunctionTester._test_basic_loss(
            TverskyLoss(softmax=True, to_onehot_y=True),
            'TverskyLoss',
            device=device
        )

        # ContrastiveLoss (special case - 2D features)
        results['ContrastiveLoss'] = _LossFunctionTester._test_contrastive_loss(device)

        # DiceLoss
        results['DiceLoss'] = _LossFunctionTester._test_basic_loss(
            DiceLoss(softmax=True, to_onehot_y=True),
            'DiceLoss',
            device=device
        )

        # MaskedDiceLoss
        results['MaskedDiceLoss'] = _LossFunctionTester._test_basic_loss(
            MaskedDiceLoss(softmax=True, to_onehot_y=True),
            'MaskedDiceLoss',
            device=device
        )

        # DeepSupervisionDiceLoss
        results['DeepSupervisionDiceLoss'] = _LossFunctionTester._test_deep_supervision_loss(
            DeepSupervisionDiceLoss(softmax=True, to_onehot_y=True),
            'DeepSupervisionDiceLoss',
            device=device
        )

        # GeneralizedDiceLoss
        results['GeneralizedDiceLoss'] = _LossFunctionTester._test_basic_loss(
            GeneralizedDiceLoss(softmax=True, to_onehot_y=True),
            'GeneralizedDiceLoss',
            device=device
        )

        # DeepSupervisionGeneralizedDiceLoss
        results['DeepSupervisionGeneralizedDiceLoss'] = _LossFunctionTester._test_deep_supervision_loss(
            DeepSupervisionGeneralizedDiceLoss(softmax=True, to_onehot_y=True),
            'DeepSupervisionGeneralizedDiceLoss',
            device=device
        )

        # DiceCELoss
        results['DiceCELoss'] = _LossFunctionTester._test_basic_loss(
            DiceCELoss(softmax=True, to_onehot_y=True),
            'DiceCELoss',
            device=device
        )

        # DeepSupervisionDiceCELoss
        results['DeepSupervisionDiceCELoss'] = _LossFunctionTester._test_deep_supervision_loss(
            DeepSupervisionDiceCELoss(softmax=True, to_onehot_y=True),
            'DeepSupervisionDiceCELoss',
            device=device
        )

        # DiceFocalLoss
        results['DiceFocalLoss'] = _LossFunctionTester._test_basic_loss(
            DiceFocalLoss(softmax=True, to_onehot_y=True),
            'DiceFocalLoss',
            device=device
        )

        # DeepSupervisionDiceFocalLoss
        results['DeepSupervisionDiceFocalLoss'] = _LossFunctionTester._test_deep_supervision_loss(
            DeepSupervisionDiceFocalLoss(softmax=True, to_onehot_y=True),
            'DeepSupervisionDiceFocalLoss',
            device=device
        )

        # GeneralizedDiceFocalLoss
        results['GeneralizedDiceFocalLoss'] = _LossFunctionTester._test_basic_loss(
            GeneralizedDiceFocalLoss(softmax=True, to_onehot_y=True),
            'GeneralizedDiceFocalLoss',
            device=device
        )

        # DeepSupervisionGeneralizedDiceFocalLoss
        results['DeepSupervisionGeneralizedDiceFocalLoss'] = _LossFunctionTester._test_deep_supervision_loss(
            DeepSupervisionGeneralizedDiceFocalLoss(softmax=True, to_onehot_y=True),
            'DeepSupervisionGeneralizedDiceFocalLoss',
            device=device
        )

        # HausdorffDTLoss
        results['HausdorffDTLoss'] = _LossFunctionTester._test_basic_loss(
            HausdorffDTLoss(softmax=True, to_onehot_y=True),
            'HausdorffDTLoss',
            device=device
        )

        # Test reconstruction losses
        print("\n[Reconstruction Losses]")
        print("-" * 40)

        # SSIMLoss
        results['SSIMLoss'] = _LossFunctionTester._test_ssim_loss(device)

        # PerceptualLoss
        results['PerceptualLoss'] = _LossFunctionTester._test_perceptual_loss(device)

        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        passed = sum(results.values())
        total = len(results)
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")

        if passed < total:
            print("\nFailed Tests:")
            for name, result in results.items():
                if not result:
                    print(f"  - {name}")

        print("=" * 80)

        return results


def _test_loss_functions():
    """
    Main test function to verify all loss functions
    
    This function tests all loss functions in the module to ensure:
    1. Input tensors have correct shape (B, C, H, W, D) or (B, C, H, W)
    2. Target tensors have correct shape (B, C, H, W, D) or (B, 1, H, W, D)
    3. Output is a scalar tensor (when reduction is 'mean' or 'sum')
    4. No NaN or Inf values in output
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run all tests
    results = _LossFunctionTester.run_all_tests(device=device)

    # Return overall success
    return all(results.values())


# endregion

if __name__ == "__main__":
    success = _test_loss_functions()
    exit(0 if success else 1)
