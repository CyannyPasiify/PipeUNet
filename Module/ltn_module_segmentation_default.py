# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor, optim
from typing import TypeVar, Optional, Dict, Any, Union, List, Literal, Type, Set, Collection, Tuple, Callable
import lightning as L
from dataclasses import dataclass
from monai.inferers import Inferer, SimpleInferer
from Inferer.inferer_configurer import ConfigInfererSlidingWindow, ConfigInfererMainWithAuxSlidingWindow, ConfigInfererBase, ConfigInfererSimple
from monai.utils import MetricReduction, BlendMode, PytorchPadMode
from torchmetrics import Metric

T = TypeVar("T")
TLSeq = Union[List[T], Tuple[T, ...]]

from Network.network_configurer import ConfigNetworkUNet, ConfigNetworkBase, ConfigNetworkUNet
from Loss.loss_configurer import (
    ConfigLossDice, ConfigLossDeepSupervisionDice,
    ConfigLossDiceCE, ConfigLossDeepSupervisionDiceCE,
    ConfigLossDiceFocal, ConfigLossDeepSupervisionDiceFocal,
    ConfigLossHausdorffDT, ConfigLossBase
)
from Operator.operator_configurer import ConfigOperatorIdentity, ConfigOperatorDisplayConfMat, ConfigOperatorDisplayDictKeys, \
    ConfigOperatorMonaiAsDiscrete, ConfigOperatorTorchSoftmax
from Optimizer.optimizer_configurer import ConfigOptimizerSGD, ConfigOptimizerAdamW
from LRScheduler.lrscheduler_configurer import (
    ConfigLRSchedulerLinear,
    ConfigLRSchedulerCosineAnnealing,
    ConfigLRSchedulerCosineAnnealingWarmRestarts,
    ConfigLRSchedulerOneCycleConfigLR,
    ConfigLRSchedulerReduceConfigLROnPlateau, ConfigLRSchedulerBase
)
from Metric.metric_configurer import (
    BACC, BPREC, BREC, BF1, BAUROC, BCM, BSPE, BROC, BPRC,
    MCACC, MCPREC, MCRECALL, MCF1, MCAUROC, MCCM, MCSPEC, MCROC, MCPRC,
    MLACC, MLPREC, MLREC, MLF1, MLAUROC, MLCM, MLSPE, MLROC, MLPRC,
    Dice, IoU, HD, SD, NSD,
    ConfigMetricEfficiency, VPS, ConfigMetricBase, ConfigMetricDiceScore, SupportedMetric
)

PhaseLike = Literal['train', 'val', 'test', 'predict']

SupportedNetwork = Union[ConfigNetworkUNet]
SupportedLoss = Union[
    ConfigLossDice, ConfigLossDeepSupervisionDiceCE,
    ConfigLossDiceCE, ConfigLossDeepSupervisionDiceCE,
    ConfigLossDiceFocal, ConfigLossDeepSupervisionDiceFocal,
    ConfigLossHausdorffDT
]
SupportedOptimizer = Union[ConfigOptimizerSGD, ConfigOptimizerAdamW]
SupportedLRScheduler = Union[
    ConfigLRSchedulerLinear,
    ConfigLRSchedulerCosineAnnealing,
    ConfigLRSchedulerCosineAnnealingWarmRestarts,
    ConfigLRSchedulerOneCycleConfigLR,
    ConfigLRSchedulerReduceConfigLROnPlateau
]


@dataclass
class NamedInitArgs:
    name: str = "<Name ID>"
    description_info: Union[str, Any] = "<Description>"

    def __repr__(self):
        desc: str = f'{self.name}:\n'
        if self.description_info is not None:
            desc += f'  {self.description_info}'
        return desc


@dataclass
class NamedNetworkInitArgs(NamedInitArgs):
    network_wrapper: ConfigNetworkBase = ConfigNetworkUNet()


@dataclass
class NamedLossInitArgs(NamedInitArgs):
    loss_wrapper: ConfigLossBase = ConfigLossDice()
    postprocess_func: Optional[Union[Callable[[Tensor], Tensor], Callable[[Tensor], Dict[str, Tensor]]]] = None
    logger: Optional[bool] = True
    on_step: Optional[bool] = True
    on_epoch: Optional[bool] = True
    prog_bar: bool = True
    reduce_fx: Union[str, Callable[[Tensor], Tensor]] = 'mean'
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.postprocess_func is None:
            self.postprocess_func = ConfigOperatorIdentity()

    def get_logging_args(self, dict_logging: bool = False) -> Dict[str, Any]:
        logging_args: Dict[str, Any] = {}
        if not dict_logging:
            logging_args['name'] = self.name
        logging_args.update({
            'logger': self.logger,
            'on_step': self.on_step,
            'on_epoch': self.on_epoch,
            'prog_bar': self.prog_bar,
            'reduce_fx': self.reduce_fx,
        })
        if self.kwargs is not None:
            logging_args.update(self.kwargs)
        return logging_args


@dataclass
class NamedOptimizerInitArgs(NamedInitArgs):
    optimizer_wrapper: SupportedOptimizer = ConfigOptimizerAdamW()


@dataclass
class LRSchedulerConfig:
    # The unit of the scheduler's step size, could also be 'step'.
    # 'epoch' updates the scheduler on epoch end whereas 'step'
    # updates it after an optimizer update.
    interval: Literal["epoch", "step"] = 'epoch'
    # How many epochs/steps should pass between calls to
    # `scheduler.step()`. 1 corresponds to updating the learning
    # rate after every epoch/step.
    frequency: int = 1
    # Metric to monitor for schedulers like `ReduceLROnPlateau`
    monitor: str = 'val_loss'
    # If set to `True`, will enforce that the value specified 'monitor'
    # is available when the scheduler is updated, thus stopping
    # training if not found. If set to `False`, it will only produce a warning
    strict: bool = True
    # If using the `LearningRateMonitor` callback to monitor the
    # learning rate progress, this keyword can be used to specify
    # a custom logged name
    name: Optional[str] = None


@dataclass
class NamedLRSchedulerInitArgs(NamedInitArgs):
    lr_scheduler_wrapper: ConfigLRSchedulerBase = ConfigLRSchedulerCosineAnnealing()
    config_args: LRSchedulerConfig = LRSchedulerConfig()

    def __post_init__(self):
        pass


@dataclass
class NamedMetricInitArgs(NamedInitArgs):
    metric_wrapper: ConfigMetricBase = Dice()
    # The preprocess function to apply to logits/gt-mask to generate valid prediction input,
    # may be identity, sigmoid, softmax, argmax or any other functions
    preprocess_pred_func: Optional[Callable[[Tensor], Tensor]] = None
    preprocess_gt_func: Optional[Callable[[Tensor], Tensor]] = None
    postprocess_metric_func: Optional[Union[Callable[[Tensor], Tensor], Callable[[Tensor], Dict[str, Tensor]]]] = None
    logger: Optional[bool] = True
    on_step: Optional[bool] = True
    on_epoch: Optional[bool] = True
    prog_bar: bool = True
    reduce_fx: Union[str, Callable[[Tensor], Tensor]] = 'mean'
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.preprocess_pred_func is None:
            self.preprocess_pred_func = ConfigOperatorIdentity()
        if self.preprocess_gt_func is None:
            self.preprocess_gt_func = ConfigOperatorIdentity()
        if self.postprocess_metric_func is None:
            self.postprocess_metric_func = ConfigOperatorIdentity()
        assert self.name is not None, f'You must specify a name for the metric.'

    def get_logging_args(self, dict_logging: bool = False) -> Dict[str, Any]:
        logging_args: Dict[str, Any] = {}
        if not dict_logging:
            logging_args['name'] = self.name
        logging_args.update({
            'logger': self.logger,
            'on_step': self.on_step,
            'on_epoch': self.on_epoch,
            'prog_bar': self.prog_bar,
            'reduce_fx': self.reduce_fx,
        })
        if self.kwargs is not None:
            logging_args.update(self.kwargs)
        return logging_args


@dataclass
class ModuleStepAdditionArgs:
    inferer_wrapper: ConfigInfererBase = SimpleInferer()
    metric_init_args_collection: TLSeq[NamedMetricInitArgs] = ()
    # Hook functions will leverage return dict in module steps for custom purposes
    hook_functions: Optional[Collection[Callable[[Dict[str, Any]], Any]]] = None


@dataclass
class ModuleStepWithLossAdditionArgs(ModuleStepAdditionArgs):
    loss_init_args: NamedLossInitArgs = NamedLossInitArgs()


@dataclass
class ModuleTrainingStepAdditionArgs(ModuleStepWithLossAdditionArgs):
    optimizer_init_args: NamedOptimizerInitArgs = NamedOptimizerInitArgs()
    lrscheduler_init_args: NamedLRSchedulerInitArgs = NamedLRSchedulerInitArgs()
    volume_key: str = 'volume'
    mask_key: str = 'mask'


@dataclass
class ModuleValidationStepAdditionArgs(ModuleStepWithLossAdditionArgs):
    volume_key: str = 'volume'
    mask_key: str = 'mask'


@dataclass
class ModuleTestStepAdditionArgs(ModuleStepWithLossAdditionArgs):
    volume_key: str = 'volume'
    mask_key: str = 'mask'


@dataclass
class ModulePredictStepAdditionArgs(ModuleStepAdditionArgs):
    volume_key: str = 'volume'


class ModuleSegmentationDefault(L.LightningModule):
    def __init__(
            self,
            network_init_args: NamedNetworkInitArgs = NamedNetworkInitArgs(),
            module_training_step_addition_args: Optional[ModuleTrainingStepAdditionArgs] = None,
            module_validation_step_addition_args: Optional[ModuleValidationStepAdditionArgs] = None,
            module_test_step_addition_args: Optional[ModuleTestStepAdditionArgs] = None,
            module_predict_step_addition_args: Optional[ModulePredictStepAdditionArgs] = None
    ):
        super().__init__()
        self.network_init_args: NamedNetworkInitArgs = network_init_args
        self.module_training_step_addition_args = module_training_step_addition_args
        self.module_validation_step_addition_args = module_validation_step_addition_args
        self.module_test_step_addition_args = module_test_step_addition_args
        self.module_predict_step_addition_args = module_predict_step_addition_args
        self.save_hyperparameters()

        self.available_phases: Set[PhaseLike] = set()

        # Initialization
        self.network: nn.Module = self.network_init_args.network_wrapper.get_network_module()

        if self.module_training_step_addition_args is not None:
            try:
                step_args: ModuleTrainingStepAdditionArgs = self.module_training_step_addition_args
                self.training_inferer: Inferer = step_args.inferer_wrapper.get_inferer_operator()
                self.training_loss: nn.Module = step_args.loss_init_args.loss_wrapper.get_loss_operator()
                self.training_metrics: Dict[str, SupportedMetric] = {
                    args.name: args.metric_wrapper.get_metric_operator()
                    for args in step_args.metric_init_args_collection
                }
                self.training_metrics_desc: Dict[str, NamedMetricInitArgs] = {
                    args.name: args for args in step_args.metric_init_args_collection
                }
                self.training_hook_funcs: List[Callable[[Dict[str, Any]], Any]] = \
                    [] if step_args.hook_functions is None else list(step_args.hook_functions)
                self.available_phases.add('train')
            except Exception as e:
                print(e)

        if module_validation_step_addition_args is not None:
            try:
                step_args: ModuleValidationStepAdditionArgs = module_validation_step_addition_args
                self.validation_inferer: Inferer = step_args.inferer_wrapper.get_inferer_operator()
                self.validation_loss: nn.Module = step_args.loss_init_args.loss_wrapper.get_loss_operator()
                self.validation_metrics: Dict[str, SupportedMetric] = {
                    args.name: args.metric_wrapper.get_metric_operator()
                    for args in step_args.metric_init_args_collection
                }
                self.validation_metrics_desc: Dict[str, NamedMetricInitArgs] = {
                    args.name: args for args in step_args.metric_init_args_collection
                }
                self.validation_hook_funcs: List[Callable[[Dict[str, Any]], Any]] = \
                    [] if step_args.hook_functions is None else list(step_args.hook_functions)
                self.available_phases.add('val')
            except Exception as e:
                print(e)

        if module_test_step_addition_args is not None:
            try:
                step_args: ModuleTestStepAdditionArgs = module_test_step_addition_args
                self.test_inferer: Inferer = step_args.inferer_wrapper.get_inferer_operator()
                self.test_loss: nn.Module = step_args.loss_init_args.loss_wrapper.get_loss_operator()
                self.test_metrics: Dict[str, SupportedMetric] = {
                    args.name: args.metric_wrapper.get_metric_operator()
                    for args in step_args.metric_init_args_collection
                }
                self.test_metrics_desc: Dict[str, NamedMetricInitArgs] = {
                    args.name: args for args in step_args.metric_init_args_collection
                }
                self.test_hook_funcs: List[Callable[[Dict[str, Any]], Any]] = \
                    [] if step_args.hook_functions is None else list(step_args.hook_functions)
                self.available_phases.add('test')
            except Exception as e:
                print(e)

        if module_predict_step_addition_args is not None:
            try:
                step_args: ModulePredictStepAdditionArgs = module_predict_step_addition_args
                self.predict_inferer: Inferer = step_args.inferer_wrapper.get_inferer_operator()
                self.predict_metrics: Dict[str, SupportedMetric] = {
                    args.name: args.metric_wrapper.get_metric_operator()
                    for args in step_args.metric_init_args_collection
                }
                self.predict_metrics_desc: Dict[str, NamedMetricInitArgs] = {
                    args.name: args for args in step_args.metric_init_args_collection
                }
                self.predict_hook_funcs: List[Callable[[Dict[str, Any]], Any]] = \
                    [] if step_args.hook_functions is None else list(step_args.hook_functions)
                self.available_phases.add('predict')
            except Exception as e:
                print(e)

    def get_available_phases(self) -> Set[PhaseLike]:
        self._assert_init_essentials()
        if not hasattr(self, 'available_phases'):
            return set()
        return self.available_phases

    def forward(self, input_source: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass through network

        Args:
            input_source: Input tensor of shape (B, C, X, Y, Z)

        Returns:
            Tuple of (main_logits, auxiliary_logits_list)
            - main_logits: Final segmentation output (B, num_classes, X, Y, Z)
            - auxiliary_logits_list: List of auxiliary outputs for deep supervision
        """
        self._assert_init_essentials()
        return self.network(input_source)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        self._assert_init_essentials()
        assert 'train' in self.get_available_phases(), f'train phase is not available in {self.get_available_phases()}'
        step_args: ModuleTrainingStepAdditionArgs = self.module_training_step_addition_args
        ret: Dict[str, Tensor] = {}

        volume: Tensor = batch[step_args.volume_key]
        mask: Tensor = batch[step_args.mask_key]
        sz_volume: Tuple[int, ...] = tuple(volume.size())
        sz_mask: Tuple[int, ...] = tuple(mask.size())
        assert volume.ndim == 5, f'volume [ndim={volume.ndim}] shall be 5 (B, C, X, Y, Z)'
        assert mask.ndim == 5, f'mask [ndim={volume.ndim}] shall be 5 (B, C, X, Y, Z)'
        assert sz_volume[0] == sz_mask[0] and sz_volume[2:] == sz_mask[2:], \
            f'mask [size={sz_mask}] size shall be compatible with volume [size={sz_volume}]'

        metrics: Dict[str, Any] = {}
        # For performance metric
        for name, metric_func in self.training_metrics.items():
            if not isinstance(metric_func, ConfigMetricEfficiency):
                continue
            metric_func()  # First call

        # Forward
        cls_logits: Tensor
        aux_cls_logits: List[Tensor]
        cls_logits, aux_cls_logits = self.training_inferer(volume, self)
        # Reordered logits
        logits: List[Tensor] = [cls_logits] + list(reversed(aux_cls_logits))

        # Calculate loss
        loss: Tensor = self.training_loss(logits, mask)
        ret.update({
            'loss': loss, 'cls_logits': cls_logits, 'aux_cls_logits': aux_cls_logits,
            'volume': volume, 'mask': mask
        })

        loss_init_args: NamedLossInitArgs = self.module_training_step_addition_args.loss_init_args
        loss_post: Union[Tensor, Dict[str, Tensor]] = loss_init_args.postprocess_func(loss)
        if isinstance(loss, dict):
            self.log_dict(loss_post, **loss_init_args.get_logging_args(dict_logging=True))
            ret.update(loss_post)
        else:
            self.log(value=loss_post, **loss_init_args.get_logging_args())
            ret['loss_post'] = loss_post

        # Calculate and log metrics
        metrics_desc: Dict[str, NamedMetricInitArgs] = self.training_metrics_desc
        for name, metric_func in self.training_metrics.items():
            desc: NamedMetricInitArgs = metrics_desc[name]
            pred, gt = logits[0], mask  # logits[0] the main prediction mask
            if desc.preprocess_pred_func is not None:
                pred: Tensor = desc.preprocess_pred_func(pred)
            if desc.preprocess_gt_func is not None:
                gt: Tensor = desc.preprocess_gt_func(gt)
            value = metric_func(pred, gt)
            if desc.postprocess_metric_func is not None:
                value = desc.postprocess_metric_func(value)
            if isinstance(value, dict):
                metrics.update(value)
                self.log_dict(value, **desc.get_logging_args(dict_logging=True))
            else:
                metrics[name] = value
                if isinstance(metric_func, Metric):  # Use internal logging mode defined by TorchMetrics
                    self.log(value=metric_func, **desc.get_logging_args())
                else:
                    self.log(value=value, **desc.get_logging_args())

        ret.update(metrics)

        for hook in self.training_hook_funcs:
            hook(ret)

        return ret

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        self._assert_init_essentials()
        assert 'val' in self.get_available_phases(), \
            f'val phase is not available in {self.get_available_phases()}'
        step_args: ModuleValidationStepAdditionArgs = self.module_validation_step_addition_args
        ret: Dict[str, Tensor] = {}

        volume: Tensor = batch[step_args.volume_key]
        mask: Tensor = batch[step_args.mask_key]
        sz_volume: Tuple[int, ...] = tuple(volume.size())
        sz_mask: Tuple[int, ...] = tuple(mask.size())
        assert volume.ndim == 5, f'volume [ndim={volume.ndim}] shall be 5 (1, C, X, Y, Z)'
        assert sz_volume[0] == 1, f'volume [size={sz_volume}] shall have batch_size == 1 as (1, C, X, Y, Z)'
        assert mask.ndim == 5, f'mask [ndim={volume.ndim}] shall be 5 (1, C, X, Y, Z)'
        assert sz_mask[0] == 1, f'mask [size={sz_mask}] shall have batch_size == 1 as (1, C, X, Y, Z)'
        assert sz_volume[0] == sz_mask[0] and sz_volume[2:] == sz_mask[2:], \
            f'mask [size={sz_mask}] size shall be compatible with volume [size={sz_volume}]'

        metrics: Dict[str, Any] = {}
        # For performance metric
        for name, metric_func in self.validation_metrics.items():
            if not isinstance(metric_func, ConfigMetricEfficiency):
                continue
            metric_func()  # First call

        # Forward
        cls_logits: Tensor
        aux_cls_logits: List[Tensor]
        cls_logits, aux_cls_logits = self.validation_inferer(volume, self)
        # Reordered logits
        logits: List[Tensor] = [cls_logits] + list(reversed(aux_cls_logits))

        # Calculate loss
        loss: Tensor = self.validation_loss(logits, mask)
        ret.update({
            'loss': loss, 'cls_logits': cls_logits, 'aux_cls_logits': aux_cls_logits,
            'volume': volume, 'mask': mask
        })
        loss_init_args: NamedLossInitArgs = self.module_validation_step_addition_args.loss_init_args
        loss_post: Union[Tensor, Dict[str, Tensor]] = loss_init_args.postprocess_func(loss)
        if isinstance(loss, dict):
            self.log_dict(loss_post, **loss_init_args.get_logging_args(dict_logging=True))
            ret.update(loss_post)
        else:
            self.log(value=loss_post, **loss_init_args.get_logging_args())
            ret['loss_post'] = loss_post

        # Calculate and log metrics
        metrics_desc: Dict[str, NamedMetricInitArgs] = self.validation_metrics_desc
        for name, metric_func in self.validation_metrics.items():
            desc: NamedMetricInitArgs = metrics_desc[name]
            pred, gt = logits[0], mask
            if desc.preprocess_pred_func is not None:
                pred: Tensor = desc.preprocess_pred_func(pred)
            if desc.preprocess_gt_func is not None:
                gt: Tensor = desc.preprocess_gt_func(gt)
            value = metric_func(pred, gt)
            if desc.postprocess_metric_func is not None:
                value = desc.postprocess_metric_func(value)
            if isinstance(value, dict):
                metrics.update(value)
                self.log_dict(value, **desc.get_logging_args(dict_logging=True))
            else:
                metrics[name] = value
                if isinstance(metric_func, Metric):  # Use internal logging mode defined by TorchMetrics
                    self.log(value=metric_func, **desc.get_logging_args())
                else:
                    self.log(value=value, **desc.get_logging_args())

        ret.update(metrics)

        for hook in self.validation_hook_funcs:
            hook(ret)

        return ret

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        self._assert_init_essentials()
        assert 'test' in self.get_available_phases(), f'test phase is not available in {self.get_available_phases()}'
        step_args: ModuleTestStepAdditionArgs = self.module_test_step_addition_args
        ret: Dict[str, Tensor] = {}

        volume: Tensor = batch[step_args.volume_key]
        mask: Tensor = batch[step_args.mask_key]
        sz_volume: Tuple[int, ...] = tuple(volume.size())
        sz_mask: Tuple[int, ...] = tuple(mask.size())
        assert volume.ndim == 5, f'volume [ndim={volume.ndim}] shall be 5 (1, C, X, Y, Z)'
        assert sz_volume[0] == 1, f'volume [size={sz_volume}] shall have batch_size == 1 as (1, C, X, Y, Z)'
        assert mask.ndim == 5, f'mask [ndim={volume.ndim}] shall be 5 (1, C, X, Y, Z)'
        assert sz_mask[0] == 1, f'mask [size={sz_mask}] shall have batch_size == 1 as (1, C, X, Y, Z)'
        assert sz_volume[0] == sz_mask[0] and sz_volume[2:] == sz_mask[2:], \
            f'mask [size={sz_mask}] size shall be compatible with volume [size={sz_volume}]'

        metrics: Dict[str, Any] = {}
        # For performance metric
        for name, metric_func in self.test_metrics.items():
            if not isinstance(metric_func, ConfigMetricEfficiency):
                continue
            metric_func()  # First call

        # Forward
        cls_logits: Tensor
        aux_cls_logits: List[Tensor]
        cls_logits, aux_cls_logits = self.test_inferer(volume, self)
        # Reordered logits
        logits: List[Tensor] = [cls_logits] + list(reversed(aux_cls_logits))

        # Calculate loss
        loss: Tensor = self.test_loss(logits, mask)
        ret.update({
            'loss': loss, 'cls_logits': cls_logits, 'aux_cls_logits': aux_cls_logits,
            'volume': volume, 'mask': mask
        })

        loss_init_args: NamedLossInitArgs = self.module_test_step_addition_args.loss_init_args
        loss_post: Union[Tensor, Dict[str, Tensor]] = loss_init_args.postprocess_func(loss)
        if isinstance(loss, dict):
            self.log_dict(loss_post, **loss_init_args.get_logging_args(dict_logging=True))
            ret.update(loss_post)
        else:
            self.log(value=loss_post, **loss_init_args.get_logging_args())
            ret['loss_post'] = loss_post

        # Calculate and log metrics
        metrics_desc: Dict[str, NamedMetricInitArgs] = self.test_metrics_desc
        for name, metric_func in self.test_metrics.items():
            desc: NamedMetricInitArgs = metrics_desc[name]
            pred, gt = logits[0], mask
            if desc.preprocess_pred_func is not None:
                pred: Tensor = desc.preprocess_pred_func(pred)
            if desc.preprocess_gt_func is not None:
                gt: Tensor = desc.preprocess_gt_func(gt)
            value = metric_func(pred, gt)
            if desc.postprocess_metric_func is not None:
                value = desc.postprocess_metric_func(value)
            if isinstance(value, dict):
                metrics.update(value)
                self.log_dict(value, **desc.get_logging_args(dict_logging=True))
            else:
                metrics[name] = value
                if isinstance(metric_func, Metric):  # Use internal logging mode defined by TorchMetrics
                    self.log(value=metric_func, **desc.get_logging_args())
                else:
                    self.log(value=value, **desc.get_logging_args())

        ret.update(metrics)

        for hook in self.test_hook_funcs:
            hook(ret)

        return ret

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        self._assert_init_essentials()
        assert 'predict' in self.get_available_phases(), \
            f'predict phase is not available in {self.get_available_phases()}'
        step_args: ModulePredictStepAdditionArgs = self.module_predict_step_addition_args
        ret: Dict[str, Tensor] = {}

        volume: Tensor = batch[step_args.volume_key]
        assert volume.ndim == 5, f'volume [ndim={volume.ndim}] shall be 5 (1, C, X, Y, Z)'
        assert volume.size(0) == 1, \
            f'volume [size={tuple(volume.size())}] shall have batch_size == 1 as (1, C, X, Y, Z)'

        metrics: Dict[str, Any] = {}
        # For performance metric
        for name, metric_func in self.predict_metrics.items():
            if not isinstance(metric_func, ConfigMetricEfficiency):
                continue
            metric_func()  # First call

        # Forward
        cls_logits: Tensor
        aux_cls_logits: List[Tensor]
        cls_logits, aux_cls_logits = self.predict_inferer(volume, self)
        # Reordered logits
        logits: List[Tensor] = [cls_logits] + list(reversed(aux_cls_logits))
        ret.update({
            'cls_logits': cls_logits, 'aux_cls_logits': aux_cls_logits,
            'volume': volume
        })

        # Calculate and log metrics
        metrics_desc: Dict[str, NamedMetricInitArgs] = self.predict_metrics_desc
        for name, metric_func in self.predict_metrics.items():
            desc: NamedMetricInitArgs = metrics_desc[name]
            pred = logits[0]
            if desc.preprocess_pred_func is not None:
                pred: Tensor = desc.preprocess_pred_func(pred)
            value = metric_func(pred)
            if desc.postprocess_metric_func is not None:
                value = desc.postprocess_metric_func(value)
            if isinstance(value, dict):
                metrics.update(value)
                self.log_dict(value, **desc.get_logging_args(dict_logging=True))
            else:
                metrics[name] = value
                if isinstance(metric_func, Metric):  # Use internal logging mode defined by TorchMetrics
                    self.log(value=metric_func, **desc.get_logging_args())
                else:
                    self.log(value=value, **desc.get_logging_args())

        ret.update(metrics)

        for hook in self.predict_hook_funcs:
            hook(ret)

        return ret

    def configure_optimizers(self) -> Dict[str, Any]:
        self._assert_init_essentials()
        # Optimizer
        opt_init_args: NamedOptimizerInitArgs = self.module_training_step_addition_args.optimizer_init_args
        optimizer: optim.Optimizer = opt_init_args.optimizer_wrapper.get_optimizer(params=self.parameters())

        # Scheduler (generate lr_scheduler_config)
        lrsch_init_args: NamedLRSchedulerInitArgs = self.module_training_step_addition_args.lrscheduler_init_args
        scheduler: optim.lr_scheduler.LRScheduler = lrsch_init_args.lr_scheduler_wrapper.get_lr_scheduler(
            optimizer=optimizer
        )
        lr_scheduler_config: Dict[str, Any] = {'scheduler': scheduler}
        lr_scheduler_config.update(vars(lrsch_init_args.config_args))

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler_config
        }


def get_default_config(num_sequence: int = 1, num_classes: int = 2) -> Dict[str, Any]:
    network_init_args = NamedNetworkInitArgs(
        name='UNet',
        network_wrapper=ConfigNetworkUNet(
            focuser_in_channels=num_sequence,  # Assume (num_sequence) sequence input
            focuser_out_channels=16,
            encoder_primary_in_channels=(16, 32),
            encoder_primary_out_channels=(32, 64),
            encoder_primary_depth=2,
            encoder_advanced_in_channels=(64, 128),
            encoder_advanced_out_channels=(128, 256),
            encoder_advanced_depth=2,
            bottleneck_in_channels=256,
            bottleneck_out_channels=512,
            bottleneck_depth=2,
            decoder_advanced_in_channels=(512, 256),
            decoder_advanced_upsample_channels=(256, 128),
            decoder_advanced_bridge_channels=(256, 128),
            decoder_advanced_out_channels=(256, 128),
            decoder_advanced_depth=2,
            decoder_primary_in_channels=(128, 64),
            decoder_primary_upsample_channels=(64, 32),
            decoder_primary_bridge_channels=(64, 32),
            decoder_primary_out_channels=(64, 32),
            decoder_primary_depth=2,
            auxiliary_classifier_in_channels=(256, 128, 64, 32),
            auxiliary_classifier_out_channels=(num_classes, num_classes, num_classes, num_classes),
            distributor_in_channels=32,
            distributor_out_channels=16,
            classifier_in_channels=16,
            classifier_out_channels=num_classes,  # Assume N classes (background & N-1 foregrounds)
            reserve_io=True
        ),
        description_info='Basic multi-organ segmentation UNet'
    )
    module_training_step_addition_args = ModuleTrainingStepAdditionArgs(
        inferer_wrapper=ConfigInfererSimple(),
        metric_init_args_collection=[
            # Dice Similarity Coefficient
            NamedMetricInitArgs(
                name='train/DSC',
                metric_wrapper=Dice(
                    include_background=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=None,
                    return_with_label=False
                ),
                description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # Normalized Surface Dice
            NamedMetricInitArgs(
                name='train/NSD',
                metric_wrapper=NSD(
                    # Tolerance of at most 3.0 distance error in index space
                    # First threshold is for background, this is nonsense in case background is excluded
                    class_thresholds=[0., 3.],
                    include_background=False,
                    distance_metric='euclidean',
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    use_subvoxels=False
                ),
                description_info='Normalized surface dice metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # 95% percentile Hausdorff Distance
            NamedMetricInitArgs(
                name='train/HD95',
                metric_wrapper=HD(
                    include_background=False,
                    distance_metric='euclidean',
                    percentile=95.0,
                    directed=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False
                ),
                description_info='95% percentile Hausdorff distance metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # Confusion Matrix
            # For Binary, CM is M(2*2): [[TN,FP],[FN,TP]]
            # For Multi-class, CM is M(num_classes*num_classes): E[i,j] denotes the i-th gt class is predicted as j-th class
            NamedMetricInitArgs(
                name='train/ConfMat',  # Nonsense, handled by postprocess_metric_func, which will return a dict
                metric_wrapper=MCCM(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    normalize='none'
                ),
                description_info='Confusion Matrix for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                postprocess_metric_func=ConfigOperatorDisplayConfMat(
                    'train',
                    'ConfMat',
                    ((0, 'gt'), (1, 'pred'))
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='sum'  # Elements shall be summed up
            ),
            # Classification global metrics
            # Acc: Multi-class calculation shall always accumulate all classes
            # Prec Recall Spec F1 AUROC: Shall keep metrics per class, and do post reduce as per class metrics
            NamedMetricInitArgs(
                name='train/Acc',
                metric_wrapper=MCACC(  # Accuracy shall calculate across all classes
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='micro',
                    multidim_average='global'
                ),
                description_info='Accuracy metric for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='train/Prec',
                metric_wrapper=MCPREC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Precision metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='train/Recall',
                metric_wrapper=MCRECALL(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Recall metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='train/Spec',
                metric_wrapper=MCSPEC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Specificity metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='train/F1',
                metric_wrapper=MCF1(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='F1-Score metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='train/AUROC',
                metric_wrapper=MCAUROC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    ignore_index=0  # Ignoring background
                ),
                description_info='AUROC metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTorchSoftmax(dim=1),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='train/VPS',
                metric_wrapper=VPS(),
                description_info='Voxel Processing Per Second metric',
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            )
        ],
        loss_init_args=NamedLossInitArgs(
            name='train/loss',
            loss_wrapper=ConfigLossDeepSupervisionDiceCE(
                include_background=False,  # Foregrounds are small
                to_onehot_y=False,  # We use (B, C, X, Y, Z) C-binary map as mask
                sigmoid=False,
                softmax=True,  # Assume multi-class (organs not overlapped) segmentation
                jaccard=False,
                reduction="mean",
                batch=False,
                weight=None,
                lambda_dice=1.0,
                lambda_ce=1.0,
                label_smoothing=0.0,
                ds_weight_mode='exp',
                ds_weights=None
            ),
            description_info='Dice + Cross Entropy compounded loss for deep supervision',
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            reduce_fx='mean'
        ),
        optimizer_init_args=NamedOptimizerInitArgs(
            name='AdamW',
            optimizer_wrapper=ConfigOptimizerAdamW(
                # 'params': module.parameters()  # Shall ignore this argument, will be set at configure_optimizers()
                lr=1e-4,  # May be overwritten by LRScheduler
                amsgrad=False
            ),
            description_info='AdamW optimizer'
        ),
        lrscheduler_init_args=NamedLRSchedulerInitArgs(
            name='OneCycleLR',
            lr_scheduler_wrapper=ConfigLRSchedulerOneCycleConfigLR(  # Shall step() per batch
                # 'optimizer': optimizer  # Shall ignore this argument, will be set at configure_optimizers()
                max_lr=0.01,
                total_steps=None,  # Infer from epochs * steps_per_epoch
                epochs=100,
                steps_per_epoch=5,  # Practically, shall infer from len(dataloader)
                pct_start=0.3,  # Increasing part occupies the first 30% steps
                div_factor=25,
                final_div_factor=1e4
            ),
            description_info='OneCycleLR scheduler'
        ),
        volume_key='volume',
        mask_key='mask',
        hook_functions=[ConfigOperatorDisplayDictKeys(('Train', 'Step returns'))]
    )
    module_validation_step_addition_args = ModuleValidationStepAdditionArgs(
        inferer_wrapper=ConfigInfererMainWithAuxSlidingWindow(
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            overlap=0.5,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            padding_mode=PytorchPadMode.REPLICATE,
            progress=True
        ),
        metric_init_args_collection=[
            # Dice Similarity Coefficient
            NamedMetricInitArgs(
                name='val/DSC',
                metric_wrapper=Dice(
                    include_background=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=None,
                    return_with_label=False
                ),
                description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # Normalized Surface Dice
            NamedMetricInitArgs(
                name='val/NSD',
                metric_wrapper=NSD(
                    # Tolerance of at most 3.0 distance error in index space
                    # First threshold is for background, this is nonsense in case background is excluded
                    class_thresholds=[0., 3.],
                    include_background=False,
                    distance_metric='euclidean',
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    use_subvoxels=False
                ),
                description_info='Normalized surface dice metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # 95% percentile Hausdorff Distance
            NamedMetricInitArgs(
                name='val/HD95',
                metric_wrapper=HD(
                    include_background=False,
                    distance_metric='euclidean',
                    percentile=95.0,
                    directed=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False
                ),
                description_info='95% percentile Hausdorff distance metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # Confusion Matrix
            # For Binary, CM is M(2*2): [[TN,FP],[FN,TP]]
            # For Multi-class, CM is M(num_classes*num_classes): E[i,j] denotes the i-th gt class is predicted as j-th class
            NamedMetricInitArgs(
                name='val/ConfMat',  # Nonsense, handled by postprocess_metric_func, which will return a dict
                metric_wrapper=MCCM(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    normalize='none'
                ),
                description_info='Confusion Matrix for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                postprocess_metric_func=ConfigOperatorDisplayConfMat(
                    'val',
                    'ConfMat',
                    ((0, 'gt'), (1, 'pred'))
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='sum'  # Elements shall be summed up
            ),
            # Classification global metrics
            # Acc: Multi-class calculation shall always accumulate all classes
            # Prec Recall Spec F1 AUROC: Shall keep metrics per class, and do post reduce as per class metrics
            NamedMetricInitArgs(
                name='val/Acc',
                metric_wrapper=MCACC(  # Accuracy shall calculate across all classes
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='micro',
                    multidim_average='global'
                ),
                description_info='Accuracy metric for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='val/Prec',
                metric_wrapper=MCPREC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Precision metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='val/Recall',
                metric_wrapper=MCRECALL(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Recall metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='val/Spec',
                metric_wrapper=MCSPEC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Specificity metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='val/F1',
                metric_wrapper=MCF1(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='F1-Score metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='val/AUROC',
                metric_wrapper=MCAUROC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    ignore_index=0  # Ignoring background
                ),
                description_info='AUROC metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTorchSoftmax(dim=1),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='val/VPS',
                metric_wrapper=VPS(),
                description_info='Voxel Processing Per Second metric',
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            )
        ],
        loss_init_args=NamedLossInitArgs(
            name='val/loss',
            loss_wrapper=ConfigLossDeepSupervisionDiceCE(
                include_background=False,  # Foregrounds are small
                to_onehot_y=False,  # We use (B, C, X, Y, Z) C-binary map as mask
                sigmoid=False,
                softmax=True,  # Assume multi-class (organs not overlapped) segmentation
                jaccard=False,
                reduction="mean",
                batch=False,
                weight=None,
                lambda_dice=1.0,
                lambda_ce=1.0,
                label_smoothing=0.0,
                ds_weight_mode='exp',
                ds_weights=None
            ),
            description_info='Dice + Cross Entropy compounded loss for deep supervision',
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            reduce_fx='mean'
        ),
        volume_key='volume',
        mask_key='mask',
        hook_functions=[ConfigOperatorDisplayDictKeys(('Val', 'Step returns'))]
    )
    module_test_step_addition_args = ModuleTestStepAdditionArgs(
        inferer_wrapper=ConfigInfererMainWithAuxSlidingWindow(
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            overlap=0.5,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            padding_mode=PytorchPadMode.REPLICATE,
            progress=True
        ),
        metric_init_args_collection=[
            # Dice Similarity Coefficient
            NamedMetricInitArgs(
                name='test/DSC',
                metric_wrapper=Dice(
                    include_background=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=None,
                    return_with_label=False
                ),
                description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # Normalized Surface Dice
            NamedMetricInitArgs(
                name='test/NSD',
                metric_wrapper=NSD(
                    # Tolerance of at most 3.0 distance error in index space
                    # First threshold is for background, this is nonsense in case background is excluded
                    class_thresholds=[0., 3.],
                    include_background=False,
                    distance_metric='euclidean',
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    use_subvoxels=False
                ),
                description_info='Normalized surface dice metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # 95% percentile Hausdorff Distance
            NamedMetricInitArgs(
                name='test/HD95',
                metric_wrapper=HD(
                    include_background=False,
                    distance_metric='euclidean',
                    percentile=95.0,
                    directed=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False
                ),
                description_info='95% percentile Hausdorff distance metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1,
                                                                   dtype=torch.int),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, to_onehot=num_classes, dim=1, dtype=torch.int),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # Confusion Matrix
            # For Binary, CM is M(2*2): [[TN,FP],[FN,TP]]
            # For Multi-class, CM is M(num_classes*num_classes): E[i,j] denotes the i-th gt class is predicted as j-th class
            NamedMetricInitArgs(
                name='test/ConfMat',  # Nonsense, handled by postprocess_metric_func, which will return a dict
                metric_wrapper=MCCM(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    normalize='none'
                ),
                description_info='Confusion Matrix for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                postprocess_metric_func=ConfigOperatorDisplayConfMat(
                    'test',
                    'ConfMat',
                    ((0, 'gt'), (1, 'pred'))
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='sum'  # Elements shall be summed up
            ),
            # Classification global metrics
            # Acc: Multi-class calculation shall always accumulate all classes
            # Prec Recall Spec F1 AUROC: Shall keep metrics per class, and do post reduce as per class metrics
            NamedMetricInitArgs(
                name='test/Acc',
                metric_wrapper=MCACC(  # Accuracy shall calculate across all classes
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='micro',
                    multidim_average='global'
                ),
                description_info='Accuracy metric for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='test/Prec',
                metric_wrapper=MCPREC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Precision metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='test/Recall',
                metric_wrapper=MCRECALL(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Recall metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='test/Spec',
                metric_wrapper=MCSPEC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Specificity metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='test/F1',
                metric_wrapper=MCF1(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='F1-Score metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='test/AUROC',
                metric_wrapper=MCAUROC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    ignore_index=0  # Ignoring background
                ),
                description_info='AUROC metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTorchSoftmax(dim=1),
                preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(argmax=True, dim=1, dtype=torch.int, keepdim=False),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                # reduce_fx='mean'  # Nonsense, handled by Metric class own reduce logic
            ),
            NamedMetricInitArgs(
                name='test/VPS',
                metric_wrapper=VPS(),
                description_info='Voxel Processing Per Second metric',
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            )
        ],
        loss_init_args=NamedLossInitArgs(
            name='test/loss',
            loss_wrapper=ConfigLossDeepSupervisionDiceCE(
                include_background=False,  # Foregrounds are small
                to_onehot_y=False,  # We use (B, C, X, Y, Z) C-binary map as mask
                sigmoid=False,
                softmax=True,  # Assume multi-class (organs not overlapped) segmentation
                jaccard=False,
                reduction="mean",
                batch=False,
                weight=None,
                lambda_dice=1.0,
                lambda_ce=1.0,
                label_smoothing=0.0,
                ds_weight_mode='exp',
                ds_weights=None
            ),
            description_info='Dice + Cross Entropy compounded loss for deep supervision',
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            reduce_fx='mean'
        ),
        volume_key='volume',
        mask_key='mask',
        hook_functions=[ConfigOperatorDisplayDictKeys(('Test', 'Step returns'))]
    )
    module_predict_step_addition_args = ModulePredictStepAdditionArgs(
        inferer_wrapper=ConfigInfererMainWithAuxSlidingWindow(
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            overlap=0.5,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            padding_mode=PytorchPadMode.REPLICATE,
            progress=True
        ),
        metric_init_args_collection=[
            NamedMetricInitArgs(
                name='predict/VPS',
                metric_wrapper=VPS(),
                description_info='Voxel Processing Per Second metric',
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            )
        ],
        volume_key='volume',
        hook_functions=[ConfigOperatorDisplayDictKeys(('Predict', 'Step returns'))]
    )

    config: Dict[str, Any] = {
        'network_init_args': network_init_args,
        'module_training_step_addition_args': module_training_step_addition_args,
        'module_validation_step_addition_args': module_validation_step_addition_args,
        'module_test_step_addition_args': module_test_step_addition_args,
        'module_predict_step_addition_args': module_predict_step_addition_args
    }
    return config


if __name__ == "__main__":
    num_sequence: int = 1
    num_classes: int = 3
    B, X, Y, Z = (2, 512, 496, 374)
    config: Dict[str, Any] = get_default_config(num_sequence, num_classes)
    torch.manual_seed(0)
    volume: Tensor = torch.rand(B, num_sequence, X, Y, Z)
    # Simulate mask
    class_step: float = 1. / num_classes
    mask: Tensor = torch.zeros(B, num_classes, X, Y, Z)
    for idx in range(1, num_classes):
        low: float = idx * class_step
        high: float = low + class_step
        if idx == num_classes - 1:
            high = 1.0
        mask[:, idx] = torch.logical_and(volume > low, volume <= high).float().squeeze(1)

    # crop center (128, 128, 128)
    center_cropped_volume: Tensor = volume[
        :, :,
        X // 2 - 64:X // 2 + 64,
        Y // 2 - 64:Y // 2 + 64,
        Z // 2 - 64:Z // 2 + 64
    ]
    center_cropped_mask: Tensor = mask[
        :, :,
        X // 2 - 64:X // 2 + 64,
        Y // 2 - 64:Y // 2 + 64,
        Z // 2 - 64:Z // 2 + 64
    ]
    module: ModuleSegmentationDefault = ModuleSegmentationDefault(**config)
    opt_lr: Dict[str, Any] = module.configure_optimizers()

    # Train
    print('[Train]')
    print(f'Input: volume={tuple(center_cropped_volume.size())}, mask={tuple(center_cropped_mask.size())}')
    train_batch = {'volume': center_cropped_volume, 'mask': center_cropped_mask}
    module.train()
    train_output: Dict[str, Tensor] = module.training_step(train_batch, 0)
    train_output['cls_logits'] = tuple(train_output['cls_logits'].size())
    train_output['aux_cls_logits'] = [tuple(ts.size()) for ts in train_output['aux_cls_logits']]
    train_output['volume'] = tuple(train_output['volume'].size())
    train_output['mask'] = tuple(train_output['mask'].size())
    print(f'{train_output}\n')

    # Val
    print('[Validation]')
    module.eval()
    for batch_idx in range(B):
        val_batch = {'volume': volume[batch_idx:batch_idx + 1], 'mask': mask[batch_idx:batch_idx + 1]}
        print(f'Input: volume={tuple(val_batch["volume"].size())}, mask={tuple(val_batch["mask"].size())}')
        val_output: Dict[str, Tensor] = module.validation_step(val_batch, batch_idx)
        val_output['cls_logits'] = tuple(val_output['cls_logits'].size())
        val_output['aux_cls_logits'] = [tuple(ts.size()) for ts in val_output['aux_cls_logits']]
        val_output['volume'] = tuple(val_output['volume'].size())
        val_output['mask'] = tuple(val_output['mask'].size())
        print(f'[{batch_idx}] {val_output}\n')

    # Test
    print('[Test]')
    module.eval()
    for batch_idx in range(B):
        test_batch = {'volume': volume[batch_idx:batch_idx + 1], 'mask': mask[batch_idx:batch_idx + 1]}
        print(f'Input: volume={tuple(test_batch["volume"].size())}, mask={tuple(test_batch["mask"].size())}')
        test_output: Dict[str, Tensor] = module.test_step(test_batch, batch_idx)
        test_output['cls_logits'] = tuple(test_output['cls_logits'].size())
        test_output['aux_cls_logits'] = [tuple(ts.size()) for ts in test_output['aux_cls_logits']]
        test_output['volume'] = tuple(test_output['volume'].size())
        test_output['mask'] = tuple(test_output['mask'].size())
        print(f'[{batch_idx}] {test_output}\n')

    # Predict
    print(f'[Predict]')
    module.eval()
    for batch_idx in range(B):
        predict_batch = {'volume': volume[batch_idx:batch_idx + 1]}
        predict_output: Dict[str, Tensor] = module.predict_step(predict_batch, batch_idx)
        predict_output['cls_logits'] = tuple(predict_output['cls_logits'].size())
        predict_output['aux_cls_logits'] = [tuple(ts.size()) for ts in predict_output['aux_cls_logits']]
        predict_output['volume'] = tuple(predict_output['volume'].size())
        print(f'[{batch_idx}] {predict_output}\n')
