# -*- coding: utf-8 -*-
import os
from pathlib import Path
import yaml
from yaml import Node, ScalarNode
from yaml.representer import Representer
from yaml.emitter import Emitter
from yaml.serializer import Serializer
from yaml.resolver import Resolver
from yaml.constructor import Constructor
from yaml.reader import Reader
from yaml.scanner import Scanner
from yaml.parser import Parser
from yaml.composer import Composer
from dataclasses import dataclass
import torch
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.utilities import GradClipAlgorithmType
from typing import Optional, Dict, Any, Union, Literal, Type, Mapping
from monai.inferers import Inferer, SimpleInferer
from yaml.constructor import ConstructorError

from Callback.callback_configurer import (
    ConfigCallbackBase,
    ConfigCallbackModelCheckpoint,
    ConfigCallbackDeviceStatsMonitor,
    ConfigCallbackEarlyStopping,
    ConfigCallbackLearningRateMonitor,
    ConfigCallbackRichModelSummary,
    ConfigCallbackRichProgressBar
)
from DataModule.data_module_segmentation_default import (
    DataModuleSegmentationDefault,
    DataModuleSegmentationDefaultInitArgs
)
from Inferer.inferer_configurer import (
    ConfigInfererBase,
    ConfigInfererSimple,
    ConfigInfererSlidingWindow,
    ConfigInfererMainWithAuxSlidingWindow
)
from monai.utils import MetricReduction, BlendMode, PytorchPadMode, GridSampleMode, GridSamplePadMode
import monai.data.dataset as mD

from Module.ltn_module_segmentation_default import (
    NamedNetworkInitArgs,
    ModuleTrainingStepAdditionArgs,
    NamedMetricInitArgs,
    NamedLossInitArgs,
    NamedOptimizerInitArgs,
    NamedLRSchedulerInitArgs,
    ModuleTestStepAdditionArgs,
    ModuleValidationStepAdditionArgs,
    ModulePredictStepAdditionArgs,
    ModuleSegmentationDefault
)
from Network.network_configurer import (
    ConfigNetworkBase,
    ConfigNetworkUNet
)
from Loss.loss_configurer import (
    ConfigLossBase,
    ConfigLossDice,
    ConfigLossDeepSupervisionDice,
    ConfigLossDiceCE,
    ConfigLossDeepSupervisionDiceCE,
    ConfigLossDiceFocal,
    ConfigLossDeepSupervisionDiceFocal,
    ConfigLossHausdorffDT
)
from Operator.operator_configurer import (
    ConfigOperatorBase,
    ConfigOperatorDisplayDictKeys,
    ConfigOperatorDisplayConfMat,
    ConfigOperatorMonaiAsDiscrete,
    ConfigOperatorTorchSoftmax
)
from Optimizer.optimizer_configurer import (
    ConfigOptimizerBase,
    ConfigOptimizerSGD,
    ConfigOptimizerAdamW
)
from LRScheduler.lrscheduler_configurer import (
    ConfigLRSchedulerBase,
    ConfigLRSchedulerLinear,
    ConfigLRSchedulerCosineAnnealing,
    ConfigLRSchedulerCosineAnnealingWarmRestarts,
    ConfigLRSchedulerOneCycleConfigLR,
    ConfigLRSchedulerReduceConfigLROnPlateau
)
from Metric.metric_configurer import (
    ConfigMetricBase,
    BACC, BPREC, BREC, BF1, BAUROC, BCM, BSPE, BROC, BPRC,
    MCACC, MCPREC, MCRECALL, MCF1, MCAUROC, MCCM, MCSPEC, MCROC, MCPRC,
    MLACC, MLPREC, MLREC, MLF1, MLAUROC, MLCM, MLSPE, MLROC, MLPRC,
    Dice, IoU, HD, SD, NSD,
    ConfigMetricEfficiency, VPS
)
from Trainer.trainer_configurer import (
    TrainerInitArgs,
    CallbackInitArgs,
    LoggerInitArgs,
    ConfigTrainerSegmentationDefault
)
from Transform.transform_configurer import (
    ConfigTransformSegmentationDefaultTrain,
    ConfigTransformSegmentationDefaultInferencePre
)
from Launcher.Parser.parser_ABC import ParserABC

PhaseLike = Literal['train', 'val', 'test', 'predict']

SupportedNetwork = Union[UNet]
SupportedLoss = Union[
    ConfigLossDice, DeepSupervisionDiceLoss,
    ConfigLossDiceCE, ConfigLossDeepSupervisionDiceCE,
    ConfigLossDiceFocal, ConfigLossDeepSupervisionDiceFocal,
    ConfigLossHausdorffDT
]
SupportedOptimizer = Union[SGD, AdamW]
SupportedLRScheduler = Union[
    ConfigLRSchedulerLinear,
    ConfigLRSchedulerCosineAnnealing,
    ConfigLRSchedulerCosineAnnealingWarmRestarts,
    OneCycleLR,
    ConfigLRSchedulerReduceConfigLROnPlateau
]
SupportedMetric = Union[
    BACC, BPREC, BREC, BF1, BAUROC, BCM, BSPE, BROC, BPRC,
    MCACC, MCPREC, MCRECALL, MCF1, MCAUROC, MCCM, MCSPEC, MCROC, MCPRC,
    MLACC, MLPREC, MLREC, MLF1, MLAUROC, MLCM, MLSPE, MLROC, MLPRC,
    Dice, IoU, HD, SD, NSD,
    VPS
]


class YamlDumperSegmentationDefault(yaml.Dumper):
    def __init__(
            self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True
    ):
        Emitter.__init__(
            self, stream, canonical=canonical,
            indent=indent, width=width,
            allow_unicode=allow_unicode, line_break=line_break
        )
        Serializer.__init__(
            self, encoding=encoding,
            explicit_start=explicit_start, explicit_end=explicit_end,
            version=version, tags=tags
        )
        Representer.__init__(
            self, default_style=default_style,
            default_flow_style=default_flow_style, sort_keys=sort_keys
        )
        Resolver.__init__(self)

    def ignore_aliases(self, data):
        return True

    def represent_torch_dtype(self, data: torch.dtype) -> Node:
        name: str = data.__reduce__()
        node: ScalarNode = self.represent_scalar('tag:yaml.org,2002:torch/dtype:' + name, '')
        return node


YamlDumperSegmentationDefault.add_representer(
    torch.dtype,
    YamlDumperSegmentationDefault.represent_torch_dtype  # noqa
)


class YamlLoaderSegmentationDefault(yaml.Loader):
    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        Constructor.__init__(self)
        Resolver.__init__(self)

    def construct_torch_dtype(self, suffix: str, node: ScalarNode) -> ScalarNode:
        value: str = self.construct_scalar(node)
        if value:
            raise ConstructorError("while constructing a Torch dtype", node.start_mark,
                                   "expected the empty value, but found %r" % value, node.start_mark)
        if not hasattr(torch, suffix):
            raise ConstructorError("while constructing a Torch dtype", node.start_mark,
                                   "type %r is not supported" % suffix, node.start_mark)
        return getattr(torch, suffix)


YamlLoaderSegmentationDefault.add_multi_constructor(
    'tag:yaml.org,2002:torch/dtype:',
    YamlLoaderSegmentationDefault.construct_torch_dtype  # noqa
)


@dataclass
class ParserSegmentationDefault(ParserABC):
    @staticmethod
    def default_experiment_root_dir() -> str:
        return 'Experiments'

    @staticmethod
    def default_experiment_name() -> str:
        return 'PipeUNet-3DUNet'

    @staticmethod
    def default_experiment_version() -> str:
        return 'Train-001'

    @staticmethod
    def default_trainer_class() -> Type[ConfigTrainerSegmentationDefault]:
        return ConfigTrainerSegmentationDefault

    @staticmethod
    def default_trainer_init_args() -> TrainerInitArgs:
        return TrainerInitArgs(
            # Platform control
            accelerator='gpu',
            devices=[0],
            precision=32,
            enable_distributed_data_parallel=False,
            # Routine control
            max_epochs=100,
            check_val_every_n_epoch=1,
            # Gradient control
            accumulate_grad_batches=16,  # Simulating batch_size*16
            gradient_clip_val=None,
            gradient_clip_algorithm=GradClipAlgorithmType.NORM,
            # Logging control
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            # Reproducibility control
            deterministic=None,  # We do not control CUDA operators' randomness, if you need, set to 'warn' or True
            # Debugging
            detect_anomaly=True,
            num_sanity_val_steps=2,
            fast_dev_run=False,
            overfit_batches=0.0
        )

    @staticmethod
    def default_callback_init_args() -> CallbackInitArgs:
        return CallbackInitArgs(
            enable_device_stats_monitor=True,
            callback_device_stats_monitor=DeviceStatsMonitorInitArgs(cpu_stats=True),
            enable_early_stopping=True,
            callback_early_stopping=EarlyStoppingInitArgs(
                monitor='val/loss',
                patience=20,
                mode='min',
                verbose=True
            ),
            enable_learning_rate_monitor=True,
            callback_learning_rate_monitor=LearningRateMonitorInitArgs(
                logging_interval='epoch',
                log_momentum=True,
                log_weight_decay=True
            ),
            enable_rich_model_summary=True,
            callback_rich_model_summary=RichModelSummaryInitArgs(max_depth=5),
            enable_rich_progressbar=True,
            callback_rich_progressbar=RichProgressBarInitArgs(
                refresh_rate=1,
                leave=True,
                theme=RichProgressBarTheme(),
                console_kwargs=None
            ),
            callback_model_checkpoints=[
                ModelCheckpointInitArgs(
                    dirpath='milestone',
                    filename='{epoch:03d}-loss={val/loss:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='epoch',
                    save_top_k=-1,
                    mode='max',
                    save_last=False,
                    every_n_epochs=10
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/DSC',
                    filename='{epoch:03d}-loss={val/loss:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/DSC',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/HD95',
                    filename='{epoch:03d}-loss={val/loss:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/HD95',
                    save_top_k=5,
                    mode='min',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/NSD',
                    filename='{epoch:03d}-NSD={val/NSD:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/NSD',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/Acc',
                    filename='{epoch:03d}-Acc={val/Acc:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/Acc',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/Prec',
                    filename='{epoch:03d}-Prec={val/Prec:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/Prec',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/Spec',
                    filename='{epoch:03d}-spe={val/Spec:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/Spec',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/Recall',
                    filename='{epoch:03d}-Recall={val/Recall:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/Recall',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/F1',
                    filename='{epoch:03d}-F1={val/F1:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/F1',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                ModelCheckpointInitArgs(
                    dirpath='val/AUROC',
                    filename='{epoch:03d}-AUROC={val/AUROC:4f}-DSC={val/DSC:4f}-HD95{val/HD95:4f}',
                    monitor='val/AUROC',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                )
            ]
        )

    @staticmethod
    def default_logger_init_args() -> LoggerInitArgs:
        return LoggerInitArgs(
            enable_csv_logger=True,
            enable_tensorboard_logger=True,
            enable_wandb_logger=True,
            wandb_project='PipeUNet'
        )

    @staticmethod
    def default_datamodule_class() -> Type[DataModuleSegmentationDefault]:
        return DataModuleSegmentationDefault

    @staticmethod
    def default_module_class() -> Type[ModuleSegmentationDefault]:
        return ModuleSegmentationDefault

    @staticmethod
    def default_network_init_args(num_sequence: int = 1, num_classes: int = 2) -> NamedNetworkInitArgs:
        return NamedNetworkInitArgs(
            name='UNet',
            class_type=UNet,
            init_args={
                'focuser_in_channels': num_sequence,  # Assume (num_sequence) sequence input
                'focuser_out_channels': 16,
                'encoder_primary_in_channels': (16, 32),
                'encoder_primary_out_channels': (32, 64),
                'encoder_primary_depth': 2,
                'encoder_advanced_in_channels': (64, 128),
                'encoder_advanced_out_channels': (128, 256),
                'encoder_advanced_depth': 2,
                'bottleneck_in_channels': 256,
                'bottleneck_out_channels': 512,
                'bottleneck_depth': 2,
                'decoder_advanced_in_channels': (512, 256),
                'decoder_advanced_upsample_channels': (256, 128),
                'decoder_advanced_bridge_channels': (256, 128),
                'decoder_advanced_out_channels': (256, 128),
                'decoder_advanced_depth': 2,
                'decoder_primary_in_channels': (128, 64),
                'decoder_primary_upsample_channels': (64, 32),
                'decoder_primary_bridge_channels': (64, 32),
                'decoder_primary_out_channels': (64, 32),
                'decoder_primary_depth': 2,
                'auxiliary_classifier_in_channels': (256, 128, 64, 32),
                'auxiliary_classifier_out_channels': (2, 2, 2, 2),
                'distributor_in_channels': 32,
                'distributor_out_channels': 16,
                'classifier_in_channels': 16,
                'classifier_out_channels': num_classes,  # Assume (C=num_classes) classes (background & C-1 foreground)
                'reserve_io': False  # Do not reserve io, except for indepth inspecting
            },
            description_info=f'Basic background/{num_classes}-foreground segmentation UNet'
        )

    @staticmethod
    def default_datamodule_training_init_args() -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            root_dir='Samples',
            manifest_file='Samples/split/split_train.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_ROI': 'mask_1',
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None,
            dataset_cls=mD.PersistentDataset,
            dataset_params={
                'cache_dir': 'Samples/cache'
            },
            transform_cls=ConfigTransformSegmentationDefaultTrain,
            transform_params={
                'volume_key': 'volume',
                'mask_key': 'mask',
                'param_volume_tf_duplicate_items_dup_keys_volume': None,
                'param_mask_tf_duplicate_items_dup_keys_mask': None,
                'param_tf_spacing_pixdim': (1.0, 1.0, 1.0),
                'param_tf_spacing_mode_volume': GridSampleMode.BILINEAR,
                'param_tf_spacing_mode_mask': GridSampleMode.NEAREST,
                'param_tf_padding_mode_volume': GridSamplePadMode.BORDER,
                'param_tf_padding_mode_mask': GridSamplePadMode.BORDER,
                'param_tf_spatial_pad_spatial_size': (128, 128, 128),
                'param_tf_spatial_pad_mode': PytorchPadMode.REPLICATE,
                'param_tf_rand_crop_by_label_classes_spatial_size': (128, 128, 128),
                'param_tf_rand_crop_by_label_classes_ratios': (0.0, 1.0),
                'param_tf_rand_crop_by_label_classes_num_classes': 2,
                'param_tf_rand_crop_by_label_classes_num_samples': 2,
                'param_tf_scale_intensity_range_a_min': -1000,
                'param_tf_scale_intensity_range_a_max': 1000,
                'param_tf_scale_intensity_range_b_min': 0.0,
                'param_tf_scale_intensity_range_b_max': 1.0,
                'param_tf_scale_intensity_range_clip': True,
                'param_tf_allow_missing_keys': False,
                'random_seed': 0,
            },
            batch_size=2,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            # Set persistent_workers=false to enable resumable reproducibility.
            # When false, Dataloader will use the specified RNG to init each worker every epoch,
            # so by recording the RNG state, worker states are recorded properly.
            # But if persistent_workers=true, workers will be initialized only one time at the beginning, after which
            # their states can not be reached anymore, so can not be identified by recording the RNG state.
            # After resuming, workers will be reinitialized by the recoded RNG, which differs from the former state.
            persistent_workers=False,
            generator_random_seed=0
        )

    @staticmethod
    def default_datamodule_validation_init_args() -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            root_dir='Samples',
            manifest_file='Samples/split/split_val.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_ROI': 'mask_1'
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None,
            dataset_cls=mD.PersistentDataset,
            dataset_params={
                'cache_dir': 'Samples/cache'
            },
            transform_cls=ConfigTransformSegmentationDefaultInferencePre,
            transform_params={
                'volume_key': 'volume',
                'mask_key': 'mask',
                'param_volume_tf_duplicate_items_dup_keys_volume': 'volume_raw',
                'param_mask_tf_duplicate_items_dup_keys_mask': 'mask_raw',
                'param_tf_spacing_pixdim': (1.0, 1.0, 1.0),
                'param_tf_spacing_mode_volume': GridSampleMode.BILINEAR,
                'param_tf_spacing_mode_mask': GridSampleMode.NEAREST,
                'param_tf_padding_mode_volume': GridSamplePadMode.BORDER,
                'param_tf_padding_mode_mask': GridSamplePadMode.BORDER,
                'param_tf_scale_intensity_range_a_min': -1000,
                'param_tf_scale_intensity_range_a_max': 1000,
                'param_tf_scale_intensity_range_b_min': 0.0,
                'param_tf_scale_intensity_range_b_max': 1.0,
                'param_tf_scale_intensity_range_clip': True
            },
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def default_datamodule_test_init_args() -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            root_dir='Samples',
            manifest_file='Samples/split/split_test.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_ROI': 'mask_1'
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None,
            dataset_cls=mD.PersistentDataset,
            dataset_params={
                'cache_dir': 'Samples/cache'
            },
            transform_cls=ConfigTransformSegmentationDefaultInferencePre,
            transform_params={
                'volume_key': 'volume',
                'mask_key': 'mask',
                'param_volume_tf_duplicate_items_dup_keys_volume': 'volume_raw',
                'param_mask_tf_duplicate_items_dup_keys_mask': 'mask_raw',
                'param_tf_spacing_pixdim': (1.0, 1.0, 1.0),
                'param_tf_spacing_mode_volume': GridSampleMode.BILINEAR,
                'param_tf_spacing_mode_mask': GridSampleMode.NEAREST,
                'param_tf_padding_mode_volume': GridSamplePadMode.BORDER,
                'param_tf_padding_mode_mask': GridSamplePadMode.BORDER,
                'param_tf_scale_intensity_range_a_min': -1000,
                'param_tf_scale_intensity_range_a_max': 1000,
                'param_tf_scale_intensity_range_b_min': 0.0,
                'param_tf_scale_intensity_range_b_max': 1.0,
                'param_tf_scale_intensity_range_clip': True
            },
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def default_datamodule_predict_init_args() -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            root_dir='Samples',
            manifest_file='Samples/split/split_predict.xlsx',
            column_key_map={
                'volume': 'volume'
            },
            column_key_relative_path=['volume'],
            column_group_map={'volume': ['volume']},
            column_dtype_map=None,
            dataset_cls=mD.PersistentDataset,
            dataset_params={
                'cache_dir': 'Samples/cache'
            },
            transform_cls=ConfigTransformSegmentationDefaultInferencePre,
            transform_params={
                'volume_key': 'volume',
                'param_volume_tf_duplicate_items_dup_keys_volume': 'volume_raw',
                'param_tf_spacing_pixdim': (1.0, 1.0, 1.0),
                'param_tf_spacing_mode_volume': GridSampleMode.BILINEAR,
                'param_tf_padding_mode_volume': GridSamplePadMode.BORDER,
                'param_tf_scale_intensity_range_a_min': -1000,
                'param_tf_scale_intensity_range_a_max': 1000,
                'param_tf_scale_intensity_range_b_min': 0.0,
                'param_tf_scale_intensity_range_b_max': 1.0,
                'param_tf_scale_intensity_range_clip': True
            },
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def default_module_training_step_addition_args(num_classes: int = 2) -> ModuleTrainingStepAdditionArgs:
        return ModuleTrainingStepAdditionArgs(
            inferer=SimpleInferer,
            inferer_init_args={},
            metric_init_args_collection=[
                # Dice Similarity Coefficient
                NamedMetricInitArgs(
                    name='train/DSC',
                    class_type=Dice,
                    init_args={
                        'include_background': False,
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False,
                        'ignore_empty': True,
                        'num_classes': None,
                        'return_with_label': False
                    },
                    description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                ),
                # Normalized Surface Dice
                NamedMetricInitArgs(
                    name='train/NSD',
                    class_type=NSD,
                    init_args={
                        # Tolerance of at most 3.0 distance error in index space
                        # First threshold is for background, this is nonsense in case background is excluded
                        'class_thresholds': [0., 3.],
                        'include_background': False,
                        'distance_metric': 'euclidean',
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False,
                        'use_subvoxels': False
                    },
                    description_info='Normalized surface Dice metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                ),
                # 95% percentile Hausdorff Distance
                NamedMetricInitArgs(
                    name='train/HD95',
                    class_type=HD,
                    init_args={
                        'include_background': False,
                        'distance_metric': 'euclidean',
                        'percentile': 95.0,
                        'directed': False,
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False
                    },
                    description_info='95% percentile Hausdorff distance metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
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
                    class_type=MCCM,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'normalize': 'none'
                    },
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
                    class_type=MCACC,  # Accuracy shall calculate across all classes
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'micro',
                        'multidim_average': 'global'
                    },
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
                    class_type=MCPREC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCRECALL,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCSPEC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCF1,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCAUROC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=VPS,
                    init_args={},
                    description_info='Voxel Processing Per Second metric',
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                )
            ],
            loss_init_args=NamedLossInitArgs(
                name='train/loss',
                class_type=ConfigLossDeepSupervisionDiceCE,
                init_args={
                    'include_background': False,  # Foregrounds are small
                    'to_onehot_y': False,  # We use (B, C, X, Y, Z) C-binary map as mask
                    'sigmoid': False,
                    'softmax': True,  # Assume multi-class (organs not overlapped) segmentation
                    'jaccard': False,
                    'reduction': "mean",
                    'batch': False,
                    'weight': None,
                    'lambda_dice': 1.0,
                    'lambda_ce': 1.0,
                    'label_smoothing': 0.0,
                    'ds_weight_mode': 'exp',
                    'ds_weights': None
                },
                description_info='Dice + Cross Entropy compounded loss for deep supervision',
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                reduce_fx='mean'
            ),
            optimizer_init_args=NamedOptimizerInitArgs(
                name='AdamW',
                class_type=AdamW,
                init_args={
                    # 'params': module.parameters()  # Shall ignore this argument, will be set at configure_optimizers()
                    'lr': 1e-4,  # May be overwritten by LRScheduler
                    'amsgrad': False
                },
                description_info='AdamW optimizer'
            ),
            lrscheduler_init_args=NamedLRSchedulerInitArgs(
                name='OneCycleLR',
                class_type=OneCycleLR,  # Shall step() per batch
                init_args={
                    # 'optimizer': optimizer  # Shall ignore this argument, will be set at configure_optimizers()
                    'max_lr': 0.01,
                    # Set total_steps to None, so as to infer from epochs * steps_per_epoch,
                    # otherwise set both epochs & steps_per_epoch to None, and directly specify total_steps
                    'total_steps': None,
                    'epochs': 100,
                    # Practically, steps_per_epoch can be inferred from len(Dataloader),
                    # but it may not be available all the time, in case that Dataloader do not report len(),
                    # and for Trainer routine, you shall initialize the LR-Scheduler before getting Dataloader
                    # Anyway, please specify it manually, it's your responsibility!
                    'steps_per_epoch': 5,
                    'pct_start': 0.3,  # Increasing part occupies the first 30% steps
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                description_info='OneCycleLR scheduler'
            ),
            volume_key='volume',
            mask_key='mask',
            hook_functions=[ConfigOperatorDisplayDictKeys(('Train', 'Step returns'))]
        )

    @staticmethod
    def default_module_validation_step_addition_args(num_classes: int = 2) -> ModuleValidationStepAdditionArgs:
        return ModuleValidationStepAdditionArgs(
            inferer=ConfigInfererMainWithAuxSlidingWindow,
            inferer_init_args={
                'roi_size': (128, 128, 128),
                'sw_batch_size': 1,
                'overlap': 0.5,
                'mode': BlendMode.GAUSSIAN,
                'sigma_scale': 0.125,
                'padding_mode': PytorchPadMode.REPLICATE,
                'progress': True
            },
            metric_init_args_collection=[
                # Dice Similarity Coefficient
                NamedMetricInitArgs(
                    name='val/DSC',
                    class_type=Dice,
                    init_args={
                        'include_background': False,
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False,
                        'ignore_empty': True,
                        'num_classes': None,
                        'return_with_label': False
                    },
                    description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                ),
                # Normalized Surface Dice
                NamedMetricInitArgs(
                    name='val/NSD',
                    class_type=NSD,
                    init_args={
                        # Tolerance of at most 3.0 distance error in index space
                        # First threshold is for background, this is nonsense in case background is excluded
                        'class_thresholds': [0., 3.],
                        'include_background': False,
                        'distance_metric': 'euclidean',
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False,
                        'use_subvoxels': False
                    },
                    description_info='Normalized surface dice metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                ),
                # 95% percentile Hausdorff Distance
                NamedMetricInitArgs(
                    name='val/HD95',
                    class_type=HD,
                    init_args={
                        'include_background': False,
                        'distance_metric': 'euclidean',
                        'percentile': 95.0,
                        'directed': False,
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False
                    },
                    description_info='95% percentile Hausdorff distance metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int)
                    ,
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
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
                    class_type=MCCM,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'normalize': 'none'
                    },
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
                    class_type=MCACC,  # Accuracy shall calculate across all classes
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'micro',
                        'multidim_average': 'global'
                    },
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
                    class_type=MCPREC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCRECALL,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCSPEC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCF1,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCAUROC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=VPS,
                    init_args={},
                    description_info='Voxel Processing Per Second metric',
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                )
            ],
            loss_init_args=NamedLossInitArgs(
                name='val/loss',
                class_type=ConfigLossDeepSupervisionDiceCE,
                init_args={
                    'include_background': False,  # Foregrounds are small
                    'to_onehot_y': False,  # We use (B, C, X, Y, Z) C-binary map as mask
                    'sigmoid': False,
                    'softmax': True,  # Assume multi-class (organs not overlapped) segmentation
                    'jaccard': False,
                    'reduction': "mean",
                    'batch': False,
                    'weight': None,
                    'lambda_dice': 1.0,
                    'lambda_ce': 1.0,
                    'label_smoothing': 0.0,
                    'ds_weight_mode': 'exp',
                    'ds_weights': None
                },
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

    @staticmethod
    def default_module_test_step_addition_args(num_classes: int = 2) -> ModuleTestStepAdditionArgs:
        return ModuleTestStepAdditionArgs(
            inferer=ConfigInfererMainWithAuxSlidingWindow,
            inferer_init_args={
                'roi_size': (128, 128, 128),
                'sw_batch_size': 1,
                'overlap': 0.5,
                'mode': BlendMode.GAUSSIAN,
                'sigma_scale': 0.125,
                'padding_mode': PytorchPadMode.REPLICATE,
                'progress': True
            },
            metric_init_args_collection=[
                # Dice Similarity Coefficient
                NamedMetricInitArgs(
                    name='test/DSC',
                    class_type=Dice,
                    init_args={
                        'include_background': False,
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False,
                        'ignore_empty': True,
                        'num_classes': None,
                        'return_with_label': False
                    },
                    description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                ),
                # Normalized Surface Dice
                NamedMetricInitArgs(
                    name='test/NSD',
                    class_type=NSD,
                    init_args={
                        # Tolerance of at most 3.0 distance error in index space
                        # First threshold is for background, this is nonsense in case background is excluded
                        'class_thresholds': [0., 3.],
                        'include_background': False,
                        'distance_metric': 'euclidean',
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False,
                        'use_subvoxels': False
                    },
                    description_info='Normalized surface dice metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                ),
                # 95% percentile Hausdorff Distance
                NamedMetricInitArgs(
                    name='test/HD95',
                    class_type=HD,
                    init_args={
                        'include_background': False,
                        'distance_metric': 'euclidean',
                        'percentile': 95.0,
                        'directed': False,
                        'reduction': MetricReduction.MEAN,
                        'get_not_nans': False
                    },
                    description_info='95% percentile Hausdorff distance metric (ignoring background) '
                                     'for multi-class (organs not overlapped) segmentation',
                    preprocess_pred_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
                    preprocess_gt_func=ConfigOperatorMonaiAsDiscrete(
                        argmax=True,
                        to_onehot=num_classes,
                        dim=1,
                        dtype=torch.int
                    ),
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
                    class_type=MCCM,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'normalize': 'none'
                    },
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
                    class_type=MCACC,  # Accuracy shall calculate across all classes
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'micro',
                        'multidim_average': 'global'
                    },
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
                    class_type=MCPREC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCRECALL,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCSPEC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCF1,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'multidim_average': 'global',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=MCAUROC,
                    init_args={
                        'num_classes': num_classes,  # Assume N classes (background & N-1 foregrounds)
                        'average': 'macro',
                        'ignore_index': 0  # Ignoring background
                    },
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
                    class_type=VPS,
                    init_args={},
                    description_info='Voxel Processing Per Second metric',
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    reduce_fx='mean'
                )
            ],
            loss_init_args=NamedLossInitArgs(
                name='test/loss',
                class_type=ConfigLossDeepSupervisionDiceCE,
                init_args={
                    'include_background': False,  # Foregrounds are small
                    'to_onehot_y': False,  # We use (B, C, X, Y, Z) C-binary map as mask
                    'sigmoid': False,
                    'softmax': True,  # Assume multi-class (organs not overlapped) segmentation
                    'jaccard': False,
                    'reduction': "mean",
                    'batch': False,
                    'weight': None,
                    'lambda_dice': 1.0,
                    'lambda_ce': 1.0,
                    'label_smoothing': 0.0,
                    'ds_weight_mode': 'exp',
                    'ds_weights': None
                },
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

    @staticmethod
    def default_module_predict_step_addition_args() -> ModulePredictStepAdditionArgs:
        return ModulePredictStepAdditionArgs(
            inferer=ConfigInfererMainWithAuxSlidingWindow,
            inferer_init_args={
                'roi_size': (128, 128, 128),
                'sw_batch_size': 1,
                'overlap': 0.5,
                'mode': BlendMode.GAUSSIAN,
                'sigma_scale': 0.125,
                'padding_mode': PytorchPadMode.REPLICATE,
                'progress': True
            },
            metric_init_args_collection=[
                NamedMetricInitArgs(
                    name='predict/VPS',
                    class_type=VPS,
                    init_args={},
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

    @staticmethod
    def dump_example_to_yaml(
            export_path: Optional[Union[str, os.PathLike, Path]],
            default_style: Optional[str] = None,
            default_flow_style: Optional[bool] = False,
            canonical: Optional[bool] = None,
            indent: Optional[int] = 2,
            width: Optional[Union[int, float]] = None,
            allow_unicode: Optional[bool] = None,
            line_break: Optional[str] = None,
            encoding: Optional[str] = None,
            explicit_start: Optional[bool] = None,
            explicit_end: Optional[bool] = None,
            version: Optional[tuple[int, int]] = None,
            tags: Optional[Mapping[str, str]] = None,
            sort_keys: bool = False
    ) -> Optional[str]:
        experiment_root_dir = ParserSegmentationDefault.default_experiment_root_dir()
        experiment_name = ParserSegmentationDefault.default_experiment_name()
        experiment_version = ParserSegmentationDefault.default_experiment_version()
        trainer_class = ParserSegmentationDefault.default_trainer_class()
        trainer_init_args = ParserSegmentationDefault.default_trainer_init_args()
        callback_init_args = ParserSegmentationDefault.default_callback_init_args()
        logger_init_args = ParserSegmentationDefault.default_logger_init_args()
        datamodule_class = ParserSegmentationDefault.default_datamodule_class()
        module_class = ParserSegmentationDefault.default_module_class()
        network_init_args = ParserSegmentationDefault.default_network_init_args()
        datamodule_training_init_args = ParserSegmentationDefault.default_datamodule_training_init_args()
        datamodule_validation_init_args = ParserSegmentationDefault.default_datamodule_validation_init_args()
        datamodule_test_init_args = ParserSegmentationDefault.default_datamodule_test_init_args()
        datamodule_predict_init_args = ParserSegmentationDefault.default_datamodule_predict_init_args()
        module_training_step_addition_args = ParserSegmentationDefault.default_module_training_step_addition_args()
        module_validation_step_addition_args = ParserSegmentationDefault.default_module_validation_step_addition_args()
        module_test_step_addition_args = ParserSegmentationDefault.default_module_test_step_addition_args()
        module_predict_step_addition_args = ParserSegmentationDefault.default_module_predict_step_addition_args()
        export_dict: Dict[str, Any] = {
            'experiment_root_dir': experiment_root_dir,
            'experiment_name': experiment_name,
            'experiment_version': experiment_version,
            'trainer_class': trainer_class,
            'trainer_init_args': trainer_init_args,
            'callback_init_args': callback_init_args,
            'logger_init_args': logger_init_args,
            'datamodule_class': datamodule_class,
            'module_class': module_class,
            'network_init_args': network_init_args,
            'datamodule_training_init_args': datamodule_training_init_args,
            'datamodule_validation_init_args': datamodule_validation_init_args,
            'datamodule_test_init_args': datamodule_test_init_args,
            'datamodule_predict_init_args': datamodule_predict_init_args,
            'module_training_step_addition_args': module_training_step_addition_args,
            'module_validation_step_addition_args': module_validation_step_addition_args,
            'module_test_step_addition_args': module_test_step_addition_args,
            'module_predict_step_addition_args': module_predict_step_addition_args
        }

        if export_path is not None:
            # Dump config to YAML file
            with open(export_path, 'w', encoding='utf-8') as file:
                yaml.dump(
                    export_dict, file, YamlDumperSegmentationDefault,
                    default_style=default_style,
                    default_flow_style=default_flow_style,
                    canonical=canonical,
                    indent=indent,
                    width=width,
                    allow_unicode=allow_unicode,
                    line_break=line_break,
                    encoding=encoding,
                    explicit_start=explicit_start,
                    explicit_end=explicit_end,
                    version=version,
                    tags=tags,
                    sort_keys=sort_keys
                )
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            print(f'Example config of {ParserSegmentationDefault.__module__}'
                  f'.{ParserSegmentationDefault.__name__} is saved to {export_path}.')
        else:
            return yaml.dump(
                export_dict, None, YamlDumperSegmentationDefault,
                default_style=default_style,
                default_flow_style=default_flow_style,
                canonical=canonical,
                indent=indent,
                width=width,
                allow_unicode=allow_unicode,
                line_break=line_break,
                encoding=encoding,
                explicit_start=explicit_start,
                explicit_end=explicit_end,
                version=version,
                tags=tags,
                sort_keys=sort_keys
            )

    experiment_root_dir: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_version: Optional[str] = None
    # Trainer shall be subtype of TrainerSegmentationDefault
    trainer_class: Optional[Type[ConfigTrainerSegmentationDefault]] = None
    trainer_init_args: Optional[TrainerInitArgs] = None
    callback_init_args: Optional[CallbackInitArgs] = None
    logger_init_args: Optional[LoggerInitArgs] = None
    datamodule_class: Optional[Type[DataModuleSegmentationDefault]] = None
    # Main Module shall be subtype of ModuleSegmentationDefault
    module_class: Optional[Type[ModuleSegmentationDefault]] = None
    network_init_args: Optional[NamedNetworkInitArgs] = None
    # Phase specific args
    datamodule_training_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_validation_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    module_training_step_addition_args: Optional[ModuleTrainingStepAdditionArgs] = None
    module_validation_step_addition_args: Optional[ModuleValidationStepAdditionArgs] = None
    module_test_step_addition_args: Optional[ModuleTestStepAdditionArgs] = None
    module_predict_step_addition_args: Optional[ModulePredictStepAdditionArgs] = None

    def to_yaml(
            self,
            export_path: Optional[Union[str, os.PathLike, Path]],
            default_style: Optional[str] = None,
            default_flow_style: Optional[bool] = False,
            canonical: Optional[bool] = None,
            indent: Optional[int] = 2,
            width: Optional[Union[int, float]] = None,
            allow_unicode: Optional[bool] = None,
            line_break: Optional[str] = None,
            encoding: Optional[str] = None,
            explicit_start: Optional[bool] = None,
            explicit_end: Optional[bool] = None,
            version: Optional[tuple[int, int]] = None,
            tags: Optional[Mapping[str, str]] = None,
            sort_keys: bool = False
    ) -> Optional[str]:
        export_dict: Dict[str, Any] = {
            'experiment_root_dir': self.experiment_root_dir,
            'experiment_name': self.experiment_name,
            'experiment_version': self.experiment_version,
            'trainer_class': self.trainer_class,
            'trainer_init_args': self.trainer_init_args,
            'callback_init_args': self.callback_init_args,
            'logger_init_args': self.logger_init_args,
            'datamodule_class': self.datamodule_class,
            'module_class': self.module_class,
            'network_init_args': self.network_init_args,
            'datamodule_training_init_args': self.datamodule_training_init_args,
            'datamodule_validation_init_args': self.datamodule_validation_init_args,
            'datamodule_test_init_args': self.datamodule_test_init_args,
            'datamodule_predict_init_args': self.datamodule_predict_init_args,
            'module_training_step_addition_args': self.module_training_step_addition_args,
            'module_validation_step_addition_args': self.module_validation_step_addition_args,
            'module_test_step_addition_args': self.module_test_step_addition_args,
            'module_predict_step_addition_args': self.module_predict_step_addition_args
        }

        if export_path is not None:
            # Dump config to YAML file, return None
            with open(export_path, 'w', encoding='utf-8') as file:
                yaml.dump(
                    export_dict, file, YamlDumperSegmentationDefault,
                    default_style=default_style,
                    default_flow_style=default_flow_style,
                    canonical=canonical,
                    indent=indent,
                    width=width,
                    allow_unicode=allow_unicode,
                    line_break=line_break,
                    encoding=encoding,
                    explicit_start=explicit_start,
                    explicit_end=explicit_end,
                    version=version,
                    tags=tags,
                    sort_keys=sort_keys
                )
            print(f'Config generated by {ParserSegmentationDefault.__module__}'
                  f'.{ParserSegmentationDefault.__name__} is saved to {export_path}.')
        else:
            # Return str
            return yaml.dump(
                export_dict, None, YamlDumperSegmentationDefault,
                default_style=default_style,
                default_flow_style=default_flow_style,
                canonical=canonical,
                indent=indent,
                width=width,
                allow_unicode=allow_unicode,
                line_break=line_break,
                encoding=encoding,
                explicit_start=explicit_start,
                explicit_end=explicit_end,
                version=version,
                tags=tags,
                sort_keys=sort_keys
            )

    def from_yaml(self, import_path: Optional[Union[str, os.PathLike, Path]] = None):
        # Load YAML contents
        with open(import_path, 'r', encoding='utf-8') as file:
            import_dict: Dict[str, Any] = yaml.load(file, YamlLoaderSegmentationDefault)

        self.experiment_root_dir = import_dict['experiment_root_dir']
        self.experiment_name = import_dict['experiment_name']
        self.experiment_version = import_dict['experiment_version']
        self.trainer_class = import_dict['trainer_class']
        self.trainer_init_args = import_dict['trainer_init_args']
        self.callback_init_args = import_dict['callback_init_args']
        self.logger_init_args = import_dict['logger_init_args']
        self.datamodule_class = import_dict['datamodule_class']
        self.module_class = import_dict['module_class']
        self.network_init_args = import_dict['network_init_args']
        self.datamodule_training_init_args = import_dict['datamodule_training_init_args']
        self.datamodule_validation_init_args = import_dict['datamodule_validation_init_args']
        self.datamodule_test_init_args = import_dict['datamodule_test_init_args']
        self.datamodule_predict_init_args = import_dict['datamodule_predict_init_args']
        self.module_training_step_addition_args = import_dict['module_training_step_addition_args']
        self.module_validation_step_addition_args = import_dict['module_validation_step_addition_args']
        self.module_test_step_addition_args = import_dict['module_test_step_addition_args']
        self.module_predict_step_addition_args = import_dict['module_predict_step_addition_args']

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


if __name__ == '__main__':
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=f'Dump example config of {ParserSegmentationDefault.__name__} to YAML file'
    )
    parser.add_argument('-o', '--export_path', type=str,
                        default='Launcher/Configs/SegmentationDefault/config_example.yaml')
    args: argparse.Namespace = parser.parse_args()
    Path(args.export_path).parent.mkdir(parents=True, exist_ok=True)
    ParserSegmentationDefault.dump_example_to_yaml(args.export_path)
