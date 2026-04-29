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
from dataclasses import dataclass, field
import torch
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.utilities import GradClipAlgorithmType
from typing import TypeVar, Optional, List, Tuple, Dict, Any, Union, Literal, Type, Mapping
from yaml.constructor import ConstructorError

from DataModule.data_module_configurer import ConfigDataModuleSegmentationDefault

T = TypeVar("T")
TLSeq = Union[List[T], Tuple[T, ...]]

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
from Dataset.dataset_configurer import (
    ConfigDatasetBase,
    ConfigDatasetCache,
    ConfigDatasetPersistent,
    ConfigDatasetLMDB
)
from Dataset.dataset_manifest_retriever_configurer import ConfigDatasetManifestRetrieverSegmentationDefault
from Inferer.inferer_configurer import (
    ConfigInfererBase,
    ConfigInfererSimple,
    ConfigInfererSlidingWindow,
    ConfigInfererMainWithAuxSlidingWindow
)
from monai.utils import MetricReduction, BlendMode, PytorchPadMode, GridSampleMode, GridSamplePadMode
from Module.ltn_module_configurer import ConfigLightningModuleSegmentationDefault
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
    LightningModuleSegmentationDefault, LRSchedulerLightningConfig
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
from Operator import (
    ConfigOperatorBase,
    ConfigOperatorTensorProcessBase,
    ConfigOperatorTensorProcessIdentity,
    ConfigOperatorTensorProcessMonaiAsDiscrete,
    ConfigOperatorTensorProcessTorchSoftmax,
    ConfigOperatorTensorRemapBase,
    ConfigOperatorTensorRemapConfMat,
    ConfigOperatorTensorRemapClassWise,
    ConfigOperatorHookStepBase,
    ConfigOperatorHookStepDisplayDictKeys
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
    ConfigLRSchedulerOneCycle,
    ConfigLRSchedulerReduceLROnPlateau
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
    ConfigTransformSegmentationDefaultInferencePre, ConfigTransformBase
)
from Launcher.Parser.parser_ABC import ParserABC

PhaseLike = Literal['train', 'val', 'test', 'predict']


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
    def default_trainer_init_args() -> TrainerInitArgs:
        return TrainerInitArgs(
            # Platform control
            accelerator='gpu',
            devices=[0],
            precision='bf16',
            enable_distributed_data_parallel=False,
            # Routine control
            max_epochs=100,
            check_val_every_n_epoch=1,
            # Gradient control
            accumulate_grad_batches=10,  # Simulating batch_size*10
            gradient_clip_val=None,
            gradient_clip_algorithm=GradClipAlgorithmType.NORM,
            # Logging control
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            # Reproducibility control
            deterministic=None,  # We do not control CUDA operators' randomness, if you need, set to 'warn' or True
            # Debugging
            detect_anomaly=False,
            num_sanity_val_steps=2,
            fast_dev_run=False,
            overfit_batches=0.0,
        )

    @staticmethod
    def default_callback_init_args() -> CallbackInitArgs:
        return CallbackInitArgs(
            # callback_device_stats_monitor=ConfigCallbackDeviceStatsMonitor(cpu_stats=True),
            callback_device_stats_monitor=None,
            # callback_early_stopping=ConfigCallbackEarlyStopping(
            #     monitor='val/loss',
            #     patience=20,
            #     mode='min',
            #     verbose=True
            # ),
            callback_early_stopping=None,
            callback_learning_rate_monitor=ConfigCallbackLearningRateMonitor(
                logging_interval='step',
                log_momentum=True,
                log_weight_decay=True
            ),
            # callback_learning_rate_monitor=None,
            callback_rich_model_summary=ConfigCallbackRichModelSummary(max_depth=5),
            # callback_rich_model_summary=None,
            callback_rich_progressbar=ConfigCallbackRichProgressBar(
                refresh_rate=1,
                leave=True,
                theme=RichProgressBarTheme(),
                console_kwargs=None
            ),
            # callback_rich_progressbar=None,
            callback_model_checkpoints=[
                ConfigCallbackModelCheckpoint(
                    dirpath='milestone',
                    filename='{epoch:03d}-loss={val/loss:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                    monitor='epoch',
                    save_top_k=-1,
                    mode='max',
                    save_last=False,
                    every_n_epochs=10
                ),
                ConfigCallbackModelCheckpoint(
                    dirpath='val/DSC',
                    filename='{epoch:03d}-loss={val/loss:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                    monitor='val/DSC',
                    save_top_k=5,
                    mode='max',
                    save_last=False,
                    every_n_epochs=1
                ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/HD95',
                #     filename='{epoch:03d}-loss={val/loss:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/HD95',
                #     save_top_k=5,
                #     mode='min',
                #     save_last=False,
                #     every_n_epochs=1
                # ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/NSD',
                #     filename='{epoch:03d}-NSD={val/NSD:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/NSD',
                #     save_top_k=5,
                #     mode='max',
                #     save_last=False,
                #     every_n_epochs=1
                # ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/Acc',
                #     filename='{epoch:03d}-Acc={val/Acc:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/Acc',
                #     save_top_k=5,
                #     mode='max',
                #     save_last=False,
                #     every_n_epochs=1
                # ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/Prec',
                #     filename='{epoch:03d}-Prec={val/Prec:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/Prec',
                #     save_top_k=5,
                #     mode='max',
                #     save_last=False,
                #     every_n_epochs=1
                # ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/Spec',
                #     filename='{epoch:03d}-Spec={val/Spec:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/Spec',
                #     save_top_k=5,
                #     mode='max',
                #     save_last=False,
                #     every_n_epochs=1
                # ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/Recall',
                #     filename='{epoch:03d}-Recall={val/Recall:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/Recall',
                #     save_top_k=5,
                #     mode='max',
                #     save_last=False,
                #     every_n_epochs=1
                # ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/F1',
                #     filename='{epoch:03d}-F1={val/F1:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/F1',
                #     save_top_k=5,
                #     mode='max',
                #     save_last=False,
                #     every_n_epochs=1
                # ),
                # ConfigCallbackModelCheckpoint(
                #     dirpath='val/AUROC',
                #     filename='{epoch:03d}-AUROC={val/AUROC:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                #     monitor='val/AUROC',
                #     save_top_k=5,
                #     mode='max',
                #     save_last=False,
                #     every_n_epochs=1
                # )
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
    def default_config_trainer() -> ConfigTrainerSegmentationDefault:
        return ConfigTrainerSegmentationDefault(
            experiment_root_dir=ParserSegmentationDefault.default_experiment_root_dir(),
            experiment_name=ParserSegmentationDefault.default_experiment_name(),
            experiment_version=ParserSegmentationDefault.default_experiment_version(),
            trainer_init_args=ParserSegmentationDefault.default_trainer_init_args(),
            callback_init_args=ParserSegmentationDefault.default_callback_init_args(),
            logger_init_args=ParserSegmentationDefault.default_logger_init_args()
        )

    @staticmethod
    def default_train_data_manifest_retriever() -> ConfigDatasetManifestRetrieverSegmentationDefault:
        return ConfigDatasetManifestRetrieverSegmentationDefault(
            root_dir='Samples',
            manifest_file='Samples/split/split_train.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_ROI': 'mask_1',
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None
        )

    @staticmethod
    def default_train_config_dataset() -> ConfigDatasetBase:
        return ConfigDatasetPersistent()

    @staticmethod
    def default_train_config_transform(
            roi_size: Tuple[int, int, int] = (64, 64, 64),
            num_classes: int = 2,
            crop_per_sample: int = 2
    ) -> ConfigTransformSegmentationDefaultTrain:
        return ConfigTransformSegmentationDefaultTrain(
            volume_key='volume',
            mask_key='mask',
            param_volume_tf_duplicate_items_dup_keys_volume=None,
            param_mask_tf_duplicate_items_dup_keys_mask=None,
            param_tf_spacing_pixdim=(1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume=GridSampleMode.BILINEAR,
            param_tf_spacing_mode_mask=GridSampleMode.NEAREST,
            param_tf_padding_mode_volume=GridSamplePadMode.BORDER,
            param_tf_padding_mode_mask=GridSamplePadMode.BORDER,
            param_tf_spatial_pad_spatial_size=roi_size,
            param_tf_spatial_pad_mode=PytorchPadMode.REPLICATE,
            param_tf_rand_crop_by_label_classes_spatial_size=roi_size,
            param_tf_rand_crop_by_label_classes_ratios=[0.0] + [1.0] * (num_classes - 1),
            param_tf_rand_crop_by_label_classes_num_classes=num_classes,
            param_tf_rand_crop_by_label_classes_num_samples=crop_per_sample,
            param_tf_scale_intensity_range_a_min=-1000,
            param_tf_scale_intensity_range_a_max=1000,
            param_tf_scale_intensity_range_b_min=0.0,
            param_tf_scale_intensity_range_b_max=1.0,
            param_tf_scale_intensity_range_clip=True,
            param_tf_allow_missing_keys=False,
            random_seed=0
        )

    @staticmethod
    def default_data_module_train_init_args(
            roi_size: Tuple[int, int, int] = (64, 64, 64),
            num_classes: int = 2,
            crop_per_sample: int = 2,
            batch_size: int = 2
    ) -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            config_retriever=ParserSegmentationDefault.default_train_data_manifest_retriever(),
            config_dataset=ParserSegmentationDefault.default_train_config_dataset(),
            config_transform=ParserSegmentationDefault.default_train_config_transform(
                roi_size,
                num_classes,
                crop_per_sample
            ),
            batch_size=batch_size,
            shuffle=True,
            # Multiprocessing note:
            # Keep in mind that there shall occur worker thread unexpected exiting if you wrap PersistentDataset with monai.DataLoader
            # and set num_workers > 0 for the first run, which might caused by cache file writing issues.
            # It is doubtful whether this error has something to do with CUDA cooperation, because I only encounter this error when
            # using GPU devices while setting num_workers > 0 for the first run.
            # To handle this:
            # You may first set num_workers = 0 to run a pre-epoch which generates all cache files, and then launch formal experiments.
            # After all cache files are generated, set num_workers > 0 won't encounter errors anymore.
            #
            # If you modify transform parameters which affects cache (i.e., all non-randomness transforms before the first random transform),
            # then you should delete old cache files and regenerate new ones.
            # Alternatively, you may use CacheDataset or torch Dataset to avoid these matters, at the cost of efficiency loss.
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
    def default_val_data_manifest_retriever() -> ConfigDatasetManifestRetrieverSegmentationDefault:
        return ConfigDatasetManifestRetrieverSegmentationDefault(
            root_dir='Samples',
            manifest_file='Samples/split/split_val.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_ROI': 'mask_1'
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None
        )

    @staticmethod
    def default_val_config_dataset() -> ConfigDatasetBase:
        return ConfigDatasetPersistent()

    @staticmethod
    def default_val_config_transform() -> ConfigTransformSegmentationDefaultInferencePre:
        return ConfigTransformSegmentationDefaultInferencePre(
            volume_key='volume',
            mask_key='mask',
            param_volume_tf_duplicate_items_dup_keys_volume='volume_raw',
            param_mask_tf_duplicate_items_dup_keys_mask='mask_raw',
            param_tf_spacing_pixdim=(1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume=GridSampleMode.BILINEAR,
            param_tf_spacing_mode_mask=GridSampleMode.NEAREST,
            param_tf_padding_mode_volume=GridSamplePadMode.BORDER,
            param_tf_padding_mode_mask=GridSamplePadMode.BORDER,
            param_tf_scale_intensity_range_a_min=-1000,
            param_tf_scale_intensity_range_a_max=1000,
            param_tf_scale_intensity_range_b_min=0.0,
            param_tf_scale_intensity_range_b_max=1.0,
            param_tf_scale_intensity_range_clip=True
        )

    @staticmethod
    def default_data_module_val_init_args() -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            config_retriever=ParserSegmentationDefault.default_val_data_manifest_retriever(),
            config_dataset=ParserSegmentationDefault.default_val_config_dataset(),
            config_transform=ParserSegmentationDefault.default_val_config_transform(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def default_test_data_manifest_retriever() -> ConfigDatasetManifestRetrieverSegmentationDefault:
        return ConfigDatasetManifestRetrieverSegmentationDefault(
            root_dir='Samples',
            manifest_file='Samples/split/split_test.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_ROI': 'mask_1'
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None
        )

    @staticmethod
    def default_test_config_dataset() -> ConfigDatasetBase:
        return ConfigDatasetPersistent()

    @staticmethod
    def default_test_config_transform() -> ConfigTransformSegmentationDefaultInferencePre:
        return ConfigTransformSegmentationDefaultInferencePre(
            volume_key='volume',
            mask_key='mask',
            param_volume_tf_duplicate_items_dup_keys_volume='volume_raw',
            param_mask_tf_duplicate_items_dup_keys_mask='mask_raw',
            param_tf_spacing_pixdim=(1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume=GridSampleMode.BILINEAR,
            param_tf_spacing_mode_mask=GridSampleMode.NEAREST,
            param_tf_padding_mode_volume=GridSamplePadMode.BORDER,
            param_tf_padding_mode_mask=GridSamplePadMode.BORDER,
            param_tf_scale_intensity_range_a_min=-1000,
            param_tf_scale_intensity_range_a_max=1000,
            param_tf_scale_intensity_range_b_min=0.0,
            param_tf_scale_intensity_range_b_max=1.0,
            param_tf_scale_intensity_range_clip=True
        )

    @staticmethod
    def default_data_module_test_init_args() -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            config_retriever=ParserSegmentationDefault.default_test_data_manifest_retriever(),
            config_dataset=ParserSegmentationDefault.default_test_config_dataset(),
            config_transform=ParserSegmentationDefault.default_test_config_transform(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def default_predict_data_manifest_retriever() -> ConfigDatasetManifestRetrieverSegmentationDefault:
        return ConfigDatasetManifestRetrieverSegmentationDefault(
            root_dir='Samples',
            manifest_file='Samples/split/split_predict.xlsx',
            column_key_map={
                'volume': 'volume'
            },
            column_key_relative_path=['volume'],
            column_group_map={'volume': ['volume']},
            column_dtype_map=None
        )

    @staticmethod
    def default_predict_config_dataset() -> ConfigDatasetBase:
        return ConfigDatasetPersistent()

    @staticmethod
    def default_predict_config_transform() -> ConfigTransformSegmentationDefaultInferencePre:
        return ConfigTransformSegmentationDefaultInferencePre(
            volume_key='volume',
            param_volume_tf_duplicate_items_dup_keys_volume='volume_raw',
            param_tf_spacing_pixdim=(1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume=GridSampleMode.BILINEAR,
            param_tf_padding_mode_volume=GridSamplePadMode.BORDER,
            param_tf_scale_intensity_range_a_min=-1000,
            param_tf_scale_intensity_range_a_max=1000,
            param_tf_scale_intensity_range_b_min=0.0,
            param_tf_scale_intensity_range_b_max=1.0,
            param_tf_scale_intensity_range_clip=True
        )

    @staticmethod
    def default_data_module_predict_init_args() -> DataModuleSegmentationDefaultInitArgs:
        return DataModuleSegmentationDefaultInitArgs(
            config_retriever=ParserSegmentationDefault.default_predict_data_manifest_retriever(),
            config_dataset=ParserSegmentationDefault.default_predict_config_dataset(),
            config_transform=ParserSegmentationDefault.default_predict_config_transform(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def default_config_data_module(
            roi_size: Tuple[int, int, int] = (64, 64, 64),
            num_classes: int = 2,
            crop_per_sample: int = 2,
            batch_size: int = 2
    ) -> ConfigDataModuleSegmentationDefault:
        return ConfigDataModuleSegmentationDefault(
            train_init_args=ParserSegmentationDefault.default_data_module_train_init_args(
                roi_size,
                num_classes,
                crop_per_sample,
                batch_size
            ),
            val_init_args=ParserSegmentationDefault.default_data_module_val_init_args(),
            test_init_args=ParserSegmentationDefault.default_data_module_test_init_args(),
            predict_init_args=ParserSegmentationDefault.default_data_module_predict_init_args()
        )

    @staticmethod
    def default_network_init_args(num_sequence: int = 1, num_classes: int = 2) -> NamedNetworkInitArgs:
        return NamedNetworkInitArgs(
            name='UNet',
            config_network=ConfigNetworkUNet(
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
                classifier_out_channels=num_classes,  # Assume (C=num_classes) classes (background & C-1 foreground)
                reserve_io=False  # Do not reserve io, except for indepth inspecting
            ),
            description_info=f'Basic background/{num_classes}-foreground segmentation UNet'
        )

    @staticmethod
    def default_train_config_inferer() -> ConfigInfererBase:
        return ConfigInfererSimple()

    @staticmethod
    def default_train_metric_init_args_collection(num_classes: int) -> TLSeq[NamedMetricInitArgs]:
        return [
            # Dice Similarity Coefficient
            NamedMetricInitArgs(
                name='train/DSC',
                config_metric=Dice(
                    include_background=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=num_classes,
                    return_with_label=False
                ),
                description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                reduce_fx='mean'
            ),
            # # Normalized Surface Dice
            # NamedMetricInitArgs(
            #     name='train/NSD',
            #     config_metric=NSD(
            #         # Tolerance of at most 3.0 distance error in index space
            #         # First threshold is for background, this is nonsense in case background is excluded
            #         class_thresholds=[3.] * (num_classes - 1),
            #         include_background=False,
            #         distance_metric='euclidean',
            #         reduction=MetricReduction.MEAN,
            #         get_not_nans=False,
            #         use_subvoxels=False
            #     ),
            #     description_info='Normalized surface Dice metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=True,
            #     reduce_fx='mean'
            # ),
            # # 95% percentile Hausdorff Distance
            # NamedMetricInitArgs(
            #     name='train/HD95',
            #     config_metric=HD(
            #         include_background=False,
            #         distance_metric='euclidean',
            #         percentile=95.0,
            #         directed=False,
            #         reduction=MetricReduction.MEAN,
            #         get_not_nans=False
            #     ),
            #     description_info='95% percentile Hausdorff distance metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=True,
            #     reduce_fx='mean'
            # ),
            # Confusion Matrix
            # For Binary, CM is M(2*2): [[TN,FP],[FN,TP]]
            # For Multi-class, CM is M(num_classes*num_classes): E[i,j] denotes the i-th gt class is predicted as j-th class
            NamedMetricInitArgs(
                name='train/ConfMat',  # Nonsense, handled by postprocess_metric_func, which will return a dict
                config_metric=MCCM(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    normalize='none'
                ),
                description_info='Confusion Matrix for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                postprocess_metric_func=ConfigOperatorTensorRemapConfMat(
                    'train',
                    'ConfMat',
                    ((0, 'gt'), (1, 'pred'))
                ),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx='sum'  # Elements shall be summed up
            ),
            # # Classification global metrics
            # # Acc: Multi-class calculation shall always accumulate all classes
            # # Prec Recall Spec F1 AUROC: Shall keep metrics per class, and do post reduce as per class metrics
            # NamedMetricInitArgs(
            #     name='train/Acc',
            #     config_metric=MCACC(  # Accuracy shall calculate across all classes
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='micro',
            #         multidim_average='global'
            #     ),
            #     description_info='Accuracy metric for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            NamedMetricInitArgs(
                name='train/Prec',
                config_metric=MCPREC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Precision metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx='mean'
            ),
            NamedMetricInitArgs(
                name='train/Recall',
                config_metric=MCRECALL(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Recall metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx='mean'
            ),
            # NamedMetricInitArgs(
            #     name='train/Spec',
            #     config_metric=MCSPEC(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         multidim_average='global',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='Specificity metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='train/F1',
            #     config_metric=MCF1(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         multidim_average='global',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='F1-Score metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='train/AUROC',
            #     config_metric=MCAUROC(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='AUROC metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessTorchSoftmax(dim=1),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='train/VPS',
            #     config_metric=VPS(),
            #     description_info='Voxel Processing Per Second metric',
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # )
        ]

    @staticmethod
    def default_config_loss() -> ConfigLossDeepSupervisionDiceCE:
        return ConfigLossDeepSupervisionDiceCE(
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
        )

    @staticmethod
    def default_config_optimizer() -> ConfigOptimizerAdamW:
        return ConfigOptimizerAdamW(
            lr=1e-4,  # May be overwritten by LRScheduler
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False
        )

    @staticmethod
    def default_config_lr_scheduler() -> ConfigLRSchedulerOneCycle:
        return ConfigLRSchedulerOneCycle(
            max_lr=0.01,
            # Set total_steps to None, so as to infer from epochs * steps_per_epoch,
            # otherwise set both epochs & steps_per_epoch to None, and directly specify total_steps
            total_steps=None,
            epochs=100,
            # Practically, steps_per_epoch can be inferred from len(Dataloader),
            # but it may not be available all the time, in case that Dataloader do not report len(),
            # and for Trainer routine, you shall initialize the LR-Scheduler before getting Dataloader
            # Anyway, please specify it manually, it's your responsibility!
            steps_per_epoch=5,
            pct_start=0.3,  # Increasing part occupies the first 30% steps
            div_factor=25,
            final_div_factor=1e4
        )

    @staticmethod
    def default_module_training_step_addition_args(num_classes: int = 2) -> ModuleTrainingStepAdditionArgs:
        return ModuleTrainingStepAdditionArgs(
            config_inferer=ParserSegmentationDefault.default_train_config_inferer(),
            metric_init_args_collection= \
                ParserSegmentationDefault.default_train_metric_init_args_collection(num_classes),
            loss_init_args=NamedLossInitArgs(
                name='train/loss',
                config_loss=ParserSegmentationDefault.default_config_loss(),
                description_info='Dice + Cross Entropy compounded loss for deep supervision',
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                reduce_fx='mean'
            ),
            optimizer_init_args=NamedOptimizerInitArgs(
                name='AdamW',
                config_optimizer=ParserSegmentationDefault.default_config_optimizer(),
                description_info='AdamW optimizer'
            ),
            lr_scheduler_init_args=NamedLRSchedulerInitArgs(
                name='OneCycleLR',
                config_lr_scheduler=ParserSegmentationDefault.default_config_lr_scheduler(),  # Shall step() per batch
                config_lr_scheduler_ltn_control=LRSchedulerLightningConfig(interval='step', monitor='val/loss'),
                description_info='OneCycleLR scheduler'
            ),
            volume_key='volume',
            mask_key='mask',
            # hook_functions=[ConfigOperatorHookDisplayDictKeys(('Train', 'Step returns'))]
        )

    @staticmethod
    def default_val_config_inferer() -> ConfigInfererBase:
        return ConfigInfererMainWithAuxSlidingWindow(
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            overlap=0.5,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            padding_mode=PytorchPadMode.REPLICATE,
            progress=True
        )

    @staticmethod
    def default_val_metric_init_args_collection(num_classes: int) -> TLSeq[NamedMetricInitArgs]:
        return [
            # Dice Similarity Coefficient
            NamedMetricInitArgs(
                name='val/DSC',
                config_metric=Dice(
                    include_background=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=num_classes,
                    return_with_label=False
                ),
                description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
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
            NamedMetricInitArgs(
                name='val-class-wise/DSC',
                config_metric=Dice(
                    include_background=True,
                    reduction=MetricReduction.MEAN_BATCH,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=num_classes,
                    return_with_label=False
                ),
                description_info='Class-wise Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                postprocess_metric_func=ConfigOperatorTensorRemapClassWise(
                    'val-class-wise',
                    'DSC',
                    True
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            # # Normalized Surface Dice
            # NamedMetricInitArgs(
            #     name='val/NSD',
            #     config_metric=NSD(
            #         # Tolerance of at most 3.0 distance error in index space
            #         # First threshold is for background, this is nonsense in case background is excluded
            #         class_thresholds=[3.] * (num_classes - 1),
            #         include_background=False,
            #         distance_metric='euclidean',
            #         reduction=MetricReduction.MEAN,
            #         get_not_nans=False,
            #         use_subvoxels=False
            #     ),
            #     description_info='Normalized surface dice metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=True,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='val-class-wise/NSD',
            #     config_metric=NSD(
            #         # Tolerance of at most 3.0 distance error in index space
            #         # First threshold is for background, this is nonsense in case background is excluded
            #         class_thresholds=[3.] * num_classes,
            #         include_background=True,
            #         distance_metric='euclidean',
            #         reduction=MetricReduction.MEAN_BATCH,
            #         get_not_nans=False,
            #         use_subvoxels=False
            #     ),
            #     description_info='Class-wise Normalized surface dice metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     postprocess_metric_func=ConfigOperatorTensorRemapClassWise(
            #         'val-class-wise',
            #         'NSD',
            #         True
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # # 95% percentile Hausdorff Distance
            # NamedMetricInitArgs(
            #     name='val/HD95',
            #     config_metric=HD(
            #         include_background=False,
            #         distance_metric='euclidean',
            #         percentile=95.0,
            #         directed=False,
            #         reduction=MetricReduction.MEAN,
            #         get_not_nans=False
            #     ),
            #     description_info='95% percentile Hausdorff distance metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=True,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='val-class-wise/HD95',
            #     config_metric=HD(
            #         include_background=True,
            #         distance_metric='euclidean',
            #         percentile=95.0,
            #         directed=False,
            #         reduction=MetricReduction.MEAN_BATCH,
            #         get_not_nans=False
            #     ),
            #     description_info='Class-wise 95% percentile Hausdorff distance metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True,
            #         to_onehot=num_classes,
            #         dim=1,
            #         dtype=torch.int
            #     ),
            #     postprocess_metric_func=ConfigOperatorTensorRemapClassWise(
            #         'val-class-wise',
            #         'HD95',
            #         True
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # # Confusion Matrix
            # # For Binary, CM is M(2*2): [[TN,FP],[FN,TP]]
            # # For Multi-class, CM is M(num_classes*num_classes): E[i,j] denotes the i-th gt class is predicted as j-th class
            # NamedMetricInitArgs(
            #     name='val/ConfMat',  # Nonsense, handled by postprocess_metric_func, which will return a dict
            #     config_metric=MCCM(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         normalize='none'
            #     ),
            #     description_info='Confusion Matrix for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     postprocess_metric_func=ConfigOperatorTensorRemapConfMat(
            #         'val',
            #         'ConfMat',
            #         ((0, 'gt'), (1, 'pred'))
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='sum'  # Elements shall be summed up
            # ),
            # Classification global metrics
            # Acc: Multi-class calculation shall always accumulate all classes
            # Prec Recall Spec F1 AUROC: Shall keep metrics per class, and do post reduce as per class metrics
            # NamedMetricInitArgs(
            #     name='val/Acc',
            #     config_metric=MCACC(  # Accuracy shall calculate across all classes
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='micro',
            #         multidim_average='global'
            #     ),
            #     description_info='Accuracy metric for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='val/Prec',
            #     config_metric=MCPREC(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         multidim_average='global',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='Precision metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='val/Recall',
            #     config_metric=MCRECALL(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         multidim_average='global',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='Recall metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='val/Spec',
            #     config_metric=MCSPEC(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         multidim_average='global',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='Specificity metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='val/F1',
            #     config_metric=MCF1(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         multidim_average='global',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='F1-Score metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            # NamedMetricInitArgs(
            #     name='val/AUROC',
            #     config_metric=MCAUROC(
            #         num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
            #         average='macro',
            #         ignore_index=0  # Ignoring background
            #     ),
            #     description_info='AUROC metric (ignoring background) '
            #                      'for multi-class (organs not overlapped) segmentation',
            #     preprocess_pred_func=ConfigOperatorTensorProcessTorchSoftmax(dim=1),
            #     preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
            #         argmax=True, dim=1, dtype=torch.int, keepdim=False
            #     ),
            #     on_step=True,
            #     on_epoch=True,
            #     prog_bar=False,
            #     reduce_fx='mean'
            # ),
            NamedMetricInitArgs(
                name='val/VPS',
                config_metric=VPS(),
                description_info='Voxel Processing Per Second metric',
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            )
        ]

    @staticmethod
    def default_module_validation_step_addition_args(num_classes: int = 2) -> ModuleValidationStepAdditionArgs:
        return ModuleValidationStepAdditionArgs(
            config_inferer=ParserSegmentationDefault.default_val_config_inferer(),
            metric_init_args_collection= \
                ParserSegmentationDefault.default_val_metric_init_args_collection(num_classes),
            loss_init_args=NamedLossInitArgs(
                name='val/loss',
                config_loss=ParserSegmentationDefault.default_config_loss(),
                description_info='Dice + Cross Entropy compounded loss for deep supervision',
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                reduce_fx='mean'
            ),
            volume_key='volume',
            mask_key='mask',
            # hook_functions=[ConfigOperatorHookDisplayDictKeys(('Val', 'Step returns'))]
        )

    @staticmethod
    def default_test_config_inferer() -> ConfigInfererBase:
        return ConfigInfererMainWithAuxSlidingWindow(
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            overlap=0.5,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            padding_mode=PytorchPadMode.REPLICATE,
            progress=True
        )

    @staticmethod
    def default_test_metric_init_args_collection(num_classes: int) -> TLSeq[NamedMetricInitArgs]:
        return [
            # Dice Similarity Coefficient
            NamedMetricInitArgs(
                name='test/DSC',
                config_metric=Dice(
                    include_background=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=num_classes,
                    return_with_label=False
                ),
                description_info='Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
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
            NamedMetricInitArgs(
                name='test-class-wise/DSC',
                config_metric=Dice(
                    include_background=True,
                    reduction=MetricReduction.MEAN_BATCH,
                    get_not_nans=False,
                    ignore_empty=True,
                    num_classes=num_classes,
                    return_with_label=False
                ),
                description_info='Class-wise Dice similarity coefficient (also known as DSC) metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                postprocess_metric_func=ConfigOperatorTensorRemapClassWise(
                    'test-class-wise',
                    'DSC',
                    True
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            # Normalized Surface Dice
            NamedMetricInitArgs(
                name='test/NSD',
                config_metric=NSD(
                    # Tolerance of at most 3.0 distance error in index space
                    # First threshold is for background, this is nonsense in case background is excluded
                    class_thresholds=[3.] * (num_classes - 1),
                    include_background=False,
                    distance_metric='euclidean',
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False,
                    use_subvoxels=False
                ),
                description_info='Normalized surface dice metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
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
            NamedMetricInitArgs(
                name='test-class-wise/NSD',
                config_metric=NSD(
                    # Tolerance of at most 3.0 distance error in index space
                    class_thresholds=[3.] * num_classes,
                    include_background=True,
                    distance_metric='euclidean',
                    reduction=MetricReduction.MEAN_BATCH,
                    get_not_nans=False,
                    use_subvoxels=False
                ),
                description_info='Class-wise Normalized surface dice metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                postprocess_metric_func=ConfigOperatorTensorRemapClassWise(
                    'test-class-wise',
                    'NSD',
                    True
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            # 95% percentile Hausdorff Distance
            NamedMetricInitArgs(
                name='test/HD95',
                config_metric=HD(
                    include_background=False,
                    distance_metric='euclidean',
                    percentile=95.0,
                    directed=False,
                    reduction=MetricReduction.MEAN,
                    get_not_nans=False
                ),
                description_info='95% percentile Hausdorff distance metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
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
            NamedMetricInitArgs(
                name='test-class-wise/HD95',
                config_metric=HD(
                    include_background=True,
                    distance_metric='euclidean',
                    percentile=95.0,
                    directed=False,
                    reduction=MetricReduction.MEAN_BATCH,
                    get_not_nans=False
                ),
                description_info='Class-wise 95% percentile Hausdorff distance metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True,
                    to_onehot=num_classes,
                    dim=1,
                    dtype=torch.int
                ),
                postprocess_metric_func=ConfigOperatorTensorRemapClassWise(
                    'test-class-wise',
                    'HD95',
                    True
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            # Confusion Matrix
            # For Binary, CM is M(2*2): [[TN,FP],[FN,TP]]
            # For Multi-class, CM is M(num_classes*num_classes): E[i,j] denotes the i-th gt class is predicted as j-th class
            NamedMetricInitArgs(
                name='test/ConfMat',  # Nonsense, handled by postprocess_metric_func, which will return a dict
                config_metric=MCCM(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    normalize='none'
                ),
                description_info='Confusion Matrix for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                postprocess_metric_func=ConfigOperatorTensorRemapConfMat(
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
                config_metric=MCACC(  # Accuracy shall calculate across all classes
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='micro',
                    multidim_average='global'
                ),
                description_info='Accuracy metric for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            NamedMetricInitArgs(
                name='test/Prec',
                config_metric=MCPREC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Precision metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            NamedMetricInitArgs(
                name='test/Recall',
                config_metric=MCRECALL(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Recall metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            NamedMetricInitArgs(
                name='test/Spec',
                config_metric=MCSPEC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='Specificity metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            NamedMetricInitArgs(
                name='test/F1',
                config_metric=MCF1(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    multidim_average='global',
                    ignore_index=0  # Ignoring background
                ),
                description_info='F1-Score metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            NamedMetricInitArgs(
                name='test/AUROC',
                config_metric=MCAUROC(
                    num_classes=num_classes,  # Assume N classes (background & N-1 foregrounds)
                    average='macro',
                    ignore_index=0  # Ignoring background
                ),
                description_info='AUROC metric (ignoring background) '
                                 'for multi-class (organs not overlapped) segmentation',
                preprocess_pred_func=ConfigOperatorTensorProcessTorchSoftmax(dim=1),
                preprocess_gt_func=ConfigOperatorTensorProcessMonaiAsDiscrete(
                    argmax=True, dim=1, dtype=torch.int, keepdim=False
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                reduce_fx='mean'
            ),
            NamedMetricInitArgs(
                name='test/VPS',
                config_metric=VPS(),
                description_info='Voxel Processing Per Second metric',
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            )
        ]

    @staticmethod
    def default_module_test_step_addition_args(num_classes: int = 2) -> ModuleTestStepAdditionArgs:
        return ModuleTestStepAdditionArgs(
            config_inferer=ParserSegmentationDefault.default_test_config_inferer(),
            metric_init_args_collection=ParserSegmentationDefault.default_test_metric_init_args_collection(num_classes),
            loss_init_args=NamedLossInitArgs(
                name='test/loss',
                config_loss=ParserSegmentationDefault.default_config_loss(),
                description_info='Dice + Cross Entropy compounded loss for deep supervision',
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                reduce_fx='mean'
            ),
            volume_key='volume',
            mask_key='mask',
            hook_functions=[
                # ConfigOperatorHookDisplayDictKeys(('Test', 'Step returns')),
            ]
        )

    @staticmethod
    def default_predict_config_inferer() -> ConfigInfererBase:
        return ConfigInfererMainWithAuxSlidingWindow(
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            overlap=0.5,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            padding_mode=PytorchPadMode.REPLICATE,
            progress=True
        )

    @staticmethod
    def default_predict_metric_init_args_collection() -> TLSeq[NamedMetricInitArgs]:
        return [
            NamedMetricInitArgs(
                name='predict/VPS',
                config_metric=VPS(),
                description_info='Voxel Processing Per Second metric',
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                reduce_fx='mean'
            )
        ]

    @staticmethod
    def default_module_predict_step_addition_args() -> ModulePredictStepAdditionArgs:
        return ModulePredictStepAdditionArgs(
            config_inferer=ParserSegmentationDefault.default_predict_config_inferer(),
            metric_init_args_collection=ParserSegmentationDefault.default_predict_metric_init_args_collection(),
            volume_key='volume',
            # hook_functions=[ConfigOperatorHookDisplayDictKeys(('Predict', 'Step returns'))]
        )

    @staticmethod
    def default_config_ltn_module(
            num_sequence: int = 1,
            num_classes: int = 2
    ) -> ConfigLightningModuleSegmentationDefault:
        return ConfigLightningModuleSegmentationDefault(
            network_init_args=ParserSegmentationDefault.default_network_init_args(num_sequence, num_classes),
            module_training_step_addition_args= \
                ParserSegmentationDefault.default_module_training_step_addition_args(num_classes),
            module_validation_step_addition_args= \
                ParserSegmentationDefault.default_module_validation_step_addition_args(num_classes),
            module_test_step_addition_args= \
                ParserSegmentationDefault.default_module_test_step_addition_args(num_classes),
            module_predict_step_addition_args= \
                ParserSegmentationDefault.default_module_predict_step_addition_args()
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
        config_trainer: ConfigTrainerSegmentationDefault = ParserSegmentationDefault.default_config_trainer()
        config_data_module: ConfigDataModuleSegmentationDefault = ParserSegmentationDefault.default_config_data_module()
        config_ltn_module: ConfigLightningModuleSegmentationDefault = \
            ParserSegmentationDefault.default_config_ltn_module()
        export_dict: Dict[str, Any] = {
            'config_trainer': config_trainer,
            'config_data_module': config_data_module,
            'config_ltn_module': config_ltn_module
        }

        if export_path is not None:
            # Dump config to YAML file
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
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
            print(f'Example config of {ParserSegmentationDefault.__module__}'
                  f'.{ParserSegmentationDefault.__name__} is saved to {export_path}.')
            return None
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

    # Dataclass Attributes: Trainer, DataModule and LightningModule
    config_trainer: ConfigTrainerSegmentationDefault = ConfigTrainerSegmentationDefault()
    config_data_module: ConfigDataModuleSegmentationDefault = ConfigDataModuleSegmentationDefault()
    config_ltn_module: ConfigLightningModuleSegmentationDefault = ConfigLightningModuleSegmentationDefault()

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
            'config_trainer': self.config_trainer,
            'config_data_module': self.config_data_module,
            'config_ltn_module': self.config_ltn_module
        }

        if export_path is not None:
            # Dump config to YAML file, return None
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
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

        self.config_trainer = import_dict['config_trainer']
        self.config_data_module = import_dict['config_data_module']
        self.config_ltn_module = import_dict['config_ltn_module']

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
