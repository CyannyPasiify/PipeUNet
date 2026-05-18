# -*- coding: utf-8 -*-
import copy
import os
import torch
from pathlib import Path
from lightning import Trainer, LightningDataModule, LightningModule
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE
from typing import Optional, Dict, Any, Union, List, Callable
from typing_extensions import override
from DataModule.data_module_configurer import ConfigDataModuleBase, ConfigDataModuleSegmentationDefault
from Dataset.dataset_configurer import (
    ConfigDatasetCache,
    ConfigDatasetPersistent,
    ConfigDatasetLMDB
)
from Inferer.inferer_configurer import ConfigInfererMainWithAuxSlidingWindow
from LRScheduler.lrscheduler_configurer import ConfigLRSchedulerOneCycle
from Module.ltn_module_configurer import ConfigLightningModuleBase, ConfigLightningModuleSegmentationDefault
from Operator import (
    ConfigOperatorHookStepExportMulticlassPredWithMaskResults,
    ConfigOperatorHookStepExportMulticlassPredOnlyResults
)
from Trainer.trainer_configurer import (
    ConfigTrainerSegmentationDefault,
    ConfigTrainerBase
)
from Launcher.Parser.parser_segmentation_default import ParserSegmentationDefault
from Launcher.launcher_ABC import LauncherABC
from dataclasses import dataclass, field

from Transform.transform_configurer import ConfigTransformSegmentationDefaultTrain

# Torch Settings
torch.set_float32_matmul_precision('medium')

@dataclass
class LauncherSegmentationDefault(LauncherABC):
    config_trainer: ConfigTrainerBase = field(default_factory=ConfigTrainerSegmentationDefault)
    config_data_module: ConfigDataModuleBase = field(default_factory=ConfigDataModuleSegmentationDefault)
    config_ltn_module: ConfigLightningModuleBase = field(default_factory=ConfigLightningModuleSegmentationDefault)

    @override
    def init_essentials(self) -> 'LauncherSegmentationDefault':
        self.trainer: Trainer = self.config_trainer.get_trainer()
        self.data_module: LightningDataModule = self.config_data_module.get_data_module()
        self.ltn_module: LightningModule = self.config_ltn_module.get_ltn_module()

        # Mark as ready
        self._is_ready = True
        print(f'{self.__class__.__name__} initialized. Available phases {self.ltn_module.get_available_phases()}')
        return self

    @override
    def fit(self, checkpoint: Optional[Union[str, os.PathLike, Path]] = None) -> Dict[str, Any]:
        self._assert_init_essentials()
        return self.config_trainer.fit(
            model=self.ltn_module,
            datamodule=self.data_module,
            ckpt_path=checkpoint  # You may use checkpoint to resume training.
        )

    @override
    def finetune(
            self,
            checkpoint: Optional[Union[str, os.PathLike, Path]],
            finetune_map_location: _MAP_LOCATION_TYPE = None,
            finetune_hparams_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        self._assert_init_essentials()
        return self.config_trainer.fit(
            model=self.ltn_module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            # Checkpoint is loaded at the beginning for initialization
            ckpt_path=str(checkpoint) if checkpoint is not None else None,
            finetune=True,
            finetune_map_location=finetune_map_location,
            finetune_hparams_file=finetune_hparams_file
        )

    @override
    def validation(
            self,
            checkpoint: Optional[Union[str, os.PathLike, Path]]
    ) -> Dict[str, Any]:
        self._assert_init_essentials()
        return self.config_trainer.validate(
            model=self.ltn_module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            ckpt_path=str(checkpoint) if checkpoint is not None else None
        )

    @override
    def test(
            self,
            checkpoint: Optional[Union[str, os.PathLike, Path]]
    ) -> Dict[str, Any]:
        self._assert_init_essentials()
        return self.config_trainer.test(
            model=self.ltn_module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            ckpt_path=str(checkpoint) if checkpoint is not None else None
        )

    @override
    def predict(
            self,
            checkpoint: Optional[Union[str, os.PathLike, Path]]
    ) -> Dict[str, Any]:
        self._assert_init_essentials()
        return self.config_trainer.predict(
            model=self.ltn_module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            ckpt_path=str(checkpoint) if checkpoint is not None else None
        )

    @override
    def detect(self) -> Dict[str, int]:
        self._assert_init_essentials()
        self.data_module.prepare_data()

        from DataModule.data_module_segmentation_default import DataModuleSegmentationDefault
        from typing import cast
        dm: DataModuleSegmentationDefault = cast(DataModuleSegmentationDefault, self.data_module)
        dm.setup('fit')
        steps_per_epoch_d: Dict[str, int] = {}
        phases: Dict[str, Callable] = {
            'train': dm.train_dataloader,
            'val': dm.val_dataloader,
        }
        for phase, get_loader in phases.items():
            if not phase in dm.datasets: continue
            steps_per_epoch_d[phase] = len(get_loader())

        print('Detected data loader steps_per_epoch:')
        for phase, length in steps_per_epoch_d.items():
            print(f'  {phase}: {length}')

        return steps_per_epoch_d


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands',
        help='additional help',
        dest='routine'
    )

    # Common: Experiment
    parser_common: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    parser_common.add_argument('-r', '--experiment_root_dir', type=str, required=True, help='experiment_root_dir')
    parser_common.add_argument('-e', '--experiment_name', type=str, required=True, help='experiment_name')
    parser_common.add_argument('-v', '--experiment_version', type=str, required=True, help='experiment_version')
    # Common: Routine
    parser_common.add_argument('--accelerator', type=str, choices=['cpu', 'gpu'], required=True, help='accelerator')
    parser_common.add_argument('--devices', type=int, nargs='+', required=True,
                               help='devices, gpu [multi], cpu single num')
    parser_common.add_argument('--deterministic', choices=['none', 'warn', 'true', 'false'], default='warn',
                               help='deterministic, \'warn\' try it best to be deterministic')
    # Common: Logger
    parser_common.add_argument('--wandb_project', type=str, default='PipeUNet', help='wandb_project')
    # Common: DataModule (specified main phase)
    # fit: train (val will have another set of args)
    # finetune: train (val will have another set of args)
    # validation: val
    # test: test
    # predict: predict
    parser_common.add_argument('--dataset_root_dir', type=str, required=True, help='dataset_root_dir')
    parser_common.add_argument('--dataset_manifest_file', type=str, required=True, help='manifest_file')
    parser_common.add_argument('--volume_keys', type=str, nargs='+', required=True, help='volume_keys')
    parser_common.add_argument('--mask_keys', type=str, nargs='*', required=True, help='mask_keys')
    parser_common.add_argument(
        '--cache_dir', type=str, default=None,
        help='cache_dir, absolute path, default at {experiment_root_dir}/{experiment_name}/{experiment_version}/cache'
    )
    parser_common.add_argument(
        '--roi_size', type=int, nargs=3, default=(64, 64, 64),
        help='roi_size for cropping or sliding window inference'
    )
    parser_common.add_argument('--num_workers', type=int, default=4, help='workers for data loading')
    parser_common.add_argument('--batch_size', type=int, default=1, help='batch_size, 1')
    # Common: Network
    parser_common.add_argument('--num_sequence', '--num_modality', type=int, required=True, help='num_sequence')
    parser_common.add_argument('--num_classes', type=int, required=True, help='num_classes')

    # region Fit
    parser_fit: argparse.ArgumentParser = subparsers.add_parser('fit', parents=[parser_common], help='fit help')
    parser_fit.add_argument('-ckpt', '--resume_checkpoint', type=str, default=None,
                            help='checkpoint for resuming training')
    # Fit: Trainer
    parser_fit.add_argument('--epochs', type=int, required=True, help='epochs, 100')
    parser_fit.add_argument('--accumulate_grad_batches', type=int, default=1, help='accumulate_grad_batches, 1')
    # Fit: Callback
    parser_fit.add_argument('--early_stopping', type=int, default=None,
                            help='early_stopping, if specified, it equals patience')
    # Fit: DataModule (train phase)
    parser_fit.add_argument('--crop_per_sample', type=int, default=1, help='Patch to crop from each sample')
    # Fit: DataModule (val phase)
    parser_fit.add_argument('--val_dataset_root_dir', type=str, required=True, help='dataset_root_dir')
    parser_fit.add_argument('--val_dataset_manifest_file', type=str, required=True, help='manifest_file')
    parser_fit.add_argument('--val_volume_keys', type=str, nargs='+', required=True, help='volume_keys')
    parser_fit.add_argument('--val_mask_keys', type=str, nargs='*', required=True, help='mask_keys')
    parser_fit.add_argument('--val_cache_dir', type=str, default=None,
                            help='cache_dir, absolute path, default at {experiment_root_dir}/{experiment_name}/{experiment_version}/cache')
    parser_fit.add_argument('--val_batch_size', type=int, default=1, help='batch_size, 1')
    # Fit: Optimizer & LR-Scheduler
    parser_fit.add_argument('--max_lr', type=float, default=0.01, help='OneCycleLR.max_lr')
    parser_fit.add_argument('--steps_per_epoch', type=int, required=True, help='OneCycleLR.steps_per_epoch')
    parser_fit.add_argument('--final_div_factor', type=float, default=1e4, help='OneCycleLR.final_div_factor')
    # Fit: val Inferer
    parser_fit.add_argument(
        '--val_roi_size', type=int, nargs=3, default=(32, 32, 32),
        help='roi_size for sliding window inference'
    )
    parser_fit.add_argument('--sw_batch_size', type=int, default=4, help='sw_batch_size for sliding window inference')
    parser_fit.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')
    # Fit: val export hook
    parser_fit.add_argument('--export_val_results', action='store_true', default=False)
    parser_fit.add_argument('--export_root_dir', type=str, default=None,
                            help='export_root_dir, absolute path, '
                                 'default at {experiment_root_dir}/{experiment_name}/{experiment_version}/hook_fit_val/exported_pred_results')
    parser_fit.add_argument('--id_keys', type=str, nargs="+", default=['ID'],
                            help='The columns to identify samples, usually it takes ID as primary key')
    parser_fit.add_argument('--combined_mask_key', type=str, default="mask", help='combined_mask_key')
    parser_fit.add_argument('--save_option', type=str, nargs="*",
                            choices=("volume", "mask", "pred", "diff"),
                            default=["mask", "pred", "diff"],
                            help='save_option:\n'
                                 ' volume: save the original volumes, this can be very large\n'
                                 ' mask: save the original masks (binary and combined)\n'
                                 ' pred: save the pred_softmax_logits, pred_mask (binary and combined)\n'
                                 ' diff: save diff mask derived by pred & gt mask'
                            )
    # endregion
    # region Finetune
    parser_finetune: argparse.ArgumentParser = subparsers.add_parser('finetune', parents=[parser_common],
                                                                     help='finetune help')
    parser_finetune.add_argument('-ckpt', '--init_checkpoint', type=str, default=None,
                                 help='checkpoint for initializing module')
    # Finetune: Trainer
    parser_finetune.add_argument('--epochs', type=int, required=True, help='epochs, 100')
    parser_finetune.add_argument('--accumulate_grad_batches', type=int, default=16, help='accumulate_grad_batches, 16')
    # Finetune: Callback
    parser_finetune.add_argument('--early_stopping', type=int, default=None,
                                 help='early_stopping, if specified, it equals patience')
    # Finetune: DataModule (train phase)
    parser_finetune.add_argument('--crop_per_sample', type=int, default=1, help='Patch to crop from each sample')
    # Finetune: DataModule (val phase)
    parser_finetune.add_argument('--val_dataset_root_dir', type=str, required=True, help='dataset_root_dir')
    parser_finetune.add_argument('--val_dataset_manifest_file', type=str, required=True, help='manifest_file')
    parser_finetune.add_argument('--val_volume_keys', type=str, nargs='+', required=True, help='volume_keys')
    parser_finetune.add_argument('--val_mask_keys', type=str, nargs='+', required=True, help='mask_keys')
    parser_finetune.add_argument('--val_cache_dir', type=str, default=None,
                                 help='cache_dir, absolute path, default at {experiment_root_dir}/{experiment_name}/{experiment_version}/cache')
    parser_finetune.add_argument('--val_batch_size', type=int, default=1, help='batch_size, 1')
    # Finetune: Optimizer & LR-Scheduler
    parser_finetune.add_argument('--max_lr', type=float, default=0.01, help='OneCycleLR.max_lr')
    parser_finetune.add_argument('--steps_per_epoch', type=int, required=True, help='OneCycleLR.steps_per_epoch')
    parser_finetune.add_argument('--final_div_factor', type=float, default=1e4, help='OneCycleLR.final_div_factor')
    # Finetune: val Inferer
    parser_finetune.add_argument(
        '--val_roi_size', type=int, nargs=3, default=(32, 32, 32),
        help='roi_size for sliding window inference'
    )
    parser_finetune.add_argument('--sw_batch_size', type=int, default=1,
                                 help='sw_batch_size for sliding window inference')
    parser_finetune.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')
    # Finetune: val export hook
    parser_finetune.add_argument('--export_val_results', action='store_true', default=False)
    parser_finetune.add_argument('--export_root_dir', type=str, default=None,
                                 help='export_root_dir, absolute path, '
                                      'default at {experiment_root_dir}/{experiment_name}/{experiment_version}/hook_finetune_val/exported_pred_results')
    parser_finetune.add_argument('--id_keys', type=str, nargs="+", default=['ID'],
                                 help='The columns to identify samples, usually it takes ID as primary key')
    parser_finetune.add_argument('--combined_mask_key', type=str, default="mask", help='combined_mask_key')
    parser_finetune.add_argument('--save_option', type=str, nargs="*",
                                 choices=("volume", "mask", "pred", "diff"),
                                 default=["mask", "pred", "diff"],
                                 help='save_option:\n'
                                      ' volume: save the original volumes, this can be very large\n'
                                      ' mask: save the original masks (binary and combined)\n'
                                      ' pred: save the pred_softmax_logits, pred_mask (binary and combined)\n'
                                      ' diff: save diff mask derived by pred & gt mask'
                                 )
    # Finetune: resuming
    parser_finetune.add_argument('--map_location', action='append', nargs=2, default=[], type=str, help='map_location')
    # endregion
    # region Validation
    parser_validation: argparse.ArgumentParser = subparsers.add_parser('validation', parents=[parser_common],
                                                                       help='validation help')
    parser_validation.add_argument('-ckpt', '--init_checkpoint', type=str, required=True,
                                   help='checkpoint for initializing module')
    # Validation: val Inferer
    parser_validation.add_argument('--sw_batch_size', type=int, default=1,
                                   help='sw_batch_size for sliding window inference')
    parser_validation.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')
    # Validation: val export hook
    parser_validation.add_argument('--export_results', action='store_true', default=False)
    parser_validation.add_argument('--export_root_dir', type=str, default=None,
                                   help='export_root_dir, absolute path, '
                                        'default at {experiment_root_dir}/{experiment_name}/{experiment_version}/hook_val/exported_pred_results')
    parser_validation.add_argument('--id_keys', type=str, nargs="+", default=['ID'],
                                   help='The columns to identify samples, usually it takes ID as primary key')
    parser_validation.add_argument('--combined_mask_key', type=str, default="mask", help='combined_mask_key')
    parser_validation.add_argument('--save_option', type=str, nargs="*",
                                   choices=("volume", "mask", "pred", "diff"),
                                   default=["mask", "pred", "diff"],
                                   help='save_option:\n'
                                        ' volume: save the original volumes, this can be very large\n'
                                        ' mask: save the original masks (binary and combined)\n'
                                        ' pred: save the pred_softmax_logits, pred_mask (binary and combined)\n'
                                        ' diff: save diff mask derived by pred & gt mask'
                                   )
    # endregion
    # region Test
    parser_test: argparse.ArgumentParser = subparsers.add_parser('test', parents=[parser_common], help='test help')
    parser_test.add_argument('-ckpt', '--init_checkpoint', type=str, required=True,
                             help='checkpoint for initializing module')
    # Test: test Inferer
    parser_test.add_argument('--sw_batch_size', type=int, default=1, help='sw_batch_size for sliding window inference')
    parser_test.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')
    # Test: test export hook
    parser_test.add_argument('--export_results', action='store_true', default=False)
    parser_test.add_argument('--export_root_dir', type=str, default=None,
                             help='export_root_dir, absolute path, '
                                  'default at {experiment_root_dir}/{experiment_name}/{experiment_version}/hook_test/exported_pred_results')
    parser_test.add_argument('--id_keys', type=str, nargs="+", default=['ID'],
                             help='The columns to identify samples, usually it takes ID as primary key')
    parser_test.add_argument('--combined_mask_key', type=str, default="mask", help='combined_mask_key')
    parser_test.add_argument('--save_option', type=str, nargs="*",
                             choices=("volume", "mask", "pred", "diff"),
                             default=["mask", "pred", "diff"],
                             help='save_option:\n'
                                  ' volume: save the original volumes, this can be very large\n'
                                  ' mask: save the original masks (binary and combined)\n'
                                  ' pred: save the pred_softmax_logits, pred_mask (binary and combined)\n'
                                  ' diff: save diff mask derived by pred & gt mask'
                             )
    # endregion
    # region Predict
    parser_predict: argparse.ArgumentParser = subparsers.add_parser('predict', parents=[parser_common], help='predict help')
    parser_predict.add_argument('-ckpt', '--init_checkpoint', type=str, required=True,
                                help='checkpoint for initializing module')
    # Predict: predict Inferer
    parser_predict.add_argument('--sw_batch_size', type=int, default=1,
                                help='sw_batch_size for sliding window inference')
    parser_predict.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')
    # Predict: predict export hook
    parser_predict.add_argument('--export_results', action='store_true', default=False)
    parser_predict.add_argument('--export_root_dir', type=str, default=None,
                                help='export_root_dir, absolute path, '
                                     'default at {experiment_root_dir}/{experiment_name}/{experiment_version}/hook_predict/exported_pred_results')
    parser_predict.add_argument('--id_keys', type=str, nargs="+", default=['ID'],
                                help='The columns to identify samples, usually it takes ID as primary key')
    parser_predict.add_argument('--combined_mask_key', type=str, default="mask", help='combined_mask_key')
    parser_predict.add_argument('--save_option', type=str, nargs="*",
                                choices=("volume", "pred"),
                                default=["pred"],
                                help='save_option:\n'
                                     ' volume: save the original volumes, this can be very large\n'
                                     ' pred: save the pred_softmax_logits, pred_mask (binary and combined)'
                                )
    # endregion
    # region Detect
    # Detect: data loader length
    parser_detect: argparse.ArgumentParser = subparsers.add_parser('detect', parents=[parser_common], help='detect help')
    # Fit: DataModule (val phase)
    parser_detect.add_argument('--val_dataset_root_dir', type=str, required=True, help='dataset_root_dir')
    parser_detect.add_argument('--val_dataset_manifest_file', type=str, required=True, help='manifest_file')
    parser_detect.add_argument('--val_volume_keys', type=str, nargs='+', required=True, help='volume_keys')
    parser_detect.add_argument('--val_mask_keys', type=str, nargs='*', required=True, help='mask_keys')
    parser_detect.add_argument('--val_cache_dir', type=str, default=None,
                               help='cache_dir, absolute path, default at {experiment_root_dir}/{experiment_name}/{experiment_version}/cache')
    parser_detect.add_argument('--val_batch_size', type=int, default=1, help='batch_size, 1')
    # endregion
    args: argparse.Namespace = parser.parse_args()

    # Common: Experiment defaults
    config_trainer: ConfigTrainerSegmentationDefault = ParserSegmentationDefault.default_config_trainer()
    """
        experiment_root_dir: Root directory where experiment logs and checkpoints will be saved.
        experiment_name: Name of the experiment for organizing logs. Defaults to 'pipeunet'.
        experiment_version: Version string for the experiment run. Defaults to '001'.
        trainer_init_args: Configuration for PyTorch Lightning Trainer. Defaults to CPU training with 32-bit precision.
        callback_init_args: Configuration for training callbacks. Defaults to no callbacks.
        logger_init_args: Configuration for experiment loggers. Defaults to CSV logger only.
    """

    config_data_module: ConfigDataModuleSegmentationDefault = ParserSegmentationDefault.default_config_data_module(
        roi_size=args.roi_size,
        num_classes=args.num_classes,
        crop_per_sample=args.crop_per_sample if hasattr(args, 'crop_per_sample') else 1,
        batch_size=args.batch_size
    )
    """
        train_init_args: Initialization arguments for the training phase
        val_init_args: Initialization arguments for the validation phase
        test_init_args: Initialization arguments for the testing phase
        predict_init_args: Initialization arguments for the prediction phase
    """

    config_ltn_module: ConfigLightningModuleSegmentationDefault = \
        ParserSegmentationDefault.default_config_ltn_module(args.num_sequence, args.num_classes)
    """
        network_init_args: NamedNetworkInitArgs = NamedNetworkInitArgs()
        # Specify as needed
        module_training_step_addition_args: Optional[ModuleTrainingStepAdditionArgs] = None
        module_validation_step_addition_args: Optional[ModuleValidationStepAdditionArgs] = None
        module_test_step_addition_args: Optional[ModuleTestStepAdditionArgs] = None
        module_predict_step_addition_args: Optional[ModulePredictStepAdditionArgs] = None
    """

    # Common: Modify configs
    # Common: Trainer
    config_trainer.experiment_root_dir = args.experiment_root_dir
    config_trainer.experiment_name = args.experiment_name
    config_trainer.experiment_version = args.experiment_version
    config_trainer.trainer_init_args.accelerator = args.accelerator
    config_trainer.trainer_init_args.devices = \
        args.devices[0] if config_trainer.trainer_init_args.accelerator == 'cpu' else args.devices
    deterministic_map: Dict[str, Optional[Union[str, bool]]] = {
        'none': None,
        'true': True,
        'false': False,
        'warn': 'warn'
    }
    config_trainer.trainer_init_args.deterministic = deterministic_map[args.deterministic]
    # Common: Logger
    config_trainer.logger_init_args.wandb_project = args.wandb_project

    # Prepare params for specific phase
    if args.routine in {'fit', 'finetune'}:
        # Fit/Finetune: Trainer
        config_trainer.trainer_init_args.max_epochs = args.epochs
        config_trainer.trainer_init_args.accumulate_grad_batches = args.accumulate_grad_batches

        # Fit/Finetune: Callback
        if args.early_stopping is None:
            config_trainer.callback_init_args.callback_early_stopping = None
        else:
            config_trainer.callback_init_args.callback_early_stopping.patience = args.early_stopping

        # Fit/Finetune: DataModule (train phase)
        config_data_module.train_init_args.config_retriever.root_dir = args.dataset_root_dir
        config_data_module.train_init_args.config_retriever.manifest_file = args.dataset_manifest_file
        config_data_module.train_init_args.config_retriever.column_key_map = {}
        config_data_module.train_init_args.config_retriever.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in config_data_module.train_init_args.config_retriever.column_group_map:
                config_data_module.train_init_args.config_retriever.column_group_map['volume'] = []
            config_data_module.train_init_args.config_retriever.column_key_map[volume_key] = f'volume_{idx}'
            config_data_module.train_init_args.config_retriever.column_group_map['volume'].append(f'volume_{idx}')
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in config_data_module.train_init_args.config_retriever.column_group_map:
                config_data_module.train_init_args.config_retriever.column_group_map['mask'] = []
            config_data_module.train_init_args.config_retriever.column_key_map[mask_key] = f'mask_{idx}'
            config_data_module.train_init_args.config_retriever.column_group_map['mask'].append(f'mask_{idx}')
        config_data_module.train_init_args.config_retriever.column_key_relative_path = list(
            config_data_module.train_init_args.config_retriever.column_key_map.values())
        if isinstance(config_data_module.train_init_args.config_dataset, (ConfigDatasetPersistent, ConfigDatasetLMDB)):
            if args.cache_dir is None:
                args.cache_dir = \
                    Path(args.experiment_root_dir) / args.experiment_name / args.experiment_version / 'cache'
            setattr(config_data_module.train_init_args.config_dataset, 'cache_dir', str(args.cache_dir))
        config_data_module.train_init_args.batch_size = args.batch_size
        config_data_module.train_init_args.num_workers = args.num_workers
        assert isinstance(config_data_module.train_init_args.config_transform, ConfigTransformSegmentationDefaultTrain)
        config_train_transform: ConfigTransformSegmentationDefaultTrain = config_data_module.train_init_args.config_transform
        config_train_transform.param_tf_rand_crop_by_label_classes_num_classes = args.num_classes
        config_train_transform.param_tf_rand_crop_by_label_classes_spatial_size = args.roi_size

        # Fit/Finetune: DataModule (val phase)
        config_data_module.val_init_args.config_retriever.root_dir = args.val_dataset_root_dir
        config_data_module.val_init_args.config_retriever.manifest_file = args.val_dataset_manifest_file
        config_data_module.val_init_args.config_retriever.column_key_map = {}
        config_data_module.val_init_args.config_retriever.column_group_map = {}
        for idx, volume_key in enumerate(args.val_volume_keys):
            if 'volume' not in config_data_module.val_init_args.config_retriever.column_group_map:
                config_data_module.val_init_args.config_retriever.column_group_map['volume'] = []
            config_data_module.val_init_args.config_retriever.column_key_map[volume_key] = f'volume_{idx}'
            config_data_module.val_init_args.config_retriever.column_group_map['volume'].append(f'volume_{idx}')
        for idx, mask_key in enumerate(args.val_mask_keys):
            if 'mask' not in config_data_module.val_init_args.config_retriever.column_group_map:
                config_data_module.val_init_args.config_retriever.column_group_map['mask'] = []
            config_data_module.val_init_args.config_retriever.column_key_map[mask_key] = f'mask_{idx}'
            config_data_module.val_init_args.config_retriever.column_group_map['mask'].append(f'mask_{idx}')
        config_data_module.val_init_args.config_retriever.column_key_relative_path = list(
            config_data_module.val_init_args.config_retriever.column_key_map.values())
        if isinstance(config_data_module.val_init_args.config_dataset, (ConfigDatasetPersistent, ConfigDatasetLMDB)):
            if args.val_cache_dir is None:
                args.val_cache_dir = \
                    Path(args.experiment_root_dir) / args.experiment_name / args.experiment_version / 'cache'
            setattr(config_data_module.val_init_args.config_dataset, 'cache_dir', str(args.val_cache_dir))
        assert args.val_batch_size == 1, f'Routine {args.routine}: val_batch_size must be equal to 1'
        config_data_module.val_init_args.batch_size = args.val_batch_size
        config_data_module.val_init_args.num_workers = args.num_workers

        config_data_module.test_init_args = None
        config_data_module.predict_init_args = None

        # Fit/Finetune: Optimizer & LR-Scheduler
        assert isinstance(
            config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler,
            ConfigLRSchedulerOneCycle
        ), (f'config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler'
            f'({type(config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler)}) '
            f'must be ConfigLRSchedulerOneCycle.')
        config_lr_scheduler_one_cycle: ConfigLRSchedulerOneCycle = \
            config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler
        config_lr_scheduler_one_cycle.max_lr = args.max_lr
        config_lr_scheduler_one_cycle.epochs = args.epochs
        # OneCycleLR.steps_per_epoch shall also be specified, you must do it manually
        config_lr_scheduler_one_cycle.steps_per_epoch = args.steps_per_epoch
        config_lr_scheduler_one_cycle.final_div_factor = args.final_div_factor

        # Fit/Finetune: Inferer (val phase)
        assert isinstance(
            config_ltn_module.module_validation_step_addition_args.config_inferer,
            ConfigInfererMainWithAuxSlidingWindow
        ), (f'config_ltn_module.module_validation_step_addition_args.config_inferer'
            f'({type(config_ltn_module.module_validation_step_addition_args.config_inferer)}) '
            f'must be ConfigInfererMainWithAuxSlidingWindow.')
        config_validation_inferer: ConfigInfererMainWithAuxSlidingWindow = config_ltn_module.module_validation_step_addition_args.config_inferer
        config_validation_inferer.roi_size = args.val_roi_size
        config_validation_inferer.sw_batch_size = args.sw_batch_size
        config_validation_inferer.overlap = args.overlap

        # Fit/Finetune: Export pred hook
        if args.export_val_results:
            export_path = args.export_root_dir
            if export_path is None:
                export_path: Path = (
                        Path(args.experiment_root_dir) /
                        args.experiment_name /
                        args.experiment_version /
                        ('hook_fit_val' if args.routine == 'fit' else 'hook_finetune_val') /
                        'exported_pred_results'
                )
            config_ltn_module.module_validation_step_addition_args.hook_functions.append(
                ConfigOperatorHookStepExportMulticlassPredWithMaskResults(
                    export_root_dir=str(export_path),
                    dataset_root_dir=args.val_dataset_root_dir,
                    manifest_file=args.val_dataset_manifest_file,
                    pred_key=config_ltn_module.module_validation_step_addition_args.pred_key,
                    id_keys=args.id_keys,
                    volume_keys=args.val_volume_keys,
                    mask_keys=args.val_mask_keys,
                    combined_mask_key=args.combined_mask_key,
                    save_option=args.save_option
                )
            )

        config_ltn_module.module_test_step_addition_args = None
        config_ltn_module.module_predict_step_addition_args = None

    elif args.routine == 'validation':
        # Val: DataModule
        config_data_module.val_init_args.config_retriever.root_dir = args.dataset_root_dir
        config_data_module.val_init_args.config_retriever.manifest_file = args.dataset_manifest_file
        config_data_module.val_init_args.config_retriever.column_key_map = {}
        config_data_module.val_init_args.config_retriever.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in config_data_module.val_init_args.config_retriever.column_group_map:
                config_data_module.val_init_args.config_retriever.column_group_map['volume'] = []
            config_data_module.val_init_args.config_retriever.column_key_map[volume_key] = f'volume_{idx}'
            config_data_module.val_init_args.config_retriever.column_group_map['volume'].append(f'volume_{idx}')
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in config_data_module.val_init_args.config_retriever.column_group_map:
                config_data_module.val_init_args.config_retriever.column_group_map['mask'] = []
            config_data_module.val_init_args.config_retriever.column_key_map[mask_key] = f'mask_{idx}'
            config_data_module.val_init_args.config_retriever.column_group_map['mask'].append(f'mask_{idx}')
        config_data_module.val_init_args.config_retriever.column_key_relative_path = list(
            config_data_module.val_init_args.config_retriever.column_key_map.values())
        if isinstance(config_data_module.val_init_args.config_dataset, (ConfigDatasetPersistent, ConfigDatasetLMDB)):
            if args.cache_dir is None:
                args.cache_dir = \
                    Path(args.experiment_root_dir) / args.experiment_name / args.experiment_version / 'cache'
            setattr(config_data_module.val_init_args.config_dataset, 'cache_dir', str(args.cache_dir))
        assert args.batch_size == 1, f'Routine {args.routine}: batch_size must be equal to 1'
        config_data_module.val_init_args.batch_size = args.batch_size
        config_data_module.val_init_args.num_workers = args.num_workers

        config_data_module.train_init_args = None
        config_data_module.test_init_args = None
        config_data_module.predict_init_args = None

        # Val: Inferer
        assert isinstance(
            config_ltn_module.module_validation_step_addition_args.config_inferer,
            ConfigInfererMainWithAuxSlidingWindow
        ), (f'config_ltn_module.module_validation_step_addition_args.config_inferer'
            f'({type(config_ltn_module.module_validation_step_addition_args.config_inferer)}) '
            f'must be ConfigInfererMainWithAuxSlidingWindow.')
        config_validation_inferer: ConfigInfererMainWithAuxSlidingWindow = config_ltn_module.module_validation_step_addition_args.config_inferer
        config_validation_inferer.roi_size = args.roi_size
        config_validation_inferer.sw_batch_size = args.sw_batch_size
        config_validation_inferer.overlap = args.overlap

        # Val: Export pred hook
        if args.export_results:
            export_path = args.export_root_dir
            if export_path is None:
                export_path: Path = (
                        Path(args.experiment_root_dir) /
                        args.experiment_name /
                        args.experiment_version /
                        'hook_val' /
                        'exported_pred_results'
                )
            config_ltn_module.module_validation_step_addition_args.hook_functions.append(
                ConfigOperatorHookStepExportMulticlassPredWithMaskResults(
                    export_root_dir=str(export_path),
                    dataset_root_dir=args.dataset_root_dir,
                    manifest_file=args.dataset_manifest_file,
                    pred_key=config_ltn_module.module_validation_step_addition_args.pred_key,
                    id_keys=args.id_keys,
                    volume_keys=args.volume_keys,
                    mask_keys=args.mask_keys,
                    combined_mask_key=args.combined_mask_key,
                    save_option=args.save_option
                )
            )

        config_ltn_module.module_train_step_addition_args = None
        config_ltn_module.module_test_step_addition_args = None
        config_ltn_module.module_predict_step_addition_args = None

    elif args.routine == 'test':
        # Test: DataModule
        config_data_module.test_init_args.config_retriever.root_dir = args.dataset_root_dir
        config_data_module.test_init_args.config_retriever.manifest_file = args.dataset_manifest_file
        config_data_module.test_init_args.config_retriever.column_key_map = {}
        config_data_module.test_init_args.config_retriever.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in config_data_module.test_init_args.config_retriever.column_group_map:
                config_data_module.test_init_args.config_retriever.column_group_map['volume'] = []
            config_data_module.test_init_args.config_retriever.column_key_map[volume_key] = f'volume_{idx}'
            config_data_module.test_init_args.config_retriever.column_group_map['volume'].append(f'volume_{idx}')
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in config_data_module.test_init_args.config_retriever.column_group_map:
                config_data_module.test_init_args.config_retriever.column_group_map['mask'] = []
            config_data_module.test_init_args.config_retriever.column_key_map[mask_key] = f'mask_{idx}'
            config_data_module.test_init_args.config_retriever.column_group_map['mask'].append(f'mask_{idx}')
        config_data_module.test_init_args.config_retriever.column_key_relative_path = list(
            config_data_module.test_init_args.config_retriever.column_key_map.values())
        if isinstance(config_data_module.test_init_args.config_dataset, (ConfigDatasetPersistent, ConfigDatasetLMDB)):
            if args.cache_dir is None:
                args.cache_dir = \
                    Path(args.experiment_root_dir) / args.experiment_name / args.experiment_version / 'cache'
            setattr(config_data_module.test_init_args.config_dataset, 'cache_dir', str(args.cache_dir))
        assert args.batch_size == 1, f'Routine {args.routine}: batch_size must be equal to 1'
        config_data_module.test_init_args.batch_size = args.batch_size
        config_data_module.test_init_args.num_workers = args.num_workers

        config_data_module.train_init_args = None
        config_data_module.val_init_args = None
        config_data_module.predict_init_args = None

        # Test: Inferer
        assert isinstance(
            config_ltn_module.module_test_step_addition_args.config_inferer,
            ConfigInfererMainWithAuxSlidingWindow
        ), (f'config_ltn_module.module_test_step_addition_args.config_inferer'
            f'({type(config_ltn_module.module_test_step_addition_args.config_inferer)}) '
            f'must be ConfigInfererMainWithAuxSlidingWindow.')
        config_test_inferer: ConfigInfererMainWithAuxSlidingWindow = config_ltn_module.module_test_step_addition_args.config_inferer
        config_test_inferer.roi_size = args.roi_size
        config_test_inferer.sw_batch_size = args.sw_batch_size
        config_test_inferer.overlap = args.overlap

        # Test: Export pred hook
        if args.export_results:
            export_path = args.export_root_dir
            if export_path is None:
                export_path: Path = (
                        Path(args.experiment_root_dir) /
                        args.experiment_name /
                        args.experiment_version /
                        'hook_test' /
                        'exported_pred_results'
                )
            config_ltn_module.module_test_step_addition_args.hook_functions.append(
                ConfigOperatorHookStepExportMulticlassPredWithMaskResults(
                    export_root_dir=str(export_path),
                    dataset_root_dir=args.dataset_root_dir,
                    manifest_file=args.dataset_manifest_file,
                    pred_key=config_ltn_module.module_test_step_addition_args.pred_key,
                    id_keys=args.id_keys,
                    volume_keys=args.volume_keys,
                    mask_keys=args.mask_keys,
                    combined_mask_key=args.combined_mask_key,
                    save_option=args.save_option
                )
            )

        config_ltn_module.module_train_step_addition_args = None
        config_ltn_module.module_validation_step_addition_args = None
        config_ltn_module.module_predict_step_addition_args = None

    elif args.routine == 'predict':
        # Predict: DataModule
        config_data_module.predict_init_args.config_retriever.root_dir = args.dataset_root_dir
        config_data_module.predict_init_args.config_retriever.manifest_file = args.dataset_manifest_file
        config_data_module.predict_init_args.config_retriever.column_key_map = {}
        config_data_module.predict_init_args.config_retriever.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in config_data_module.predict_init_args.config_retriever.column_group_map:
                config_data_module.predict_init_args.config_retriever.column_group_map['volume'] = []
            config_data_module.predict_init_args.config_retriever.column_key_map[volume_key] = f'volume_{idx}'
            config_data_module.predict_init_args.config_retriever.column_group_map['volume'].append(f'volume_{idx}')
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in config_data_module.predict_init_args.config_retriever.column_group_map:
                config_data_module.predict_init_args.config_retriever.column_group_map['mask'] = []
            config_data_module.predict_init_args.config_retriever.column_key_map[mask_key] = f'mask_{idx}'
            config_data_module.predict_init_args.config_retriever.column_group_map['mask'].append(f'mask_{idx}')
        config_data_module.predict_init_args.config_retriever.column_key_relative_path = list(
            config_data_module.predict_init_args.config_retriever.column_key_map.values())
        if isinstance(config_data_module.predict_init_args.config_dataset,
                      (ConfigDatasetPersistent, ConfigDatasetLMDB)):
            if args.cache_dir is None:
                args.cache_dir = \
                    Path(args.experiment_root_dir) / args.experiment_name / args.experiment_version / 'cache'
            setattr(config_data_module.predict_init_args.config_dataset, 'cache_dir', str(args.cache_dir))
        assert args.batch_size == 1, f'Routine {args.routine}: batch_size must be equal to 1'
        config_data_module.predict_init_args.batch_size = args.batch_size
        config_data_module.predict_init_args.num_workers = args.num_workers

        config_data_module.train_init_args = None
        config_data_module.val_init_args = None
        config_data_module.test_init_args = None

        # Predict: Inferer
        assert isinstance(
            config_ltn_module.module_predict_step_addition_args.config_inferer,
            ConfigInfererMainWithAuxSlidingWindow
        ), (f'config_ltn_module.module_predict_step_addition_args.config_inferer'
            f'({type(config_ltn_module.module_predict_step_addition_args.config_inferer)}) '
            f'must be ConfigInfererMainWithAuxSlidingWindow.')
        config_predict_inferer: ConfigInfererMainWithAuxSlidingWindow = config_ltn_module.module_predict_step_addition_args.config_inferer
        config_predict_inferer.roi_size = args.roi_size
        config_predict_inferer.sw_batch_size = args.sw_batch_size
        config_predict_inferer.overlap = args.overlap

        # Predict: Export pred hook
        if args.export_results:
            export_path = args.export_root_dir
            if export_path is None:
                export_path: Path = (
                        Path(args.experiment_root_dir) /
                        args.experiment_name /
                        args.experiment_version /
                        'hook_predict' /
                        'exported_pred_results'
                )
            config_ltn_module.module_predict_step_addition_args.hook_functions.append(
                ConfigOperatorHookStepExportMulticlassPredOnlyResults(
                    export_root_dir=str(export_path),
                    dataset_root_dir=args.dataset_root_dir,
                    manifest_file=args.dataset_manifest_file,
                    pred_key=config_ltn_module.module_predict_step_addition_args.pred_key,
                    id_keys=args.id_keys,
                    volume_keys=args.volume_keys,
                    mask_keys=args.mask_keys,
                    combined_mask_key=args.combined_mask_key,
                    save_option=args.save_option
                )
            )

        config_ltn_module.module_train_step_addition_args = None
        config_ltn_module.module_validation_step_addition_args = None
        config_ltn_module.module_test_step_addition_args = None

    elif args.routine == 'detect':
        # Detect: DataModule (train phase)
        config_data_module.train_init_args.config_retriever.root_dir = args.dataset_root_dir
        config_data_module.train_init_args.config_retriever.manifest_file = args.dataset_manifest_file
        config_data_module.train_init_args.config_retriever.column_key_map = {}
        config_data_module.train_init_args.config_retriever.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in config_data_module.train_init_args.config_retriever.column_group_map:
                config_data_module.train_init_args.config_retriever.column_group_map['volume'] = []
            config_data_module.train_init_args.config_retriever.column_key_map[volume_key] = f'volume_{idx}'
            config_data_module.train_init_args.config_retriever.column_group_map['volume'].append(f'volume_{idx}')
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in config_data_module.train_init_args.config_retriever.column_group_map:
                config_data_module.train_init_args.config_retriever.column_group_map['mask'] = []
            config_data_module.train_init_args.config_retriever.column_key_map[mask_key] = f'mask_{idx}'
            config_data_module.train_init_args.config_retriever.column_group_map['mask'].append(f'mask_{idx}')
        config_data_module.train_init_args.config_retriever.column_key_relative_path = list(
            config_data_module.train_init_args.config_retriever.column_key_map.values())
        if isinstance(config_data_module.train_init_args.config_dataset, (ConfigDatasetPersistent, ConfigDatasetLMDB)):
            if args.cache_dir is None:
                args.cache_dir = \
                    Path(args.experiment_root_dir) / args.experiment_name / args.experiment_version / 'cache'
            setattr(config_data_module.train_init_args.config_dataset, 'cache_dir', str(args.cache_dir))
        config_data_module.train_init_args.batch_size = args.batch_size

        # Detect: DataModule (val phase)
        config_data_module.val_init_args.config_retriever.root_dir = args.val_dataset_root_dir
        config_data_module.val_init_args.config_retriever.manifest_file = args.val_dataset_manifest_file
        config_data_module.val_init_args.config_retriever.column_key_map = {}
        config_data_module.val_init_args.config_retriever.column_group_map = {}
        for idx, volume_key in enumerate(args.val_volume_keys):
            if 'volume' not in config_data_module.val_init_args.config_retriever.column_group_map:
                config_data_module.val_init_args.config_retriever.column_group_map['volume'] = []
            config_data_module.val_init_args.config_retriever.column_key_map[volume_key] = f'volume_{idx}'
            config_data_module.val_init_args.config_retriever.column_group_map['volume'].append(f'volume_{idx}')
        for idx, mask_key in enumerate(args.val_mask_keys):
            if 'mask' not in config_data_module.val_init_args.config_retriever.column_group_map:
                config_data_module.val_init_args.config_retriever.column_group_map['mask'] = []
            config_data_module.val_init_args.config_retriever.column_key_map[mask_key] = f'mask_{idx}'
            config_data_module.val_init_args.config_retriever.column_group_map['mask'].append(f'mask_{idx}')
        config_data_module.val_init_args.config_retriever.column_key_relative_path = list(
            config_data_module.val_init_args.config_retriever.column_key_map.values())
        if isinstance(config_data_module.val_init_args.config_dataset, (ConfigDatasetPersistent, ConfigDatasetLMDB)):
            if args.val_cache_dir is None:
                args.val_cache_dir = Path(
                    args.experiment_root_dir) / args.experiment_name / args.experiment_version / 'cache'
            setattr(config_data_module.val_init_args.config_dataset, 'cache_dir', str(args.val_cache_dir))
        assert args.val_batch_size == 1, f'Routine {args.routine}: val_batch_size must be equal to 1'
        config_data_module.val_init_args.batch_size = args.val_batch_size

        config_data_module.test_init_args = None
        config_data_module.predict_init_args = None

        config_ltn_module.module_test_step_addition_args = None
        config_ltn_module.module_predict_step_addition_args = None

    else:
        raise ValueError(f'routine {args.routine!r} not supported')

    # Export as YAML config file
    parser: ParserSegmentationDefault = ParserSegmentationDefault(
        config_trainer=copy.deepcopy(config_trainer),
        config_data_module=copy.deepcopy(config_data_module),
        config_ltn_module=copy.deepcopy(config_ltn_module)
    )
    yaml_save_path: Path = (
            Path(args.experiment_root_dir) /
            args.experiment_name /
            args.experiment_version /
            f'config-{args.experiment_name}-{args.experiment_version}.yaml'
    )

    if args.routine not in {'detect'}:
        parser.to_yaml(
            yaml_save_path,
            allow_unicode=True
        )

    # Initialize the Launcher
    config_launcher: LauncherSegmentationDefault = LauncherSegmentationDefault(
        config_trainer=config_trainer,
        config_data_module=config_data_module,
        config_ltn_module=config_ltn_module
    )

    # Launch specified routine
    # To remove open file (file_descriptor) limit, use file_system for multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.routine == 'fit':
        config_launcher.fit(args.resume_checkpoint)
    elif args.routine == 'finetune':
        map_location: Dict[int, int] = {item[0]: item[1] for item in args.map_location}
        config_launcher.finetune(args.init_checkpoint, finetune_map_location=map_location)
    elif args.routine == 'validation':
        config_launcher.validation(args.init_checkpoint)
    elif args.routine == 'test':
        config_launcher.test(args.init_checkpoint)
    elif args.routine == 'predict':
        config_launcher.predict(args.init_checkpoint)
    elif args.routine == 'detect':
        steps_per_epoch_d: Dict[str, int] = config_launcher.detect()
        train_steps_per_epoch: int = steps_per_epoch_d['train']
        assert isinstance(
            parser.config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler,
            ConfigLRSchedulerOneCycle
        ), (
            f'parser.config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler'
            f'({type(parser.config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler)}) '
            f'must be ConfigLRSchedulerOneCycle.')
        config_lr_scheduler_one_cycle: ConfigLRSchedulerOneCycle = \
            parser.config_ltn_module.module_training_step_addition_args.lr_scheduler_init_args.config_lr_scheduler
        # Set detected steps_per_epoch
        config_lr_scheduler_one_cycle.steps_per_epoch = train_steps_per_epoch
        parser.to_yaml(
            yaml_save_path,
            allow_unicode=True
        )

    print(f'{config_launcher.__class__.__qualname__} has finished {args.routine!r} routine')
