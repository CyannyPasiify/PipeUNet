# -*- coding: utf-8 -*-
import os
from pathlib import Path
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE
from typing import Optional, Dict, Any, Union, Type
from DataModule.data_module_segmentation_default import DataModuleSegmentationDefault, DataModuleSegmentationDefaultInitArgs
from Module.ltn_module_segmentation_default import NamedNetworkInitArgs, ModuleTrainingStepAdditionArgs, \
    NamedMetricInitArgs, NamedLossInitArgs, NamedOptimizerInitArgs, NamedLRSchedulerInitArgs, \
    ModuleTestStepAdditionArgs, ModuleValidationStepAdditionArgs, ModulePredictStepAdditionArgs, \
    ModuleSegmentationDefault
from Trainer.trainer_configurer import TrainerInitArgs, CallbackInitArgs, LoggerInitArgs, ConfigTrainerSegmentationDefault, \
    ConfigTrainerBase
from Launcher.Parser.parser_segmentation_default import ParserSegmentationDefault
from Launcher.launcher_ABC import LauncherABC
from dataclasses import dataclass


@dataclass
class LauncherSegmentationDefault(LauncherABC):
    experiment_root_dir: Union[str, os.PathLike, Path]
    experiment_name: str
    experiment_version: str
    trainer_wrapper: ConfigTrainerBase
    callback_init_args: CallbackInitArgs
    logger_init_args: LoggerInitArgs
    data_module_wrapper: DataModule
    module_class: Type[ModuleSegmentationDefault]  # Main Module shall be subtype of ModuleSegmentationDefault
    network_init_args: NamedNetworkInitArgs
    # Optional args
    datamodule_training_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_validation_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    module_training_step_addition_args: Optional[ModuleTrainingStepAdditionArgs] = None
    module_validation_step_addition_args: Optional[ModuleValidationStepAdditionArgs] = None
    module_test_step_addition_args: Optional[ModuleTestStepAdditionArgs] = None
    module_predict_step_addition_args: Optional[ModulePredictStepAdditionArgs] = None
    def __init__(
            self,
            experiment_root_dir: Union[str, os.PathLike, Path],
            experiment_name: str,
            experiment_version: str,
            trainer_class: Type[ConfigTrainerSegmentationDefault],  # Trainer shall be subtype of TrainerSegmentationDefault
            trainer_init_args: TrainerInitArgs,
            callback_init_args: CallbackInitArgs,
            logger_init_args: LoggerInitArgs,
            datamodule_class: Type[DataModuleSegmentationDefault],
            module_class: Type[ModuleSegmentationDefault],  # Main Module shall be subtype of ModuleSegmentationDefault
            network_init_args: NamedNetworkInitArgs,
            # Optional args
            datamodule_training_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            datamodule_validation_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            datamodule_test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            datamodule_predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            module_training_step_addition_args: Optional[ModuleTrainingStepAdditionArgs] = None,
            module_validation_step_addition_args: Optional[ModuleValidationStepAdditionArgs] = None,
            module_test_step_addition_args: Optional[ModuleTestStepAdditionArgs] = None,
            module_predict_step_addition_args: Optional[ModulePredictStepAdditionArgs] = None
    ):
        assert experiment_root_dir is not None, \
            (f'experiment_root_dir must be specified, '
             f'see {ParserSegmentationDefault.default_experiment_root_dir.__qualname__} for example.')
        assert experiment_name is not None, \
            (f'experiment_name must be specified, '
             f'see {ParserSegmentationDefault.default_experiment_name.__qualname__} for example.')
        assert experiment_version is not None, \
            (f'experiment_version must be specified, '
             f'see {ParserSegmentationDefault.default_experiment_version.__qualname__} for example.')
        assert trainer_class is not None, \
            (f'trainer_class must be specified, '
             f'see {ParserSegmentationDefault.default_trainer_class.__qualname__} for example.')
        assert trainer_init_args is not None, \
            (f'trainer_init_args must be specified, '
             f'see {ParserSegmentationDefault.default_trainer_init_args.__qualname__} for example.')
        assert callback_init_args is not None, \
            (f'callback_init_args must be specified, '
             f'see {ParserSegmentationDefault.default_callback_init_args.__qualname__} for example.')
        assert logger_init_args is not None, \
            (f'logger_init_args must be specified, '
             f'see {ParserSegmentationDefault.default_logger_init_args.__qualname__} for example.')
        assert datamodule_class is not None, \
            (f'module_class must be specified, '
             f'see {ParserSegmentationDefault.default_datamodule_class.__qualname__} for example.')
        assert module_class is not None, \
            (f'module_class must be specified, '
             f'see {ParserSegmentationDefault.default_module_class.__qualname__} for example.')
        assert network_init_args is not None, \
            (f'network_init_args must be specified, '
             f'see {ParserSegmentationDefault.default_network_init_args.__qualname__} for example.')
        assert not (datamodule_training_init_args is None and
                    datamodule_validation_init_args is None and
                    datamodule_test_init_args is None and
                    datamodule_predict_init_args is None), \
            (f'All datamodule_{{phase}}_init_args are None. You must at least specify one of them.'
             f'see '
             f'{ParserSegmentationDefault.default_datamodule_training_init_args.__qualname__}, '
             f'{ParserSegmentationDefault.default_datamodule_validation_init_args.__qualname__}, '
             f'{ParserSegmentationDefault.default_datamodule_test_init_args.__qualname__}, '
             f'{ParserSegmentationDefault.default_datamodule_predict_init_args.__qualname__}'
             f' for example.')
        assert not (module_training_step_addition_args is None and
                    module_validation_step_addition_args is None and
                    module_test_step_addition_args is None and
                    module_predict_step_addition_args is None), \
            (f'All module_{{phase}}_step_addition_args are None. You must at least specify one of them.'
             f'see '
             f'{ParserSegmentationDefault.default_module_training_step_addition_args.__qualname__}, '
             f'{ParserSegmentationDefault.default_module_validation_step_addition_args.__qualname__}, '
             f'{ParserSegmentationDefault.default_module_test_step_addition_args.__qualname__}, '
             f'{ParserSegmentationDefault.default_module_predict_step_addition_args.__qualname__}'
             f' for example.')

        self.experiment_root_dir: Path = Path(experiment_root_dir)
        self.experiment_name: str = experiment_name
        self.experiment_version: str = experiment_version

        self.trainer_class: Type[ConfigTrainerSegmentationDefault] = trainer_class
        self.trainer_init_args: TrainerInitArgs = trainer_init_args
        self.callback_init_args: CallbackInitArgs = callback_init_args
        self.logger_init_args: LoggerInitArgs = logger_init_args

        self.datamodule_class: Type[DataModuleSegmentationDefault] = datamodule_class
        self.datamodule_training_init_args: DataModuleSegmentationDefaultInitArgs = datamodule_training_init_args
        self.datamodule_validation_init_args: DataModuleSegmentationDefaultInitArgs = datamodule_validation_init_args
        self.datamodule_test_init_args: DataModuleSegmentationDefaultInitArgs = datamodule_test_init_args
        self.datamodule_predict_init_args: DataModuleSegmentationDefaultInitArgs = datamodule_predict_init_args

        self.module_class: Type[ModuleSegmentationDefault] = module_class
        self.network_init_args: NamedNetworkInitArgs = network_init_args
        self.module_training_step_addition_args: ModuleTrainingStepAdditionArgs = module_training_step_addition_args
        self.module_validation_step_addition_args: ModuleValidationStepAdditionArgs = module_validation_step_addition_args
        self.module_test_step_addition_args: ModuleTestStepAdditionArgs = module_test_step_addition_args
        self.module_predict_step_addition_args: ModulePredictStepAdditionArgs = module_predict_step_addition_args

        self.trainer: ConfigTrainerSegmentationDefault = trainer_class(
            experiment_root_dir=experiment_root_dir,
            experiment_name=experiment_name,
            experiment_version=experiment_version,
            trainer_init_args=trainer_init_args,
            callback_init_args=callback_init_args,
            logger_init_args=logger_init_args
        )

        self.data_module: DataModuleSegmentationDefault = datamodule_class(
            train_init_args=datamodule_training_init_args,
            val_init_args=datamodule_validation_init_args,
            test_init_args=datamodule_test_init_args,
            predict_init_args=datamodule_predict_init_args
        )

        self.module: ModuleSegmentationDefault = module_class(
            network_init_args=network_init_args,
            module_training_step_addition_args=module_training_step_addition_args,
            module_validation_step_addition_args=module_validation_step_addition_args,
            module_test_step_addition_args=module_test_step_addition_args,
            module_predict_step_addition_args=module_predict_step_addition_args
        )

        print(f'{self.__class__.__name__} initialized. Available phases {self.module.get_available_phases()}')

    def fit(self, checkpoint: Optional[Union[str, os.PathLike, Path]] = None) -> Dict[str, Any]:
        return self.trainer.fit(
            model=self.module,
            datamodule=self.data_module,
            ckpt_path=checkpoint  # You may use checkpoint to resume training.
        )

    def finetune(
            self,
            checkpoint: Optional[Union[str, os.PathLike, Path]],
            finetune_map_location: _MAP_LOCATION_TYPE = None,
            finetune_hparams_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        return self.trainer.fit(
            model=self.module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            # Checkpoint is loaded at the beginning for initialization
            ckpt_path=str(checkpoint) if checkpoint is not None else None,
            finetune=True,
            finetune_map_location=finetune_map_location,
            finetune_hparams_file=finetune_hparams_file
        )

    def validation(self, checkpoint: Optional[Union[str, os.PathLike, Path]]) -> Dict[str, Any]:
        return self.trainer.validate(
            model=self.module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            ckpt_path=str(checkpoint) if checkpoint is not None else None
        )

    def test(self, checkpoint: Optional[Union[str, os.PathLike, Path]]) -> Dict[str, Any]:
        return self.trainer.test(
            model=self.module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            ckpt_path=str(checkpoint) if checkpoint is not None else None
        )

    def predict(self, checkpoint: Optional[Union[str, os.PathLike, Path]]) -> Dict[str, Any]:
        return self.trainer.predict(
            model=self.module,
            datamodule=self.data_module,
            # Usually you shall specify a checkpoint to load,
            # but if you initialized the Module manually, you can ignore it.
            ckpt_path=str(checkpoint) if checkpoint is not None else None
        )


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    # Common: Experiment
    parser.add_argument('-r', '--experiment_root_dir', type=str, required=True, help='experiment_root_dir')
    parser.add_argument('-e', '--experiment_name', type=str, required=True, help='experiment_name')
    parser.add_argument('-v', '--experiment_version', type=str, required=True, help='experiment_version')
    # Common: Routine
    parser.add_argument('--accelerator', type=str, required=True, help='accelerator')
    parser.add_argument('--device', type=int, nargs='+', required=True, help='device, gpu [multi], cpu single num')
    parser.add_argument('--deterministic', choice=['none', 'warn', 'true', 'false'], default='warn',
                        help='deterministic, \'warn\' try it best to be deterministic')
    # Common: Logger
    parser.add_argument('--wandb_project', type=str, default='PipeUNet', help='wandb_project')
    # Common: DataModule (specified main phase)
    # fit: train (val will have another set of args)
    # finetune: train (val will have another set of args)
    # validation: val
    # test: test
    # predict: predict
    parser.add_argument('--dataset_root_dir', type=str, required=True, help='dataset_root_dir')
    parser.add_argument('--dataset_manifest_file', type=str, required=True, help='manifest_file')
    parser.add_argument('--volume_keys', type=str, nargs='+', required=True, help='volume_keys')
    parser.add_argument('--mask_keys', type=str, nargs='*', required=True, help='mask_keys')
    parser.add_argument('--cache_dir', type=str, required=True, help='cache_dir')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size, 1')
    # Common: Network
    parser.add_argument('--num_sequence', '--num_modality', type=int, required=True, help='num_sequence')
    parser.add_argument('--num_classes', type=int, required=True, help='num_classes')

    subparsers = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands',
        help='additional help',
        dest='routine'
    )

    # Fit
    parser_fit: argparse.ArgumentParser = subparsers.add_parser('fit', help='fit help')
    parser_fit.add_argument('-ckpt', '--resume_checkpoint', type=str, default=None,
                            help='checkpoint for resuming training')
    # Fit: Trainer
    parser_fit.add_argument('--epochs', type=int, required=True, help='epochs, 100')
    parser_fit.add_argument('--accumulate_grad_batches', type=int, default=16, help='accumulate_grad_batches, 16')
    # Fit: Callback
    parser_fit.add_argument('--early_stopping', type=int, default=None,
                            help='early_stopping, if specified, it equals patience')
    # Fit: DataModule (val phase)
    parser_fit.add_argument('--val_dataset_root_dir', type=str, required=True, help='dataset_root_dir')
    parser_fit.add_argument('--val_dataset_manifest_file', type=str, required=True, help='manifest_file')
    parser_fit.add_argument('--val_volume_keys', type=str, nargs='+', required=True, help='volume_keys')
    parser_fit.add_argument('--val_mask_keys', type=str, nargs='*', required=True, help='mask_keys')
    parser_fit.add_argument('--val_cache_dir', type=str, required=True, help='cache_dir')
    parser_fit.add_argument('--val_batch_size', type=int, default=1, help='batch_size, 1')
    # Fit: Optimizer & LR-Scheduler
    parser_fit.add_argument('--max_lr', type=float, default=0.01, help='OneCycleLR.max_lr')
    parser_fit.add_argument('--steps_per_epoch', type=int, required=True, help='OneCycleLR.steps_per_epoch')
    parser_fit.add_argument('--final_div_factor', type=float, default=1e4, help='OneCycleLR.final_div_factor')
    # Fit: val Inferer
    parser_fit.add_argument('--roi_size', type=int, nargs=3, default=(128, 128, 128),
                            help='roi_size for sliding window inference')
    parser_fit.add_argument('--sw_batch_size', type=int, default=1, help='sw_batch_size for sliding window inference')
    parser_fit.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')

    # Finetune
    parser_finetune: argparse.ArgumentParser = subparsers.add_parser('finetune', help='finetune help')
    parser_finetune.add_argument('-ckpt', '--init_checkpoint', type=str, default=None,
                                 help='checkpoint for initializing module')
    # Finetune: Trainer
    parser_finetune.add_argument('--epochs', type=int, required=True, help='epochs, 100')
    parser_finetune.add_argument('--accumulate_grad_batches', type=int, default=16, help='accumulate_grad_batches, 16')
    # Finetune: Callback
    parser_finetune.add_argument('--early_stopping', type=int, default=None,
                                 help='early_stopping, if specified, it equals patience')
    # Finetune: DataModule (val phase)
    parser_finetune.add_argument('--val_dataset_root_dir', type=str, required=True, help='dataset_root_dir')
    parser_finetune.add_argument('--val_dataset_manifest_file', type=str, required=True, help='manifest_file')
    parser_finetune.add_argument('--val_volume_keys', type=str, nargs='+', required=True, help='volume_keys')
    parser_finetune.add_argument('--val_mask_keys', type=str, nargs='*', required=True, help='mask_keys')
    parser_finetune.add_argument('--val_cache_dir', type=str, required=True, help='cache_dir')
    parser_finetune.add_argument('--val_batch_size', type=int, default=1, help='batch_size, 1')
    # Finetune: Optimizer & LR-Scheduler
    parser_finetune.add_argument('--max_lr', type=float, default=0.01, help='OneCycleLR.max_lr')
    parser_finetune.add_argument('--steps_per_epoch', type=int, required=True, help='OneCycleLR.steps_per_epoch')
    parser_finetune.add_argument('--final_div_factor', type=float, default=1e4, help='OneCycleLR.final_div_factor')
    # Finetune: val Inferer
    parser_finetune.add_argument('--roi_size', type=int, nargs=3, default=(128, 128, 128),
                                 help='roi_size for sliding window inference')
    parser_finetune.add_argument('--sw_batch_size', type=int, default=1,
                                 help='sw_batch_size for sliding window inference')
    parser_finetune.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')

    # Validation
    parser_validation: argparse.ArgumentParser = subparsers.add_parser('validation', help='validation help')
    parser_validation.add_argument('-ckpt', '--init_checkpoint', type=str, required=True,
                                   help='checkpoint for initializing module')
    # Validation: val Inferer
    parser_validation.add_argument('--roi_size', type=int, nargs=3, default=(128, 128, 128),
                                   help='roi_size for sliding window inference')
    parser_validation.add_argument('--sw_batch_size', type=int, default=1,
                                   help='sw_batch_size for sliding window inference')
    parser_validation.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')

    # Test
    parser_test: argparse.ArgumentParser = subparsers.add_parser('test', help='test help')
    parser_test.add_argument('-ckpt', '--init_checkpoint', type=str, required=True,
                             help='checkpoint for initializing module')
    # Test: test Inferer
    parser_test.add_argument('--roi_size', type=int, nargs=3, default=(128, 128, 128),
                             help='roi_size for sliding window inference')
    parser_test.add_argument('--sw_batch_size', type=int, default=1, help='sw_batch_size for sliding window inference')
    parser_test.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')

    # Predict
    parser_predict: argparse.ArgumentParser = subparsers.add_parser('predict', help='predict help')
    parser_predict.add_argument('-ckpt', '--init_checkpoint', type=str, required=True,
                                help='checkpoint for initializing module')
    # Predict: predict Inferer
    parser_predict.add_argument('--roi_size', type=int, nargs=3, default=(128, 128, 128),
                                help='roi_size for sliding window inference')
    parser_predict.add_argument('--sw_batch_size', type=int, default=1,
                                help='sw_batch_size for sliding window inference')
    parser_predict.add_argument('--overlap', type=float, default=0.5, help='overlap for sliding window inference')

    args: argparse.Namespace = parser.parse_args()

    # Common: Experiment
    experiment_root_dir: str = args.experiment_root_dir
    experiment_name: str = args.experiment_name
    experiment_version: str = args.experiment_version

    # Common: Load defaults
    trainer_class: Type[ConfigTrainerSegmentationDefault] = ParserSegmentationDefault.default_trainer_class()
    trainer_init_args: TrainerInitArgs = ParserSegmentationDefault.default_trainer_init_args()
    logger_init_args: LoggerInitArgs = ParserSegmentationDefault.default_logger_init_args()
    datamodule_class: Type[DataModuleSegmentationDefault] = ParserSegmentationDefault.default_datamodule_class()
    module_class: Type[ModuleSegmentationDefault] = ParserSegmentationDefault.default_module_class()
    network_init_args: NamedNetworkInitArgs = \
        ParserSegmentationDefault.default_network_init_args(args.num_sequence, args.num_classes)
    # Followings shall be further loaded in phase logics, here provide blank defaults
    callback_init_args: CallbackInitArgs = CallbackInitArgs()
    datamodule_training_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_validation_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    datamodule_predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    module_training_step_addition_args: Optional[ModuleTrainingStepAdditionArgs] = None
    module_validation_step_addition_args: Optional[ModuleValidationStepAdditionArgs] = None
    module_test_step_addition_args: Optional[ModuleTestStepAdditionArgs] = None
    module_predict_step_addition_args: Optional[ModulePredictStepAdditionArgs] = None

    # Common: Modify configs
    # Common: Trainer
    trainer_init_args.accelerator = args.accelerator
    trainer_init_args.device = args.device[0] if trainer_init_args.accelerator == 'cpu' else args.device
    deterministicMap: Dict[str, Optional[Union[str, bool]]] = {
        'none': None,
        'true': True,
        'false': False,
        'warn': 'warn'
    }
    trainer_init_args.deterministic = deterministicMap[args.deterministic]
    trainer_init_args.accumulate_grad_batches = args.accumulate_grad_batches
    # Common: Logger
    logger_init_args.wandb_project = args.wandb_project

    # Prepare params for specific phase
    if args.routine in {'fit', 'finetune'}:
        # Fit/Finetune: Trainer
        trainer_init_args.max_epochs = args.epochs
        trainer_init_args.accumulate_grad_batches = args.accumulate_grad_batches

        # Fit/Finetune: Callback
        callback_init_args: CallbackInitArgs = ParserSegmentationDefault.default_callback_init_args()
        if args.early_stopping is None:
            callback_init_args.enable_early_stopping = False
            callback_init_args.callback_early_stopping = None
        else:
            callback_init_args.callback_early_stopping.patience = args.early_stopping

        # Fit/Finetune: DataModule (train phase)
        datamodule_training_init_args: DataModuleSegmentationDefaultInitArgs = \
            ParserSegmentationDefault.default_datamodule_training_init_args()
        datamodule_training_init_args.root_dir = args.dataset_root_dir
        datamodule_training_init_args.manifest_file = args.dataset_manifest_file
        datamodule_training_init_args.column_key_map = {}
        datamodule_training_init_args.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in datamodule_training_init_args.column_group_map:
                datamodule_training_init_args.column_group_map['volume'] = []
            datamodule_training_init_args.column_key_map[volume_key] = f'volume_{idx}'
            datamodule_training_init_args.column_group_map['volume'] += f'volume_{idx}'
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in datamodule_training_init_args.column_group_map:
                datamodule_training_init_args.column_group_map['mask'] = []
            datamodule_training_init_args.column_key_map[mask_key] = f'mask_{idx}'
            datamodule_training_init_args.column_group_map['mask'] += f'mask_{idx}'
        datamodule_training_init_args.column_key_relative_path = list(
            datamodule_training_init_args.column_key_map.values())
        datamodule_training_init_args.dataset_params['cache_dir'] = args.cache_dir
        datamodule_training_init_args.batch_size = args.batch_size

        # Fit/Finetune: DataModule (val phase)
        datamodule_validation_init_args = ParserSegmentationDefault.default_datamodule_validation_init_args()
        datamodule_validation_init_args.root_dir = args.val_dataset_root_dir
        datamodule_validation_init_args.manifest_file = args.val_dataset_manifest_file
        datamodule_validation_init_args.column_key_map = {}
        datamodule_validation_init_args.column_group_map = {}
        for idx, volume_key in enumerate(args.val_volume_keys):
            if 'volume' not in datamodule_validation_init_args.column_group_map:
                datamodule_validation_init_args.column_group_map['volume'] = []
            datamodule_validation_init_args.column_key_map[volume_key] = f'volume_{idx}'
            datamodule_validation_init_args.column_group_map['volume'] += f'volume_{idx}'
        for idx, mask_key in enumerate(args.val_mask_keys):
            if 'mask' not in datamodule_validation_init_args.column_group_map:
                datamodule_validation_init_args.column_group_map['mask'] = []
            datamodule_validation_init_args.column_key_map[mask_key] = f'mask_{idx}'
            datamodule_validation_init_args.column_group_map['mask'] += f'mask_{idx}'
        datamodule_validation_init_args.column_key_relative_path = list(
            datamodule_validation_init_args.column_key_map.values())
        datamodule_validation_init_args.dataset_params['cache_dir'] = args.val_cache_dir
        assert args.val_batch_size == 1, f'Routine {args.routine}: val_batch_size must be equal to 1'
        datamodule_validation_init_args.batch_size = args.val_batch_size

        # Fit/Finetune: Optimizer & LR-Scheduler
        module_training_step_addition_args: ModuleTrainingStepAdditionArgs = \
            ParserSegmentationDefault.default_module_training_step_addition_args(args.num_classes)
        module_training_step_addition_args.lrscheduler_init_args.init_args['max_lr'] = args.max_lr
        module_training_step_addition_args.lrscheduler_init_args.init_args['epochs'] = args.epochs
        # OneCycleLR.steps_per_epoch shall also be specified, you must do it manually
        module_training_step_addition_args.lrscheduler_init_args.init_args['steps_per_epoch'] = args.steps_per_epoch
        module_training_step_addition_args.lrscheduler_init_args.init_args['final_div_factor'] = args.final_div_factor

        # Fit/Finetune: Inferer (val phase)
        module_validation_step_addition_args = \
            ParserSegmentationDefault.default_module_validation_step_addition_args(args.num_classes)
        module_validation_step_addition_args.inferer_init_args['roi_size'] = args.roi_size
        module_validation_step_addition_args.inferer_init_args['sw_batch_size'] = args.sw_batch_size
        module_validation_step_addition_args.inferer_init_args['overlap'] = args.overlap

    elif args.routine == 'validation':
        # Val: DataModule
        datamodule_validation_init_args = ParserSegmentationDefault.default_datamodule_validation_init_args()
        datamodule_validation_init_args.root_dir = args.dataset_root_dir
        datamodule_validation_init_args.manifest_file = args.dataset_manifest_file
        datamodule_validation_init_args.column_key_map = {}
        datamodule_validation_init_args.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in datamodule_validation_init_args.column_group_map:
                datamodule_validation_init_args.column_group_map['volume'] = []
            datamodule_validation_init_args.column_key_map[volume_key] = f'volume_{idx}'
            datamodule_validation_init_args.column_group_map['volume'] += f'volume_{idx}'
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in datamodule_validation_init_args.column_group_map:
                datamodule_validation_init_args.column_group_map['mask'] = []
            datamodule_validation_init_args.column_key_map[mask_key] = f'mask_{idx}'
            datamodule_validation_init_args.column_group_map['mask'] += f'mask_{idx}'
        datamodule_validation_init_args.column_key_relative_path = list(
            datamodule_validation_init_args.column_key_map.values())
        datamodule_validation_init_args.dataset_params['cache_dir'] = args.cache_dir
        assert args.batch_size == 1, f'Routine {args.routine}: batch_size must be equal to 1'
        datamodule_validation_init_args.batch_size = args.batch_size

        # Val: Inferer
        module_validation_step_addition_args = \
            ParserSegmentationDefault.default_module_validation_step_addition_args(args.num_classes)
        module_validation_step_addition_args.inferer_init_args['roi_size'] = args.roi_size
        module_validation_step_addition_args.inferer_init_args['sw_batch_size'] = args.sw_batch_size
        module_validation_step_addition_args.inferer_init_args['overlap'] = args.overlap

    elif args.routine == 'test':
        # Test: DataModule
        datamodule_test_init_args = ParserSegmentationDefault.default_datamodule_test_init_args()
        datamodule_test_init_args.root_dir = args.dataset_root_dir
        datamodule_test_init_args.manifest_file = args.dataset_manifest_file
        datamodule_test_init_args.column_key_map = {}
        datamodule_test_init_args.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in datamodule_test_init_args.column_group_map:
                datamodule_test_init_args.column_group_map['volume'] = []
            datamodule_test_init_args.column_key_map[volume_key] = f'volume_{idx}'
            datamodule_test_init_args.column_group_map['volume'] += f'volume_{idx}'
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in datamodule_test_init_args.column_group_map:
                datamodule_test_init_args.column_group_map['mask'] = []
            datamodule_test_init_args.column_key_map[mask_key] = f'mask_{idx}'
            datamodule_test_init_args.column_group_map['mask'] += f'mask_{idx}'
        datamodule_test_init_args.column_key_relative_path = list(datamodule_test_init_args.column_key_map.values())
        datamodule_test_init_args.dataset_params['cache_dir'] = args.cache_dir
        assert args.batch_size == 1, f'Routine {args.routine}: batch_size must be equal to 1'
        datamodule_test_init_args.batch_size = args.batch_size

        # Test: Inferer
        module_test_step_addition_args = \
            ParserSegmentationDefault.default_module_test_step_addition_args(args.num_classes)
        module_test_step_addition_args.inferer_init_args['roi_size'] = args.roi_size
        module_test_step_addition_args.inferer_init_args['sw_batch_size'] = args.sw_batch_size
        module_test_step_addition_args.inferer_init_args['overlap'] = args.overlap

    elif args.routine == 'predict':
        # Predict: DataModule
        datamodule_predict_init_args = ParserSegmentationDefault.default_datamodule_predict_init_args()
        datamodule_predict_init_args.root_dir = args.dataset_root_dir
        datamodule_predict_init_args.manifest_file = args.dataset_manifest_file
        datamodule_predict_init_args.column_key_map = {}
        datamodule_predict_init_args.column_group_map = {}
        idx: int
        volume_key: str
        mask_key: str
        for idx, volume_key in enumerate(args.volume_keys):
            if 'volume' not in datamodule_predict_init_args.column_group_map:
                datamodule_predict_init_args.column_group_map['volume'] = []
            datamodule_predict_init_args.column_key_map[volume_key] = f'volume_{idx}'
            datamodule_predict_init_args.column_group_map['volume'] += f'volume_{idx}'
        for idx, mask_key in enumerate(args.mask_keys):
            if 'mask' not in datamodule_predict_init_args.column_group_map:
                datamodule_predict_init_args.column_group_map['mask'] = []
            datamodule_predict_init_args.column_key_map[mask_key] = f'mask_{idx}'
            datamodule_predict_init_args.column_group_map['mask'] += f'mask_{idx}'
        datamodule_predict_init_args.column_key_relative_path = list(
            datamodule_predict_init_args.column_key_map.values())
        datamodule_predict_init_args.dataset_params['cache_dir'] = args.cache_dir
        assert args.batch_size == 1, f'Routine {args.routine}: batch_size must be equal to 1'
        datamodule_predict_init_args.batch_size = args.batch_size

        # Predict: Inferer
        module_predict_step_addition_args = \
            ParserSegmentationDefault.default_module_predict_step_addition_args()
        module_predict_step_addition_args.inferer_init_args['roi_size'] = args.roi_size
        module_predict_step_addition_args.inferer_init_args['sw_batch_size'] = args.sw_batch_size
        module_predict_step_addition_args.inferer_init_args['overlap'] = args.overlap

    else:
        raise ValueError(f'routine {args.routine!r} not supported')

    # Initialize the Launcher
    launcher: LauncherSegmentationDefault = LauncherSegmentationDefault(
        experiment_root_dir=experiment_root_dir,
        experiment_name=experiment_name,
        experiment_version=experiment_version,
        trainer_class=trainer_class,
        trainer_init_args=trainer_init_args,
        callback_init_args=callback_init_args,
        logger_init_args=logger_init_args,
        module_class=module_class,
        network_init_args=network_init_args,
        datamodule_class=datamodule_class,
        module_training_step_addition_args=module_training_step_addition_args,
        module_validation_step_addition_args=module_validation_step_addition_args,
        module_test_step_addition_args=module_test_step_addition_args,
        module_predict_step_addition_args=module_predict_step_addition_args,
        datamodule_training_init_args=datamodule_training_init_args,
        datamodule_validation_init_args=datamodule_validation_init_args,
        datamodule_test_init_args=datamodule_test_init_args,
        datamodule_predict_init_args=datamodule_predict_init_args
    )

    # Launch specified routine
    if args.routine == 'fit':
        launcher.fit(args.resume_checkpoint)
    elif args.routine == 'finetune':
        launcher.finetune(args.init_checkpoint)
    elif args.routine == 'validation':
        launcher.validation(args.init_checkpoint)
    elif args.routine == 'test':
        launcher.test(args.init_checkpoint)
    elif args.routine == 'predict':
        launcher.predict(args.init_checkpoint)

    print(f'{launcher.__class__.__qualname__} has finished {args.routine!r} routine')
