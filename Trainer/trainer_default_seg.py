# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Literal, Mapping, Collection
import torch
import lightning as L
import lightning.pytorch.loggers
from lightning import Trainer
from pathlib import Path

from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE

from Logger.logger_configurer import CSVLogger, TensorBoardLogger, WandbLogger
from Callback.callback_configurer import (
    DeviceStatsMonitorInitArgs, DeviceStatsMonitor,
    EarlyStoppingInitArgs, EarlyStopping,
    LearningRateMonitorInitArgs, LearningRateMonitor,
    ModelCheckpointInitArgs, ModelCheckpoint,
    ModelSummaryInitArgs, ModelSummary,
    RichModelSummaryInitArgs, RichModelSummary,
    RichProgressBarInitArgs, RichProgressBar,
    TQDMProgressBarInitArgs, TQDMProgressBar
)
from lightning.pytorch.strategies import Strategy
from dataclasses import dataclass

SupportedPrecision = Optional[Union[Literal[64, 32, 16,
"transformer-engine", "transformer-engine-float16",
"16-true", "16-mixed", "bf16-true", "bf16-mixed",
"32-true", "64-true", "64", "32", "16", "bf16"]]]


@dataclass(frozen=True)
class TrainerInitArgs:
    """
    Initialization arguments for Trainer.
    Attributes:
        [Platform control]
        accelerator: Type of accelerator to use for training. 'cpu' for CPU training, 'gpu' for GPU training.
        devices: Number or list of devices to use. For CPU, specify an integer count. For GPU, specify a list of CUDA device IDs like [0], [0,1,2,3].
        precision: Training precision. Supports 64, 32, 16-bit precision and various mixed precision modes like '16-mixed', 'bf16-mixed'.
        enable_distributed_data_parallel: Whether to enable Distributed Data Parallel (DDP) for multi-GPU training.
        [Routine control]
        max_epochs: Maximum number of training epochs.
        check_val_every_n_epoch: Run validation every N epochs. Default is 1 (every epoch).
        [Gradient control]
        accumulate_grad_batches: Accumulate gradients over N batches before performing a backward pass. Useful for simulating larger batch sizes.
        gradient_clip_val: Value for gradient clipping. None means no clipping.
        gradient_clip_algorithm: Algorithm for gradient clipping. 'value' clips by value, 'norm' clips by norm.
        [Logging control]
        log_every_n_steps: Log metrics every N steps.
        enable_progress_bar: Whether to display the default progress bar during training.
        enable_model_summary: Whether to print a default summary of the model architecture.
        enable_checkpointing: Whether to enable default model checkpointing, save last model.
        [Reproducibility control]
        deterministic: If True, enables deterministic algorithms for reproducibility (may raise exceptions). 'warn' warns if deterministic operations are not available.
        [Debugging]
        detect_anomaly: If True, enables autograd anomaly detection to help debug NaN or Inf issues.
        num_sanity_val_steps: Number of validation steps to run before training starts to check for errors.
        fast_dev_run: If True, runs a quick test with only a few batches to verify the training loop works.
        overfit_batches: Percentage or number of batches to overfit on for debugging. 0.0 means disabled.
    """
    # Platform control
    accelerator: Literal["cpu", "gpu"]
    devices: Union[int, List[int], str]
    precision: SupportedPrecision = 32
    enable_distributed_data_parallel: bool = False
    
    # Routine control
    max_epochs: int = 100
    check_val_every_n_epoch: int = 1

    # Gradient control
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[Union[int, float]] = None
    gradient_clip_algorithm: Literal[GradClipAlgorithmType.VALUE, GradClipAlgorithmType.NORM] = \
        GradClipAlgorithmType.NORM

    # Logging control
    log_every_n_steps: int = 50
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    enable_checkpointing: bool = True

    # Reproducibility control
    deterministic: Optional[Union[bool, Literal["warn"]]] = None

    # Debugging
    detect_anomaly: bool = True
    num_sanity_val_steps: int = 2
    fast_dev_run: bool = False
    overfit_batches: Union[int, float] = 0.0


@dataclass(frozen=True)
class CallbackInitArgs:
    """
    Initialization arguments for Callback bundle.
    Attributes:
        [DeviceStatsMonitor]
        enable_device_stats_monitor: Whether to enable DeviceStatsMonitor callback for monitoring CPU/GPU usage.
        device_stats_monitor_init_args: Initialization arguments for DeviceStatsMonitor. Required if enable_device_stats_monitor is True.
        [EarlyStopping]
        enable_early_stopping: Whether to enable EarlyStopping callback to stop training when metric stops improving.
        early_stopping_init_args: Initialization arguments for EarlyStopping. Required if enable_early_stopping is True.
        [LearningRateMonitor]
        enable_learning_rate_monitor: Whether to enable LearningRateMonitor callback for logging learning rate.
        learning_rate_monitor_init_args: Initialization arguments for LearningRateMonitor. Required if enable_learning_rate_monitor is True.
        [ModelSummary]
        enable_model_summary: Whether to enable ModelSummary callback for printing model architecture.
        model_summary_init_args: Initialization arguments for ModelSummary. Required if enable_model_summary is True.
        [RichModelSummary]
        enable_rich_model_summary: Whether to enable RichModelSummary callback for rich-formatted model summary.
        rich_model_summary_init_args: Initialization arguments for RichModelSummary. Required if enable_rich_model_summary is True.
            Note: Cannot enable both ModelSummary and RichModelSummary at the same time.
        [RichProgressBar]
        enable_rich_progressbar: Whether to enable RichProgressBar for rich-formatted progress bar.
        rich_progressbar_init_args: Initialization arguments for RichProgressBar. Required if enable_rich_progressbar is True.
        [TQDMProgressBar]
        enable_tqdm_progressbar: Whether to enable TQDMProgressBar for standard progress bar.
        tqdm_progressbar_init_args: Initialization arguments for TQDMProgressBar. Required if enable_tqdm_progressbar is True.
            Note: Cannot enable both RichProgressBar and TQDMProgressBar at the same time.
        [ModelCheckpoint] List
        model_checkpoint_init_args_collection: List of ModelCheckpoint initialization arguments for saving checkpoints.
            Note: The dirpath in each ModelCheckpointInitArgs should be relative to experiment_root_dir/experiment_name/experiment_version.
    """
    enable_device_stats_monitor: bool = False
    device_stats_monitor_init_args: Optional[DeviceStatsMonitorInitArgs] = None
    enable_early_stopping: bool = False
    early_stopping_init_args: Optional[EarlyStoppingInitArgs] = None
    enable_learning_rate_monitor: bool = False
    learning_rate_monitor_init_args: Optional[LearningRateMonitorInitArgs] = None
    #  Select only one model summary callback
    enable_model_summary: bool = False
    model_summary_init_args: Optional[ModelSummaryInitArgs] = None
    enable_rich_model_summary: bool = False
    rich_model_summary_init_args: Optional[RichModelSummaryInitArgs] = None
    # Select only one progressbar
    enable_rich_progressbar: bool = False
    rich_progressbar_init_args: Optional[RichProgressBarInitArgs] = None
    enable_tqdm_progressbar: bool = False
    tqdm_progressbar_init_args: Optional[TQDMProgressBarInitArgs] = None
    # Model checkpoints
    # Note: shall specify dirpath for ModelCheckpoint relative to
    # experiment_root_dir/experiment_name/experiment_version
    model_checkpoint_init_args_collection: Optional[List[ModelCheckpointInitArgs]] = None

    def __post_init__(self):
        assert not (self.enable_model_summary and self.enable_rich_model_summary), \
            'Can not enable multiple model summaries at the same time'
        assert not (self.enable_rich_progressbar and self.enable_tqdm_progressbar), \
            'Can not enable multiple progressbars at the same time'


@dataclass(frozen=True)
class LoggerInitArgs:
    """
    Initialization arguments for Logger.
    Attributes:
        enable_csv_logger: Whether to enable CSVLogger for logging metrics to CSV files.
        enable_tensorboard_logger: Whether to enable TensorBoardLogger for logging metrics to TensorBoard.
        enable_wandb_logger: Whether to enable WandbLogger for logging metrics to Weights & Biases.
        wandb_project: Name of the Wandb project. Required if enable_wandb_logger is True.
    """
    enable_csv_logger: bool = True
    enable_tensorboard_logger: bool = False
    enable_wandb_logger: bool = False
    wandb_project: Optional[str] = None

    def __post_init__(self):
        if self.enable_wandb_logger:
            assert self.wandb_project is not None, f'When enable_wandb_logger={self.enable_wandb_logger}, you must specify wandb_project'


class TrainerSegmentationDefault:
    """
    Default trainer class for segmentation tasks.

    Provides a high-level interface for training, validating, testing, and predicting with segmentation models.
    Integrates PyTorch Lightning Trainer with custom callbacks and loggers for comprehensive experiment tracking.

    Attributes:
        experiment_root_dir: Root directory for all experiments.
        experiment_name: Name of the current experiment.
        experiment_version: Version identifier for the current experiment run.
        trainer_init_args: Configuration arguments for the PyTorch Lightning Trainer.
        callback_init_args: Configuration arguments for training callbacks.
        logger_init_args: Configuration arguments for experiment loggers.
        trainer: The initialized PyTorch Lightning Trainer instance.
    """

    def __init__(
            self,
            experiment_root_dir: Union[str, os.PathLike, Path],
            experiment_name: str = 'pipeunet',
            experiment_version: str = '001',
            trainer_init_args: TrainerInitArgs = TrainerInitArgs(
                accelerator='cpu',  # or 'gpu'
                # if 'cpu', specify an int device count for usage
                # if 'gpu', specify a CUDA device list, such as [0], [0,1,2,3]
                devices=1,
                precision=32,
                max_epochs=100,
            ),
            callback_init_args: CallbackInitArgs = CallbackInitArgs(),
            logger_init_args: LoggerInitArgs = LoggerInitArgs(),

    ):
        """
        Initialize the TrainerSegmentationDefault instance.

        Args:
            experiment_root_dir: Root directory where experiment logs and checkpoints will be saved.
            experiment_name: Name of the experiment for organizing logs. Defaults to 'pipeunet'.
            experiment_version: Version string for the experiment run. Defaults to '001'.
            trainer_init_args: Configuration for PyTorch Lightning Trainer. Defaults to CPU training with 32-bit precision.
            callback_init_args: Configuration for training callbacks. Defaults to no callbacks.
            logger_init_args: Configuration for experiment loggers. Defaults to CSV logger only.
        """
        assert experiment_root_dir is not None, 'experiment_root_dir must be specified'
        self.experiment_root_dir: Path = Path(experiment_root_dir)
        self.experiment_name: str = experiment_name
        self.experiment_version: str = experiment_version
        self.trainer_init_args: TrainerInitArgs = trainer_init_args
        self.callback_init_args: CallbackInitArgs = callback_init_args
        self.logger_init_args: LoggerInitArgs = logger_init_args

        self.trainer: Trainer = self._get_trainer()

    def _get_trainer(self) -> Trainer:
        if hasattr(self, 'trainer') and self.trainer is not None:
            return self.trainer

        # If not exists, creating new Trainer
        callbacks: List[L.pytorch.callbacks.Callback] = self._get_callbacks()
        loggers: List[L.pytorch.loggers.Logger] = self._get_loggers()
        strategy: Union[Strategy, str] = self._get_strategy()

        assert self.trainer_init_args is not None, 'trainer_init_args must be specified before _get_trainer'
        init: TrainerInitArgs = self.trainer_init_args
        self.trainer = Trainer(
            # Routine control
            accelerator=init.accelerator,
            devices=init.devices,
            precision=init.precision,
            strategy=strategy,
            max_epochs=init.max_epochs,
            check_val_every_n_epoch=init.check_val_every_n_epoch,

            # Loggers and Callbacks
            logger=loggers,
            callbacks=callbacks,

            # Gradient control
            accumulate_grad_batches=init.accumulate_grad_batches,
            gradient_clip_val=init.gradient_clip_val,
            gradient_clip_algorithm=init.gradient_clip_algorithm,

            # Logging control
            log_every_n_steps=init.log_every_n_steps,
            enable_progress_bar=init.enable_progress_bar,
            enable_model_summary=init.enable_model_summary,
            enable_checkpointing=init.enable_checkpointing,
            default_root_dir=self.experiment_root_dir / self.experiment_name / self.experiment_version,

            # Reproducibility control
            deterministic=init.deterministic,

            # Distributed training control
            sync_batchnorm=True if init.enable_distributed_data_parallel else False,

            # Debugging
            detect_anomaly=init.detect_anomaly,
            num_sanity_val_steps=init.num_sanity_val_steps,
            fast_dev_run=init.fast_dev_run,
            overfit_batches=init.overfit_batches
        )

        return self.trainer

    def _get_callbacks(self) -> List[L.pytorch.callbacks.Callback]:
        callbacks = []
        if self.callback_init_args is None:
            return callbacks

        init: CallbackInitArgs = self.callback_init_args

        # Add required singleton callbacks
        if init.enable_device_stats_monitor:
            assert init.device_stats_monitor_init_args is not None, f'When enabled DeviceStatsMonitor (enable_device_stats_monitor={init.enable_device_stats_monitor}), you must specify device_stats_monitor_init_args'
            callbacks.append(DeviceStatsMonitor(**vars(init.device_stats_monitor_init_args)))
        if init.enable_early_stopping:
            assert init.early_stopping_init_args is not None, f'When enabled EarlyStopping (enable_early_stopping={init.enable_early_stopping}), you must specify early_stopping_init_args)'
            callbacks.append(EarlyStopping(**vars(init.early_stopping_init_args)))
        if init.enable_learning_rate_monitor:
            assert init.learning_rate_monitor_init_args is not None, f'When enabled LearningRateMonitor (enable_learning_rate_monitor={init.enable_learning_rate_monitor}), you must specify learning_rate_monitor_init_args)'
            callbacks.append(LearningRateMonitor(**vars(init.learning_rate_monitor_init_args)))
        if init.enable_model_summary:
            assert init.model_summary_init_args is not None, f'When enabled ModelSummary (enable_model_summary={init.enable_model_summary}), you must specify model_summary_init_args)'
            callbacks.append(ModelSummary(**vars(init.model_summary_init_args)))
        if init.enable_rich_model_summary:
            assert init.rich_model_summary_init_args is not None, f'When enabled RichModelSummary (enable_rich_model_summary={init.enable_rich_model_summary}), you must specify rich_model_summary_init_args)'
            callbacks.append(RichModelSummary(**vars(init.rich_model_summary_init_args)))
        if init.enable_rich_progressbar:
            assert init.rich_progressbar_init_args is not None, f'When enabled RichProgressBar (enable_rich_progressbar={init.enable_rich_progressbar}), you must specify rich_progressbar_init_args'
            callbacks.append(RichProgressBar(**vars(init.rich_progressbar_init_args)))
        if init.enable_tqdm_progressbar:
            assert init.tqdm_progressbar_init_args is not None, f'When enabled TQDMProgressBar (enable_tqdm_progressbar={init.enable_tqdm_progressbar}), you must specify tqdm_progressbar_init_args'
            callbacks.append(TQDMProgressBar(**vars(init.tqdm_progressbar_init_args)))

        # Add ModelCheckpoints
        model_checkpoints: List[L.pytorch.callbacks.ModelCheckpoint] = self._get_model_checkpoints()
        callbacks.extend(model_checkpoints)

        return callbacks

    def _get_model_checkpoints(self) -> List[ModelCheckpoint]:
        assert self.callback_init_args is not None, f'callback_init_args must be specified before _get_model_checkpoints'
        checkpoints = []
        checkpoint_inits: List[ModelCheckpointInitArgs] = self.callback_init_args.model_checkpoint_init_args_collection
        if checkpoint_inits is None or len(checkpoint_inits) == 0:
            return checkpoints

        model_checkpoint_dir: Path = self.experiment_root_dir

        for init in checkpoint_inits:
            if init is None:
                continue
            init_dict: Dict[str, Any] = vars(init)
            # Inject experiment_name and experiment_version as sub-dirs
            init_dict['dirpath'] = (model_checkpoint_dir / self.experiment_name /
                                    self.experiment_version / init_dict['dirpath'])
            checkpoints.append(ModelCheckpoint(**init_dict))

        return checkpoints

    def _get_loggers(self) -> List[L.pytorch.loggers.Logger]:
        loggers: List[L.pytorch.loggers.Logger] = []
        if self.logger_init_args is None:
            return loggers

        init: LoggerInitArgs = self.logger_init_args

        # CSVLogger
        if init.enable_csv_logger:
            csv_log_dir: Path = self.experiment_root_dir
            csv_logger: L.pytorch.loggers.CSVLogger = CSVLogger(
                save_dir=csv_log_dir,
                name=self.experiment_name,
                version=self.experiment_version,
                flush_logs_every_n_steps=self.trainer_init_args.log_every_n_steps
            )
            loggers.append(csv_logger)

        # TensorBoardLogger
        if init.enable_tensorboard_logger:
            tb_log_dir: Path = self.experiment_root_dir
            tb_logger: L.pytorch.loggers.TensorBoardLogger = TensorBoardLogger(
                save_dir=tb_log_dir,
                name=self.experiment_name,
                version=self.experiment_version,
                sub_dir='tensorboard'
            )
            loggers.append(tb_logger)

        # WandbLogger
        wb_log_dir: Path = self.experiment_root_dir / self.experiment_name / self.experiment_version
        wandb_logger: L.pytorch.loggers.WandbLogger = WandbLogger(
            project=init.wandb_project,
            name=f"{self.experiment_name}_{self.experiment_version}",
            save_dir=wb_log_dir,
            log_model=False
        )
        loggers.append(wandb_logger)

        return loggers

    def _get_strategy(self) -> Union[Strategy, str]:
        assert self.trainer_init_args is not None, f'trainer_init_args must be specified before _get_strategy'
        init: TrainerInitArgs = self.trainer_init_args
        if init.enable_distributed_data_parallel and init.accelerator == 'gpu':
            strategy = L.pytorch.strategies.DDPStrategy()
            return strategy
        return 'auto'

    def fit(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None,  # shall always be a file path, not
            finetune: bool = False,
            finetune_map_location: _MAP_LOCATION_TYPE = None,
            finetune_hparams_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        trainer: Trainer = self._get_trainer()

        print(f'[Fit Starts] {self.experiment_name}-{self.experiment_version}')
        print(f'  Accelerator: {self.trainer_init_args.accelerator}\n'
              f'  Devices: {self.trainer_init_args.devices}')

        if ckpt_path is not None:
            if finetune:
                print('  [Finetune Mode Registered]')
                print(f'  Checkpoint (pre-trained): {ckpt_path}')
                if finetune_map_location is not None:
                    print(f'  Device Mapping: {finetune_map_location}')
                if finetune_hparams_file is not None:
                    print(f'  Hyper-Params File: {finetune_hparams_file}')
                model.load_from_checkpoint(ckpt_path, finetune_map_location, finetune_hparams_file)
            else:
                print('  [Resume Fitting]')
                print(f'  Checkpoint (resumed): {ckpt_path}')

        # Start fitting
        start_time: datetime = datetime.now()
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path if not finetune else None)
        end_time: datetime = datetime.now()

        training_time: timedelta = end_time - start_time
        print(f"[Fit Ends] Time consumed: {training_time}")

        # Fitting results
        results = {
            'experiment_root_dir': self.experiment_root_dir,
            'experiment_name': self.experiment_name,
            'experiment_version': self.experiment_version,
            'accelerator': self.trainer_init_args.accelerator,
            'devices': self.trainer_init_args.devices,
            'max_epochs': self.trainer_init_args.max_epochs,
            'training_time': str(training_time),
            'metrics': trainer.logged_metrics
        }

        return results

    def validate(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[str] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        trainer: Trainer = self._get_trainer()

        print(f"[Validate Starts] {self.experiment_name}-{self.experiment_version}")
        print(f"  Accelerator: {self.trainer_init_args.accelerator}\n"
              f"  Devices: {self.trainer_init_args.devices}")

        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

        # Start validating
        start_time: datetime = datetime.now()
        val_results = trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=verbose)
        end_time: datetime = datetime.now()

        validating_time: timedelta = end_time - start_time
        print(f"[Validate Ends] Time consumed: {validating_time}")

        # Validating results
        results = {
            'experiment_root_dir': self.experiment_root_dir,
            'experiment_name': self.experiment_name,
            'experiment_version': self.experiment_version,
            'accelerator': self.trainer_init_args.accelerator,
            'devices': self.trainer_init_args.devices,
            'validating_time': str(validating_time),
            'metrics': val_results
        }

        return results

    def test(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[str] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        trainer: Trainer = self._get_trainer()

        print(f"[Test Starts] {self.experiment_name}-{self.experiment_version}")
        print(f"  Accelerator: {self.trainer_init_args.accelerator}\n"
              f"  Devices: {self.trainer_init_args.devices}")

        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

        # Start testing
        start_time: datetime = datetime.now()
        val_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=verbose)
        end_time: datetime = datetime.now()

        testing_time: timedelta = end_time - start_time
        print(f"[Test Ends] Time consumed: {testing_time}")

        # Testing results
        results = {
            'experiment_root_dir': self.experiment_root_dir,
            'experiment_name': self.experiment_name,
            'experiment_version': self.experiment_version,
            'accelerator': self.trainer_init_args.accelerator,
            'devices': self.trainer_init_args.devices,
            'testing_time': str(testing_time),
            'metrics': val_results
        }

        return results

    def predict(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        trainer: Trainer = self._get_trainer()

        print(f"[Predict Starts] {self.experiment_name}-{self.experiment_version}")
        print(f"  Accelerator: {self.trainer_init_args.accelerator}\n"
              f"  Devices: {self.trainer_init_args.devices}")

        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

        # Start predicting
        start_time: datetime = datetime.now()
        predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        end_time = datetime.now()

        predicting_time: timedelta = end_time - start_time
        print(f"[Predict Ends] Time consumed: {predicting_time}")

        results = {
            'experiment_root_dir': self.experiment_root_dir,
            'experiment_name': self.experiment_name,
            'experiment_version': self.experiment_version,
            'accelerator': self.trainer_init_args.accelerator,
            'devices': self.trainer_init_args.devices,
            'predicting_time': str(predicting_time),
            'predictions': predictions
        }

        return results


if __name__ == "__main__":
    from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

    # Test Trainer init on CPU
    trainer_init_args: TrainerInitArgs = TrainerInitArgs(
        # Platform control
        accelerator='cpu',
        devices=8,
        precision=32,
        enable_distributed_data_parallel=False,
        # Routine control
        max_epochs=100,
        check_val_every_n_epoch=1,
        # Gradient control
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm=GradClipAlgorithmType.NORM,
        # Logging control
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        # Reproducibility control
        deterministic='warn',
        # Debugging
        detect_anomaly=True,
        num_sanity_val_steps=2,
        fast_dev_run=False,
        overfit_batches=0.0
    )

    model_checkpoint_init_args_list: List[ModelCheckpointInitArgs] = []

    callback_init_args: CallbackInitArgs = CallbackInitArgs(
        enable_device_stats_monitor=True,
        device_stats_monitor_init_args=DeviceStatsMonitorInitArgs(cpu_stats=True),
        enable_early_stopping=True,
        early_stopping_init_args=EarlyStoppingInitArgs(
            monitor='val/loss',
            patience=100,
            mode='min',
            verbose=True
        ),
        enable_learning_rate_monitor=True,
        learning_rate_monitor_init_args=LearningRateMonitorInitArgs(
            logging_interval='epoch',
            log_momentum=True,
            log_weight_decay=True
        ),
        enable_rich_model_summary=True,
        rich_model_summary_init_args=RichModelSummaryInitArgs(max_depth=5),
        enable_rich_progressbar=True,
        rich_progressbar_init_args=RichProgressBarInitArgs(
            refresh_rate=1,
            leave=True,
            theme=RichProgressBarTheme(),
            console_kwargs=None
        ),
        model_checkpoint_init_args_collection=[
            ModelCheckpointInitArgs(
                dirpath='milestone',
                filename='{epoch:03d}-loss={val/loss:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='epoch',
                save_top_k=-1,
                mode='max',
                save_last=False,
                every_n_epochs=10
            ),
            ModelCheckpointInitArgs(
                dirpath='val/dice',
                filename='{epoch:03d}-loss={val/loss:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/dice',
                save_top_k=3,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/hd95',
                filename='{epoch:03d}-loss={val/loss:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/hd95',
                save_top_k=3,
                mode='min',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/assd',
                filename='{epoch:03d}-assd={val/assd:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/assd',
                save_top_k=3,
                mode='min',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/nsd',
                filename='{epoch:03d}-nsd={val/nsd:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/nsd',
                save_top_k=3,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/accuracy',
                filename='{epoch:03d}-accuracy={val/accuracy:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/acc',
                save_top_k=3,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/precision',
                filename='{epoch:03d}-precision={val/precision:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/precision',
                save_top_k=3,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/specificity',
                filename='{epoch:03d}-spe={val/specificity:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/specificity',
                save_top_k=3,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/recall',
                filename='{epoch:03d}-recall={val/recall:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/recall',
                save_top_k=3,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ModelCheckpointInitArgs(
                dirpath='val/auroc',
                filename='{epoch:03d}-auroc={val/auroc:4f}-dice={val/dice:4f}-hd95{val/hd95:4f}',
                monitor='val/auroc',
                save_top_k=3,
                mode='max',
                save_last=False,
                every_n_epochs=1
            )
        ]
    )

    logger_init_args: LoggerInitArgs = LoggerInitArgs(
        enable_csv_logger=True,
        enable_tensorboard_logger=True,
        enable_wandb_logger=True,
        wandb_project='pipeunet-trainer-test-000-tutorial'
    )

    trainer: TrainerSegmentationDefault = TrainerSegmentationDefault(
        experiment_root_dir='./Samples/trainer_test',
        experiment_name='pipeunet-trainer-test',
        experiment_version='000-tutorial',
        trainer_init_args=trainer_init_args,
        callback_init_args=callback_init_args,
        logger_init_args=logger_init_args
    )

    if torch.cuda.is_available():
        # Test Trainer init on GPU
        trainer_init_args: TrainerInitArgs = TrainerInitArgs(
            # Platform control
            accelerator='gpu',
            devices=[0],
            precision=32,
            enable_distributed_data_parallel=False,
            # Routine control
            max_epochs=100,
            check_val_every_n_epoch=1,
            # Gradient control
            accumulate_grad_batches=1,
            gradient_clip_val=None,
            gradient_clip_algorithm=GradClipAlgorithmType.NORM,
            # Logging control
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            # Reproducibility control
            deterministic='warn',
            # Debugging
            detect_anomaly=True,
            num_sanity_val_steps=2,
            fast_dev_run=False,
            overfit_batches=0.0
        )

        trainer: TrainerSegmentationDefault = TrainerSegmentationDefault(
            experiment_root_dir='./Samples/trainer_test',
            experiment_name='pipeunet-trainer-test',
            experiment_version='000-gpu',
            trainer_init_args=trainer_init_args,
            callback_init_args=callback_init_args,
            logger_init_args=logger_init_args
        )


    # Use trainer for specified purposes
    # trainer.fit(...)
    # trainer.validate(...)
    # trainer.test(...)
    # trainer.predict(...)