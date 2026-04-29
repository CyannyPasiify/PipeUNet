# -*- coding: utf-8 -*-
import gc
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Literal, Mapping, Collection, cast
import torch
import lightning as L
import lightning.pytorch.loggers
from lightning import Trainer
from pathlib import Path
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE
from typing_extensions import override

from Logger.logger_configurer import ConfigLoggerCSV, ConfigLoggerTensorBoard, ConfigLoggerWandb
from Callback.callback_configurer import (
    ConfigCallbackDeviceStatsMonitor,
    ConfigCallbackEarlyStopping,
    ConfigCallbackLearningRateMonitor,
    ConfigCallbackModelCheckpoint,
    ConfigCallbackModelSummary,
    ConfigCallbackRichModelSummary,
    ConfigCallbackRichProgressBar,
    ConfigCallbackTQDMProgressBar
)
from lightning.pytorch.strategies import Strategy
from dataclasses import dataclass

from Module.ltn_module_segmentation_default import LightningModuleSegmentationDefault

SupportedPrecision = Optional[Union[Literal[64, 32, 16,
"transformer-engine", "transformer-engine-float16",
"16-true", "16-mixed", "bf16-true", "bf16-mixed",
"32-true", "64-true", "64", "32", "16", "bf16"]]]


@dataclass
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
    accelerator: Literal["cpu", "gpu"] = 'cpu'
    devices: Union[int, List[int], str] = 1
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
    num_sanity_val_steps: int = 0
    fast_dev_run: bool = False
    overfit_batches: Union[int, float] = 0.0


@dataclass
class CallbackInitArgs:
    """
    Initialization for Callback bundle.
    Attributes:
        [DeviceStatsMonitor]
        callback_device_stats_monitor: Initialization for DeviceStatsMonitor. Required if enable_device_stats_monitor is True.
        [EarlyStopping]
        callback_early_stopping: Initialization for EarlyStopping. Required if enable_early_stopping is True.
        [LearningRateMonitor]
        callback_learning_rate_monitor: Initialization for LearningRateMonitor. Required if enable_learning_rate_monitor is True.
        [ModelSummary]
        callback_model_summary: Initialization for ModelSummary. Required if enable_model_summary is True.
        [RichModelSummary]
        callback_rich_model_summary: Initialization for RichModelSummary. Required if enable_rich_model_summary is True.
            Note: Cannot enable both ModelSummary and RichModelSummary at the same time.
        [RichProgressBar]
        callback_rich_progressbar: Initialization for RichProgressBar. Required if enable_rich_progressbar is True.
        [TQDMProgressBar]
        callback_tqdm_progressbar: Initialization for TQDMProgressBar. Required if enable_tqdm_progressbar is True.
            Note: Cannot enable both RichProgressBar and TQDMProgressBar at the same time.
        [ModelCheckpoint] List
        callback_model_checkpoints: List of ModelCheckpoint Initialization for saving checkpoints.
            Note: The dirpath in each ModelCheckpoint should be relative to experiment_root_dir/experiment_name/experiment_version.
    """
    callback_device_stats_monitor: Optional[ConfigCallbackDeviceStatsMonitor] = None
    callback_early_stopping: Optional[ConfigCallbackEarlyStopping] = None
    callback_learning_rate_monitor: Optional[ConfigCallbackLearningRateMonitor] = None
    #  Select only one model summary callback
    callback_model_summary: Optional[ConfigCallbackModelSummary] = None
    callback_rich_model_summary: Optional[ConfigCallbackRichModelSummary] = None
    # Select only one progressbar
    callback_rich_progressbar: Optional[ConfigCallbackRichProgressBar] = None
    callback_tqdm_progressbar: Optional[ConfigCallbackTQDMProgressBar] = None
    # Model checkpoints
    # Note: shall specify dirpath for ModelCheckpoint relative to
    # experiment_root_dir/experiment_name/experiment_version
    callback_model_checkpoints: Optional[List[ConfigCallbackModelCheckpoint]] = None

    def __post_init__(self):
        assert self.callback_model_summary is None or self.callback_rich_model_summary is None, \
            'Can not enable multiple model summaries at the same time'
        assert self.callback_tqdm_progressbar is None or self.callback_rich_progressbar is None, \
            'Can not enable multiple progressbars at the same time'


@dataclass
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


@dataclass
class ConfigTrainerBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "trainer")

    def _assert_init_essentials(
            self,
            *args,
            **kwargs
    ) -> None:
        if self.is_ready(): return
        self.init_essentials(*args, **kwargs)

    @abstractmethod
    def init_essentials(
            self,
            *args,
            **kwargs
    ) -> 'ConfigTrainerBase':
        self.trainer = None  # Just placeholder
        return self

    def get_trainer(self) -> Trainer:
        self._assert_init_essentials()
        return self.trainer

    @abstractmethod
    def fit(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None,
            finetune: bool = False,
            finetune_map_location: _MAP_LOCATION_TYPE = None,
            finetune_hparams_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def test(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        pass


@dataclass
class ConfigTrainerSegmentationDefault(ConfigTrainerBase):
    """
    Default trainer class for segmentation tasks.

    Provides a high-level interface for training, validating, testing, and predicting with segmentation models.
    Integrates PyTorch Lightning Trainer with custom callbacks and loggers for comprehensive experiment tracking.

    Attributes:
        experiment_root_dir: Root directory where experiment logs and checkpoints will be saved.
        experiment_name: Name of the experiment for organizing logs. Defaults to 'pipeunet'.
        experiment_version: Version string for the experiment run. Defaults to '001'.
        trainer_init_args: Configuration for PyTorch Lightning Trainer. Defaults to CPU training with 32-bit precision.
        callback_init_args: Configuration for training callbacks. Defaults to no callbacks.
        logger_init_args: Configuration for experiment loggers. Defaults to CSV logger only.
    """
    experiment_root_dir: Union[str, os.PathLike, Path] = 'Experiments'
    experiment_name: str = 'pipeunet'
    experiment_version: str = '001'
    trainer_init_args: TrainerInitArgs = TrainerInitArgs(
        accelerator='cpu',  # or 'gpu'
        # if 'cpu', specify an int device count for usage
        # if 'gpu', specify a CUDA device list, such as [0], [0,1,2,3]
        devices=1,
        precision=32,
        max_epochs=100
    )
    callback_init_args: CallbackInitArgs = CallbackInitArgs()
    logger_init_args: LoggerInitArgs = LoggerInitArgs()

    @override
    def init_essentials(self) -> 'ConfigTrainerSegmentationDefault':
        assert self.experiment_root_dir is not None, 'experiment_root_dir must be specified'
        self.experiment_root_dir: Path = Path(self.experiment_root_dir)
        self.trainer: Trainer = self._get_trainer()

        return self

    def _get_trainer(self) -> Trainer:
        # If not exists, creating new Trainer
        callbacks: List[L.pytorch.callbacks.Callback] = self._get_callbacks()
        loggers: List[L.pytorch.loggers.Logger] = self._get_loggers()
        strategy: Union[Strategy, str] = self._get_strategy()

        assert self.trainer_init_args is not None, 'trainer_init_args must be specified before _get_trainer'
        init: TrainerInitArgs = self.trainer_init_args
        trainer = Trainer(
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

        return trainer

    def _get_callbacks(self) -> List[L.pytorch.callbacks.Callback]:
        callbacks: List[L.pytorch.callbacks.Callback] = []
        if self.callback_init_args is None:
            return callbacks

        init: CallbackInitArgs = self.callback_init_args

        # Add required singleton callbacks
        if init.callback_device_stats_monitor is not None:
            callbacks.append(init.callback_device_stats_monitor.get_callback_hooker())
        if init.callback_early_stopping is not None:
            callbacks.append(init.callback_early_stopping.get_callback_hooker())
        if init.callback_learning_rate_monitor is not None:
            callbacks.append(init.callback_learning_rate_monitor.get_callback_hooker())
        if init.callback_model_summary is not None:
            callbacks.append(init.callback_model_summary.get_callback_hooker())
        if init.callback_rich_model_summary is not None:
            callbacks.append(init.callback_rich_model_summary.get_callback_hooker())
        if init.callback_rich_progressbar is not None:
            callbacks.append(init.callback_rich_progressbar.get_callback_hooker())
        if init.callback_tqdm_progressbar is not None:
            callbacks.append(init.callback_tqdm_progressbar.get_callback_hooker())

        # Add ModelCheckpoints
        model_checkpoints: List[L.pytorch.callbacks.ModelCheckpoint] = self._get_model_checkpoints()
        callbacks.extend(model_checkpoints)

        return callbacks

    def _get_model_checkpoints(self) -> List[L.pytorch.callbacks.ModelCheckpoint]:
        assert self.callback_init_args is not None, f'callback_init_args must be specified before _get_model_checkpoints'
        checkpoints: List[L.pytorch.callbacks.ModelCheckpoint] = []
        checkpoint_inits: List[ConfigCallbackModelCheckpoint] = self.callback_init_args.callback_model_checkpoints
        if checkpoint_inits is None or len(checkpoint_inits) == 0:
            return checkpoints

        model_checkpoint_dir: Path = self.experiment_root_dir

        init: ConfigCallbackModelCheckpoint
        for init in checkpoint_inits:
            if init is None:
                continue
            # Inject experiment_name and experiment_version as sub-dirs
            init.dirpath = model_checkpoint_dir / self.experiment_name / self.experiment_version / init.dirpath
            checkpoints.append(cast(L.pytorch.callbacks.ModelCheckpoint, init.get_callback_hooker()))

        return checkpoints

    def _get_loggers(self) -> List[L.pytorch.loggers.Logger]:
        loggers: List[L.pytorch.loggers.Logger] = []
        if self.logger_init_args is None:
            return loggers

        init: LoggerInitArgs = self.logger_init_args

        # CSVLogger
        if init.enable_csv_logger:
            csv_log_dir: Path = self.experiment_root_dir
            csv_logger: L.pytorch.loggers.CSVLogger = cast(
                L.pytorch.loggers.CSVLogger,
                ConfigLoggerCSV(
                    save_dir=csv_log_dir,
                    name=self.experiment_name,
                    version=self.experiment_version,
                    flush_logs_every_n_steps=self.trainer_init_args.log_every_n_steps
                ).get_logger()
            )
            loggers.append(csv_logger)

        # TensorBoardLogger
        if init.enable_tensorboard_logger:
            tb_log_dir: Path = self.experiment_root_dir
            tb_logger: L.pytorch.loggers.TensorBoardLogger = cast(
                L.pytorch.loggers.TensorBoardLogger,
                ConfigLoggerTensorBoard(
                    save_dir=tb_log_dir,
                    name=self.experiment_name,
                    version=self.experiment_version,
                    sub_dir='tensorboard'
                ).get_logger()
            )
            loggers.append(tb_logger)

        # WandbLogger
        wb_log_dir: Path = self.experiment_root_dir / self.experiment_name / self.experiment_version
        wandb_logger: L.pytorch.loggers.WandbLogger = cast(
            L.pytorch.loggers.WandbLogger,
            ConfigLoggerWandb(
                project=init.wandb_project,
                name=f"{self.experiment_name}_{self.experiment_version}",
                save_dir=wb_log_dir,
                log_model=False
            ).get_logger()
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

    @override
    def fit(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None,
            finetune: bool = False,
            finetune_map_location: _MAP_LOCATION_TYPE = None,
            finetune_hparams_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        self._assert_init_essentials()

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
                model: LightningModuleSegmentationDefault
                ckpt = torch.load(ckpt_path, finetune_map_location)
                model.load_state_dict(ckpt["state_dict"])
                # model= type(model).load_from_checkpoint(
                #     checkpoint_path=ckpt_path,
                #     map_location=finetune_map_location,
                #     hparams_file=finetune_hparams_file
                # )
                del ckpt
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print('  [Resume Fitting]')
                print(f'  Checkpoint (resumed): {ckpt_path}')

        # Start fitting
        start_time: datetime = datetime.now()
        self.trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path if not finetune else None)
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
            'metrics': self.trainer.logged_metrics
        }

        return results

    @override
    def validate(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        self._assert_init_essentials()

        print(f"[Validate Starts] {self.experiment_name}-{self.experiment_version}")
        print(f"  Accelerator: {self.trainer_init_args.accelerator}\n"
              f"  Devices: {self.trainer_init_args.devices}")

        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

        # Start validating
        start_time: datetime = datetime.now()
        val_results = self.trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=verbose)
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

    @override
    def test(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        self._assert_init_essentials()

        print(f"[Test Starts] {self.experiment_name}-{self.experiment_version}")
        print(f"  Accelerator: {self.trainer_init_args.accelerator}\n"
              f"  Devices: {self.trainer_init_args.devices}")

        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

        # Start testing
        start_time: datetime = datetime.now()
        val_results = self.trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=verbose)
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

    @override
    def predict(
            self,
            model: L.LightningModule,
            datamodule: L.LightningDataModule,
            ckpt_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        self._assert_init_essentials()

        print(f"[Predict Starts] {self.experiment_name}-{self.experiment_version}")
        print(f"  Accelerator: {self.trainer_init_args.accelerator}\n"
              f"  Devices: {self.trainer_init_args.devices}")

        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

        # Start predicting
        start_time: datetime = datetime.now()
        predictions = self.trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
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

    model_checkpoint_init_args_list: List[ConfigCallbackModelCheckpoint] = []

    callback_init_args: CallbackInitArgs = CallbackInitArgs(
        callback_device_stats_monitor=ConfigCallbackDeviceStatsMonitor(cpu_stats=True),
        callback_early_stopping=ConfigCallbackEarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
            verbose=True
        ),
        callback_learning_rate_monitor=ConfigCallbackLearningRateMonitor(
            logging_interval='epoch',
            log_momentum=True,
            log_weight_decay=True
        ),
        callback_rich_model_summary=ConfigCallbackRichModelSummary(max_depth=5),
        callback_rich_progressbar=ConfigCallbackRichProgressBar(
            refresh_rate=1,
            leave=True,
            theme=RichProgressBarTheme(),
            console_kwargs=None
        ),
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
            ConfigCallbackModelCheckpoint(
                dirpath='val/HD95',
                filename='{epoch:03d}-loss={val/loss:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/HD95',
                save_top_k=5,
                mode='min',
                save_last=False,
                every_n_epochs=1
            ),
            ConfigCallbackModelCheckpoint(
                dirpath='val/NSD',
                filename='{epoch:03d}-NSD={val/NSD:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/NSD',
                save_top_k=5,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ConfigCallbackModelCheckpoint(
                dirpath='val/Acc',
                filename='{epoch:03d}-Acc={val/Acc:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/Acc',
                save_top_k=5,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ConfigCallbackModelCheckpoint(
                dirpath='val/Prec',
                filename='{epoch:03d}-Prec={val/Prec:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/Prec',
                save_top_k=5,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ConfigCallbackModelCheckpoint(
                dirpath='val/Spec',
                filename='{epoch:03d}-Spec={val/Spec:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/Spec',
                save_top_k=5,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ConfigCallbackModelCheckpoint(
                dirpath='val/Recall',
                filename='{epoch:03d}-Recall={val/Recall:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/Recall',
                save_top_k=5,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ConfigCallbackModelCheckpoint(
                dirpath='val/F1',
                filename='{epoch:03d}-F1={val/F1:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/F1',
                save_top_k=5,
                mode='max',
                save_last=False,
                every_n_epochs=1
            ),
            ConfigCallbackModelCheckpoint(
                dirpath='val/AUROC',
                filename='{epoch:03d}-AUROC={val/AUROC:4f}-DSC={val/DSC:4f}-HD95={val/HD95:4f}',
                monitor='val/AUROC',
                save_top_k=5,
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

    trainer: ConfigTrainerSegmentationDefault = ConfigTrainerSegmentationDefault(
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

        trainer: ConfigTrainerSegmentationDefault = ConfigTrainerSegmentationDefault(
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
