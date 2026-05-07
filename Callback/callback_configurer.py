# -*- coding: utf-8 -*-
"""
Callback Configuration Module

Overview:
    Provides wrapper interfaces for PyTorch Lightning callbacks, simplifying the configuration process for common callbacks
    Each wrapper function hides most infrequently used parameters and provides a quick configuration interface.
"""

import lightning.pytorch.callbacks as callbacks
from typing import Optional, Union, Literal, Dict, Any
from pathlib import Path
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from Logger.logger_configurer import ConfigLoggerCSV
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from typing_extensions import override


@dataclass
class ConfigCallbackBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "callback")

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
    ) -> 'ConfigCallbackBase':
        self.callback: callbacks.Callback = callbacks.Callback()  # Just placeholder
        return self

    def get_callback_hooker(self, *args, **kwargs) -> callbacks.Callback:
        self._assert_init_essentials(*args, **kwargs)
        return self.callback


@dataclass
class ConfigCallbackDeviceStatsMonitor(ConfigCallbackBase):
    """
    Initialization arguments for DeviceStatsMonitor.
    Attributes:
        cpu_stats: if None, it will log CPU stats only if the accelerator is CPU. If True, it will log CPU stats regardless of the accelerator. If False, it will not log CPU stats regardless of the accelerator.
    """
    cpu_stats: Optional[bool] = None

    @override
    def init_essentials(self) -> 'ConfigCallbackDeviceStatsMonitor':
        self.callback: callbacks.DeviceStatsMonitor = callbacks.DeviceStatsMonitor(cpu_stats=self.cpu_stats)
        return self


@dataclass
class ConfigCallbackEarlyStopping(ConfigCallbackBase):
    """
    Initialization arguments for EarlyStopping.
    Attributes:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement.
        patience: number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch. However, the frequency of validation can be modified by setting various parameters on the Trainer, for example check_val_every_n_epoch and val_check_interval.
        mode: Monitoring mode for the metric, 'min' means smaller is better, 'max' means larger is better.
        verbose: Whether to output detailed information.
    """
    monitor: str = ''
    min_delta: float = 0.0
    patience: int = 3
    mode: Literal['min', 'max'] = 'min'
    verbose: bool = False

    @override
    def init_essentials(self) -> 'ConfigCallbackEarlyStopping':
        self.callback: callbacks.EarlyStopping = callbacks.EarlyStopping(
            monitor=self.monitor,
            min_delta=self.min_delta,
            patience=self.patience,
            mode=self.mode,
            verbose=self.verbose
        )
        return self


@dataclass
class ConfigCallbackLearningRateMonitor(ConfigCallbackBase):
    """
    Initialization arguments for LearningRateMonitor.
    Attributes:
        logging_interval: set to 'epoch' or 'step' to log lr of all optimizers at the same interval, set to None to log at individual interval according to the interval key of each scheduler. Defaults to None.
        log_momentum: option to also log the momentum values of the optimizer, if the optimizer has the momentum or betas attribute. Defaults to False.
        log_weight_decay: option to also log the weight decay values of the optimizer. Defaults to False.
    """
    logging_interval: Optional[Literal['step', 'epoch']] = None
    log_momentum: bool = False
    log_weight_decay: bool = False

    @override
    def init_essentials(self) -> 'ConfigCallbackLearningRateMonitor':
        self.callback: callbacks.LearningRateMonitor = callbacks.LearningRateMonitor(
            logging_interval=self.logging_interval,
            log_momentum=self.log_momentum,
            log_weight_decay=self.log_weight_decay
        )
        return self


@dataclass
class ConfigCallbackModelCheckpoint(ConfigCallbackBase):
    """
    Initialization arguments for ModelCheckpoint.
    Attributes:
        dirpath: directory to save the model file.
        filename: checkpoint filename. Can contain named formatting options to be auto-filled.
            # save any arbitrary metrics like `val_loss`, etc. in name
            # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt
            >>> checkpoint_callback = ConfigCallbackModelCheckpoint(
            >>>     dirpath='my/path',
            >>>     filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
            >>> )
        By default, filename is None and will be set to '{epoch}-{step}', where "epoch" and "step" match the number of finished epoch and optimizer steps respectively.
        monitor: quantity to monitor. By default, it is None which saves a checkpoint only for the last epoch.
        save_top_k: if save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. If save_top_k >= 2 and the callback is called multiple times inside an epoch, and the filename remains unchanged, the name of the saved file will be appended with a version count starting with v1 to avoid collisions. The version counter is unrelated to the top-k ranking of the checkpoint, and we recommend formatting the filename to include the monitored metric to avoid collisions.
        mode: one of {min, max}. If save_top_k != 0, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For 'val_acc', this should be 'max', for 'val_loss' this should be 'min', etc.
        save_last: When True, saves a last.ckpt copy whenever a checkpoint file gets saved. Can be set to 'link' on a local filesystem to create a symbolic link. This allows accessing the latest checkpoint in a deterministic manner. Default: None.
        every_n_epochs: Number of epochs between checkpoints. This value must be None or non-negative. To disable saving top-k checkpoints, set every_n_epochs = 0. This argument does not impact the saving of save_last=True checkpoints. If all of every_n_epochs is None, we save a checkpoint at the end of every epoch (equivalent to every_n_epochs = 1). Setting both ModelCheckpoint(..., every_n_epochs=V) and Trainer(max_epochs=N, check_val_every_n_epoch=M) will only save checkpoints at epochs 0 < E <= N where both values for every_n_epochs and check_val_every_n_epoch evenly divide E.
    """
    dirpath: Optional[Union[str, Path]] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    save_top_k: int = 1
    mode: Literal['min', 'max'] = 'min'
    save_last: Optional[Union[bool, Literal['link']]] = None
    every_n_epochs: Optional[int] = None

    @override
    def init_essentials(self) -> 'ConfigCallbackModelCheckpoint':
        self.callback: callbacks.ModelCheckpoint = callbacks.ModelCheckpoint(
            dirpath=self.dirpath,
            filename=self.filename,
            monitor=self.monitor,
            save_top_k=self.save_top_k,
            mode=self.mode,
            save_last=self.save_last,
            auto_insert_metric_name=False,
            every_n_epochs=self.every_n_epochs
        )
        return self


@dataclass
class ConfigCallbackModelSummary(ConfigCallbackBase):
    """
    Initialization arguments for ModelSummary.
    Attributes:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the layer summary off.
    """
    max_depth: int = 1

    @override
    def init_essentials(self) -> 'ConfigCallbackModelSummary':
        self.callback: callbacks.ModelSummary = callbacks.ModelSummary(max_depth=self.max_depth)
        return self


@dataclass
class ConfigCallbackRichModelSummary(ConfigCallbackBase):
    """
    Initialization arguments for RichModelSummary
    Attributes:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the layer summary off.
    """
    max_depth: int = 1

    @override
    def init_essentials(self) -> 'ConfigCallbackRichModelSummary':
        self.callback: callbacks.RichModelSummary = callbacks.RichModelSummary(max_depth=self.max_depth)
        return self


@dataclass
class ConfigCallbackRichProgressBar(ConfigCallbackBase):
    """
    Initialization arguments for RichProgressBar
    Attributes:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated. Set it to 0 to disable the display.
        leave: Leaves the finished progress bar in the terminal at the end of the epoch. Default: False.
        theme: Contains styles used to stylize the progress bar.
        console_kwargs: Args for constructing a Console. Please refer to rich.console.Console.
    """
    refresh_rate: int = 1
    leave: bool = False
    theme: RichProgressBarTheme = RichProgressBarTheme()
    console_kwargs: Optional[dict[str, Any]] = None

    @override
    def init_essentials(self) -> 'ConfigCallbackRichProgressBar':
        self.callback: callbacks.RichProgressBar = callbacks.RichProgressBar(
            refresh_rate=self.refresh_rate,
            leave=self.leave,
            theme=self.theme,
            console_kwargs=self.console_kwargs
        )
        return self


@dataclass
class ConfigCallbackTQDMProgressBar(ConfigCallbackBase):
    """
    Initialization arguments for TQDMProgressBar
    Attributes:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated. Set it to 0 to disable the display.
        process_position: Set this to a value greater than 0 to offset the progress bars by this many lines. This is useful when you have progress bars defined elsewhere and want to show all of them together.
        leave: If set to True, leaves the finished progress bar in the terminal at the end of the epoch. Default: False.
    """
    refresh_rate: int = 1
    process_position: int = 0
    leave: bool = False

    @override
    def init_essentials(self) -> 'ConfigCallbackTQDMProgressBar':
        self.callback: callbacks.TQDMProgressBar = callbacks.TQDMProgressBar(
            refresh_rate=self.refresh_rate,
            process_position=self.process_position,
            leave=self.leave
        )
        return self


if __name__ == '__main__':
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import lightning.pytorch as pl


    class SimpleModel(pl.LightningModule):
        """Simple neural network model for testing callback functions"""

        def __init__(self, input_dim: int = 20, hidden_dim: int = 10, output_dim: int = 2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)

            # Record training metrics
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log('train_loss', loss)
            self.log('train_acc', acc)

            return loss

        def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)

            # Record validation metrics
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log('val_loss', loss)
            self.log('val_acc', acc)

            return loss

        def configure_optimizers(self) -> Dict[str, Any]:
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            scheduler = {
                'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1),
                'name': 'learning_rate'
            }
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }


    def create_datasets(input_dim: int = 20, num_samples: int = 1000, batch_size: int = 32):
        """Creates a simple dataset for testing"""
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create random data
        x = torch.randn(num_samples, input_dim)
        y = torch.randint(0, 2, (num_samples,))

        # Split dataset
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size

        train_x, val_x = torch.split(x, [train_size, val_size])
        train_y, val_y = torch.split(y, [train_size, val_size])

        # Create datasets
        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader


    def create_callback(callback_type: str) -> Optional[ConfigCallbackBase]:
        """Creates a callback instance based on the specified type"""
        if callback_type == 'DeviceStatsMonitor':
            print("Creating DeviceStatsMonitor callback")
            return ConfigCallbackDeviceStatsMonitor(cpu_stats=True)

        elif callback_type == 'EarlyStopping':
            print("Creating EarlyStopping callback")
            return ConfigCallbackEarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=True)

        elif callback_type == 'LearningRateMonitor':
            print("Creating LearningRateMonitor callback")
            return ConfigCallbackLearningRateMonitor(logging_interval='step', log_weight_decay=True)

        elif callback_type == 'ModelCheckpoint':
            print("Creating ModelCheckpoint callback")
            return ConfigCallbackModelCheckpoint(
                dirpath='./Samples/callback_test/ckpt',
                filename='epoch={epoch}-val_loss={val_loss:.2f}-val_acc={val_acc:.2f}',
                monitor='val_acc',
                save_top_k=3,
                mode='max',
                save_last=True
            )

        elif callback_type == 'ModelSummary':
            print("Creating ModelSummary callback")
            return ConfigCallbackModelSummary(max_depth=2)

        elif callback_type == 'RichModelSummary':
            print("Creating RichModelSummary callback")
            try:
                return ConfigCallbackRichModelSummary(max_depth=2)
            except ModuleNotFoundError:
                print("Warning: 'rich' package is not installed, please install it using 'pip install rich'")
                return None

        elif callback_type == 'RichProgressBar':
            print("Creating RichProgressBar callback")
            try:
                return ConfigCallbackRichProgressBar(refresh_rate=2, leave=True)
            except ModuleNotFoundError:
                print("Warning: 'rich' package is not installed, please install it using 'pip install rich'")
                return None

        elif callback_type == 'TQDMProgressBar':
            print("Creating TQDMProgressBar callback")
            return ConfigCallbackTQDMProgressBar(refresh_rate=2, leave=True)

        else:
            print(f"Unknown callback type: {callback_type}")
            return None


    def run_test(callback_type: str):
        """Runs the test for the specified callback function"""
        print(f"\n===== Testing callback: {callback_type} =====")

        # Create callback
        callback = create_callback(callback_type)
        if callback is None:
            return

        # Create dataset
        train_loader, val_loader = create_datasets()

        # Create model
        model = SimpleModel()

        # Create Trainer with the specified callback
        trainer = pl.Trainer(
            max_epochs=100,
            callbacks=[callback.get_callback_hooker()],
            accelerator='auto',
            devices=1,
            log_every_n_steps=5,
            check_val_every_n_epoch=2,
            logger=ConfigLoggerCSV(save_dir='./Samples/callback_test', name=callback_type)
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        print(f"\n===== Callback {callback_type} test completed =====\n")


    def main_test():
        """Main function"""
        parser = argparse.ArgumentParser(description='Test PyTorch Lightning callback functions')
        parser.add_argument('--callback', type=str, default='all',
                            help='Callback function type to test, available options: DeviceStatsMonitor, EarlyStopping, LearningRateMonitor, ' \
                                 'ModelCheckpoint, ModelSummary, RichModelSummary, RichProgressBar, TQDMProgressBar, all')

        args = parser.parse_args()

        # Define all available callback types
        all_callbacks = [
            'DeviceStatsMonitor',
            'EarlyStopping',
            'LearningRateMonitor',
            'ModelCheckpoint',
            'ModelSummary',
            'RichModelSummary',
            'RichProgressBar',
            'TQDMProgressBar'
        ]

        # Run tests
        if args.callback == 'all':
            print("Testing all callback functions")
            for callback_type in all_callbacks:
                run_test(callback_type)
        else:
            run_test(args.callback)


    main_test()
