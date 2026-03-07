# -*- coding: utf-8 -*-
"""
Callback Configuration Module

Overview:
    Provides wrapper interfaces for PyTorch Lightning callbacks, simplifying the configuration process for common callbacks
    Each wrapper function hides most infrequently used parameters and provides a quick configuration interface

Core Features:
    1. Quick Configuration: Simplifies the callback creation process by hiding most infrequently used parameters
    2. Parameter Encapsulation: Encapsulates validated common parameter combinations to reduce code duplication
    3. Unified Interface: Provides a consistent API style for ease of use and maintenance
"""

import lightning.pytorch.callbacks as callbacks
from typing import Optional, Union, List, Literal, Dict, Any
from pathlib import Path


def DeviceStatsMonitor(cpu_stats: Optional[bool] = None) -> callbacks.DeviceStatsMonitor:
    """
    Creates a device status monitoring callback instance

    Args:
        cpu_stats: Whether to monitor CPU status. If None, only monitor when using CPU accelerator;
                  if True, monitor CPU status regardless of which accelerator is used;
                  if False, don't monitor CPU status

    Returns:
        callbacks.DeviceStatsMonitor: Configured device status monitoring callback instance
    """
    return callbacks.DeviceStatsMonitor(cpu_stats=cpu_stats)


def EarlyStopping(monitor: str,
                  min_delta: float = 0.0,
                  patience: int = 3,
                  mode: Literal['min', 'max'] = 'min',
                  verbose: bool = False) -> callbacks.EarlyStopping:
    """
    Creates an early stopping callback instance
    
    Args:
        monitor: Name of the metric to monitor
        min_delta: Minimum change considered as an improvement
        patience: Number of consecutive checks with no improvement before stopping training
        mode: Monitoring mode for the metric, 'min' means smaller is better, 'max' means larger is better
        verbose: Whether to output detailed information
    
    Returns:
        callbacks.EarlyStopping: Configured early stopping callback instance
    """
    return callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode,
        verbose=verbose
    )


def LearningRateMonitor(
        logging_interval: Optional[Literal['step', 'epoch']] = None,
        log_momentum: bool = False,
        log_weight_decay: bool = False
) -> callbacks.LearningRateMonitor:
    """
    Creates a learning rate monitoring callback instance
    
    Args:
        logging_interval: Logging interval, can be 'step', 'epoch', or None
        log_momentum: Whether to log optimizer momentum values
        log_weight_decay: Whether to log optimizer weight decay values
    
    Returns:
        callbacks.LearningRateMonitor: Configured learning rate monitoring callback instance
    """
    return callbacks.LearningRateMonitor(
        logging_interval=logging_interval,
        log_momentum=log_momentum,
        log_weight_decay=log_weight_decay
    )


def ModelCheckpoint(
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        save_top_k: int = 1,
        mode: Literal['min', 'max'] = 'min',
        save_last: Optional[Union[bool, Literal['link']]] = None,
        every_n_epochs: Optional[int] = None
) -> callbacks.ModelCheckpoint:
    """
    Creates a model checkpoint callback instance
    
    Args:
        dirpath: Directory path to save checkpoints
        filename: Checkpoint filename template
        monitor: Name of the metric to monitor, None means only save the last epoch's checkpoint
        save_top_k: Save the top k best models, 0 means no saving, -1 means save all
        mode: Monitoring mode for the metric, 'min' means smaller is better, 'max' means larger is better
        save_last: Whether to save the last checkpoint, 'link' means create a symbolic link
        every_n_epochs: Check whether to save checkpoint every n epochs
    
    Returns:
        callbacks.ModelCheckpoint: Configured model checkpoint callback instance
    """
    return callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        save_top_k=save_top_k,
        mode=mode,
        save_last=save_last,
        auto_insert_metric_name=False,
        every_n_epochs=every_n_epochs
    )


def ModelSummary(max_depth: int = 1) -> callbacks.ModelSummary:
    """
    Creates a model summary callback instance
    
    Args:
        max_depth: Maximum depth of layer nesting, 0 means disable layer summary
    
    Returns:
        callbacks.ModelSummary: Configured model summary callback instance
    """
    return callbacks.ModelSummary(max_depth=max_depth)


def RichModelSummary(max_depth: int = 1) -> callbacks.RichModelSummary:
    """
    Creates a Rich-formatted model summary callback instance
    
    Args:
        max_depth: Maximum depth of layer nesting, 0 means disable layer summary
    
    Returns:
        callbacks.RichModelSummary: Configured Rich model summary callback instance
    """
    return callbacks.RichModelSummary(max_depth=max_depth)


def RichProgressBar(refresh_rate: int = 1, leave: bool = False) -> callbacks.RichProgressBar:
    """
    Creates a Rich-formatted progress bar callback instance
    
    Args:
        refresh_rate: Progress bar update frequency (batch count)
        leave: Whether to keep the progress bar in the terminal after training ends
    
    Returns:
        callbacks.RichProgressBar: Configured Rich progress bar callback instance
    """
    return callbacks.RichProgressBar(refresh_rate=refresh_rate, leave=leave)


def TQDMProgressBar(
        refresh_rate: int = 1,
        process_position: int = 0,
        leave: bool = False
) -> callbacks.TQDMProgressBar:
    """
    Creates a TQDM progress bar callback instance
    
    Args:
        refresh_rate: Progress bar update frequency (batch count)
        process_position: Position offset of the progress bar in the terminal
        leave: Whether to keep the progress bar in the terminal after training ends
    
    Returns:
        callbacks.TQDMProgressBar: Configured TQDM progress bar callback instance
    """

    return callbacks.TQDMProgressBar(
        refresh_rate=refresh_rate,
        process_position=process_position,
        leave=leave
    )


# Import all functions
__all__ = [
    'DeviceStatsMonitor',
    'EarlyStopping',
    'LearningRateMonitor',
    'ModelCheckpoint',
    'ModelSummary',
    'RichModelSummary',
    'RichProgressBar',
    'TQDMProgressBar'
]

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


    def create_callback(callback_type: str) -> Optional[pl.Callback]:
        """Creates a callback instance based on the specified type"""
        if callback_type == 'DeviceStatsMonitor':
            print("Creating DeviceStatsMonitor callback")
            return DeviceStatsMonitor(cpu_stats=True)

        elif callback_type == 'EarlyStopping':
            print("Creating EarlyStopping callback")
            return EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=True)

        elif callback_type == 'LearningRateMonitor':
            print("Creating LearningRateMonitor callback")
            return LearningRateMonitor(logging_interval='step', log_weight_decay=True)

        elif callback_type == 'ModelCheckpoint':
            print("Creating ModelCheckpoint callback")
            return ModelCheckpoint(
                dirpath='./Samples/callback_test/ckpt',
                filename='epoch={epoch}-val_loss={val_loss:.2f}-val_acc={val_acc:.2f}',
                monitor='val_acc',
                save_top_k=3,
                mode='max',
                save_last=True
            )

        elif callback_type == 'ModelSummary':
            print("Creating ModelSummary callback")
            return ModelSummary(max_depth=2)

        elif callback_type == 'RichModelSummary':
            print("Creating RichModelSummary callback")
            try:
                return RichModelSummary(max_depth=2)
            except ModuleNotFoundError:
                print("Warning: 'rich' package is not installed, please install it using 'pip install rich'")
                return None

        elif callback_type == 'RichProgressBar':
            print("Creating RichProgressBar callback")
            try:
                return RichProgressBar(refresh_rate=2, leave=True)
            except ModuleNotFoundError:
                print("Warning: 'rich' package is not installed, please install it using 'pip install rich'")
                return None

        elif callback_type == 'TQDMProgressBar':
            print("Creating TQDMProgressBar callback")
            return TQDMProgressBar(refresh_rate=2, leave=True)

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
            callbacks=[callback],
            accelerator='auto',
            devices=1,
            log_every_n_steps=5,
            check_val_every_n_epoch=2,
            logger=pl.loggers.CSVLogger(save_dir='./Samples/callback_test', name=callback_type)
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
