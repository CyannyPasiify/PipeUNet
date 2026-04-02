# -*- coding: utf-8 -*-
"""
Learning Rate Scheduler Configurer

Overview:
    Provides five learning rate scheduler creation functions:
    1. LinearLR: Creates a configured linear learning rate scheduler with parameter validation
    2. CosineAnnealingLR: Creates a configured cosine annealing learning rate scheduler with parameter validation
    3. CosineAnnealingWarmRestarts: Creates a configured cosine annealing with warm restarts learning rate scheduler with parameter validation
    4. OneCycleLR: Creates a configured OneCycle learning rate scheduler with parameter validation
    5. ReduceLROnPlateau: Creates a configured performance-based learning rate scheduler with parameter validation
    
Core Features:
    1. Quick Configuration: Simplifies the scheduler creation process by hiding most infrequently used parameters
    2. Parameter Encapsulation: Encapsulates validated common parameter combinations to reduce code duplication
    3. Command-line Testing: Supports testing different scheduler learning rate curves through command-line parameters

Parameter Description:

Usage Examples:
    # Create linear scheduler
    linear_scheduler = LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=100
    )
    
    # Create cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=100,
        eta_min=0.0001
    )
    
    # Command-line testing
    # python lrscheduler_configurer.py --scheduler LinearLR --step 100 --lr 0.1
    # python lrscheduler_configurer.py --scheduler CosineAnnealingLR --step 100 --lr 0.1
    # python lrscheduler_configurer.py --scheduler CosineAnnealingWarmRestarts --step 100 --lr 0.1
    # python lrscheduler_configurer.py --scheduler OneCycleLR --step 100 --lr 0.1
    # python lrscheduler_configurer.py --scheduler ReduceLROnPlateau --step 100 --lr 0.1
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Union, List, Tuple, Literal
torch.mean

def LinearLR(
        optimizer: torch.optim.Optimizer,
        start_factor: float,
        end_factor: float,
        total_iters: int,
        last_epoch: int = -1
) -> torch.optim.lr_scheduler.LinearLR:
    """
    Creates a configured linear learning rate scheduler instance
    
    Args:
        optimizer: Optimizer whose learning rate is to be adjusted
        start_factor: Starting factor
        end_factor: Ending factor
        total_iters: Total number of iterations
        last_epoch: Last training epoch, default is -1
        
    Returns:
        torch.optim.lr_scheduler.LinearLR: Configured linear learning rate scheduler instance
    """
    # Create and return LinearLR scheduler
    return lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters,
        last_epoch=last_epoch
    )


def CosineAnnealingLR(
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """
    Creates a configured cosine annealing learning rate scheduler instance
    
    Args:
        optimizer: Optimizer whose learning rate is to be adjusted
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate, default is 0
        last_epoch: Last training epoch, default is -1
        
    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR: Configured cosine annealing learning rate scheduler instance
    """
    # Create and return CosineAnnealingLR scheduler
    return lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=T_max,
        eta_min=eta_min,
        last_epoch=last_epoch
    )


def CosineAnnealingWarmRestarts(
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1
) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """
    Creates a configured cosine annealing with warm restarts learning rate scheduler instance
    
    Args:
        optimizer: Optimizer whose learning rate is to be adjusted
        T_0: Initial period
        T_mult: Period multiplier, default is 1
        eta_min: Minimum learning rate, default is 0
        last_epoch: Last training epoch, default is -1
        
    Returns:
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts: Configured cosine annealing with warm restarts learning rate scheduler instance
    """
    # Create and return CosineAnnealingWarmRestarts scheduler
    return lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        last_epoch=last_epoch
    )


def OneCycleLR(
        optimizer: torch.optim.Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: int,
        epochs: int,
        steps_per_epoch: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        last_epoch: int = -1
) -> torch.optim.lr_scheduler.OneCycleLR:
    """
    Creates a configured OneCycle learning rate scheduler instance
    
    Args:
        optimizer: Optimizer whose learning rate is to be adjusted
        max_lr: Maximum learning rate
        total_steps: Total steps in the cycle. Note that if no value is provided here, it must be inferred by providing values for epochs and steps_per_epoch
        epochs: Number of training epochs. This parameter, together with steps_per_epoch, is used to infer the total number of steps in the cycle, provided that no total_steps value is provided
        steps_per_epoch: Number of training steps per epoch. This parameter, together with epochs, is used to infer the total number of steps in the cycle, provided that no total_steps value is provided
        pct_start: Warm-up proportion, default is 0.3
        div_factor: Ratio of maximum learning rate to initial learning rate, default is 25.0
        final_div_factor: Ratio of initial learning rate to minimum learning rate, default is 10000.0
        last_epoch: Last training epoch, default is -1
        
    Returns:
        torch.optim.lr_scheduler.OneCycleLR: Configured OneCycle learning rate scheduler instance
    """
    # Create and return OneCycleLR scheduler
    return lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        last_epoch=last_epoch
    )


def ReduceLROnPlateau(
        optimizer: torch.optim.Optimizer,
        mode: Literal["min", "max"] = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = 'rel',
        cooldown: int = 0,
        min_lr: float = 0
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Creates a configured performance-based learning rate scheduler instance

    Args:
        optimizer: Optimizer whose learning rate is to be adjusted
        mode: Mode selection, 'min' means reduce learning rate when the metric stops decreasing, 'max' means reduce learning rate when the metric stops increasing, default is 'min'
        factor: Factor by which the learning rate is reduced, new learning rate = old learning rate * factor, default is 0.1
        patience: Number of epochs with no improvement after which learning rate will be reduced, default is 10
        threshold: Threshold for measuring new best values, default is 1e-4
        threshold_mode: Threshold mode, 'rel' for relative, 'abs' for absolute, default is 'rel'
        cooldown: Number of epochs to wait before resuming normal operation after a learning rate reduction, default is 0
        min_lr: Lower bound for learning rate, default is 0

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: Configured performance-based learning rate scheduler instance
    """
    # Create and return ReduceLROnPlateau scheduler
    return lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr
    )


if __name__ == "__main__":
    def plot_learning_rate_curve(scheduler_type: str, total_steps: int, initial_lr: float) -> None:
        """
        Plots learning rate curve

        Args:
            scheduler_type: LRScheduler type
            total_steps: Total number of iterations
            initial_lr: Initial learning rate
        """
        import matplotlib.pyplot as plt
        import torch.optim as optim

        # Create a simple model and optimizer
        model = torch.nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)

        # Create corresponding scheduler
        if scheduler_type == 'LinearLR':
            scheduler = LinearLR(
                optimizer=optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps
            )
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_steps,
                eta_min=0
            )
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=total_steps // 7,
                T_mult=2,
                eta_min=0
            )
        elif scheduler_type == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=initial_lr * 10,
                total_steps=total_steps,
                epochs=1,
                steps_per_epoch=total_steps,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=10000.0
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=3,
                min_lr=0
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        # Record learning rate changes
        lrs = []
        for i in range(total_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            # For ReduceLROnPlateau, we simulate metric decrease and stagnation
            if scheduler_type == 'ReduceLROnPlateau':
                # First 200 steps simulate metric decrease
                if i < 200:
                    scheduler.step(1.0 - i * 0.004)  # Metric decreases from 1.0 to 0.2
                # Next 400 steps simulate metric stagnation
                elif i < 600:
                    scheduler.step(0.2)  # Metric remains unchanged
                # After that, simulate metric slightly increasing then decreasing
                else:
                    # Simulate occasional increases and decreases to trigger learning rate adjustment
                    if i % 30 == 0:
                        scheduler.step(0.3)  # Small increase
                    else:
                        scheduler.step(0.2 - (i % 20) * 0.01)  # Slow decrease
            else:
                scheduler.step()

        # Plot learning rate curve
        plt.figure(figsize=(10, 6))
        plt.plot(lrs)
        plt.title(f'{scheduler_type} LR Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.grid(True)

        # Display plot
        plt.show()


    def main_test():
        """
        Main test function, handles command-line arguments and plots learning rate curves
        """
        import argparse

        parser = argparse.ArgumentParser(description='Learning Rate Scheduler Test Tool')
        parser.add_argument('--scheduler', type=str, required=True,
                            choices=['LinearLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR',
                                     'ReduceLROnPlateau'],
                            help='LRScheduler type')
        parser.add_argument('--step', type=int, required=True, help='Number of iterations')
        parser.add_argument('--lr', type=float, required=True, help='Initial learning rate')

        args = parser.parse_args()

        try:
            plot_learning_rate_curve(args.scheduler, args.step, args.lr)
        except Exception as e:
            print(f"Error: {e}")
            exit(1)


    main_test()
