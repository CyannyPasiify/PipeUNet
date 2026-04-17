# -*- coding: utf-8 -*-
"""
Optimizer Configurer

Overview:
    Provides two optimizer creation functions, with the core purpose of hiding most infrequently used parameters to provide a quick configuration interface
    1. SGD: Creates a configured SGD optimizer with preset common parameters
    2. AdamW: Creates a configured AdamW optimizer with preset common parameters
    
Core Features:
    1. Quick Configuration: Simplifies the optimizer creation process by hiding most infrequently used parameters
    2. Parameter Encapsulation: Encapsulates validated common parameter combinations to reduce code duplication

Parameter Description:
    - SGD parameters:
      * params: Model parameters to be optimized
      * lr: Learning rate
    - AdamW parameters:
      * params: Model parameters to be optimized
      * lr: Learning rate
      * amsgrad: Whether to use AMSGrad variant, default is False
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Iterable, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing_extensions import override


@dataclass
class ConfigOptimizerBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "optimizer")

    def _assert_init_essentials(
            self,
            params: Iterable[torch.nn.parameter.Parameter]
    ) -> None:
        if self.is_ready(): return
        self.init_essentials(params)

    @abstractmethod
    def init_essentials(
            self,
            params: Iterable[torch.nn.parameter.Parameter]
    ) -> 'ConfigOptimizerBase':
        self.optimizer: optim.Optimizer = optim.Optimizer(params, {})  # Just placeholder
        return self

    def get_optimizer(
            self,
            params: Optional[Iterable[torch.nn.parameter.Parameter]] = None
    ) -> Optional[optim.Optimizer]:
        if params is None:
            if self.is_ready():
                return self.optimizer
            else:
                return None
        self._assert_init_essentials(params)
        return self.optimizer


@dataclass
class ConfigOptimizerSGD(ConfigOptimizerBase):
    """
    Creates a configured SGD optimizer instance
    
    Args:
        lr: Learning rate
        momentum: Momentum for updating
        weight_decay: Weight decay for updating
        
    Returns:
        torch.optim.SGD: Configured SGD optimizer instance
    """
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4

    @override
    def init_essentials(
            self,
            params: Iterable[torch.nn.parameter.Parameter]
    ) -> 'ConfigOptimizerSGD':
        # Init SGD optimizer
        self.optimizer = optim.SGD(
            params=params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )
        return self


@dataclass
class ConfigOptimizerAdamW(ConfigOptimizerBase):
    """
    Creates a configured AdamW optimizer instance
    
    Args:
        lr: Learning rate
        betas: β1 and β2 weight
        eps: ε for stability
        weight_decay: Weight decay for updating
        amsgrad: Whether to use AMSGrad variant, default is False
        
    Returns:
        torch.optim.AdamW: Configured AdamW optimizer instance
    """

    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False

    @override
    def init_essentials(
            self,
            params: Iterable[torch.nn.parameter.Parameter]
    ) -> 'ConfigOptimizerAdamW':
        # Init AdamW optimizer
        self.optimizer = optim.AdamW(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )
        return self


if __name__ == "__main__":
    def test_optimizer_consistency() -> None:
        """
        Test optimizer consistency: results should be the same for two optimization processes with the same configuration
        Construct and initialize a neural network with only one linear layer and optimizer, optimize for 3 steps, record results;
        then construct and initialize a new neural network and optimizer, optimize for 3 steps, and compare if the results are the same.
        """
        print("===== Optimizer Consistency Test =====")

        # Define model and optimizer parameters
        input_dim: int = 10
        output_dim: int = 1
        batch_size: int = 8

        # Create test data
        def create_test_data() -> Tuple[torch.Tensor, torch.Tensor]:
            """Create identical test data"""
            torch.manual_seed(0)  # Ensure same data is created each time
            x: torch.Tensor = torch.randn(batch_size, input_dim)
            y: torch.Tensor = torch.randn(batch_size, output_dim)
            return x, y

        # Test function: train model for 3 steps and return final weights
        def train_model_steps(optimizer: ConfigOptimizerBase) -> Dict[str, torch.Tensor]:
            """Train model for 3 steps and return final weights"""
            # Create new model
            model: nn.Linear = nn.Linear(input_dim, output_dim)
            # Initialize weights (use fixed seed to ensure same initialization twice)
            torch.manual_seed(0)
            for param in model.parameters():
                nn.init.normal_(param)

            # Init optimizer
            opt: optim.Optimizer = optimizer.get_optimizer(model.parameters())

            # Train for 3 steps
            for step in range(3):
                x: torch.Tensor
                y: torch.Tensor
                x, y = create_test_data()

                # Forward pass
                outputs: torch.Tensor = model(x)
                loss: torch.Tensor = nn.MSELoss()(outputs, y)

                # Backward pass and optimization
                opt.zero_grad()
                loss.backward()
                opt.step()

                print(f"  Step {step + 1} loss: {loss.item():.6f}")

            # Return final weights
            return {
                'weight': model.weight.clone().detach(),
                'bias': model.bias.clone().detach()
            }

        # Test SGD optimizer
        print("-" * 100)
        print("[1] SGD Optimizer Consistency Test")
        print("1st training:")
        weights1: Dict[str, torch.Tensor] = train_model_steps(ConfigOptimizerSGD(lr=0.01))
        print("2nd training:")
        weights2: Dict[str, torch.Tensor] = train_model_steps(ConfigOptimizerSGD(lr=0.01))

        # Compare weights from both trainings
        weight_diff: torch.Tensor = torch.norm(weights1['weight'] - weights2['weight'])
        bias_diff: torch.Tensor = torch.norm(weights1['bias'] - weights2['bias'])

        print(f"SGD optimizer weight difference: {weight_diff.item():.10f}")
        print(f"SGD optimizer bias difference: {bias_diff.item():.10f}")
        print(f"SGD optimizer consistency test: {'Passed' if weight_diff < 1e-6 and bias_diff < 1e-6 else 'Failed'}")

        # Test AdamW optimizer
        print("-" * 100)
        print("[2] AdamW Optimizer Consistency Test")
        print("1st training:")
        weights3: Dict[str, torch.Tensor] = train_model_steps(ConfigOptimizerAdamW(lr=0.001))
        print("2nd training:")
        weights4: Dict[str, torch.Tensor] = train_model_steps(ConfigOptimizerAdamW(lr=0.001))

        # Compare weights from both trainings
        weight_diff: torch.Tensor = torch.norm(weights3['weight'] - weights4['weight'])
        bias_diff: torch.Tensor = torch.norm(weights3['bias'] - weights4['bias'])

        print(f"AdamW optimizer weight difference: {weight_diff.item():.10f}")
        print(f"AdamW optimizer bias difference: {bias_diff.item():.10f}")
        print(f"AdamW optimizer consistency test: {'Passed' if weight_diff < 1e-6 and bias_diff < 1e-6 else 'Failed'}")

        print()
        print("===== Optimizer Consistency Test Completed =====")


    # Run optimizer consistency test
    test_optimizer_consistency()
