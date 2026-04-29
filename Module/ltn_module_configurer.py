# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Optional, Literal, Set
import lightning as L
from dataclasses import dataclass
from typing_extensions import override

from Module.ltn_module_segmentation_default import (
    NamedNetworkInitArgs,
    ModuleTrainingStepAdditionArgs,
    ModuleValidationStepAdditionArgs,
    ModuleTestStepAdditionArgs,
    ModulePredictStepAdditionArgs,
    LightningModuleSegmentationDefault
)

PhaseLike = Literal['train', 'val', 'test', 'predict']


@dataclass
class ConfigLightningModuleBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "ltn_module")

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
    ) -> 'ConfigLightningModuleBase':
        self.ltn_module: L.LightningModule = L.LightningModule()  # Just placeholder
        return self

    def get_ltn_module(self, *args, **kwargs) -> L.LightningModule:
        self._assert_init_essentials(*args, **kwargs)
        return self.ltn_module


@dataclass
class ConfigLightningModuleSegmentationDefault(ConfigLightningModuleBase):
    network_init_args: NamedNetworkInitArgs = NamedNetworkInitArgs(),
    module_training_step_addition_args: Optional[ModuleTrainingStepAdditionArgs] = None,
    module_validation_step_addition_args: Optional[ModuleValidationStepAdditionArgs] = None,
    module_test_step_addition_args: Optional[ModuleTestStepAdditionArgs] = None,
    module_predict_step_addition_args: Optional[ModulePredictStepAdditionArgs] = None

    @override
    def init_essentials(self) -> 'ConfigLightningModuleSegmentationDefault':
        self.ltn_module: L.LightningModule = LightningModuleSegmentationDefault(
            network_init_args=self.network_init_args,
            module_training_step_addition_args=self.module_training_step_addition_args,
            module_validation_step_addition_args=self.module_validation_step_addition_args,
            module_test_step_addition_args=self.module_test_step_addition_args,
            module_predict_step_addition_args=self.module_predict_step_addition_args
        )
        return self

    def get_available_phases(self) -> Set[PhaseLike]:
        self._assert_init_essentials()
        return self.ltn_module.get_available_phases()
