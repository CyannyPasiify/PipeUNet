# -*- coding: utf-8 -*-
from typing import Optional
import lightning as L
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import override
from DataModule.data_module_segmentation_default import (
    DataModuleSegmentationDefaultInitArgs,
    DataModuleSegmentationDefault
)


@dataclass
class ConfigDataModuleBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "data_module")

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
    ) -> 'ConfigDataModuleBase':
        self.data_module: L.LightningDataModule = L.LightningDataModule()  # Just placeholder
        return self

    def get_data_module(self) -> L.LightningDataModule:
        self._assert_init_essentials()
        return self.data_module


@dataclass
class ConfigDataModuleSegmentationDefault(ConfigDataModuleBase):
    """
    Wrapped PyTorch Lightning DataModule for segmentation tasks

    Manages data loading, preprocessing, and batching for training, validation,
    testing, and prediction phases

    Initialize the DataModule

    Attributes:
        train_init_args: Initialization arguments for the training phase
        val_init_args: Initialization arguments for the validation phase
        test_init_args: Initialization arguments for the testing phase
        predict_init_args: Initialization arguments for the prediction phase
    """
    train_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    val_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None

    @override
    def init_essentials(self) -> 'ConfigDataModuleSegmentationDefault':
        self.data_module: DataModuleSegmentationDefault = DataModuleSegmentationDefault(
            train_init_args=self.train_init_args,
            val_init_args=self.val_init_args,
            test_init_args=self.test_init_args,
            predict_init_args=self.predict_init_args
        )
        return self
