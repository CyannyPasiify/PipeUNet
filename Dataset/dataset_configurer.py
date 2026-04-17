# -*- coding: utf-8 -*-
import pathlib as pl
import sys
import monai.data.dataset as mD
from typing import Dict, Any, Optional, Union, Callable, Sequence, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import override
from Operator.operator_configurer import ConfigOperatorIdentity


@dataclass
class ConfigDatasetBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, 'dataset')

    def _assert_init_essentials(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable] = ConfigOperatorIdentity()
    ) -> None:
        if self.is_ready(): return
        self.init_essentials(data, transform)

    @abstractmethod
    def init_essentials(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable] = ConfigOperatorIdentity()
    ) -> 'ConfigDatasetBase':
        self.dataset: mD.Dataset = mD.Dataset(
            data=data,
            transform=transform
        )
        return self

    def get_dataset(
            self,
            data: Optional[Sequence] = None,
            transform: Union[Sequence[Callable], Callable] = ConfigOperatorIdentity()
    ) -> Optional[mD.Dataset]:
        if data is None:
            if self.is_ready():
                return self.dataset
            else:
                return None
        self._assert_init_essentials(data, transform)
        return self.dataset


@dataclass
class ConfigDatasetPersistent(ConfigDatasetBase):
    cache_dir: Union[pl.Path, str, None] = None

    @override
    def init_essentials(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable] = ConfigOperatorIdentity()
    ) -> 'ConfigDatasetPersistent':
        self.dataset: mD.PersistentDataset = mD.PersistentDataset(
            data=data,
            transform=transform,
            cache_dir=self.cache_dir
        )
        return self


@dataclass
class ConfigDatasetCache(ConfigDatasetBase):
    cache_num: int = sys.maxsize
    cache_rate: float = 1.0
    num_workers: Optional[int] = 1

    @override
    def init_essentials(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable] = ConfigOperatorIdentity()
    ) -> 'ConfigDatasetCache':
        self.dataset: mD.CacheDataset = mD.CacheDataset(
            data=data,
            transform=transform,
            cache_num=self.cache_num,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )
        return self


@dataclass
class ConfigDatasetLMDB(ConfigDatasetBase):
    cache_dir: Union[pl.Path, str] = "cache"
    db_name: str = "monai_cache"
    lmdb_kwargs: Optional[Dict[str, Any]] = None

    @override
    def init_essentials(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable] = ConfigOperatorIdentity()
    ) -> 'ConfigDatasetLMDB':
        self.dataset: mD.LMDBDataset = mD.LMDBDataset(
            data=data,
            transform=transform,
            cache_dir=self.cache_dir,
            db_name=self.db_name,
            lmdb_kwargs=self.lmdb_kwargs
        )
        return self
