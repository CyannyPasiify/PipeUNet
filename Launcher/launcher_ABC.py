# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class LauncherABC(ABC):
    @abstractmethod
    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def fit(self, checkpoint: Optional[Union[str, os.PathLike, Path]] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def finetune(self, checkpoint: Optional[Union[str, os.PathLike, Path]]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validation(self, checkpoint: Optional[Union[str, os.PathLike, Path]]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def test(self, checkpoint: Optional[Union[str, os.PathLike, Path]]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, checkpoint: Optional[Union[str, os.PathLike, Path]]) -> Dict[str, Any]:
        pass
