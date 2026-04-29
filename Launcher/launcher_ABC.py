# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

@dataclass(init=False)
class LauncherABC(ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def is_ready(self) -> bool:
        return hasattr(self, "_is_ready")

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
    ) -> 'LauncherABC':
        return self

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

    def detect(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    def debug(self, *args, **kwargs) -> Dict[str, Any]:
        pass
