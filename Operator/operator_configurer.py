import os
from abc import ABC, abstractmethod
from typing import TypeVar, Dict, Optional, Any, Union, List, Tuple
from typing_extensions import override
from dataclasses import dataclass

T = TypeVar("T")
TLSeq = Union[List[T], Tuple[T, ...]]
PathLike = Union[str, os.PathLike]


@dataclass
class ConfigOperatorBase(ABC):
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
    ) -> 'ConfigOperatorBase':
        # Do anything
        return self

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass

    def get_operator(self, *args, **kwargs) -> Any:
        self._assert_init_essentials()
        return self


@dataclass
class ConfigOperatorIdentity(ConfigOperatorBase):
    @override
    def is_ready(self) -> bool:
        # Always ok
        return True

    @override
    def init_essentials(self) -> 'ConfigOperatorIdentity':
        return self

    @override
    def __call__(self, content: Any) -> Any:
        return content
