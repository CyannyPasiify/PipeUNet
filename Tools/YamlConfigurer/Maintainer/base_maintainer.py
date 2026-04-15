from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Type, Callable
from tkinter import ttk

"""
    Maintainer必须具备以下功能：
    · 记录属性名称、属性类型、属性值
    · 提供此类型的默认值
    · 判断目标值类型是否与自身兼容
    · 判断目标类型是否与自身兼容
    · 获取预期的类型名称（化简后的）
    · 创建监视器布局
    · 创建编辑器布局
"""


class BaseMaintainer(ABC):
    """Base class for type maintainers"""

    @classmethod
    def default_standalone_window_size(cls: Type) -> Tuple[int, int]:
        # W, H
        return 500, 500

    @classmethod
    def shall_hotkey_confirm_cancel(cls: Type) -> Tuple[bool, bool]:
        # If in Standalone window, shall this type of maintainer react to
        # - Enter/Return as Confirm
        # - Esc as Cancel
        # Hotkey enabled for (Confirm, Cancel)
        return True, True

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Any,
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize Maintainer
        
        Args:
            attribute_name: Name of the attribute
            attribute_type: Type of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        self.attribute_name: str = attribute_name
        self.attribute_type: Type = attribute_type
        # If immutable type, record value
        # otherwise, record a ref (warning: may modify contents)
        self.attribute_value: Any = attribute_value
        # Copy for editor, set when create editor, you may use copy.deepcopy
        self.editor_value: Optional[Any] = None

        self.logger: Any = logger
        # 打印debug级别的日志报告创建类型和初始化参数
        if self.logger:
            assert hasattr(self.logger, "log_message"), f"Logger must implement 'log_message(message, level)' function"
            self.logger.log_message(
                f"Created {self.__class__.__name__} for attribute '{attribute_name}' "
                f"with type {repr(attribute_type)}, value: {attribute_value!r}",
                level="debug",
            )

        # Related widgets
        self.inspector: Optional[ttk.Widget] = None
        self.editor: Optional[ttk.Widget] = None
        self.on_value_change: Optional[Callable[[Any], None]] = None
        # Vars
        self.editor_state: str = ""

    def set_attribute_value(self, new_value: Any) -> None:
        self.attribute_value = new_value

    @abstractmethod
    def is_type_compatible(self) -> bool:
        pass

    @abstractmethod
    def is_value_compatible(self) -> bool:
        pass

    @abstractmethod
    def get_default_value(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_simplest_type(self, *args, **kwargs) -> Type:
        pass

    @abstractmethod
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def can_edit(self) -> bool:
        pass

    @abstractmethod
    def confirm_editor_change(self):
        pass

    @abstractmethod
    def create_inspector(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]],
    ) -> ttk.Widget:
        """
            Shall call create_editor()
        """
        pass

    @abstractmethod
    def create_editor(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]],
    ) -> ttk.Widget:
        pass

    @abstractmethod
    def editor_enable(self):
        pass

    @abstractmethod
    def editor_disable(self):
        pass

    @abstractmethod
    def editor_set_value(self, new_value: Any):
        pass

    @abstractmethod
    def config_view(self, *args, **kwargs):
        pass

    @abstractmethod
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        """Validate input value

        Args:
            input_value: Input value to validate

        Returns:
            (is_valid, validated_value)
        """
        pass

    @staticmethod
    @abstractmethod
    def is_type_compatible_static(target_type: Type) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def is_value_compatible_static(value: Any, target_type: Type = None) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def get_default_value_static(*args, **kwargs) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_simplest_type_static(*args, **kwargs) -> Type:
        pass

    @staticmethod
    @abstractmethod
    def get_simplest_type_name_static(*args, **kwargs) -> str:
        pass

    def log_message(self, message: str, level: str = "info") -> None:
        if self.logger:
            assert hasattr(self.logger, "log_message"), f"Logger must implement 'log_message(message, level)' function"
            self.logger.log_message(message, level)
