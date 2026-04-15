import copy
from abc import ABCMeta
from typing import Any, Callable, Optional, Type, Literal, Tuple
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer

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


class ContainerMaintainer(BaseMaintainer, metaclass=ABCMeta):
    """Base class for container type maintainers
    
    Container type maintainers handle container types like List, Tuple, Dict.
    """

    # Whether to expand edit frame vertically
    @classmethod
    def shall_expand_editor(cls: Type) -> bool:
        return True

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Any,
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize type Maintainer

        Args:
            attribute_name: Name of the attribute
            attribute_type: Type of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        # Vars
        self.view_mode: str = ""
        # Widgets
        self.attribute_frame: Optional[ttk.Frame] = None
        self.attribute_title_label: Optional[ttk.Label] = None
        self.attribute_content_label: Optional[ttk.Label] = None
        self.type_frame: Optional[ttk.Frame] = None
        self.type_title_label: Optional[ttk.Label] = None
        self.type_content_label: Optional[ttk.Label] = None
        self.editor_label_frame: Optional[ttk.LabelFrame] = None

    @override
    def can_edit(self) -> bool:
        return self.is_type_compatible()

    @override
    def confirm_editor_change(self):
        if self.editor is not None:
            self.attribute_value = copy.deepcopy(self.editor_value)

    @override
    def create_inspector(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        """
            Shall call create_editor()
        """
        # Clear existing inspector (if exists)
        if self.inspector is not None:
            self.inspector.destroy()

        self.inspector = ttk.Frame(parent)

        # Attribute name
        self.attribute_frame = ttk.Frame(self.inspector)
        self.attribute_frame.pack(anchor=tk.W, padx=10, pady=5)
        self.attribute_title_label = ttk.Label(self.attribute_frame, text="Attribute:")
        self.attribute_title_label.pack(side=tk.LEFT)
        self.attribute_content_label = ttk.Label(self.attribute_frame, text=f"{self.attribute_name}")
        self.attribute_content_label.pack(side=tk.LEFT)

        # Type name
        self.type_frame = ttk.Frame(self.inspector)
        self.type_frame.pack(anchor=tk.W, padx=10, pady=5)
        self.type_title_label = ttk.Label(self.type_frame, text="Type:")
        self.type_title_label.pack(side=tk.LEFT)
        self.type_content_label = ttk.Label(self.type_frame, text=f"{self.get_simplest_type_name()}")
        self.type_content_label.pack(side=tk.LEFT)

        if not self.can_edit():
            return self.inspector

        # Editor label frame and editor instance
        self.editor_label_frame = ttk.LabelFrame(self.inspector, text="Editor")
        self.editor_label_frame.pack(
            anchor=tk.N, padx=10, pady=5,
            fill=tk.BOTH,
            expand=True
        )

        self.editor = self.create_editor(self.editor_label_frame, on_value_change)
        self.editor.pack(fill=tk.BOTH, expand=True)

        return self.inspector

    @override
    def create_editor(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        # Clear existing editor (if exists)
        if self.editor is not None:
            self.editor.destroy()

        self.editor_value = copy.deepcopy(self.attribute_value)

        self.on_value_change = on_value_change
        self.editor = ttk.Frame(parent)
        self.editor_state = "Enabled"
        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.editor_state = "Enabled"

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.editor_state = "Disabled"

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            self.editor_value = new_value
            if self.editor_state == "Enabled":
                self.editor_enable()
            elif self.editor_state == "Disabled":
                self.editor_disable()

    @override
    def config_view(self, view_mode: str, *args, **kwargs):
        self.view_mode = view_mode
