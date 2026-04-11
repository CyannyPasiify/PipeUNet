from typing import Any, Tuple, Type, Callable, Optional
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
import re
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer

from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class FloatMaintainer(PrimitiveMaintainer):
    """float type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = float,
            attribute_value: Any = 0.0,
            logger: Any = None
    ):
        """Initialize Maintainer

        Args:
            attribute_name: Name of the attribute
            attribute_type: Type of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        # Widgets
        self.value_label: Optional[ttk.Label] = None
        self.value_string_var: Optional[tk.StringVar] = None
        self.entry: Optional[ttk.Entry] = None

    @override
    def is_type_compatible(self) -> bool:
        return self.attribute_type is float

    @override
    def is_value_compatible(self) -> bool:
        return self.is_type_compatible() and type(self.attribute_value) is float

    @override
    def get_default_value(self, *args, **kwargs) -> float:
        return 0.0

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type[float]:
        return float

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        return "float"

    @override
    def create_inspector(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        """
            Shall call create_editor()
        """
        return super().create_inspector(parent, on_value_change)

    @override
    def create_editor(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        super().create_editor(parent, on_value_change)

        self.value_label = ttk.Label(self.editor, text="Value:")
        self.value_label.pack(anchor=tk.W, padx=10, pady=5)

        # Regex for float: allow optional minus sign, digits, optional decimal point, optional exponent
        float_pattern = re.compile(r'^-?\d*(\.\d*)?([eE][-+]?\d*)?$')

        def on_validate(P):
            """Validate input using regex"""
            if P == "":
                return True
            return bool(float_pattern.match(P))

        validate_cmd: str = parent.register(on_validate)

        valid_value: Any = self.attribute_value
        if not self.is_value_compatible():
            valid_value = self.get_default_value()
        self.value_string_var = tk.StringVar(value=str(valid_value))

        def on_value_change(event: Optional[tk.Event] = None):
            """Handle value change"""
            input_value: str = self.value_string_var.get()
            is_valid, validated_value = self.editor_validate(input_value)
            if is_valid:
                self.editor_value = validated_value
                self.on_value_change(validated_value)

        self.entry = ttk.Entry(
            self.editor,
            textvariable=self.value_string_var,
            validate="key",
            validatecommand=(validate_cmd, '%P')
        )
        self.entry.pack(anchor=tk.W, padx=10, pady=5)
        self.entry.bind("<Return>", on_value_change)  # 回车确认
        self.entry.bind("<Escape>", on_value_change)  # ESC确认
        self.entry.bind("<FocusOut>", on_value_change)  # 失去焦点确认

        self.entry.focus_set()
        self.entry.selection_range(0, tk.END)
        self.entry.icursor(tk.END)
        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.entry.config(state='normal')
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.entry.config(state='disabled')
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: float):
        if self.editor is not None:
            self.editor_value = new_value
            self.value_string_var.set(str(new_value))
        super().editor_set_value(new_value)

    @override
    def editor_validate(self, input_value: str) -> Tuple[bool, Optional[float]]:
        """Validate input value for float using regex

        Args:
            input_value: Input value to validate

        Returns:
            (is_valid, validated_value)
        """
        if input_value == "":
            return False, None
        float_pattern = re.compile(r'^-?\d*(\.\d*)?([eE][-+]?\d*)?$')
        if float_pattern.match(input_value):
            try:
                return True, float(input_value)
            except ValueError:
                return False, None
        return False, None

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return simplify_type(target_type) is float

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = None) -> bool:
        return type(value) is float

    @staticmethod
    @override
    def get_default_value_static(*args, **kwargs) -> float:
        return 0.0

    @staticmethod
    @override
    def get_simplest_type_static(*args, **kwargs) -> Type[float]:
        return float

    @staticmethod
    @override
    def get_simplest_type_name_static(*args, **kwargs) -> str:
        return "float"
