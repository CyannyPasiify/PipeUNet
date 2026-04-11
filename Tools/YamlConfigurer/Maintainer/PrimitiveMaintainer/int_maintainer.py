from typing import Any, Tuple, Type, Callable, Optional
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
import re
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer

from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class IntMaintainer(PrimitiveMaintainer):
    """int type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = int,
            attribute_value: Any = 0,
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
        return self.attribute_type is int

    @override
    def is_value_compatible(self) -> bool:
        return self.is_type_compatible() and type(self.attribute_value) is int

    @override
    def get_default_value(self, *args, **kwargs) -> int:
        return 0

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type[int]:
        return int

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        return "int"

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

        # Regex for int: allow optional minus sign followed by digits
        int_pattern = re.compile(r'^-?\d*$')

        def on_validate(P: str):
            """Validate input using regex"""
            if P == "":
                return True
            return bool(int_pattern.match(P))

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
        self.entry.bind("<Return>", on_value_change)  # Return to confirm
        self.entry.bind("<Escape>", on_value_change)  # ESC to confirm
        self.entry.bind("<FocusOut>", on_value_change)  # Lose focus to confirm

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
    def editor_set_value(self, new_value: int):
        if self.editor is not None:
            self.editor_value = new_value
            self.value_string_var.set(str(new_value))
        super().editor_set_value(new_value)

    @override
    def editor_validate(self, input_value: str) -> Tuple[bool, Optional[int]]:
        """Validate input value for int using regex

        Args:
            input_value: Input value to validate

        Returns:
            (is_valid, validated_value)
        """
        if input_value == "":
            return False, None
        # Allow optional minus sign followed by digits
        int_pattern = re.compile(r'^-?\d+$')
        if int_pattern.match(input_value):
            try:
                return True, int(input_value)
            except ValueError:
                return False, None
        return False, None

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return simplify_type(target_type) is int

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = None) -> bool:
        return type(value) is int

    @staticmethod
    @override
    def get_default_value_static(*args, **kwargs) -> int:
        return 0

    @staticmethod
    @override
    def get_simplest_type_static(*args, **kwargs) -> Type[int]:
        return int

    @staticmethod
    @override
    def get_simplest_type_name_static(*args, **kwargs) -> str:
        return "int"
