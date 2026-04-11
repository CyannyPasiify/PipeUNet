from typing import Any, Tuple, Type, Callable, Optional
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class BoolMaintainer(PrimitiveMaintainer):
    """bool type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = bool,
            attribute_value: Any = False,
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
        self.value_label = None
        self.boolean_var: Optional[tk.BooleanVar] = None
        self.value_check_button: Optional[ttk.Checkbutton] = None

    @override
    def is_type_compatible(self) -> bool:
        return self.attribute_type is bool

    @override
    def is_value_compatible(self) -> bool:
        return self.is_type_compatible() and type(self.attribute_value) is bool

    @override
    def get_default_value(self, *args, **kwargs) -> bool:
        return False

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type[bool]:
        return bool

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        return "bool"

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

        # Create a BooleanVar to track the Checkbutton state
        valid_value: Any = self.attribute_value
        if not self.is_value_compatible():
            valid_value = self.get_default_value()
        self.boolean_var = tk.BooleanVar(value=valid_value)

        # Create a Checkbutton with dynamic label
        def update_check_button_value():
            """Update Checkbutton label based on state"""
            self.editor_value = self.boolean_var.get()
            self.value_check_button.config(text="True" if self.boolean_var.get() else "False")

        self.value_check_button = ttk.Checkbutton(
            self.editor,
            variable=self.boolean_var,
            command=lambda: [update_check_button_value(), on_value_change(self.boolean_var.get())]
        )
        # Initialize label
        update_check_button_value()
        self.value_check_button.pack(anchor=tk.W, padx=10, pady=5)

        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.value_check_button.config(state='normal')
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.value_check_button.config(state='disabled')
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: bool):
        if self.editor is not None:
            self.editor_value = new_value
            self.boolean_var.set(new_value)
            self.value_check_button.config(text="True" if new_value else "False")
        super().editor_set_value(new_value)

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        """Validate input value

        Args:
            input_value: Input value to validate

        Returns:
            (is_valid, validated_value)
        """
        if isinstance(input_value, bool):
            return True, input_value
        return False, None

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return simplify_type(target_type) is bool

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = None) -> bool:
        return type(value) is bool

    @staticmethod
    @override
    def get_default_value_static(*args, **kwargs) -> bool:
        return False

    @staticmethod
    @override
    def get_simplest_type_static(*args, **kwargs) -> Type[bool]:
        return Type[bool]

    @staticmethod
    @override
    def get_simplest_type_name_static(*args, **kwargs) -> str:
        return "bool"
