from typing import Any, Tuple, Type, Callable, Optional
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class NoneMaintainer(PrimitiveMaintainer):
    """None type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = type(None),
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
        # Simplify first !!
        attribute_type: Type = simplify_type(attribute_type)
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        # Widgets
        self.boolean_var: Optional[tk.BooleanVar] = None
        self.null_check_button: Optional[ttk.Checkbutton] = None

    @override
    def is_type_compatible(self) -> bool:
        return self.attribute_type in {None, type(None)}

    @override
    def is_value_compatible(self) -> bool:
        return self.is_type_compatible() and self.attribute_value is None

    @override
    def get_default_value(self, *args, **kwargs) -> None:
        return None

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        return type(None)

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        return "NoneType"

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

        # Create read-only checkbox that's always checked
        self.boolean_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.null_check_button = ttk.Checkbutton(
            self.editor,
            text="Null",
            variable=self.boolean_var,
            state='disabled'  # Make it read-only
        )
        self.null_check_button.state(['alternate'])
        self.null_check_button.pack(anchor=tk.W, padx=10, pady=5)
        return self.editor

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        """Validate input value

        Args:
            input_value: Input value to validate

        Returns:
            (is_valid, validated_value)
        """
        return True, None

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return simplify_type(target_type) in {None, type(None)}

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = None) -> bool:
        return value is None

    @staticmethod
    @override
    def get_default_value_static(*args, **kwargs) -> None:
        return None

    @staticmethod
    @override
    def get_simplest_type_static(*args, **kwargs) -> Type:
        return type(None)

    @staticmethod
    @override
    def get_simplest_type_name_static(*args, **kwargs) -> str:
        return "NoneType"
