import copy
import inspect
from typing import Any, Tuple, Type, Callable, Optional, get_origin, get_args, List, Dict, cast

from sympy import false
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
import enum
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class EnumMaintainer(PrimitiveMaintainer):
    """Enum type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = enum.Enum,
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize Maintainer

        Args:
            attribute_name: Name of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        attribute_type: Type = simplify_type(attribute_type)
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        self.map_enum_name: Dict[enum.Enum, str] = {}
        self.map_name_enum: Dict[str, enum.Enum] = {}
        if self.is_type_compatible():
            # Get the actual enum class
            enum_class: Type[enum.Enum] = cast(Type[enum.Enum], self.attribute_type)
            # Get all enum members
            for member_name, member in enum_class.__members__.items():
                self.map_enum_name[member] = member_name
                self.map_name_enum[member_name] = member
        # Widgets
        self.enum_string_var: Optional[tk.StringVar] = None
        self.enum_label: Optional[ttk.Label] = None
        self.enum_combobox: Optional[ttk.Combobox] = None

    @override
    def is_type_compatible(self) -> bool:
        # Check if it's an Enum subclass
        return type(self.attribute_type) is type and issubclass(self.attribute_type, enum.Enum)

    @override
    def is_value_compatible(self) -> bool:
        if not self.is_type_compatible():
            return False
        # Check if value is an instance of the enum class
        return isinstance(self.attribute_value, self.attribute_type)

    @override
    def get_default_value(self, *args, **kwargs) -> Any:
        if not self.is_type_compatible():
            return None
        # Return first enum member as default
        enum_class: Type[enum.Enum] = cast(Type[enum.Enum], self.attribute_type)
        enum_members = list(enum_class.__members__.values())
        if enum_members:
            return enum_members[0]
        return None

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        return self.attribute_type

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        if not self.is_type_compatible():
            return ""
        return f"{self.attribute_type!r}"

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

        # Enum value selection dropdown
        self.enum_string_var = tk.StringVar(value="")
        enum_names: List[str] = list(self.map_name_enum.keys())

        # Calculate max width based on longest enum name
        max_length = max(len(name) for name in enum_names) if enum_names else 8
        buffer = 2  # Add buffer to avoid truncation
        combobox_width = max_length + buffer

        # Find the matched enum value
        selected_name: Optional[str] = None
        if self.attribute_value is not None:
            selected_name = self.map_enum_name.get(self.attribute_value)

        if selected_name is not None:
            self.enum_string_var.set(selected_name)
        elif len(self.map_name_enum) > 0:
            # Get first record
            first_name = next(iter(self.map_name_enum.keys()))
            self.enum_string_var.set(first_name)
        else:  # Enum not compatible
            self.enum_string_var.set("<Not Compatible>")

        # Create dropdown
        self.enum_label = ttk.Label(self.editor, text="Enum Value:")
        self.enum_label.pack(anchor=tk.W, padx=10, pady=5)
        self.enum_combobox = ttk.Combobox(
            self.editor,
            textvariable=self.enum_string_var,
            values=enum_names,
            width=combobox_width,
            state="readonly" if self.can_edit() else "disabled"
        )
        self.enum_combobox.pack(anchor=tk.W, padx=10, pady=5)

        # Bind dropdown change event
        def on_enum_change(event: Optional[tk.Event] = None):
            # Get selected enum value
            selected_name: str = self.enum_string_var.get()
            if not self.can_edit() or selected_name not in self.map_name_enum:
                return
            selected_enum: enum.Enum = self.map_name_enum[selected_name]
            self.editor_value = selected_enum
            self.on_value_change(selected_enum)

        self.enum_combobox.bind("<<ComboboxSelected>>", on_enum_change)

        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.enum_combobox.config(state="readonly")
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.enum_combobox.config(state="disabled")
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            # Find the matched enum value
            selected_name: Optional[str] = self.map_enum_name.get(new_value)

            if selected_name is not None:
                self.enum_string_var.set(selected_name)
            elif len(self.map_name_enum) > 0:
                # Get first record
                first_name = next(iter(self.map_name_enum.keys()))
                self.enum_string_var.set(first_name)
            else:  # Enum not compatible
                self.enum_string_var.set("<Not Compatible>")
        super().editor_set_value(new_value)

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        return True, input_value

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return type(target_type) is type and issubclass(target_type, enum.Enum)

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = enum.Enum) -> bool:
        if not EnumMaintainer.is_type_compatible_static(target_type):
            return False
        return isinstance(value, target_type)

    @staticmethod
    @override
    def get_default_value_static(target_type: Type, *args, **kwargs) -> Any:
        if not EnumMaintainer.is_type_compatible_static(target_type):
            return None
        # Return first enum member as default
        enum_class: Type[enum.Enum] = cast(Type[enum.Enum], target_type)
        enum_members = list(enum_class.__members__.values())
        if enum_members:
            return enum_members[0]
        return None

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return target_type

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, *args, **kwargs) -> str:
        if not EnumMaintainer.is_type_compatible_static(target_type):
            return ""
        return f"{target_type!r}"
