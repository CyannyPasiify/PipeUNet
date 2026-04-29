import copy
from typing import Any, Tuple, Literal, Type, Callable, Optional, get_origin, get_args, List, Dict
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class LiteralMaintainer(PrimitiveMaintainer):
    """Literal type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Literal[""],
            attribute_value: Any = "",
            logger: Any = None
    ):
        """Initialize Maintainer

        Args:
            attribute_name: Name of the attribute
            attribute_type: Type of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        attribute_type: Type = simplify_type(attribute_type)
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        self.map_value_name: Dict[Any, str] = {}
        self.map_name_value: Dict[str, Any] = {}
        if self.is_type_compatible():
            args: Tuple[Any, ...] = get_args(self.attribute_type)
            for arg in args:
                value_str = repr(arg)
                self.map_value_name[arg] = value_str
                self.map_name_value[value_str] = arg
        # Widgets
        self.literal_string_var: Optional[tk.StringVar] = None
        self.literal_label: Optional[ttk.Label] = None
        self.literal_combobox: Optional[ttk.Combobox] = None

    @override
    def is_type_compatible(self) -> bool:
        # Only Literal[] will have origin as Literal
        origin: Any = get_origin(self.attribute_type)
        return origin is Literal

    @override
    def is_value_compatible(self) -> bool:
        if not self.is_type_compatible():
            return False
        # Check if value is in Literal args
        type_args: Tuple[Any, ...] = get_args(self.attribute_type)
        return self.attribute_value in type_args

    @override
    def get_default_value(self, *args, **kwargs) -> Any:
        if not self.is_type_compatible():
            return ""
        # Return first argument as default
        type_args: Tuple[Any, ...] = get_args(self.attribute_type)
        if len(type_args) > 0:
            return type_args[0]
        return ""

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        return self.attribute_type

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        if not self.is_type_compatible():
            return ""

        type_args: Tuple[Any, ...] = get_args(self.attribute_type)
        args_str = ", ".join(repr(arg) for arg in type_args)
        return f"Literal[{args_str}]"

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

        # Literal value selection dropdown
        self.literal_string_var = tk.StringVar(value="")
        literal_names: List[str] = list(self.map_name_value.keys())

        # Calculate max width based on longest literal name
        max_length = max(len(name) for name in literal_names) if literal_names else 8
        buffer = 2  # Add buffer to avoid truncation
        combobox_width = max_length + buffer

        # Find the matched literal value
        selected_name: Optional[str] = None
        for value, name in self.map_value_name.items():
            if self.attribute_value == value:
                selected_name = name
                break

        if selected_name is not None:
            self.literal_string_var.set(selected_name)
        elif len(self.map_name_value) > 0:
            # Get first record
            first_name = next(iter(self.map_name_value.keys()))
            self.literal_string_var.set(first_name)
        else:  # Literal not compatible
            self.literal_string_var.set("<Not Compatible>")

        # Create dropdown
        self.literal_label = ttk.Label(self.editor, text="Literal Value:")
        self.literal_label.pack(anchor=tk.W, padx=10, pady=5)
        self.literal_combobox = ttk.Combobox(
            self.editor,
            textvariable=self.literal_string_var,
            values=literal_names,
            width=combobox_width,
            state="readonly" if self.can_edit() else "disabled"
        )
        self.literal_combobox.pack(anchor=tk.W, padx=10, pady=5)

        # Bind dropdown change event
        def on_literal_change(event: Optional[tk.Event] = None):
            # Get selected literal value
            selected_name: str = self.literal_string_var.get()
            if not self.can_edit() or selected_name not in self.map_name_value:
                return
            selected_value: Any = self.map_name_value[selected_name]
            self.editor_value = copy.deepcopy(selected_value)
            self.on_value_change(copy.deepcopy(self.editor_value))

        self.literal_combobox.bind("<<ComboboxSelected>>", on_literal_change)

        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.literal_combobox.config(state="readonly")
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.literal_combobox.config(state="disabled")
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            # Find the matched literal value
            selected_name: Optional[str] = self.map_value_name.get(new_value)

            if selected_name is not None:
                self.literal_string_var.set(selected_name)
            elif len(self.map_name_value) > 0:
                # Get first record
                first_name = next(iter(self.map_name_value.keys()))
                self.literal_string_var.set(first_name)
            else:  # Literal not compatible
                self.literal_string_var.set("<Not Compatible>")
        super().editor_set_value(new_value)

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        return True, input_value

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return get_origin(simplify_type(target_type)) is Literal

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = Literal[""]):
        sim_type: Type = simplify_type(target_type)
        if not LiteralMaintainer.is_type_compatible_static(sim_type):
            return False
        # Check if value is in Literal args
        type_args: Tuple[Any, ...] = get_args(sim_type)
        return value in type_args

    @staticmethod
    @override
    def get_default_value_static(target_type: Type, *args, **kwargs) -> Any:
        sim_type: Type = simplify_type(target_type)
        if not LiteralMaintainer.is_type_compatible_static(sim_type):
            return ""
        # Return first argument as default
        type_args: Tuple[Any, ...] = get_args(sim_type)
        if len(type_args) > 0:
            return type_args[0]
        return ""

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return simplify_type(target_type)

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, *args, **kwargs) -> str:
        sim_type: Type = simplify_type(target_type)
        if not LiteralMaintainer.is_type_compatible_static(sim_type):
            return ""
        type_args: Tuple[Any, ...] = get_args(sim_type)
        args_str = ", ".join(repr(arg) for arg in type_args)
        return f"Literal[{args_str}]"
