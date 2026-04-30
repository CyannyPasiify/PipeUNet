import copy
from typing import Any, Tuple, Type, Callable, Optional, get_origin, get_args, Union, List, Dict
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class TypeMaintainer(PrimitiveMaintainer):
    """Type type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Type[type],
            attribute_value: Any = type,
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
        self.map_name_type: Dict[str, Type] = {}
        if self.is_type_compatible():
            from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
            args: Any = get_args(self.attribute_type)
            # There shall be only 1 arg
            assert len(args) == 1, f"Type must have only 1 argument"
            arg: Any = args[0]  # arg is subtype annotation
            sub_origin: Any = get_origin(arg)
            if arg is Any:
                # Scan all available types in current context
                from Tools.YamlConfigurer.auxiliary_functions import all_available_types
                # Get all available types, sorted by maintainer compatibility
                available_types = all_available_types()
                for t in available_types:
                    self.map_name_type[str(t)] = t
            elif sub_origin is Union:
                # Assuming the Union is simplified
                sub_args: Tuple[Any, ...] = get_args(arg)
                for t in sub_args:
                    self.map_name_type[MaintainerFactory.get_simplest_type_name(t, {"ignore_unsupported": True})] = t
            else:  # It shall be a specified type
                self.map_name_type[MaintainerFactory.get_simplest_type_name(arg, {"ignore_unsupported": True})] = arg
        # Widgets
        self.type_string_var: Optional[tk.StringVar] = None
        self.type_label: Optional[ttk.Label] = None
        self.type_combobox: Optional[ttk.Combobox] = None

    @override
    def is_type_compatible(self) -> bool:
        # Only Type[] or type[] will have origin as type
        origin: Any = get_origin(self.attribute_type)
        return origin in {type, Type}

    @override
    def is_value_compatible(self) -> bool:
        if not self.is_type_compatible():
            return False
        # attribute_value shall be specified type class, not type annotation
        type_args: Any = get_args(self.attribute_type)
        # There shall be only 1 arg
        assert len(type_args) == 1, f"Type must have only 1 argument"
        type_arg: Any = type_args[0]  # arg is subtype annotation
        sub_origin: Any = get_origin(type_arg)
        if type_arg is Any:
            return isinstance(self.attribute_value, type)
        elif sub_origin is Union:
            # Assuming the Union is simplified
            sub_args: Tuple[Any, ...] = get_args(type_arg)
            for t in sub_args:
                if self.attribute_value is t:
                    return True
            return False
        else:  # It shall be a specified type
            return self.attribute_value is type_arg

    @override
    def get_default_value(self, *args, **kwargs) -> Type:
        if not self.is_type_compatible():
            return type
        # attribute_value shall be specified type class, not type annotation
        type_args: Any = get_args(self.attribute_type)
        # There shall be only 1 arg
        assert len(type_args) == 1, f"Type must have only 1 argument"
        type_arg: Any = type_args[0]  # arg is subtype annotation
        sub_origin: Any = get_origin(type_arg)
        if type_arg is Any:
            return type
        elif sub_origin is Union:
            # Assuming the Union is simplified
            sub_args: Tuple[Any, ...] = get_args(type_arg)
            if len(sub_args) > 0:
                return sub_args[0]
            else:
                return type
        else:  # It shall be a specified type
            return type_arg

    @override
    def get_simplest_type(self, *args, **kwargs) -> type:
        return self.attribute_type

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        if not self.is_type_compatible():
            return ""

        type_args: Tuple[Any, ...] = get_args(self.attribute_type)
        # Type[T]
        assert len(type_args) == 1, f"Type must have only 1 argument"
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        subtype_name: str = MaintainerFactory.get_simplest_type_name(type_args[0], {"ignore_unsupported": True})
        return f"Type[{subtype_name}]"

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

        # Type selection dropdown
        self.type_string_var = tk.StringVar(value="")
        type_names: List[str] = list(self.map_name_type.keys())

        # Calculate max width based on longest type name
        max_length = max(len(name) for name in type_names) if type_names else 8
        buffer = 2  # Add buffer to avoid truncation
        combobox_width = min(80, max_length + buffer)

        # Find the first matched type
        first_valid_type_name: Optional[str] = None
        type_name: str
        type_cls: Type
        for type_name, type_cls in self.map_name_type.items():
            if self.attribute_value is type_cls and first_valid_type_name is None:
                first_valid_type_name = type_name

        if first_valid_type_name is not None:
            self.type_string_var.set(first_valid_type_name)
        elif len(self.map_name_type) > 0:
            # Get first record
            type_name, type_cls = next(iter(self.map_name_type.items()))
            self.type_string_var.set(type_name)
        else:  # Type not compatible
            self.type_string_var.set("<Not Compatible>")

        # Create dropdown
        self.type_label = ttk.Label(self.editor, text="Type Class:")
        self.type_label.pack(anchor=tk.W, padx=10, pady=5)
        self.type_combobox = ttk.Combobox(
            self.editor,
            textvariable=self.type_string_var,
            values=type_names,
            width=combobox_width,
            state="readonly" if self.can_edit() else "disabled"
        )
        self.type_combobox.pack(anchor=tk.W, padx=10, pady=5)

        # Bind dropdown change event
        def on_type_change(event: Optional[tk.Event] = None):
            # Get selected type class
            selected_type_name: str = self.type_string_var.get()
            if not self.can_edit() or selected_type_name not in self.map_name_type:
                return
            selected_type_cls: Type = self.map_name_type[selected_type_name]
            self.editor_value = copy.deepcopy(selected_type_cls)
            self.on_value_change(copy.deepcopy(self.editor_value))

        self.type_combobox.bind("<<ComboboxSelected>>", on_type_change)

        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.type_combobox.config(state="readonly")
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.type_combobox.config(state="disabled")
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            # Find the first matched type
            first_valid_type_name: Optional[str] = None
            type_name: str
            type_cls: Type
            for type_name, type_cls in self.map_name_type.items():
                if new_value is type_cls and first_valid_type_name is None:
                    first_valid_type_name = type_name

            if first_valid_type_name is not None:
                self.type_string_var.set(first_valid_type_name)
            elif len(self.map_name_type) > 0:
                # Get first record
                type_name, type_cls = next(iter(self.map_name_type.items()))
                self.type_string_var.set(type_name)
            else:  # Type not compatible
                self.type_string_var.set("<Not Compatible>")
        super().editor_set_value(new_value)

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        return True, input_value

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return get_origin(simplify_type(target_type)) in {type, Type}

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = Type[type]) -> bool:
        sim_type: Type = simplify_type(target_type)
        if not TypeMaintainer.is_type_compatible_static(sim_type):
            return False
        # attribute_value shall be specified type class, not type annotation
        type_args: Any = get_args(sim_type)
        # There shall be only 1 arg
        assert len(type_args) == 1, f"Type must have only 1 argument"
        type_arg: Any = type_args[0]  # arg is subtype annotation
        sub_origin: Any = get_origin(type_arg)
        if type_arg is Any:
            return isinstance(value, type)
        elif sub_origin is Union:
            # Assuming the Union is simplified
            sub_args: Tuple[Any, ...] = get_args(type_arg)
            for t in sub_args:
                if value is t:
                    return True
            return False
        else:  # It shall be a specified type
            return value is type_arg

    @staticmethod
    @override
    def get_default_value_static(target_type: Type, *args, **kwargs) -> Any:
        sim_type: Type = simplify_type(target_type)
        if not TypeMaintainer.is_type_compatible_static(sim_type):
            return None
        # attribute_value shall be specified type class, not type annotation
        type_args: Any = get_args(sim_type)
        # There shall be only 1 arg
        assert len(type_args) == 1, f"Type must have only 1 argument"
        type_arg: Any = type_args[0]  # arg is subtype annotation
        sub_origin: Any = get_origin(type_arg)
        if type_arg is Any:
            return type
        elif sub_origin is Union:
            # Assuming the Union is simplified
            sub_args: Tuple[Any, ...] = get_args(type_arg)
            if len(sub_args) > 0:
                return sub_args[0]
            else:
                return type
        else:  # It shall be a specified type
            return type_arg

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return simplify_type(target_type)

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, *args, **kwargs) -> str:
        sim_type: Type = simplify_type(target_type)
        if not TypeMaintainer.is_type_compatible_static(sim_type):
            return ""
        type_args: Tuple[Any, ...] = get_args(sim_type)
        # Type[T]
        assert len(type_args) == 1, f"Type must have only 1 argument"
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        subtype_name: str = MaintainerFactory.get_simplest_type_name(type_args[0], {"ignore_unsupported": True})
        return f"Type[{subtype_name}]"
