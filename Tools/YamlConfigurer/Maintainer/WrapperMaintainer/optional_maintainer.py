import copy
from typing import Any, List, Type, get_origin, get_args, Union, Tuple, Optional, Callable, Dict, Literal
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.unsupported_maintainer import UnsupportedMaintainer
from Tools.YamlConfigurer.Maintainer.wrapper_maintainer import WrapperMaintainer
from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class OptionalMaintainer(WrapperMaintainer):
    """Optional type wrapper Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Any,
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize
        
        Args:
            attribute_name: Name of the attribute
            attribute_type: Type of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        # Simplify first !!
        attribute_type: Type = simplify_type(attribute_type)
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        self.current_maintainer_cls: Optional[Type[BaseMaintainer]] = None
        # Always add None Maintainer
        if self.is_type_compatible():
            from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
            type_args: Tuple[Any, ...] = get_args(self.attribute_type)
            # Optional[T] = Union[T, NoneType]
            assert len(type_args) == 2 and any(
                [t is not type(None) for t in type_args]), f"Optional must have only 1 argument"
            self.current_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(
                type_args[0] if type_args[1] is type(None) else type_args[1]
            )
        # Vars
        self.view_mode: Literal["Standalone", "Packed"] = "Standalone"
        # Widgets
        self.is_null_boolean_var: Optional[tk.BooleanVar] = None
        self.is_null_check_button: Optional[ttk.Checkbutton] = None
        self.current_editor: Optional[ttk.Widget] = None
        self.current_maintainer: Optional[BaseMaintainer] = None

    @override
    def set_attribute_value(self, new_value: Any) -> None:
        super().set_attribute_value(new_value)

    @override
    def is_type_compatible(self) -> bool:
        # Assuming attribute_type is simplified
        origin: Type = get_origin(self.attribute_type)
        type_args: Tuple[Any, ...] = get_args(self.attribute_type)
        return origin is Union and len(type_args) == 2 and type(None) in type_args

    @override
    def is_value_compatible(self) -> bool:
        return self.is_type_compatible() and (
                self.attribute_value is None or
                self.current_maintainer_cls.is_value_compatible_static(self.attribute_value, self.attribute_type)
        )

    @override
    def get_default_value(self, *args, **kwargs) -> Any:
        return None

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        # Assuming attribute_type is simplified
        return self.attribute_type

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        if not self.is_type_compatible():
            return ""

        type_args: Tuple[Any, ...] = get_args(self.attribute_type)
        # Optional[T] = Union[T, NoneType]
        assert len(type_args) == 2 and any(
            [t is not type(None) for t in type_args]), f"Optional must have only 1 argument"
        not_none_type = type_args[0] if type_args[1] is type(None) else type_args[1]
        subtype_str: str = self.current_maintainer_cls.get_simplest_type_name_static(
            target_type=not_none_type,
            *args, **kwargs
        )
        return f"Optional[{subtype_str}]"

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

        # Set to null Checkbutton
        self.is_null_boolean_var = tk.BooleanVar(value=(self.attribute_value is None))

        self.is_null_check_button = ttk.Checkbutton(
            self.editor,
            text="Null",
            variable=self.is_null_boolean_var
        )
        self.is_null_check_button.pack(anchor=tk.W, padx=10, pady=5)

        if not self.can_edit() or self.current_maintainer_cls is None:
            return self.editor

        type_args: Tuple[Any, ...] = get_args(self.attribute_type)
        # Optional[T] = Union[T, NoneType]
        assert len(type_args) == 2 and any(
            [t is not type(None) for t in type_args]), f"Optional must have only 1 argument"
        not_none_type = type_args[0] if type_args[1] is type(None) else type_args[1]

        # Determine fill based on expand_edit_frame
        shall_expand_func: Callable = getattr(self.current_maintainer_cls, 'shall_expand_editor', False)
        shall_expand = False if shall_expand_func == False else shall_expand_func()
        fill_type = tk.BOTH if shall_expand else tk.X
        self.editor_label_frame.pack(anchor=tk.N, padx=10, pady=5, fill=fill_type, expand=True)

        self.current_maintainer: BaseMaintainer = self.current_maintainer_cls(
            self.attribute_name,
            not_none_type,
            self.attribute_value,
            self.logger
        )
        if not self.current_maintainer.is_value_compatible():
            self.current_maintainer.set_attribute_value(self.current_maintainer.get_default_value())

        def on_value_change(new_value: Any) -> None:
            """Handle value change"""
            # Assuming new_value is mutable
            is_valid, validated_value = self.editor_validate(new_value)
            if is_valid:
                self.current_maintainer.confirm_editor_change()
                self.editor_value = validated_value
                # Transferring a copy of editor_value, always assuming editor_value as immutable
                self.on_value_change(copy.deepcopy(self.editor_value))

        self.current_maintainer.config_view(self.view_mode)
        self.current_editor = self.current_maintainer.create_editor(self.editor, on_value_change)
        self.current_editor.pack(anchor=tk.N, fill=tk.BOTH, expand=True)

        def on_check_button_change():
            """Handle null Checkbutton change"""
            if self.is_null_boolean_var.get():
                # Disable value Maintainer's edit control
                self.current_maintainer.editor_disable()
                self.current_maintainer.editor_set_value(self.current_maintainer.get_default_value())
                self.editor_value = None
                self.on_value_change(None)
            else:
                # Enable value Maintainer's edit control
                self.current_maintainer.editor_enable()
                # Set last stored attribute value or use default
                valid_value: Any
                if not self.current_maintainer.is_value_compatible():
                    valid_value = self.current_maintainer.get_default_value()
                else:  # To mutable
                    valid_value = copy.deepcopy(self.current_maintainer.attribute_value)
                self.current_maintainer.editor_set_value(valid_value)
                self.editor_value = copy.deepcopy(valid_value)
                # Transferring a copy of editor_value, always assuming editor_value as immutable
                self.on_value_change(copy.deepcopy(self.editor_value))
            # Move focus to the Checkbutton
            self.is_null_check_button.focus_set()

        self.is_null_check_button.config(command=on_check_button_change)

        # Initialize state
        if self.editor_value is None:
            self.current_maintainer.editor_disable()

        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.is_null_check_button.config(state='normal')
            if self.current_maintainer is not None:
                if self.is_null_boolean_var.get():
                    self.current_maintainer.editor_disable()
                else:
                    self.current_maintainer.editor_enable()
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.is_null_check_button.config(state='disabled')
            if self.current_maintainer is not None:
                self.current_maintainer.editor_disable()
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            is_null = (new_value is None)
            self.is_null_boolean_var.set(is_null)
            if self.current_maintainer is not None:
                self.current_maintainer.editor_set_value(new_value)
        super().editor_set_value(new_value)

    @override
    def config_view(self, view_mode: Literal["Standalone", "Packed"], *args, **kwargs):
        self.view_mode = view_mode

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        return True, input_value

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        sim_type: Type = simplify_type(target_type)
        # Assuming attribute_type is simplified
        origin: Type = get_origin(sim_type)
        type_args: Tuple[Any, ...] = get_args(sim_type)
        return origin is Union and len(type_args) == 2 and type(None) in type_args

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = Optional[Any]) -> bool:
        sim_type: Type = simplify_type(target_type)
        if not OptionalMaintainer.is_type_compatible_static(sim_type):
            return False
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        type_args: Tuple[Any, ...] = get_args(sim_type)
        # Optional[T] = Union[T, NoneType]
        assert len(type_args) == 2 and any(
            [t is not type(None) for t in type_args]), f"Optional must have only 1 argument"
        not_none_type = type_args[0] if type_args[1] is type(None) else type_args[1]
        maintainer_cls: Type[BaseMaintainer] = MaintainerFactory.get_maintainer_cls_supported_type_value(
            not_none_type, value
        )
        if not issubclass(maintainer_cls, UnsupportedMaintainer):
            return True
        return False

    @staticmethod
    @override
    def get_default_value_static(target_type: Type, *args, **kwargs) -> Any:
        return None

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return simplify_type(target_type)

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, *args, **kwargs) -> str:
        sim_type: Type = simplify_type(target_type)
        if not OptionalMaintainer.is_type_compatible_static(sim_type):
            return ""

        type_args: Tuple[Any, ...] = get_args(sim_type)
        # Optional[T] = Union[T, NoneType]
        assert len(type_args) == 2 and any(
            [t is not type(None) for t in type_args]), f"Optional must have only 1 argument"
        not_none_type = type_args[0] if type_args[1] is type(None) else type_args[1]

        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        subtype_name: str = MaintainerFactory.get_maintainer_cls_supported_type(not_none_type) \
            .get_simplest_type_name_static(target_type=not_none_type, *args, **kwargs)
        return f"Optional[{subtype_name}]"
