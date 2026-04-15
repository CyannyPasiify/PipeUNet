import copy
from typing import Any, List, Type, get_origin, get_args, Union, Tuple, Optional, Callable, Dict, Literal
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.unsupported_maintainer import UnsupportedMaintainer
from Tools.YamlConfigurer.Maintainer.wrapper_maintainer import WrapperMaintainer
from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class UnionMaintainer(WrapperMaintainer):
    """Union type wrapper Maintainer"""

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
        self.maintainer_cls: List[Type[BaseMaintainer]] = []
        self.map_maintainer_cls: Dict[str, Tuple[Type, Type[BaseMaintainer]]] = {}
        if self.is_type_compatible():
            from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
            type_args: Tuple[Any, ...] = get_args(self.attribute_type)
            union_arg: Type
            for union_arg in type_args:
                mt_cls: Type[BaseMaintainer] = MaintainerFactory.get_maintainer_cls_supported_type(union_arg)
                self.maintainer_cls.append(mt_cls)
                self.map_maintainer_cls[mt_cls.get_simplest_type_name_static(target_type=union_arg)] = \
                    (union_arg, mt_cls)
        # Vars
        self.view_mode: Literal["Standalone", "Packed"] = "Standalone"
        # Widgets
        self.type_string_var: Optional[tk.StringVar] = None
        self.row_frame: Optional[ttk.Frame] = None
        self.type_label: Optional[ttk.Label] = None
        self.type_combobox: Optional[ttk.Combobox] = None
        self.selected_type_editor_frame: Optional[ttk.Frame] = None
        self.current_maintainer: Optional[BaseMaintainer] = None
        self.current_editor: Optional[ttk.Widget] = None

    @override
    def set_attribute_value(self, new_value: Any) -> None:
        super().set_attribute_value(new_value)

    @override
    def is_type_compatible(self) -> bool:
        # Assuming attribute_type is simplified
        origin: Type = get_origin(self.attribute_type)
        return origin is Union

    @override
    def is_value_compatible(self) -> bool:
        return self.is_type_compatible() and \
            any([
                not issubclass(m, UnsupportedMaintainer) and
                m.is_value_compatible_static(self.attribute_value, self.attribute_type)
                for m in self.maintainer_cls
            ])

    @override
    def get_default_value(self, *args, **kwargs) -> Any:
        if len(self.maintainer_cls) > 0:
            return self.maintainer_cls[0].get_default_value_static(target_type=self.attribute_type)
        return None

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        # Assuming attribute_type is simplified
        return self.attribute_type

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        if not self.is_type_compatible():
            return ""

        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        simplest_type: Type = self.get_simplest_type()
        type_args: Tuple[Any, ...] = get_args(simplest_type)
        subtype_names: List[str] = [
            MaintainerFactory.get_maintainer_cls_supported_type(t)
            .get_simplest_type_name_static(target_type=t, *args, **kwargs)
            for t in type_args
        ]
        return f"Union[{', '.join(subtype_names)}]"

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
        type_names: List[str] = list(self.map_maintainer_cls.keys())

        # Calculate max width based on longest type name
        max_length = max(len(name) for name in type_names) if type_names else 8
        buffer = 2  # Add buffer to avoid truncation
        combobox_width = max_length + buffer

        # Find the Maintainer that is compatible with the current value
        first_valid_type_name: Optional[str] = None
        type_name: str
        type_cls: Type
        mt: BaseMaintainer
        for type_name, (type_cls, mt) in self.map_maintainer_cls.items():
            if mt.is_value_compatible_static(self.attribute_value, type_cls):
                if first_valid_type_name is None:
                    first_valid_type_name = type_name

        if first_valid_type_name is not None:
            self.type_string_var.set(first_valid_type_name)
        elif len(self.maintainer_cls) > 0:
            # Get first record
            type_name, (type_cls, mt) = next(iter(self.map_maintainer_cls.items()))
            self.type_string_var.set(type_name)
        else:  # Type not compatible
            self.type_string_var.set("<Not Compatible>")

        # Create dropdown
        self.row_frame = ttk.Frame(self.editor)
        self.row_frame.pack(anchor=tk.W)
        self.type_label = ttk.Label(self.row_frame, text="Type:")
        self.type_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.type_combobox = ttk.Combobox(
            self.row_frame,
            textvariable=self.type_string_var,
            values=type_names,
            width=combobox_width,
            state="readonly" if self.can_edit() else "disabled"
        )
        self.type_combobox.pack(side=tk.LEFT, padx=10, pady=5)

        # Create edit control for the selected type
        self.selected_type_editor_frame = ttk.Frame(self.editor)
        self.selected_type_editor_frame.pack(anchor=tk.W, fill=tk.BOTH, expand=True)

        # Track current edit control and its callbacks
        self.current_editor = None

        # Initial update
        self._update_editor()

        # Bind dropdown change event
        def on_type_change(event: Optional[tk.Event] = None):
            self._update_editor()
            # Get selected Maintainer
            selected_type_name: str = self.type_string_var.get()
            if not self.can_edit() or selected_type_name not in self.map_maintainer_cls:
                return
            self.editor_value = copy.deepcopy(self.current_maintainer.attribute_value)
            self.on_value_change(copy.deepcopy(self.editor_value))

        self.type_combobox.bind("<<ComboboxSelected>>", on_type_change)

        return self.editor

    def _update_editor(self):
        """Update edit control based on selected type"""
        # Clear existing edit control
        for widget in self.selected_type_editor_frame.winfo_children():
            widget.destroy()

        # Get selected type index
        selected_type_name: str = self.type_string_var.get()
        if not self.can_edit() or selected_type_name not in self.map_maintainer_cls:
            return

        sel_type: Type
        sel_mt_cls: Type[BaseMaintainer]
        sel_type, sel_mt_cls = self.map_maintainer_cls[selected_type_name]
        # Determine fill based on expand_edit_frame
        shall_expand_func: Callable = getattr(sel_mt_cls, 'shall_expand_editor', False)
        shall_expand = False if shall_expand_func == False else shall_expand_func()
        fill_type = tk.BOTH if shall_expand else tk.X
        self.editor_label_frame.pack(anchor=tk.N, padx=10, pady=5, fill=fill_type, expand=True)

        # Create new edit control
        self.current_maintainer: BaseMaintainer = sel_mt_cls(
            self.attribute_name,
            sel_type,
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

        self.current_maintainer.config_view("Packed")
        self.current_editor = self.current_maintainer.create_editor(self.selected_type_editor_frame, on_value_change)
        self.current_editor.pack(anchor=tk.N, fill=tk.BOTH, expand=True)

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.type_combobox.config(state="readonly")
            if self.current_maintainer is not None:
                self.current_maintainer.editor_enable()
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.type_combobox.config(state="disabled")
            if self.current_maintainer is not None:
                self.current_maintainer.editor_disable()
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            # Find compatible Maintainer
            first_defined_maintainer_cls: Optional[Type[BaseMaintainer]] = None
            first_unsupported_maintainer_cls: Optional[Type[BaseMaintainer]] = None
            for m in self.maintainer_cls:
                m.attribute_value = new_value
                if m.is_value_compatible_static(new_value, self.attribute_type):
                    if issubclass(m, UnsupportedMaintainer):
                        if first_unsupported_maintainer_cls is None:
                            first_unsupported_maintainer_cls = m
                    else:
                        if first_defined_maintainer_cls is None:
                            first_defined_maintainer_cls = m
                    break
            valid_maintainer_cls: Optional[Type[BaseMaintainer]] = first_defined_maintainer_cls \
                if first_defined_maintainer_cls is not None else first_unsupported_maintainer_cls
            if valid_maintainer_cls is not None:
                self.type_string_var.set(
                    valid_maintainer_cls.get_simplest_type_name_static(target_type=self.attribute_type)
                )
                self._update_editor()
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
        return get_origin(simplify_type(target_type)) is Union

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = Union[Any]) -> bool:
        sim_type: Type = simplify_type(target_type)
        if not UnionMaintainer.is_type_compatible_static(sim_type):
            return False
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        type_args: Tuple[Any, ...] = get_args(sim_type)
        union_arg: Type
        for union_arg in type_args:
            maintainer_cls: Type[BaseMaintainer] = MaintainerFactory.get_maintainer_cls_supported_type_value(
                union_arg, value
            )
            if not issubclass(maintainer_cls, UnsupportedMaintainer):
                return True
        return False

    @staticmethod
    @override
    def get_default_value_static(target_type: Type, *args, **kwargs) -> Any:
        sim_type: Type = simplify_type(target_type)
        if not UnionMaintainer.is_type_compatible_static(sim_type):
            return None
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        type_args: Tuple[Any, ...] = get_args(sim_type)
        union_arg: Type
        for union_arg in type_args:
            maintainer_cls: Type[BaseMaintainer] = MaintainerFactory.get_maintainer_cls_supported_type(union_arg)
            if not issubclass(maintainer_cls, UnsupportedMaintainer):
                return maintainer_cls.get_default_value_static(target_type=union_arg)
        return None

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return simplify_type(target_type)

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, *args, **kwargs) -> str:
        sim_type: Type = simplify_type(target_type)
        if not UnionMaintainer.is_type_compatible_static(sim_type):
            return ""

        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        type_args: Tuple[Any, ...] = get_args(sim_type)
        subtype_names: List[str] = [
            MaintainerFactory.get_maintainer_cls_supported_type(t)
            .get_simplest_type_name_static(target_type=t, *args, **kwargs)
            for t in type_args
        ]
        return f"Union[{', '.join(subtype_names)}]"
