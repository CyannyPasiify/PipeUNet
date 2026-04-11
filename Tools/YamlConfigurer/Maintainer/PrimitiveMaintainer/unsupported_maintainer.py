from typing import Any, Tuple, Type, Callable, Optional
from typing_extensions import override
import tkinter as tk
from tkinter import ttk, font
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class UnsupportedMaintainer(PrimitiveMaintainer):
    """Unsupported type Maintainer
        Type Unsupported is compatible with all types.
    """

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Any,
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
        attribute_type: Type = simplify_type(attribute_type)
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        # Widgets
        self.boolean_var: Optional[tk.BooleanVar] = None
        self.unsupported_check_button: Optional[ttk.Checkbutton] = None

    @override
    def is_type_compatible(self) -> bool:
        return True

    @override
    def is_value_compatible(self) -> bool:
        return True

    @override
    def get_default_value(self, *args, **kwargs) -> None:
        return None

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        return self.attribute_type

    @override
    def get_simplest_type_name(self, ignore_unsupported: bool = False, *args, **kwargs) -> str:
        type_str: str = str(self.attribute_type)
        if not ignore_unsupported:
            type_str = type_str + "(Unsupported)"
        return type_str

    @override
    def create_inspector(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        """
            Shall call create_editor()
        """
        self.inspector = ttk.Frame(parent)

        # Attribute name
        self.attribute_frame = ttk.Frame(self.inspector)
        self.attribute_frame.pack(anchor=tk.W, padx=10, pady=5)
        italic_font: font.Font = font.nametofont("TkDefaultFont").copy()
        italic_font.configure(slant="italic")
        self.attribute_title_label = ttk.Label(
            self.attribute_frame,
            text="Attribute:"
        )
        self.attribute_title_label.pack(side=tk.LEFT)
        self.attribute_content_label = ttk.Label(self.attribute_frame, text=f"{self.attribute_name}")
        self.attribute_content_label.pack(side=tk.LEFT)

        # Type name
        self.type_frame = ttk.Frame(self.inspector)
        self.type_frame.pack(anchor=tk.W, padx=10, pady=5)
        self.type_title_label = ttk.Label(
            self.type_frame,
            text="Type:",
            font=italic_font,
            foreground='red'
        )
        self.type_title_label.pack(side=tk.LEFT)
        self.type_content_label = ttk.Label(
            self.type_frame,
            text=f"{self.get_simplest_type_name()}",
            font=italic_font,
            foreground='red'
        )
        self.type_content_label.pack(side=tk.LEFT)

        # Value editor in Edit panel - None type with read-only checkbox
        self.editor_label_frame = ttk.LabelFrame(self.inspector, text="Editor")
        self.editor_label_frame.pack(
            anchor=tk.N, padx=10, pady=5,
            fill=tk.BOTH if self.shall_expand_editor() else tk.X,
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
        super().create_editor(parent, on_value_change)

        # Create read-only checkbox that's always checked
        self.boolean_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.unsupported_check_button = ttk.Checkbutton(
            self.editor,
            text="Unsupported",
            variable=self.boolean_var,
            state='disabled'  # Make it read-only
        )
        self.unsupported_check_button.state(['alternate'])
        self.unsupported_check_button.pack(anchor=tk.W, padx=10, pady=5)
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
        return True

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = None) -> bool:
        return True

    @staticmethod
    @override
    def get_default_value_static(*args, **kwargs) -> None:
        return None

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return simplify_type(target_type)

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, ignore_unsupported: bool = False, *args, **kwargs) -> str:
        sim_type: Type = simplify_type(target_type)
        type_str: str = str(sim_type)
        if not ignore_unsupported:
            type_str = type_str + "(Unsupported)"
        return type_str
