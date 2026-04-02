from Tools.yaml_configurer.type_maintainer.base_maintainer import WrapperTypeMaintainer
from Tools.yaml_configurer.type_maintainer.int_maintainer import IntMaintainer
from Tools.yaml_configurer.type_maintainer.float_maintainer import FloatMaintainer
from Tools.yaml_configurer.type_maintainer.str_maintainer import StrMaintainer
from Tools.yaml_configurer.type_maintainer.bool_maintainer import BoolMaintainer
from Tools.yaml_configurer.type_maintainer.none_maintainer import NoneMaintainer
from typing import Any
import tkinter as tk
from tkinter import ttk


class AnyMaintainer(WrapperTypeMaintainer):
    """Any type maintainer"""

    def get_default_value(self) -> Any:
        """Get default value"""
        return None

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type
        
        Any type is compatible with any value
        """
        return True

    def get_expected_type_name(self) -> str:
        """Get expected type name"""
        return "Any"

    def create_edit_control(self, parent, value, on_change):
        """Create edit control for Any value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)

        # Type selection dropdown
        type_var = tk.StringVar()

        # Create maintainers for all basic types in the specified order
        type_order = ["None", "bool", "int", "float", "str"]
        basic_maintainers = {
            "None": NoneMaintainer(),
            "bool": BoolMaintainer(),
            "int": IntMaintainer(),
            "float": FloatMaintainer(),
            "str": StrMaintainer()
        }

        type_names = type_order
        type_var.set(type_names[0])  # Default to first type

        # Find the maintainer that is compatible with the current value
        current_maintainer_name = "None"
        for name, maintainer in basic_maintainers.items():
            if maintainer.is_compatible(value):
                current_maintainer_name = name
                type_var.set(name)
                break

        # Create dropdown
        type_frame = ttk.Frame(frame)
        ttk.Label(type_frame, text="Type:").pack(side=tk.LEFT, padx=10, pady=5)
        type_dropdown = ttk.Combobox(type_frame, textvariable=type_var, values=type_names, state="readonly")
        type_dropdown.pack(side=tk.LEFT, padx=10, pady=5)
        type_frame.pack(anchor=tk.W)

        # Create edit control for the selected type
        edit_control_frame = ttk.Frame(frame)
        edit_control_frame.pack(anchor=tk.W, fill=tk.BOTH, expand=True)

        # Track current edit control and its callbacks
        current_edit_control = None
        enable_callback = None
        disable_callback = None
        set_value_callback = None

        def update_edit_control():
            """Update edit control based on selected type"""
            nonlocal current_edit_control, enable_callback, disable_callback, set_value_callback

            # Clear existing edit control
            for widget in edit_control_frame.winfo_children():
                widget.destroy()

            # Get selected type
            selected_type = type_var.get()
            selected_maintainer = basic_maintainers[selected_type]
            # Determine fill based on expand_edit_frame
            fill_type = tk.BOTH if getattr(selected_maintainer, 'expand_edit_frame', False) else tk.X
            parent.pack(anchor=tk.N, padx=10, pady=5, fill=fill_type, expand=True)

            # Create new edit control
            current_value = value
            if not selected_maintainer.is_compatible(current_value):
                current_value = selected_maintainer.get_default_value()

            control, enable_cb, disable_cb, set_value_cb = selected_maintainer.create_edit_control(
                edit_control_frame, current_value, on_change
            )
            control.pack(anchor=tk.N, fill=tk.BOTH, expand=True)

            # Update callbacks
            current_edit_control = control
            enable_callback = enable_cb
            disable_callback = disable_cb
            set_value_callback = set_value_cb

        # Initial update
        update_edit_control()

        # Bind dropdown change event
        def on_type_change(event):
            update_edit_control()
            # Get selected maintainer
            selected_type = type_var.get()
            selected_maintainer = basic_maintainers[selected_type]
            # Update value to default for selected type
            default_value = selected_maintainer.get_default_value()
            on_change(default_value)

        type_dropdown.bind("<<ComboboxSelected>>", on_type_change)

        def enable():
            """Enable all controls"""
            type_dropdown.config(state="readonly")
            if enable_callback:
                enable_callback()

        def disable():
            """Disable all controls"""
            type_dropdown.config(state="disabled")
            if disable_callback:
                disable_callback()

        def set_value(new_value):
            """Set value for the current maintainer"""
            nonlocal value
            value = new_value
            # Find compatible maintainer
            for name, maintainer in basic_maintainers.items():
                if maintainer.is_compatible(new_value):
                    type_var.set(name)
                    update_edit_control()
                    break

        return frame, enable, disable, set_value

    def render_control(self, parent, attribute_name, value, on_change):
        """Render control for editing Any attribute"""
        frame = ttk.Frame(parent)

        # Attribute name
        ttk.Label(frame, text=f"Attribute: {attribute_name}").pack(anchor=tk.W, padx=10, pady=5)

        # Type
        ttk.Label(frame, text=f"Type: {self.get_expected_type_name()}").pack(anchor=tk.W, padx=10, pady=5)

        # Value editor in Edit panel
        edit_frame = ttk.LabelFrame(frame, text="Editor")

        edit_control, _, _, _ = self.create_edit_control(edit_frame, value, on_change)
        edit_control.pack(fill=tk.BOTH, expand=True)

        return frame
