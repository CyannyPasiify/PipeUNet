from Tools.yaml_configurer.type_maintainer.base_maintainer import ValueTypeMaintainer
from typing import Any, Tuple
import tkinter as tk
from tkinter import ttk


class BoolMaintainer(ValueTypeMaintainer):
    """bool type maintainer"""
    
    # Whether to expand edit frame vertically
    expand_edit_frame = False

    def get_default_value(self) -> bool:
        """Get default value"""
        return False

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        return isinstance(value, bool)

    def get_expected_type_name(self) -> str:
        """Get expected type name"""
        return "bool"

    def create_edit_control(self, parent, value, on_change):
        """Create edit control for bool value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)
        
        ttk.Label(frame, text="Value:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Create a BooleanVar to track the checkbox state
        bool_var = tk.BooleanVar(value=value)
        
        # Create a checkbox with dynamic label
        def update_checkbox_label():
            """Update checkbox label based on state"""
            checkbox.config(text="True" if bool_var.get() else "False")
        
        checkbox = ttk.Checkbutton(
            frame, 
            variable=bool_var,
            command=lambda: [update_checkbox_label(), on_change(bool_var.get())]
        )
        # Initialize label
        update_checkbox_label()
        checkbox.pack(anchor=tk.W, padx=10, pady=5)
        
        def enable():
            checkbox.config(state='normal')
        
        def disable():
            checkbox.config(state='disabled')
        
        def set_value(new_value):
            bool_var.set(new_value)
            update_checkbox_label()
        
        return frame, enable, disable, set_value

    def render_control(self, parent, attribute_name, value, on_change):
        """Render control for editing bool attribute"""
        frame = ttk.Frame(parent)

        # Attribute name
        ttk.Label(frame, text=f"Attribute: {attribute_name}").pack(anchor=tk.W, padx=10, pady=5)

        # Type
        ttk.Label(frame, text=f"Type: {self.get_expected_type_name()}").pack(anchor=tk.W, padx=10, pady=5)

        # Value editor in Edit panel
        edit_frame = ttk.LabelFrame(frame, text="Editor")
        edit_frame.pack(anchor=tk.N, padx=10, pady=5, fill=tk.X, expand=True)
        
        edit_control, _, _, _ = self.create_edit_control(edit_frame, value, on_change)
        edit_control.pack(fill=tk.BOTH, expand=True)

        return frame

    def validate_input(self, input_value) -> Tuple[bool, Any]:
        """Validate input value for bool"""
        if isinstance(input_value, bool):
            return True, input_value
        return False, None