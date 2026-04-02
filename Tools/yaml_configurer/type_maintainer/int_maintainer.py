from Tools.yaml_configurer.type_maintainer.base_maintainer import ValueTypeMaintainer
from typing import Any, Tuple
import tkinter as tk
from tkinter import ttk
import re


class IntMaintainer(ValueTypeMaintainer):
    """int type maintainer"""
    
    # Whether to expand edit frame vertically
    expand_edit_frame = False

    def get_default_value(self) -> int:
        """Get default value"""
        return 0

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        return isinstance(value, int)

    def get_expected_type_name(self) -> str:
        """Get expected type name"""
        return "int"

    def create_edit_control(self, parent, value, on_change):
        """Create edit control for int value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)
        
        ttk.Label(frame, text="Value:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Regex for int: allow optional minus sign followed by digits
        int_pattern = re.compile(r'^-?\d*$')
        
        def on_validate(P):
            """Validate input using regex"""
            if P == "":
                return True
            return bool(int_pattern.match(P))
        
        vcmd = parent.register(on_validate)
        
        value_var = tk.StringVar(value=str(value))
        
        def on_value_change(event=None):
            """Handle value change"""
            input_value = value_var.get()
            is_valid, converted_value = self.validate_input(input_value)
            if is_valid:
                on_change(converted_value)
        
        entry = ttk.Entry(frame, textvariable=value_var, validate="key", validatecommand=(vcmd, '%P'))
        entry.pack(anchor=tk.W, padx=10, pady=5)
        entry.bind("<Return>", on_value_change)  # 回车确认
        entry.bind("<Escape>", on_value_change)  # ESC确认
        entry.bind("<FocusOut>", on_value_change)  # 失去焦点确认
        
        def enable():
            entry.config(state='normal')
        
        def disable():
            entry.config(state='disabled')
        
        def set_value(new_value):
            value_var.set(str(new_value))
        
        return frame, enable, disable, set_value

    def render_control(self, parent, attribute_name, value, on_change):
        """Render control for editing int attribute"""
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
        """Validate input value for int using regex"""
        if input_value == "":
            return False, None
        # Allow optional minus sign followed by digits
        int_pattern = re.compile(r'^-?\d+$')
        if int_pattern.match(input_value):
            try:
                return True, int(input_value)
            except ValueError:
                return False, None
        return False, None
