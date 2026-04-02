from Tools.yaml_configurer.type_maintainer.base_maintainer import ValueTypeMaintainer
from typing import Any, Tuple
import tkinter as tk
from tkinter import ttk


class NoneMaintainer(ValueTypeMaintainer):
    """None type maintainer"""
    
    # Whether to expand edit frame vertically
    expand_edit_frame = False
    
    def get_default_value(self) -> None:
        """Get default value"""
        return None
    
    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        return value is None
    
    def get_expected_type_name(self) -> str:
        """Get expected type name"""
        return "None"
    
    def create_edit_control(self, parent, value, on_change):
        """Create edit control for None value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)
        
        # Create read-only checkbox that's always checked
        is_null_var = tk.BooleanVar(value=True)
        null_checkbox = ttk.Checkbutton(
            frame, 
            text="null",
            variable=is_null_var,
            state='disabled'  # Make it read-only
        )
        null_checkbox.pack(anchor=tk.W, padx=10, pady=5)
        
        def enable():
            # None maintainer is always disabled
            pass
        
        def disable():
            # None maintainer is always disabled
            pass
        
        def set_value(new_value):
            # None maintainer always has value None
            pass
        
        return frame, enable, disable, set_value
    
    def render_control(self, parent, attribute_name, value, on_change):
        """Render control for None attribute"""
        frame = ttk.Frame(parent)
        
        # Attribute name
        ttk.Label(frame, text=f"Attribute: {attribute_name}").pack(anchor=tk.W, padx=10, pady=5)
        
        # Type
        ttk.Label(frame, text=f"Type: {self.get_expected_type_name()}").pack(anchor=tk.W, padx=10, pady=5)
        
        # Value editor in Edit panel - None type with read-only checkbox
        edit_frame = ttk.LabelFrame(frame, text="Editor")
        edit_frame.pack(anchor=tk.N, padx=10, pady=5, fill=tk.X, expand=True)
        
        edit_control, _, _, _ = self.create_edit_control(edit_frame, value, on_change)
        edit_control.pack(fill=tk.BOTH, expand=True)
        
        return frame
    
    def validate_input(self, input_value) -> Tuple[bool, Any]:
        """Validate input value for None"""
        return True, None