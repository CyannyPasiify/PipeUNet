from Tools.yaml_configurer.type_maintainer.union_wrapper_maintainer import UnionWrapperMaintainer
from Tools.yaml_configurer.type_maintainer.none_maintainer import NoneMaintainer
from Tools.yaml_configurer.type_maintainer.base_maintainer import WrapperTypeMaintainer, ValueTypeMaintainer
import tkinter as tk
from tkinter import ttk


class OptionalWrapperMaintainer(UnionWrapperMaintainer):
    """Optional type wrapper maintainer"""

    def __init__(self, maintainer: ValueTypeMaintainer):
        """Initialize
        
        Args:
            maintainer: Non-None type maintainer
        """
        # Always add None maintainer
        super().__init__([maintainer, NoneMaintainer()])
        self.wrapped_maintainer = maintainer

    def get_expected_type_name(self) -> str:
        """Get expected type name"""
        # First maintainer is non-None type
        if self.maintainers:
            return f"Optional[{self.maintainers[0].get_expected_type_name()}]"
        return "Optional[Any]"

    def create_edit_control(self, parent, value, on_change):
        """Create edit control for Optional value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)
        
        # Set to null checkbox
        is_none_var = tk.BooleanVar(value=(value is None))
        
        none_checkbox = ttk.Checkbutton(
            frame, 
            text="null",
            variable=is_none_var
        )
        none_checkbox.pack(anchor=tk.W, padx=10, pady=5)
        
        # Create edit control from wrapped maintainer
        # Get non-None value for initialization
        non_none_value = value if value is not None else self.wrapped_maintainer.get_default_value()
        
        # Directly call create_edit_control - it should be available from base class
        edit_control, enable_callback, disable_callback, set_value_callback = \
            self.wrapped_maintainer.create_edit_control(frame, non_none_value, on_change)
        edit_control.pack(fill=tk.BOTH, expand=True)
        
        def on_none_checkbox_change():
            """Handle null checkbox change"""
            if is_none_var.get():
                # Disable wrapped maintainer's edit control
                set_value_callback(self.wrapped_maintainer.get_default_value())
                disable_callback()
                on_change(None)
            else:
                # Enable wrapped maintainer's edit control
                enable_callback()
                # Get current value from edit control or use default
                current_value = self.wrapped_maintainer.get_default_value()
                set_value_callback(current_value)
                on_change(current_value)
            # Move focus to the checkbox
            none_checkbox.focus_set()
        
        none_checkbox.config(command=on_none_checkbox_change)
        
        # Initialize state
        if value is None:
            disable_callback()
        
        def enable():
            none_checkbox.config(state='normal')
            if not is_none_var.get():
                enable_callback()
        
        def disable():
            none_checkbox.config(state='disabled')
            disable_callback()
        
        def set_value(new_value):
            is_none = (new_value is None)
            is_none_var.set(is_none)
            if is_none:
                disable_callback()
            else:
                enable_callback()
                set_value_callback(new_value)
        
        return frame, enable, disable, set_value
    
    def render_control(self, parent, attribute_name, value, on_change):
        """Render control for editing Optional attribute
        
        Creates a control with:
        - null checkbox
        - Edit panel from wrapped maintainer (disabled when null is selected)
        """
        frame = ttk.Frame(parent)
        
        # Attribute name
        ttk.Label(frame, text=f"Attribute: {attribute_name}").pack(anchor=tk.W, padx=10, pady=5)
        
        # Type
        ttk.Label(frame, text=f"Type: {self.get_expected_type_name()}").pack(anchor=tk.W, padx=10, pady=5)
        
        # Value editor in Edit panel
        edit_frame = ttk.LabelFrame(frame, text="Editor")
        # Determine fill based on expand_edit_frame
        fill_type = tk.BOTH if getattr(self.wrapped_maintainer, 'expand_edit_frame', False) else tk.X
        edit_frame.pack(anchor=tk.N, padx=10, pady=5, fill=fill_type, expand=True)
        
        edit_control, _, _, _ = self.create_edit_control(edit_frame, value, on_change)
        edit_control.pack(fill=tk.BOTH, expand=True)
        
        return frame