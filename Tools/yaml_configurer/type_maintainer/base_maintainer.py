from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import tkinter as tk
from tkinter import ttk


class TypeMaintainer(ABC):
    """Base class for type maintainers"""
    
    @abstractmethod
    def get_default_value(self) -> Any:
        """Get default value"""
        pass
    
    @abstractmethod
    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        pass
    
    @abstractmethod
    def get_expected_type_name(self) -> str:
        """Get expected type name"""
        pass
    
    def render_control(self, parent, attribute_name, value, on_change):
        """Render control for editing the attribute
        
        Args:
            parent: Parent widget
            attribute_name: Name of the attribute
            value: Current value
            on_change: Callback function when value changes
        
        Returns:
            Frame containing the control
        """
        # Default implementation returns empty frame
        frame = ttk.Frame(parent)
        return frame
    
    def validate_input(self, input_value) -> Tuple[bool, Any]:
        """Validate input value
        
        Args:
            input_value: Input value to validate
        
        Returns:
            (is_valid, converted_value)
        """
        return False, None
    
    def create_edit_control(self, parent, value, on_change):
        """Create edit control for the value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        # Default implementation returns empty frame and dummy callbacks
        frame = ttk.Frame(parent)
        
        def enable():
            pass
        
        def disable():
            pass
        
        def set_value(new_value):
            pass
        
        return frame, enable, disable, set_value


class ValueTypeMaintainer(TypeMaintainer):
    """Base class for value type maintainers
    
    Value type maintainers handle specific value types like None, int, bool, etc.
    """
    expand_edit_frame: bool = False


class WrapperTypeMaintainer(TypeMaintainer):
    """Base class for wrapper type maintainers
    
    Wrapper type maintainers handle types that wrap multiple optional types like Union, Optional, Any.
    """
    pass