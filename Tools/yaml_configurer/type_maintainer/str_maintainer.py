from Tools.yaml_configurer.type_maintainer.base_maintainer import ValueTypeMaintainer
from typing import Any, Tuple
import tkinter as tk
from tkinter import ttk, scrolledtext
import re


class StrMaintainer(ValueTypeMaintainer):
    """str type maintainer"""
    
    # Whether to expand edit frame vertically
    expand_edit_frame = True

    def get_default_value(self) -> str:
        """Get default value"""
        return ""

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        return isinstance(value, str)

    def get_expected_type_name(self) -> str:
        """Get expected type name"""
        return "str"

    def create_edit_control(self, parent, value, on_change):
        """Create edit control for str value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)

        # Create a line frame
        label_frame = ttk.Frame(frame)
        label_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)
        
        # Label: Value
        ttk.Label(label_frame, text="Value:").pack(side=tk.LEFT)
        
        # Button: Update
        update_button = ttk.Button(label_frame, text="Update to Attribute")
        update_button.pack(side=tk.RIGHT, padx=15)

        # Create a ScrolledText widget for multi-line input
        text_widget = scrolledtext.ScrolledText(frame, width=60, height=10, wrap=tk.WORD)
        text_widget.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Set initial value
        if value:
            text_widget.insert(tk.END, value)

        def on_value_change(event=None):
            """Handle value change"""
            input_value = text_widget.get(1.0, tk.END)
            # 只移除最后的换行符（ScrolledText会自动添加）
            if input_value.endswith('\n'):
                input_value = input_value[:-1]
            is_valid, converted_value = self.validate_input(input_value)
            if is_valid:
                on_change(converted_value)

        # Bind events for value change
        text_widget.bind("<FocusOut>", on_value_change)  # 失去焦点确认
        text_widget.bind("<Escape>", on_value_change)  # ESC确认
        update_button.config(command=on_value_change)
        
        def enable():
            text_widget.config(
                state='normal',
                takefocus=True,
                cursor='xterm',
                bg='white'
            )
            text_widget.unbind("<ButtonPress-1>")
            text_widget.unbind("<B1-Motion>")
            text_widget.unbind("<Key>")
            # 启用Update按钮
            update_button.config(state='normal')
        
        def disable():
            text_widget.config(
                state='disabled',
                takefocus=False,
                cursor='arrow',
                bg='#f0f0f0'
            )
            text_widget.bind("<ButtonPress-1>", lambda e: "break")
            text_widget.bind("<B1-Motion>", lambda e: "break")
            text_widget.bind("<Key>", lambda e: "break")
            # 禁用Update按钮
            update_button.config(state='disabled')
        
        def set_value(new_value):
            text_widget.delete(1.0, tk.END)
            if new_value:
                text_widget.insert(tk.END, new_value)
        
        return frame, enable, disable, set_value

    def render_control(self, parent, attribute_name, value, on_change):
        """Render control for editing str attribute"""
        frame = ttk.Frame(parent)

        # Attribute name
        ttk.Label(frame, text=f"Attribute: {attribute_name}").pack(anchor=tk.W, padx=10, pady=5)

        # Type
        ttk.Label(frame, text=f"Type: {self.get_expected_type_name()}").pack(anchor=tk.W, padx=10, pady=5)

        # Value editor in Edit panel
        edit_frame = ttk.LabelFrame(frame, text="Editor")
        edit_frame.pack(anchor=tk.N, padx=10, pady=5, fill=tk.BOTH, expand=True)

        edit_control, _, _, _ = self.create_edit_control(edit_frame, value, on_change)
        edit_control.pack(fill=tk.BOTH, expand=True)

        return frame

    def validate_input(self, input_value) -> Tuple[bool, Any]:
        """Validate input value for str using regex"""
        if input_value is None:
            return False, None
        # Regex for string: allow any character except ESC control character (ASCII 27)
        # This allows any Unicode character except ESC control character
        # Using a safer regex pattern
        str_pattern = re.compile(r'^[^\x1B]+$|^$', re.UNICODE)
        if str_pattern.match(input_value):
            return True, str(input_value)
        return False, None
