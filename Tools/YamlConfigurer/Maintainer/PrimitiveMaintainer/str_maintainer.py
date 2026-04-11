from typing import Any, Tuple, Type, Callable, Optional, Literal
from typing_extensions import override
import tkinter as tk
from tkinter import ttk, scrolledtext
import re
from Tools.YamlConfigurer.Maintainer.primitive_maintainer import PrimitiveMaintainer

from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class StrMaintainer(PrimitiveMaintainer):
    """str type Maintainer"""

    @classmethod
    @override
    def shall_expand_editor(cls: Type) -> bool:
        return True

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = str,
            attribute_value: Any = "",
            logger: Any = None
    ):
        """Initialize Maintainer

        Args:
            attribute_name: Name of the attribute
            attribute_type: Type of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        # Vars
        self.view_mode: Literal["Standalone", "Packed"] = "Standalone"
        # Widgets
        self.value_label: Optional[ttk.Label] = None
        self.row_frame: Optional[ttk.Frame] = None
        self.update_button: Optional[ttk.Button] = None
        self.scrolled_text: Optional[scrolledtext.ScrolledText] = None

    @override
    def is_type_compatible(self) -> bool:
        return self.attribute_type is str

    @override
    def is_value_compatible(self) -> bool:
        return self.is_type_compatible() and type(self.attribute_value) is str

    @override
    def get_default_value(self, *args, **kwargs) -> str:
        return ""

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type[str]:
        return str

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        return "str"

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
        assert self.is_type_compatible()

        super().create_editor(parent, on_value_change)

        # Create a line frame
        self.row_frame = ttk.Frame(self.editor)
        self.row_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)

        # Label: Value
        self.value_label = ttk.Label(self.row_frame, text="Value:")
        self.value_label.pack(side=tk.LEFT)

        # Button: Update
        if self.view_mode == "Packed":
            self.update_button = ttk.Button(self.row_frame, text="Update Attribute")
            self.update_button.pack(side=tk.RIGHT, padx=15)

        # Create a ScrolledText widget for multi-line input
        self.scrolled_text = scrolledtext.ScrolledText(self.editor, width=60, height=10, wrap=tk.WORD)
        self.scrolled_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Set initial value
        valid_value: Any = self.attribute_value
        if not self.is_value_compatible():
            valid_value = self.get_default_value()
        self.scrolled_text.insert(tk.END, valid_value)

        def on_value_change(event: Optional[tk.Event] = None):
            """Handle value change"""
            input_value: str = self.scrolled_text.get(1.0, tk.END)
            # Remove the last \n (ScrolledText auto-added)
            if input_value.endswith('\n'):
                input_value = input_value[:-1]
            is_valid, validated_value = self.editor_validate(input_value)
            if is_valid:
                self.editor_value = validated_value
                self.on_value_change(validated_value)

        # Bind events for value change
        self.scrolled_text.bind("<FocusOut>", on_value_change)  # 失去焦点确认
        self.scrolled_text.bind("<Escape>", on_value_change)  # ESC确认

        # 绑定Enter键事件，手动插入换行符并阻止事件继续传播
        def on_enter(event: Optional[tk.Event] = None):
            # 手动插入换行符
            self.scrolled_text.insert(tk.INSERT, "\n")
            # 阻止事件继续传播，避免触发其他组件的事件
            return "break"

        self.scrolled_text.bind("<Return>", on_enter)  # Enter键确认
        if self.view_mode == "Packed":
            self.update_button.config(command=on_value_change)

        self.scrolled_text.focus_set()
        self.scrolled_text.tag_add(tk.SEL, "1.0", "end-1c")
        return self.editor

    @override
    def editor_enable(self):
        if self.editor is not None:
            self.scrolled_text.config(
                state='normal',
                takefocus=True,
                cursor='xterm',
                bg='white'
            )
            self.scrolled_text.unbind("<ButtonPress-1>")
            self.scrolled_text.unbind("<B1-Motion>")
            self.scrolled_text.unbind("<Key>")
            # Enable update button
            if self.view_mode == "Packed":
                self.update_button.config(state='normal')
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            self.scrolled_text.config(
                state='disabled',
                takefocus=False,
                cursor='arrow',
                bg='#f0f0f0'
            )
            self.scrolled_text.bind("<ButtonPress-1>", lambda e: "break")
            self.scrolled_text.bind("<B1-Motion>", lambda e: "break")
            self.scrolled_text.bind("<Key>", lambda e: "break")
            # 禁用Update按钮
            if self.view_mode == "Packed":
                self.update_button.config(state='disabled')
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: str):
        if self.editor is not None:
            self.scrolled_text.delete(1.0, tk.END)
            if new_value:
                self.editor_value = new_value
                self.scrolled_text.insert(tk.END, new_value)
        super().editor_set_value(new_value)

    @override
    def config_view(self, view_mode: Literal["Standalone", "Packed"], *args, **kwargs):
        self.view_mode = view_mode

    @override
    def editor_validate(self, input_value: str) -> Tuple[bool, Optional[str]]:
        """Validate input value for str using regex

        Args:
            input_value: Input value to validate

        Returns:
            (is_valid, validated_value)
        """
        if input_value is None:
            return False, None
        # Regex for string: allow any character except ESC control character (ASCII 27)
        # This allows any Unicode character except ESC control character
        # Using a safer regex pattern
        str_pattern = re.compile(r'^[^\x1B]+$|^$', re.UNICODE)
        if str_pattern.match(input_value):
            return True, str(input_value)
        return False, None

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return simplify_type(target_type) is str

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = None) -> bool:
        return type(value) is str

    @staticmethod
    @override
    def get_default_value_static(*args, **kwargs) -> str:
        return ""

    @staticmethod
    @override
    def get_simplest_type_static(*args, **kwargs) -> Type[str]:
        return str

    @staticmethod
    @override
    def get_simplest_type_name_static(*args, **kwargs) -> str:
        return "str"


if __name__ == '__main__':
    mt = StrMaintainer()
    func = getattr(mt, 'shall_expand_editor')
    print(func())
