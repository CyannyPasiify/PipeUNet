from Tools.YamlConfigurer.Maintainer.container_maintainer import ContainerMaintainer
from typing import Any, Tuple, List, Type
import tkinter as tk
from tkinter import ttk


class TupleMaintainer(ContainerMaintainer):
    """Tuple type Maintainer base class"""

    # Whether to expand edit frame vertically
    expand_edit_frame = True

    def __init__(
            self,
            inner_types: List[Type] = None,
            attribute_name: str = "",
            attribute_type: Type = Tuple[Any],
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize
        
        Args:
            inner_types: List of types in the tuple
            attribute_name: Name of the tuple attribute
            attribute_type: Type of the tuple
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        self.inner_types = inner_types or []
        # 延迟导入MaintainerFactory，避免循环引用
        from Tools.yaml_configurer.maintainer_factory import MaintainerFactory
        self.inner_maintainers = [MaintainerFactory.get_maintainer(attribute_name, t, logger=logger) for t in
                                  self.inner_types]

    def get_default_value(self) -> Tuple[Any, ...]:
        """Get default value"""
        return ()

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        if not isinstance(value, tuple):
            return False
        return True

    def get_simplest_type_name(self) -> str:
        """Get expected type name"""
        return "Tuple"

    @staticmethod
    def get_simplest_type_name_static(inner_types: List[Type]) -> str:
        """Get expected type name"""
        if inner_types:
            from Tools.yaml_configurer.maintainer_factory import MaintainerFactory
            type_names = [MaintainerFactory.get_simplest_type_name(t) for t in inner_types]
            return f"Tuple[{', '.join(type_names)}]"
        return "Tuple"


class TupleEmptyMaintainer(TupleMaintainer):
    """Empty tuple type Maintainer"""

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Tuple[()],
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize
        
        Args:
            attribute_name: Name of the tuple attribute
            attribute_type: Type of the tuple
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        super().__init__([], attribute_name, attribute_type, attribute_value, logger)

    def get_default_value(self) -> Tuple[()]:
        """Get default value"""
        return ()

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        return isinstance(value, tuple) and len(value) == 0

    def get_simplest_type_name(self) -> str:
        """Get expected type name"""
        return "Tuple[()]"

    @staticmethod
    def get_simplest_type_name_static(*args, **kwargs) -> str:
        """Get expected type name"""
        return "Tuple[()]"

    def create_editor(self, parent, value, on_change, attribute_name="Tuple"):
        """Create edit control for empty tuple value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            attribute_name: Name of the tuple attribute
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)

        # Create buttons frame
        self.buttons_frame = ttk.Frame(frame)
        self.buttons_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)

        # Add buttons with horizontal stretch (all disabled)
        self.add_button = ttk.Button(self.buttons_frame, text="Add", state=tk.DISABLED)
        self.add_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.remove_button = ttk.Button(self.buttons_frame, text="Remove", state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.up_button = ttk.Button(self.buttons_frame, text="Up", state=tk.DISABLED)
        self.up_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.down_button = ttk.Button(self.buttons_frame, text="Down", state=tk.DISABLED)
        self.down_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.edit_button = ttk.Button(self.buttons_frame, text="Edit", state=tk.DISABLED)
        self.edit_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Create Treeview for tuple elements
        self.tree_frame = ttk.Frame(frame)
        self.tree_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Create Treeview with Index, Type, and Value columns
        self.tree = ttk.Treeview(self.tree_frame, columns=("index", "type", "value"), show="headings")
        self.tree.heading("index", text="Index")
        self.tree.heading("type", text="Type")
        self.tree.heading("value", text="Value")
        self.tree.column("index", minwidth=45, width=80, stretch=False, anchor=tk.W)
        self.tree.column("type", minwidth=80, width=150, stretch=False, anchor=tk.W)
        self.tree.column("value", minwidth=80, width=300, stretch=False, anchor=tk.W)

        # Add vertical scrollbar
        self.vscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.vscrollbar.set)
        self.vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add horizontal scrollbar with custom command to increase step size
        def custom_xview(*args):
            # args[0] 是 'moveto' 或 'scroll'
            if args[0] == 'scroll':
                # args[1] 是滚动单位数，args[2] 是单位类型 ('units' 或 'pages')
                # 增加滚动步长
                if args[2] == 'units':
                    # 对于单位滚动，增加步长
                    self.tree.xview_scroll(int(args[1]) * 10, 'units')
                else:
                    # 对于页滚动，保持默认行为
                    self.tree.xview_scroll(int(args[1]), 'pages')
            else:
                # 对于 moveto，保持默认行为
                self.tree.xview_moveto(args[1])

        self.hscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL, command=custom_xview)
        self.tree.configure(xscrollcommand=self.hscrollbar.set)
        self.hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Add a row with <Empty> indicator
        self.tree.insert("", tk.END, values=("<Empty>", "", ""))

        def enable():
            # All controls remain disabled for empty tuple
            pass

        def disable():
            pass

        def set_value(new_value):
            # Clear Treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Add empty indicator
            self.tree.insert("", tk.END, values=("<Empty>", "", ""))

        return frame, enable, disable, set_value

    def create_inspector(self, parent, attribute_name, value, on_change):
        """Render control for editing empty tuple attribute"""
        # Update instance attribute_name if provided
        if attribute_name:
            self.attribute_name = attribute_name

        self.frame = ttk.Frame(parent)

        # Attribute name
        self.attribute_label = ttk.Label(self.frame, text=f"Attribute: {self.attribute_name}")
        self.attribute_label.pack(anchor=tk.W, padx=10, pady=5)

        # Type
        self.type_label = ttk.Label(self.frame, text=f"Type: {self.get_simplest_type_name()}")
        self.type_label.pack(anchor=tk.W, padx=10, pady=5)

        # Value editor in Editor panel
        self.edit_frame = ttk.LabelFrame(self.frame, text="Editor")
        self.edit_frame.pack(anchor=tk.N, padx=10, pady=5, fill=tk.BOTH, expand=True)

        # 创建一个包装的on_change函数，确保元组修改后能正确更新
        def on_tuple_change(new_value):
            # 直接调用原始的on_change函数，传递新值
            on_change(new_value)

        self.edit_control, enable, disable, set_value = self.create_editor(self.edit_frame, value,
                                                                           on_tuple_change,
                                                                           self.attribute_name)
        self.edit_control.pack(fill=tk.BOTH, expand=True)

        return self.frame, enable, disable, set_value


class TupleFixedMaintainer(TupleMaintainer):
    """Fixed size tuple type Maintainer"""

    def __init__(
            self,
            inner_types: List[Type],
            attribute_name: str = "",
            attribute_type: Type = Tuple[Any, Any],
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize
        
        Args:
            inner_types: List of types in the tuple
            attribute_name: Name of the tuple attribute
            attribute_type: Type of the tuple
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        super().__init__(inner_types, attribute_name, attribute_type, attribute_value, logger)

    def get_default_value(self) -> Tuple[Any, ...]:
        """Get default value"""
        return tuple(maintainer.get_default_value() for maintainer in self.inner_maintainers)

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        if not isinstance(value, tuple) or len(value) != len(self.inner_maintainers):
            return False
        # Check if all items are compatible with their respective types
        for item, maintainer in zip(value, self.inner_maintainers):
            if not maintainer.is_compatible(item):
                return False
        return True

    def get_simplest_type_name(self) -> str:
        """Get expected type name"""
        type_names = [maintainer.get_simplest_type_name() for maintainer in self.inner_maintainers]
        return f"Tuple[{', '.join(type_names)}]"

    @staticmethod
    def get_simplest_type_name_static(inner_types: List[Type]) -> str:
        """Get expected type name"""
        if inner_types:
            from Tools.yaml_configurer.maintainer_factory import MaintainerFactory
            type_names = [MaintainerFactory.get_simplest_type_name(t) for t in inner_types]
            return f"Tuple[{', '.join(type_names)}]"
        return "Tuple"

    def create_editor(self, parent, value, on_change, attribute_name="Tuple"):
        """Create edit control for fixed size tuple value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            attribute_name: Name of the tuple attribute
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)

        # Create buttons frame
        self.buttons_frame = ttk.Frame(frame)
        self.buttons_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)

        # Add buttons with horizontal stretch (all disabled for fixed size tuple)
        self.add_button = ttk.Button(self.buttons_frame, text="Add", state=tk.DISABLED)
        self.add_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.remove_button = ttk.Button(self.buttons_frame, text="Remove", state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.up_button = ttk.Button(self.buttons_frame, text="Up", state=tk.DISABLED)
        self.up_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.down_button = ttk.Button(self.buttons_frame, text="Down", state=tk.DISABLED)
        self.down_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.edit_button = ttk.Button(self.buttons_frame, text="Edit", state=tk.DISABLED)
        self.edit_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Create Treeview for tuple elements
        self.tree_frame = ttk.Frame(frame)
        self.tree_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Create Treeview with Index, Type, and Value columns
        self.tree = ttk.Treeview(self.tree_frame, columns=("index", "type", "value"), show="headings")
        self.tree.heading("index", text="Index")
        self.tree.heading("type", text="Type")
        self.tree.heading("value", text="Value")
        self.tree.column("index", minwidth=45, width=80, stretch=False, anchor=tk.W)
        self.tree.column("type", minwidth=80, width=150, stretch=False, anchor=tk.W)
        self.tree.column("value", minwidth=80, width=300, stretch=False, anchor=tk.W)

        # Add vertical scrollbar
        self.vscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.vscrollbar.set)
        self.vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add horizontal scrollbar with custom command to increase step size
        def custom_xview(*args):
            # args[0] 是 'moveto' 或 'scroll'
            if args[0] == 'scroll':
                # args[1] 是滚动单位数，args[2] 是单位类型 ('units' 或 'pages')
                # 增加滚动步长
                if args[2] == 'units':
                    # 对于单位滚动，增加步长
                    self.tree.xview_scroll(int(args[1]) * 10, 'units')
                else:
                    # 对于页滚动，保持默认行为
                    self.tree.xview_scroll(int(args[1]), 'pages')
            else:
                # 对于 moveto，保持默认行为
                self.tree.xview_moveto(args[1])

        self.hscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL, command=custom_xview)
        self.tree.configure(xscrollcommand=self.hscrollbar.set)
        self.hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Populate Treeview with initial values
        if value:
            for i, (item, maintainer) in enumerate(zip(value, self.inner_maintainers)):
                type_name = maintainer.get_simplest_type_name()
                self.tree.insert("", tk.END, values=(i, type_name, repr(item)))
        else:
            # Use default values if no value provided
            for i, maintainer in enumerate(self.inner_maintainers):
                default_value = maintainer.get_default_value()
                type_name = maintainer.get_simplest_type_name()
                self.tree.insert("", tk.END, values=(i, type_name, repr(default_value)))

        # 跟踪当前选中的项目
        selected_item = None

        # 处理Treeview选择事件
        def on_tree_select(event):
            nonlocal selected_item
            selected_items = self.tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                # 启用Edit按钮
                self.edit_button.config(state=tk.NORMAL)
            else:
                selected_item = None
                self.edit_button.config(state=tk.DISABLED)

        # 绑定Treeview选择事件
        self.tree.bind("<<TreeviewSelect>>", on_tree_select)

        # 共用的编辑窗口创建方法
        def create_edit_window(title, item_value, item_attribute_name, maintainer):
            # 创建编辑窗口
            edit_window = tk.Toplevel(parent)
            edit_window.title(title)
            edit_window.geometry("700x500")
            edit_window.resizable(True, True)
            # 设置焦点到子窗口
            edit_window.focus_set()

            # 添加inspector LabelFrame
            inspector_frame = ttk.LabelFrame(edit_window, text="Inspector")
            inspector_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 配置inspector_frame的grid布局
            inspector_frame.grid_rowconfigure(0, weight=1)
            inspector_frame.grid_rowconfigure(1, weight=0)  # 横向滚动条固定高度
            inspector_frame.grid_columnconfigure(0, weight=1)
            inspector_frame.grid_columnconfigure(1, weight=0)  # 纵向滚动条固定宽度

            # 创建带滚动条的容器
            canvas = tk.Canvas(inspector_frame)
            scrollbar = ttk.Scrollbar(inspector_frame, orient="vertical", command=canvas.yview)
            hscrollbar = ttk.Scrollbar(inspector_frame, orient="horizontal", command=canvas.xview)
            canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=hscrollbar.set)

            # 布局 - 使用grid布局
            canvas.grid(row=0, column=0, sticky=tk.NSEW)
            scrollbar.grid(row=0, column=1, sticky=tk.NS)
            hscrollbar.grid(row=1, column=0, sticky=tk.EW)

            # 创建内部框架，用于容纳所有Inspector内容
            inner_frame = ttk.Frame(canvas, padding=10)
            # 将内部框架添加到canvas
            inner_frame_id = canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

            # 绑定事件，当内部框架大小改变时更新canvas的滚动区域
            def on_inner_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))

            inner_frame.bind("<Configure>", on_inner_configure)

            # 绑定事件，当canvas大小改变时调整内部框架的宽度
            def on_canvas_configure(event):
                """当canvas大小改变时调整内部框架的宽度和高度"""
                # 获取canvas的宽度和高度，留出滚动条的空间
                canvas_width = event.width - 5  # 留出滚动条的宽度空间
                canvas_height = event.height - 5  # 留出滚动条的高度空间

                # 获取内部框架的最小宽度和高度
                min_width = inner_frame.winfo_reqwidth()
                min_height = inner_frame.winfo_reqheight()

                # 计算内部框架的实际宽度和高度
                # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
                # 否则使用内部控件的最小宽度和高度
                actual_width = max(canvas_width, min_width)
                actual_height = max(canvas_height, min_height)
                canvas.itemconfig(inner_frame_id, width=event.width)

                # 设置内部框架的宽度和高度
                canvas.itemconfig(inner_frame_id, width=actual_width, height=actual_height)

                # 更新canvas的滚动区域
                canvas.configure(scrollregion=canvas.bbox("all"))

            canvas.bind("<Configure>", on_canvas_configure)

            # 结果变量
            result = {'value': item_value, 'confirmed': False}

            # 定义值变化的回调函数
            def on_value_change(new_val):
                result['value'] = new_val

            # 使用render_control创建编辑控件，使用inner_frame作为父容器
            control_frame, enable, disable, set_value = maintainer.create_inspector(
                inner_frame,
                item_attribute_name,
                item_value,
                on_value_change
            )
            control_frame.pack(fill=tk.BOTH, expand=True)

            # 添加按钮框架
            button_frame = ttk.Frame(edit_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)

            # 确认按钮回调
            def on_confirm():
                result['confirmed'] = True
                edit_window.destroy()

            # 取消按钮回调
            def on_cancel():
                result['confirmed'] = False
                edit_window.destroy()

            # 创建一个容器来居中按钮
            button_container = ttk.Frame(button_frame)
            button_container.pack(side=tk.TOP, anchor=tk.CENTER)

            # 添加Confirm按钮（放在前面）
            confirm_button = ttk.Button(button_container, text="Confirm", command=on_confirm)
            confirm_button.pack(side=tk.LEFT, padx=5)

            # 添加Cancel按钮
            cancel_button = ttk.Button(button_container, text="Cancel", command=on_cancel)
            cancel_button.pack(side=tk.LEFT, padx=5)

            # 绑定回车键和Esc键
            def on_return_key(event):
                on_confirm()
                return "break"  # 阻止事件继续传播

            def on_escape_key(event):
                on_cancel()
                return "break"  # 阻止事件继续传播

            # 为窗口绑定按键事件
            edit_window.bind("<Return>", on_return_key)
            edit_window.bind("<Escape>", on_escape_key)

            # 阻塞主窗口
            edit_window.transient(parent)
            edit_window.grab_set()
            parent.wait_window(edit_window)

            return result

        # 实现编辑元组元素功能
        def edit_item():
            nonlocal selected_item
            if not selected_item:
                return

            # 获取选中项的索引
            item_values = self.tree.item(selected_item, "values")
            if not item_values:
                return

            index = int(item_values[0])

            # 检查索引是否有效
            if 0 <= index < len(self.inner_maintainers):
                # 获取当前值
                current_value = value[index] if value and index < len(value) else self.inner_maintainers[
                    index].get_default_value()

                # 构建属性名称：{元组属性名称}[{此元素的下标序号}]
                item_attribute_name = f"{attribute_name}[{index}]"

                # 获取对应的维护器
                maintainer = self.inner_maintainers[index]

                # 调用共用方法创建编辑窗口
                result = create_edit_window(f"Edit {attribute_name}[{index}]", current_value, item_attribute_name,
                                            maintainer)

                # 如果用户确认，更新值
                if result['confirmed']:
                    # 确保值是元组类型
                    if not isinstance(value, tuple):
                        updated_value = list(self.get_default_value())
                    else:
                        updated_value = list(value)

                    # 更新值
                    updated_value[index] = result['value']
                    updated_value = tuple(updated_value)
                    on_change(updated_value)

                    # 更新Treeview中的显示
                    type_name = maintainer.get_simplest_type_name()
                    self.tree.item(selected_item, values=(index, type_name, repr(result['value'])))

                    # 确保选中项在视野范围内
                    self.tree.see(selected_item)
                    # 设置焦点回到Treeview并选中当前编辑的项
                    self.tree.focus_set()
                    self.tree.selection_set(selected_item)

        # 绑定Edit按钮点击事件
        self.edit_button.config(command=edit_item)

        # 绑定双击事件，双击项目打开编辑界面
        def on_tree_double_click(event):
            # 获取点击的项
            item = self.tree.identify_row(event.y)
            # 只有双击到具体项时才执行编辑操作，双击标题不触发
            if item:
                # 模拟点击Edit按钮
                edit_item()

        self.tree.bind("<Double-1>", on_tree_double_click)

        # Bind mouse wheel to horizontal scroll with increased sensitivity
        def on_mouse_wheel(event):
            # 检测Shift键是否被按下（使用更可靠的方式）
            shift_pressed = event.state & 0x0001 != 0

            if not shift_pressed:  # No modifier key
                # Vertical scroll (default behavior)
                self.tree.yview_scroll(-1 * (event.delta // 120), "units")
            else:  # Shift key pressed
                # Horizontal scroll with increased sensitivity
                self.tree.xview_scroll(-1 * (event.delta // 10), "units")

        self.tree.bind("<MouseWheel>", on_mouse_wheel)

        def enable():
            self.tree.config(state='normal')
            self.edit_button.config(state='normal')

        def disable():
            self.tree.config(state='disabled')
            self.edit_button.config(state='disabled')

        def set_value(new_value):
            # 清空Treeview
            for item in self.tree.get_children():
                self.tree.delete(item)

            # 重新填充Treeview
            if new_value:
                for i, (item, maintainer) in enumerate(zip(new_value, self.inner_maintainers)):
                    type_name = maintainer.get_simplest_type_name()
                    self.tree.insert("", tk.END, values=(i, type_name, repr(item)))
            else:
                # Use default values if no value provided
                for i, maintainer in enumerate(self.inner_maintainers):
                    default_value = maintainer.get_default_value()
                    type_name = maintainer.get_simplest_type_name()
                    self.tree.insert("", tk.END, values=(i, type_name, repr(default_value)))

        return frame, enable, disable, set_value

    def create_inspector(self, parent, attribute_name, value, on_change):
        """Render control for editing fixed size tuple attribute"""
        # Update instance attribute_name if provided
        if attribute_name:
            self.attribute_name = attribute_name

        self.frame = ttk.Frame(parent)

        # Attribute name
        self.attribute_label = ttk.Label(self.frame, text=f"Attribute: {self.attribute_name}")
        self.attribute_label.pack(anchor=tk.W, padx=10, pady=5)

        # Type
        self.type_label = ttk.Label(self.frame, text=f"Type: {self.get_simplest_type_name()}")
        self.type_label.pack(anchor=tk.W, padx=10, pady=5)

        # Value editor in Editor panel
        self.edit_frame = ttk.LabelFrame(self.frame, text="Editor")
        self.edit_frame.pack(anchor=tk.N, padx=10, pady=5, fill=tk.BOTH, expand=True)

        # 创建一个包装的on_change函数，确保元组修改后能正确更新
        def on_tuple_change(new_value):
            # 直接调用原始的on_change函数，传递新值
            on_change(new_value)

        self.edit_control, enable, disable, set_value = self.create_editor(
            self.edit_frame, value, on_tuple_change, self.attribute_name)
        self.edit_control.pack(fill=tk.BOTH, expand=True)

        return self.frame, enable, disable, set_value


class TupleVariadicMaintainer(TupleMaintainer):
    """Variadic tuple type Maintainer"""

    # Static variable for new item indicator
    NEW_ITEM_INDICATOR = "<New Item>"

    def __init__(
            self,
            inner_type: Type = Any,
            attribute_name: str = "",
            attribute_type: Type = Tuple[Any, ...],
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize
        
        Args:
            inner_type: Type of items in the tuple
            attribute_name: Name of the tuple attribute
            attribute_type: Type of the tuple
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        super().__init__([inner_type], attribute_name, attribute_type, attribute_value, logger)
        self.inner_type = inner_type
        # 延迟导入MaintainerFactory，避免循环引用
        from Tools.yaml_configurer.maintainer_factory import MaintainerFactory
        self.inner_maintainer = MaintainerFactory.get_maintainer(attribute_name, inner_type, logger=logger)

    def get_default_value(self) -> Tuple[Any, ...]:
        """Get default value"""
        return ()

    def is_compatible(self, value: Any) -> bool:
        """Check if value is compatible with the type"""
        if not isinstance(value, tuple):
            return False
        # Check if all items are compatible with inner type
        for item in value:
            if not self.inner_maintainer.is_compatible(item):
                return False
        return True

    def get_simplest_type_name(self) -> str:
        """Get expected type name"""
        inner_type_name = self.inner_maintainer.get_simplest_type_name()
        return f"Tuple[{inner_type_name}, ...]"

    @staticmethod
    def get_simplest_type_name_static(inner_type: Type) -> str:
        """Get expected type name"""
        if inner_type:
            from Tools.yaml_configurer.maintainer_factory import MaintainerFactory
            inner_type_name = MaintainerFactory.get_simplest_type_name(inner_type)
            return f"Tuple[{inner_type_name}, ...]"
        return "Tuple[Any, ...]"

    def create_editor(self, parent, value, on_change, attribute_name="Tuple"):
        """Create edit control for variadic tuple value
        
        Args:
            parent: Parent widget
            value: Initial value
            on_change: Callback when value changes
            attribute_name: Name of the tuple attribute
            
        Returns:
            Tuple of (frame, enable_callback, disable_callback, set_value_callback)
        """
        frame = ttk.Frame(parent)

        # Create buttons frame
        self.buttons_frame = ttk.Frame(frame)
        self.buttons_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)

        # Add buttons with horizontal stretch
        self.add_button = ttk.Button(self.buttons_frame, text="Add", state=tk.DISABLED)
        self.add_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.remove_button = ttk.Button(self.buttons_frame, text="Remove", state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.up_button = ttk.Button(self.buttons_frame, text="Up", state=tk.DISABLED)
        self.up_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.down_button = ttk.Button(self.buttons_frame, text="Down", state=tk.DISABLED)
        self.down_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.edit_button = ttk.Button(self.buttons_frame, text="Edit", state=tk.DISABLED)
        self.edit_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Create Treeview for tuple elements
        self.tree_frame = ttk.Frame(frame)
        self.tree_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Create Treeview with Index, Type, and Value columns
        self.tree = ttk.Treeview(self.tree_frame, columns=("index", "type", "value"), show="headings")
        self.tree.heading("index", text="Index")
        self.tree.heading("type", text="Type")
        # Add type information to Value column heading
        inner_type_name = self.inner_maintainer.get_simplest_type_name()
        self.tree.heading("value", text="Value")
        self.tree.column("index", minwidth=45, width=80, stretch=False, anchor=tk.W)
        self.tree.column("type", minwidth=80, width=150, stretch=False, anchor=tk.W)
        self.tree.column("value", minwidth=80, width=329, stretch=False, anchor=tk.W)

        # Add vertical scrollbar
        self.vscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.vscrollbar.set)
        self.vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add horizontal scrollbar with custom command to increase step size
        def custom_xview(*args):
            # args[0] 是 'moveto' 或 'scroll'
            if args[0] == 'scroll':
                # args[1] 是滚动单位数，args[2] 是单位类型 ('units' 或 'pages')
                # 增加滚动步长
                if args[2] == 'units':
                    # 对于单位滚动，增加步长
                    self.tree.xview_scroll(int(args[1]) * 10, 'units')
                else:
                    # 对于页滚动，保持默认行为
                    self.tree.xview_scroll(int(args[1]), 'pages')
            else:
                # 对于 moveto，保持默认行为
                self.tree.xview_moveto(args[1])

        self.hscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL, command=custom_xview)
        self.tree.configure(xscrollcommand=self.hscrollbar.set)
        self.hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Populate Treeview with initial values
        if value:
            self.value = value
            for i, item in enumerate(value):
                type_name = self.inner_maintainer.get_simplest_type_name()
                self.tree.insert("", tk.END, values=(i, type_name, repr(item)))

        # Add a blank row at the end for tuple
        self.tree.insert("", tk.END, values=(TupleVariadicMaintainer.NEW_ITEM_INDICATOR, "", ""))

        # 跟踪当前选中的项目
        selected_item = None

        # 处理Treeview选择事件
        def on_tree_select(event):
            nonlocal selected_item
            selected_items = self.tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                # 启用Add按钮
                self.add_button.config(state=tk.NORMAL)
                # 启用Edit按钮
                self.edit_button.config(state=tk.NORMAL)
                # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
                item_values = self.tree.item(selected_item, "values")
                if item_values and item_values[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                    self.remove_button.config(state=tk.NORMAL)
                    self.edit_button.config(state=tk.NORMAL)
                    # 检查Up按钮是否可用
                    if self.tree.prev(selected_item):
                        self.up_button.config(state=tk.NORMAL)
                    else:
                        self.up_button.config(state=tk.DISABLED)
                    # 检查Down按钮是否可用
                    next_item = self.tree.next(selected_item)
                    if next_item:
                        next_values = self.tree.item(next_item, "values")
                        if next_values and next_values[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                            self.down_button.config(state=tk.NORMAL)
                        else:
                            self.down_button.config(state=tk.DISABLED)
                    else:
                        self.down_button.config(state=tk.DISABLED)
                else:
                    self.remove_button.config(state=tk.DISABLED)
                    self.up_button.config(state=tk.DISABLED)
                    self.down_button.config(state=tk.DISABLED)
            else:
                selected_item = None
                self.add_button.config(state=tk.DISABLED)
                self.remove_button.config(state=tk.DISABLED)
                self.edit_button.config(state=tk.DISABLED)
                self.up_button.config(state=tk.DISABLED)
                self.down_button.config(state=tk.DISABLED)

        # 绑定Treeview选择事件
        self.tree.bind("<<TreeviewSelect>>", on_tree_select)

        # 实现Remove功能
        def remove_item(backspace=False):
            nonlocal selected_item
            if not selected_item:
                return

            # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
            item_values = self.tree.item(selected_item, "values")
            if item_values and item_values[0] == TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                return

            # 获取选中项的索引
            index = int(item_values[0])

            # 从原始值中移除该项
            if 0 <= index < len(self.value):
                value_list = list(self.value)
                value_list.pop(index)
                self.value = tuple(value_list)
                # 调用on_change更新Parser属性
                on_change(self.value)

                # 保存当前选中项的前一个项目
                prev_item = self.tree.prev(selected_item)

                # 直接删除选中的项目
                self.tree.delete(selected_item)

                # 更新剩余项目的索引
                type_name = self.inner_maintainer.get_simplest_type_name()
                for item in self.tree.get_children():
                    item_vals = self.tree.item(item, "values")
                    if item_vals and item_vals[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                        item_idx = int(item_vals[0])
                        if item_idx > index:
                            self.tree.item(item, values=(item_idx - 1, type_name, item_vals[2]))

                # 更新最后一行的显示
                last_item = self.tree.get_children()[-1]
                self.tree.item(last_item, values=(TupleVariadicMaintainer.NEW_ITEM_INDICATOR, "", ""))

                # 根据操作类型设置新的选中项
                if backspace and prev_item:
                    # Backspace: 选择上一项
                    self.tree.selection_set(prev_item)
                    selected_item = prev_item
                    # 确保选中项在视野范围内
                    self.tree.see(prev_item)
                    # 检查新选中的是否是最后一行（NEW_ITEM_INDICATOR）
                    prev_values = self.tree.item(prev_item, "values")
                    if prev_values and prev_values[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                        self.remove_button.config(state=tk.NORMAL)
                    else:
                        self.remove_button.config(state=tk.DISABLED)
                else:
                    # Delete: 保持选中项位置不变
                    # 尝试选择当前位置的项目（如果存在）
                    current_items = self.tree.get_children()
                    if index < len(current_items):
                        new_selected = current_items[index]
                        self.tree.selection_set(new_selected)
                        selected_item = new_selected
                        # 确保选中项在视野范围内
                        self.tree.see(new_selected)
                        # 检查新选中的是否是最后一行（NEW_ITEM_INDICATOR）
                        new_values = self.tree.item(new_selected, "values")
                        if new_values and new_values[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                            self.remove_button.config(state=tk.NORMAL)
                        else:
                            self.remove_button.config(state=tk.DISABLED)
                    else:
                        # 如果索引超出范围，重置选择状态
                        selected_item = None
                        self.remove_button.config(state=tk.DISABLED)

        # 实现移动项目功能
        def move_item_up():
            nonlocal selected_item
            if not selected_item:
                return

            # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
            item_values = self.tree.item(selected_item, "values")
            if item_values and item_values[0] == TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                return

            # 获取选中项的前一个项目
            prev_item = self.tree.prev(selected_item)
            if not prev_item:
                return  # 已经是第一个项目

            # 获取选中项的索引
            index = int(item_values[0])
            if index > 0:
                # 交换元组中的项目
                value_list = list(self.value)
                value_list[index], value_list[index - 1] = value_list[index - 1], value_list[index]
                self.value = tuple(value_list)
                # 调用on_change更新Parser属性
                on_change(self.value)

                # 重新排序并更新所有项目的索引
                type_name = self.inner_maintainer.get_simplest_type_name()
                for i, item in enumerate(self.tree.get_children()):
                    item_vals = self.tree.item(item, "values")
                    if item_vals and item_vals[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                        # 找到对应的值并更新显示
                        if i < len(self.value):
                            self.tree.item(item, values=(i, type_name, repr(self.value[i])))

                # 选择移动后的项目
                # 找到新位置的项目（索引为index-1）
                new_selected = None
                for item in self.tree.get_children():
                    item_vals = self.tree.item(item, "values")
                    if item_vals and item_vals[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                        # 提取索引值进行比较
                        item_idx = int(item_vals[0])
                        if item_idx == index - 1:
                            new_selected = item
                            break

                if new_selected:
                    self.tree.selection_set(new_selected)
                    selected_item = new_selected
                    # 确保选中项在视野范围内
                    self.tree.see(new_selected)

                # 更新按钮状态
                on_tree_select(None)

                # 保持焦点在Treeview上
                self.tree.focus_set()

        def move_item_down():
            nonlocal selected_item
            if not selected_item:
                return

            # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
            item_values = self.tree.item(selected_item, "values")
            if item_values and item_values[0] == TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                return

            # 获取选中项的后一个项目
            next_item = self.tree.next(selected_item)
            if not next_item:
                return  # 已经是最后一个项目

            # 检查后一个项目是否是占位符
            next_values = self.tree.item(next_item, "values")
            if next_values and next_values[0] == TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                return  # 不能移动到占位符之后

            # 获取选中项的索引
            index = int(item_values[0])
            if index < len(self.value) - 1:
                # 交换元组中的项目
                value_list = list(self.value)
                value_list[index], value_list[index + 1] = value_list[index + 1], value_list[index]
                self.value = tuple(value_list)
                # 调用on_change更新Parser属性
                on_change(self.value)

                # 重新排序并更新所有项目的索引
                type_name = self.inner_maintainer.get_simplest_type_name()
                for i, item in enumerate(self.tree.get_children()):
                    item_vals = self.tree.item(item, "values")
                    if item_vals and item_vals[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                        # 找到对应的值并更新显示
                        if i < len(self.value):
                            self.tree.item(item, values=(i, type_name, repr(self.value[i])))

                # 选择移动后的项目
                # 找到新位置的项目（索引为index+1）
                new_selected = None
                for item in self.tree.get_children():
                    item_vals = self.tree.item(item, "values")
                    if item_vals and item_vals[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                        # 提取索引值进行比较
                        item_idx = int(item_vals[0])
                        if item_idx == index + 1:
                            new_selected = item
                            break

                if new_selected:
                    self.tree.selection_set(new_selected)
                    selected_item = new_selected
                    # 确保选中项在视野范围内
                    self.tree.see(new_selected)

                # 更新按钮状态
                on_tree_select(None)

                # 保持焦点在Treeview上
                self.tree.focus_set()

        # 共用的编辑窗口创建方法
        def create_edit_window(title, item_value, item_attribute_name):
            # 创建编辑窗口
            edit_window = tk.Toplevel(parent)
            edit_window.title(title)
            edit_window.geometry("700x500")
            edit_window.resizable(True, True)
            # 设置焦点到子窗口
            edit_window.focus_set()

            # 添加inspector LabelFrame
            inspector_frame = ttk.LabelFrame(edit_window, text="Inspector")
            inspector_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 配置inspector_frame的grid布局
            inspector_frame.grid_rowconfigure(0, weight=1)
            inspector_frame.grid_rowconfigure(1, weight=0)  # 横向滚动条固定高度
            inspector_frame.grid_columnconfigure(0, weight=1)
            inspector_frame.grid_columnconfigure(1, weight=0)  # 纵向滚动条固定宽度

            # 创建带滚动条的容器
            canvas = tk.Canvas(inspector_frame)
            scrollbar = ttk.Scrollbar(inspector_frame, orient="vertical", command=canvas.yview)
            hscrollbar = ttk.Scrollbar(inspector_frame, orient="horizontal", command=canvas.xview)
            canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=hscrollbar.set)

            # 布局 - 使用grid布局
            canvas.grid(row=0, column=0, sticky=tk.NSEW)
            scrollbar.grid(row=0, column=1, sticky=tk.NS)
            hscrollbar.grid(row=1, column=0, sticky=tk.EW)

            # 创建内部框架，用于容纳所有Inspector内容
            inner_frame = ttk.Frame(canvas, padding=10)
            # 将内部框架添加到canvas
            inner_frame_id = canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

            # 绑定事件，当内部框架大小改变时更新canvas的滚动区域
            def on_inner_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))

            inner_frame.bind("<Configure>", on_inner_configure)

            # 绑定事件，当canvas大小改变时调整内部框架的宽度
            def on_canvas_configure(event):
                """当canvas大小改变时调整内部框架的宽度和高度"""
                # 获取canvas的宽度和高度，留出滚动条的空间
                canvas_width = event.width - 5  # 留出滚动条的宽度空间
                canvas_height = event.height - 5  # 留出滚动条的高度空间

                # 获取内部框架的最小宽度和高度
                min_width = inner_frame.winfo_reqwidth()
                min_height = inner_frame.winfo_reqheight()

                # 计算内部框架的实际宽度和高度
                # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
                # 否则使用内部控件的最小宽度和高度
                actual_width = max(canvas_width, min_width)
                actual_height = max(canvas_height, min_height)
                canvas.itemconfig(inner_frame_id, width=event.width)

                # 设置内部框架的宽度和高度
                canvas.itemconfig(inner_frame_id, width=actual_width, height=actual_height)

                # 更新canvas的滚动区域
                canvas.configure(scrollregion=canvas.bbox("all"))

            canvas.bind("<Configure>", on_canvas_configure)

            # 结果变量
            result = {'value': item_value, 'confirmed': False}

            # 定义值变化的回调函数
            def on_value_change(new_val):
                result['value'] = new_val

            # 使用render_control创建编辑控件，使用inner_frame作为父容器
            control_frame, enable, disable, set_value = self.inner_maintainer.create_inspector(inner_frame,
                                                                                               item_attribute_name,
                                                                                               item_value,
                                                                                               on_value_change)
            control_frame.pack(fill=tk.BOTH, expand=True)

            # 添加按钮框架
            button_frame = ttk.Frame(edit_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)

            # 确认按钮回调
            def on_confirm():
                result['confirmed'] = True
                edit_window.destroy()

            # 取消按钮回调
            def on_cancel():
                result['confirmed'] = False
                edit_window.destroy()

            # 创建一个容器来居中按钮
            button_container = ttk.Frame(button_frame)
            button_container.pack(side=tk.TOP, anchor=tk.CENTER)

            # 添加Confirm按钮（放在前面）
            confirm_button = ttk.Button(button_container, text="Confirm", command=on_confirm)
            confirm_button.pack(side=tk.LEFT, padx=5)

            # 添加Cancel按钮
            cancel_button = ttk.Button(button_container, text="Cancel", command=on_cancel)
            cancel_button.pack(side=tk.LEFT, padx=5)

            # 绑定回车键和Esc键
            def on_return_key(event):
                on_confirm()
                return "break"  # 阻止事件继续传播

            def on_escape_key(event):
                on_cancel()
                return "break"  # 阻止事件继续传播

            # 为窗口绑定按键事件
            edit_window.bind("<Return>", on_return_key)
            edit_window.bind("<Escape>", on_escape_key)

            # 阻塞主窗口
            edit_window.transient(parent)
            edit_window.grab_set()
            parent.wait_window(edit_window)

            return result

        # 实现编辑元组元素功能
        def edit_item():
            nonlocal selected_item
            if not selected_item:
                return

            # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
            item_values = self.tree.item(selected_item, "values")
            if item_values and item_values[0] == TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                # 当焦点在<New Item>上时，调用add_item函数，等同于添加新元素
                add_item()
                return

            # 获取选中项的索引
            index = int(item_values[0])

            # 获取选中项的当前值
            if 0 <= index < len(self.value):
                current_value = self.value[index]

                # 构建属性名称：{元组属性名称}[{此元素的下标序号}]
                item_attribute_name = f"{attribute_name}[{index}]"

                # 调用共用方法创建编辑窗口
                result = create_edit_window(f"Edit {attribute_name}[{index}]", current_value, item_attribute_name)

                # 如果用户确认，更新值
                if result['confirmed']:
                    value_list = list(self.value)
                    value_list[index] = result['value']
                    self.value = tuple(value_list)
                    on_change(self.value)

                    # 更新Treeview中的显示
                    type_name = self.inner_maintainer.get_simplest_type_name()
                    for i, item in enumerate(self.tree.get_children()):
                        item_vals = self.tree.item(item, "values")
                        if item_vals and item_vals[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                            if i < len(self.value):
                                self.tree.item(item, values=(i, type_name, repr(self.value[i])))

                    # 确保选中项在视野范围内
                    self.tree.see(selected_item)
                    # 设置焦点回到Treeview并选中当前编辑的项
                    self.tree.focus_set()
                    self.tree.selection_set(selected_item)

        # 绑定Up按钮点击事件
        self.up_button.config(command=move_item_up)

        # 绑定Down按钮点击事件
        self.down_button.config(command=move_item_down)

        # 绑定Edit按钮点击事件
        self.edit_button.config(command=edit_item)

        # 实现添加元组元素功能
        def add_item():
            nonlocal selected_item
            if not selected_item:
                return

            # 确定要添加的位置
            item_values = self.tree.item(selected_item, "values")
            if item_values and item_values[0] == TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                # 如果选中的是<New Item>，则添加到元组末尾
                index = len(self.value)
            else:
                # 否则，添加到当前位置，当前选中项目及其它后续项目全部后移
                index = int(item_values[0])

            # 获取默认值
            default_value = self.inner_maintainer.get_default_value()

            # 构建属性名称：{元组属性名称}[{此元素的下标序号}]
            item_attribute_name = f"{attribute_name}[{index}]"

            # 调用共用方法创建编辑窗口
            result = create_edit_window(f"Edit New {attribute_name}[{index}]", default_value, item_attribute_name)

            # 如果用户确认，添加新值
            if result['confirmed']:
                # 在指定位置插入新元素
                value_list = list(self.value)
                value_list.insert(index, result['value'])
                self.value = tuple(value_list)
                on_change(self.value)

                # 清空Treeview并重新填充
                for item in self.tree.get_children():
                    self.tree.delete(item)

                # 重新填充Treeview
                type_name = self.inner_maintainer.get_simplest_type_name()
                if self.value:
                    for i, item in enumerate(self.value):
                        self.tree.insert("", tk.END, values=(i, type_name, repr(item)))

                # 添加空白行
                self.tree.insert("", tk.END, values=(TupleVariadicMaintainer.NEW_ITEM_INDICATOR, "", ""))

                # 选择新添加的项目
                items = self.tree.get_children()
                if index < len(items):
                    self.tree.selection_set(items[index])
                    selected_item = items[index]
                    # 确保新添加的项目在视野范围内
                    self.tree.see(selected_item)
                    # 设置焦点回到Treeview并选中新插入的项
                    self.tree.focus_set()
                    self.tree.selection_set(selected_item)

        # 绑定Add按钮点击事件
        self.add_button.config(command=add_item)

        # 绑定方向键上/下
        def on_up_key(event):
            move_item_up()
            return "break"  # 阻止事件继续传播

        def on_down_key(event):
            move_item_down()
            return "break"  # 阻止事件继续传播

        # 为Treeview绑定方向键
        self.tree.bind("<Up>", on_up_key)
        self.tree.bind("<Down>", on_down_key)

        # 绑定Remove按钮点击事件
        def on_remove_button_click():
            remove_item(backspace=False)
            # 将焦点设置回Treeview，确保按键依然能被接收
            self.tree.focus_set()
            # 如果有选中项，确保选中项被高亮
            if self.tree.selection():
                self.tree.selection_set(self.tree.selection())

        # 绑定Remove按钮点击事件
        self.remove_button.config(command=on_remove_button_click)

        # 绑定Delete键
        def on_delete_key(event):
            selected_items = self.tree.selection()
            if selected_items:
                item_values = self.tree.item(selected_items[0], "values")
                if item_values and item_values[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                    remove_item(backspace=False)
            return "break"  # 阻止事件继续传播

        # 为Treeview绑定Delete键
        self.tree.bind("<Delete>", on_delete_key)

        # 绑定Backspace键
        def on_backspace_key(event):
            selected_items = self.tree.selection()
            if selected_items:
                item_values = self.tree.item(selected_items[0], "values")
                if item_values and item_values[0] != TupleVariadicMaintainer.NEW_ITEM_INDICATOR:
                    remove_item(backspace=True)
            return "break"  # 阻止事件继续传播

        # 为Treeview绑定Backspace键
        self.tree.bind("<BackSpace>", on_backspace_key)

        # 确保Treeview可以获得焦点
        self.tree.config(takefocus=True)

        # 绑定双击事件，双击项目打开编辑界面
        def on_tree_double_click(event):
            # 获取点击的项
            item = self.tree.identify_row(event.y)
            # 只有双击到具体项时才执行编辑操作，双击标题不触发
            if item:
                # 模拟点击Edit按钮
                edit_item()

        self.tree.bind("<Double-1>", on_tree_double_click)

        # Bind mouse wheel to horizontal scroll with increased sensitivity
        def on_mouse_wheel(event):
            # 检测Shift键是否被按下（使用更可靠的方式）
            shift_pressed = event.state & 0x0001 != 0

            if not shift_pressed:  # No modifier key
                # Vertical scroll (default behavior)
                self.tree.yview_scroll(-1 * (event.delta // 120), "units")
            else:  # Shift key pressed
                # Horizontal scroll with increased sensitivity
                self.tree.xview_scroll(-1 * (event.delta // 10), "units")

        self.tree.bind("<MouseWheel>", on_mouse_wheel)

        def enable():
            self.tree.config(state='normal')
            for widget in self.buttons_frame.winfo_children():
                widget.config(state='normal')

        def disable():
            self.tree.config(state='disabled')
            for widget in self.buttons_frame.winfo_children():
                widget.config(state='disabled')

        def set_value(new_value):
            # 清空Treeview
            for item in self.tree.get_children():
                self.tree.delete(item)

            # 重新填充Treeview
            type_name = self.inner_maintainer.get_simplest_type_name()
            if new_value:
                for i, item in enumerate(new_value):
                    self.tree.insert("", tk.END, values=(i, type_name, repr(item)))

            # 添加空白行
            self.tree.insert("", tk.END, values=(TupleVariadicMaintainer.NEW_ITEM_INDICATOR, "", ""))

        return frame, enable, disable, set_value

    def create_inspector(self, parent, attribute_name, value, on_change):
        """Render control for editing variadic tuple attribute"""
        # Update instance attribute_name if provided
        if attribute_name:
            self.attribute_name = attribute_name

        self.frame = ttk.Frame(parent)

        # Attribute name
        self.attribute_label = ttk.Label(self.frame, text=f"Attribute: {self.attribute_name}")
        self.attribute_label.pack(anchor=tk.W, padx=10, pady=5)

        # Type
        self.type_label = ttk.Label(self.frame, text=f"Type: {self.get_simplest_type_name()}")
        self.type_label.pack(anchor=tk.W, padx=10, pady=5)

        # Value editor in Editor panel
        self.edit_frame = ttk.LabelFrame(self.frame, text="Editor")
        self.edit_frame.pack(anchor=tk.N, padx=10, pady=5, fill=tk.BOTH, expand=True)

        # 创建一个包装的on_change函数，确保元组修改后能正确更新
        def on_tuple_change(new_value):
            # 直接调用原始的on_change函数，传递新值
            on_change(new_value)

        self.edit_control, enable, disable, set_value = self.create_editor(
            self.edit_frame, value, on_tuple_change, self.attribute_name)
        self.edit_control.pack(fill=tk.BOTH, expand=True)

        return self.frame, enable, disable, set_value
