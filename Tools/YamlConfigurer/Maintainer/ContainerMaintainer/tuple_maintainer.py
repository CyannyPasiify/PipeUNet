import copy
from typing import Any, Tuple, Type, Callable, Optional, List, Literal, get_origin, get_args, Dict
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.unsupported_maintainer import UnsupportedMaintainer
from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer
from Tools.YamlConfigurer.Maintainer.container_maintainer import ContainerMaintainer
from Tools.YamlConfigurer.auxiliary_functions import simplify_type


class TupleMaintainer(ContainerMaintainer):
    """tuple type Maintainer"""
    # Static variable for new item indicator
    NEW_ITEM_INDICATOR = "<New>"

    @classmethod
    @override
    def default_standalone_window_size(cls: Type) -> Tuple[int, int]:
        # W, H
        return 1236, 600

    @classmethod
    @override
    def shall_hotkey_confirm_cancel(cls: Type) -> Tuple[bool, bool]:
        # If in Standalone window, shall this type of maintainer react to
        # - Enter as Confirm
        # - Esc as Cancel
        # Hotkey enabled for (Confirm, Cancel)

        # Tuple may have complex sub-editors, shall not use hotkeys
        return False, False

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Tuple[Any],
            attribute_value: Any = None,
            logger: Any = None
    ):
        """Initialize Maintainer
        
        Args:
            attribute_name: Name of the tuple attribute
            attribute_type: Type of the tuple
            attribute_value: Initial value
            logger: Logger instance for logging
        """
        # Simplify first !!
        attribute_type: Type = simplify_type(attribute_type)
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        self.item_maintainer_cls_list: Optional[List[Type[BaseMaintainer]]] = None
        # Detect Maintainer for the arg type
        if self.is_type_compatible():
            from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
            type_args: Tuple[Any, ...] = get_args(self.attribute_type)
            # Check if it's Tuple[E, ...]
            if len(type_args) == 2 and type_args[1] == Ellipsis:
                # Tuple[E, Ellipsis(...)] - similar to List[E]
                self.item_maintainer_cls_list = [MaintainerFactory.get_maintainer_cls_supported_type(type_args[0])]
            # tuple[()], Tuple[()]
            elif len(type_args) == 0 or len(type_args) == 1 and type_args[0] == tuple():
                pass
            else:
                # Tuple[E1, E2, ...] - fixed length
                self.item_maintainer_cls_list = []
                for arg_type in type_args:
                    self.item_maintainer_cls_list.append(MaintainerFactory.get_maintainer_cls_supported_type(arg_type))

        # Vars
        self.view_mode: Literal["Standalone", "Packed"] = "Standalone"
        self.item_inspector_inner_frame_id: Optional[int] = None  # Standalone only
        self.popup_canvas_inner_frame_id: Optional[int] = None  # Popup only
        self.popup_wnd_result: Optional[Dict[str, Any]] = None  # Popup only
        self.item_maintainer: Optional[BaseMaintainer] = None  # Popup only
        self.current_selected_item: Optional[str] = None
        # Widgets
        ## Common
        self.editor_label_frame: Optional[ttk.LabelFrame] = None
        self.buttons_frame: Optional[ttk.Frame] = None
        self.add_button: Optional[ttk.Button] = None
        self.remove_button: Optional[ttk.Button] = None
        self.up_button: Optional[ttk.Button] = None
        self.down_button: Optional[ttk.Button] = None
        self.edit_button: Optional[ttk.Button] = None  # Packed only
        self.add_button: Optional[ttk.Button] = None
        self.tree_frame: Optional[ttk.Frame] = None
        self.list_treeview: Optional[ttk.Treeview] = None
        self.tree_frame_vscrollbar: Optional[ttk.Scrollbar] = None
        self.tree_frame_hscrollbar: Optional[ttk.Scrollbar] = None
        ## Popup
        self.popup_top_level: Optional[tk.Toplevel] = None
        self.popup_inspector_frame: Optional[ttk.Frame] = None
        self.popup_canvas: Optional[tk.Canvas] = None
        self.popup_canvas_vscrollbar: Optional[ttk.Scrollbar] = None
        self.popup_canvas_hscrollbar: Optional[ttk.Scrollbar] = None
        self.popup_canvas_inner_frame: Optional[ttk.Frame] = None
        self.popup_inspector_content: Optional[ttk.Widget] = None
        self.popup_wnd_button_container: Optional[ttk.Frame] = None
        self.popup_confirm_button: Optional[ttk.Button] = None
        self.popup_cancel_button: Optional[ttk.Button] = None
        ## Standalone only
        self.double_paned_wnd: Optional[ttk.PanedWindow] = None
        self.main_inspector_left_frame: Optional[ttk.Frame] = None
        self.item_inspector_right_frame: Optional[ttk.Frame] = None
        self.item_inspector_label_frame: Optional[ttk.LabelFrame] = None
        self.item_inspector_canvas: Optional[tk.Canvas] = None
        self.item_inspector_vscrollbar: Optional[ttk.Scrollbar] = None
        self.item_inspector_hscrollbar: Optional[ttk.Scrollbar] = None
        self.item_inspector_inner_frame: Optional[ttk.Frame] = None

    @override
    def is_type_compatible(self) -> bool:
        # Assuming attribute_type is simplified
        origin: Type = get_origin(self.attribute_type)
        return origin in {tuple, Tuple}

    @override
    def is_value_compatible(self) -> bool:
        if not self.is_type_compatible():
            return False

        # Value is not a tuple, never compatible
        if not isinstance(self.attribute_value, tuple):
            return False

        # Empty tuple () is always compatible
        if len(self.attribute_value) == 0:
            return True

        type_args: Tuple[Any, ...] = get_args(self.attribute_type)

        # tuple[()], Tuple[()]
        if len(type_args) == 0 or len(type_args) == 1 and type_args[0] == tuple():
            return False

        # Tuple[E, Ellipsis(...)]
        if len(type_args) == 2 and type_args[1] == Ellipsis:
            # Similar to List[E]
            maintainer: Type[BaseMaintainer] = self.item_maintainer_cls_list[0]

            # Unsupported element type, never compatible
            if issubclass(maintainer, UnsupportedMaintainer):
                return False

            # Check compatibility for each item
            for item in self.attribute_value:
                if not maintainer.is_value_compatible_static(item, type_args[0]):
                    return False
            return True

        # Tuple[E1, E2, ...]
        # Check if value length matches type length
        if len(self.attribute_value) != len(type_args):
            return False

        # Check compatibility for each item
        for i, item in enumerate(self.attribute_value):
            if i < len(self.item_maintainer_cls_list):
                maintainer_cls = self.item_maintainer_cls_list[i]
                if issubclass(maintainer_cls, UnsupportedMaintainer):
                    return False
                if not maintainer_cls.is_value_compatible_static(item, type_args[i]):
                    return False
        return True

    @override
    def get_default_value(self, *args, **kwargs) -> Tuple:
        type_args: Tuple[Any, ...] = get_args(self.attribute_type)

        # tuple[()], Tuple[()]
        if len(type_args) == 0 or len(type_args) == 1 and type_args[0] == tuple():
            return tuple()

        # Tuple[E, Ellipsis(...)]
        if len(type_args) == 2 and type_args[1] == Ellipsis:
            return tuple()

        # Tuple[E1, E2, ...]
        default_values = []
        for i, arg_type in enumerate(type_args):
            if i < len(self.item_maintainer_cls_list):
                default_values.append(
                    self.item_maintainer_cls_list[i].get_default_value_static(target_type=arg_type))
            else:
                default_values.append(None)
        return tuple(default_values)

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        # Assuming attribute_type is simplified
        return self.attribute_type

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        if not self.is_type_compatible():
            return ""

        type_args: Tuple[Any, ...] = get_args(self.attribute_type)

        # tuple[()], Tuple[()]
        if len(type_args) == 0 or len(type_args) == 1 and type_args[0] == tuple():
            return "Tuple[()]"

        # Tuple[E, Ellipsis(...)]
        if len(type_args) == 2 and type_args[1] == Ellipsis:
            maintainer: Type[BaseMaintainer] = self.item_maintainer_cls_list[0]
            subtype_str: str = maintainer.get_simplest_type_name_static(
                target_type=type_args[0],
                *args, **kwargs
            )
            return f"Tuple[{subtype_str}, ...]"

        # Tuple[E1, E2, ...]
        subtype_strs = []
        for i, arg_type in enumerate(type_args):
            if i < len(self.item_maintainer_cls_list):
                subtype_str = self.item_maintainer_cls_list[i].get_simplest_type_name_static(
                    target_type=arg_type,
                    *args, **kwargs
                )
                subtype_strs.append(subtype_str)
            else:
                subtype_strs.append("Any")
        return f"Tuple[{', '.join(subtype_strs)}]"

    @override
    def create_inspector(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        """
            Shall call create_editor()
        """
        # Clear existing inspector (if exists)
        if self.inspector is not None:
            self.inspector.destroy()

        self.inspector = ttk.Frame(parent)
        self.inspector.pack(fill=tk.BOTH, expand=True)

        # Editor label frame and editor instance
        if self.view_mode == "Packed":
            self._create_attribute_type_display(self.inspector)

            if not self.can_edit():
                return self.inspector

            self.editor_label_frame = ttk.LabelFrame(self.inspector, text="Editor")
            self.editor_label_frame.pack(
                anchor=tk.N, padx=10, pady=5,
                fill=tk.BOTH,
                expand=True
            )

            self.editor = self.create_editor(self.editor_label_frame, on_value_change)
            self.editor.pack(fill=tk.BOTH, expand=True)

        elif self.view_mode == "Standalone":
            # 左右分栏，使用PanedWindow实现可拖动调整
            self.double_paned_wnd = ttk.PanedWindow(self.inspector, orient=tk.HORIZONTAL)
            self.double_paned_wnd.pack(fill=tk.BOTH, expand=True)

            # 初始化分割位置50%
            def on_pane_map(event: Optional[tk.Event] = None):
                # 控件真正渲染完成了 → 此时宽度 100% 准确
                w = self.double_paned_wnd.winfo_width()
                self.double_paned_wnd.sashpos(0, w // 2)

                # 只执行一次，执行完解绑，避免多次触发
                self.double_paned_wnd.unbind("<Map>")

            self.double_paned_wnd.bind("<Map>", on_pane_map)

            # 主监视器框 - 左
            self.main_inspector_left_frame = ttk.Frame(self.double_paned_wnd)
            self.double_paned_wnd.add(self.main_inspector_left_frame, weight=1)

            # 元素监视器框 - 右
            self.item_inspector_right_frame = ttk.Frame(self.double_paned_wnd)
            self.double_paned_wnd.add(self.item_inspector_right_frame, weight=1)

            # 主监视器内容
            # Attribute and type
            self._create_attribute_type_display(self.main_inspector_left_frame)

            if not self.can_edit():
                return self.inspector

            # Tuple label frame
            self.editor_label_frame = ttk.LabelFrame(self.main_inspector_left_frame, text="Tuple")
            self.editor_label_frame.pack(
                anchor=tk.N,
                fill=tk.BOTH,
                expand=True,
                padx=(5, 0),
                pady=(5, 0)
            )

            # 元素监视器内容
            # 右侧Inspector面板 - 占据所有空间
            self.item_inspector_label_frame = ttk.LabelFrame(
                self.item_inspector_right_frame,
                text="Inspector"
            )
            self.item_inspector_label_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            self.item_inspector_label_frame.grid_rowconfigure(0, weight=1)
            self.item_inspector_label_frame.grid_rowconfigure(1, weight=0)  # 横向滚动条固定高度
            self.item_inspector_label_frame.grid_columnconfigure(0, weight=1)
            self.item_inspector_label_frame.grid_columnconfigure(1, weight=0)  # 纵向滚动条固定宽度

            # 创建带滚动条的容器
            self.item_inspector_canvas = tk.Canvas(self.item_inspector_label_frame, highlightthickness=0)
            self.item_inspector_vscrollbar = ttk.Scrollbar(
                self.item_inspector_label_frame,
                orient="vertical",
                command=self.item_inspector_canvas.yview
            )
            self.item_inspector_hscrollbar = ttk.Scrollbar(
                self.item_inspector_label_frame,
                orient="horizontal",
                command=self.item_inspector_canvas.xview
            )
            self.item_inspector_canvas.configure(
                yscrollcommand=self.item_inspector_vscrollbar.set,
                xscrollcommand=self.item_inspector_hscrollbar.set
            )

            # 布局 - 使用grid布局
            self.item_inspector_canvas.grid(row=0, column=0, sticky=tk.NSEW)
            self.item_inspector_vscrollbar.grid(row=0, column=1, sticky=tk.NS)
            self.item_inspector_hscrollbar.grid(row=1, column=0, sticky=tk.EW)

            # 创建内部框架，用于容纳所有Inspector内容
            self.item_inspector_inner_frame = ttk.Frame(self.item_inspector_canvas, padding=10)
            # 将内部框架添加到canvas
            self.item_inspector_inner_frame_id = self.item_inspector_canvas.create_window(
                (0, 0),
                window=self.item_inspector_inner_frame,
                anchor=tk.NW
            )

            # 绑定事件，当内部框架大小改变时更新canvas的滚动区域
            self.item_inspector_inner_frame.bind("<Configure>", self._on_item_inspector_inner_frame_configure)
            # 绑定事件，当canvas大小改变时调整内部框架的宽度
            self.item_inspector_canvas.bind("<Configure>", self._on_item_inspector_canvas_configure)

            # Now create editor with sub-inspector
            self.editor = self.create_editor(self.editor_label_frame, on_value_change)
            self.editor.pack(fill=tk.BOTH, expand=True)
        else:
            raise NotImplementedError(f"{self.view_mode} is not implemented")

        return self.inspector

    def _create_attribute_type_display(self, parent: ttk.Widget):
        # Attribute name
        self.attribute_frame = ttk.Frame(parent)
        self.attribute_frame.pack(anchor=tk.W, padx=10, pady=5)
        self.attribute_title_label = ttk.Label(self.attribute_frame, text="Attribute:")
        self.attribute_title_label.pack(side=tk.LEFT)
        self.attribute_content_label = ttk.Label(self.attribute_frame, text=f"{self.attribute_name}")
        self.attribute_content_label.pack(side=tk.LEFT)

        # Type name
        self.type_frame = ttk.Frame(parent)
        self.type_frame.pack(anchor=tk.W, padx=10, pady=5)
        self.type_title_label = ttk.Label(self.type_frame, text="Type:")
        self.type_title_label.pack(side=tk.LEFT)
        self.type_content_label = ttk.Label(self.type_frame, text=f"{self.get_simplest_type_name()}")
        self.type_content_label.pack(side=tk.LEFT)

    def _on_item_inspector_inner_frame_configure(self, event: Optional[tk.Event] = None) -> None:
        """当内部框架大小改变时更新canvas的滚动区域"""
        # 更新canvas的滚动区域
        self.item_inspector_canvas.configure(scrollregion=self.item_inspector_canvas.bbox("all"))

    def _on_item_inspector_canvas_configure(self, event: Optional[tk.Event] = None) -> None:
        """当canvas大小改变时调整内部框架的宽度和高度"""
        # 获取canvas的宽度和高度，留出滚动条的空间
        canvas_width: int = event.width - 5  # 留出滚动条的宽度空间
        canvas_height: int = event.height - 5  # 留出滚动条的高度空间

        # 获取内部框架的最小宽度和高度
        min_width: int = self.item_inspector_inner_frame.winfo_reqwidth()
        min_height: int = self.item_inspector_inner_frame.winfo_reqheight()

        # 计算内部框架的实际宽度和高度
        # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
        # 否则使用内部控件的最小宽度和高度
        actual_width: int = max(canvas_width, min_width)
        actual_height: int = max(canvas_height, min_height)

        # 设置内部框架的宽度和高度
        self.item_inspector_canvas.itemconfig(
            self.item_inspector_inner_frame_id,
            width=actual_width,
            height=actual_height
        )

        # 更新canvas的滚动区域
        self.item_inspector_canvas.configure(scrollregion=self.item_inspector_canvas.bbox("all"))

    @override
    def create_editor(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        super().create_editor(parent, on_value_change)

        # Create buttons frame
        self.buttons_frame = ttk.Frame(self.editor)
        self.buttons_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)

        # Add buttons with horizontal stretch
        self.add_button = ttk.Button(self.buttons_frame, text="Add", state=tk.DISABLED, width=6)
        self.add_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.remove_button = ttk.Button(self.buttons_frame, text="Remove", state=tk.DISABLED, width=6)
        self.remove_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.up_button = ttk.Button(self.buttons_frame, text="Up", state=tk.DISABLED, width=6)
        self.up_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.down_button = ttk.Button(self.buttons_frame, text="Down", state=tk.DISABLED, width=6)
        self.down_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        if self.view_mode == "Packed":
            self.edit_button = ttk.Button(self.buttons_frame, text="Edit", state=tk.DISABLED, width=6)
            self.edit_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Create Treeview for tuple elements
        self.tree_frame = ttk.Frame(self.editor)
        self.tree_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.BOTH, expand=True)

        type_args: Tuple[Any, ...] = get_args(self.attribute_type)

        # Check if it's Tuple[E, ...]
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        # Create Treeview with Index, Type and Value columns
        if is_variable_length:
            # For Tuple[E, ...], similar to List[E]
            self.list_treeview = ttk.Treeview(self.tree_frame, columns=("index", "value"), show="headings")
            self.list_treeview.heading("index", text="Index")
            # Add type information to Value column heading
            from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
            item_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(type_args[0])
            inner_type_name = item_maintainer_cls.get_simplest_type_name_static(target_type=type_args[0])
            self.list_treeview.heading("value", text=f"Value[{inner_type_name}]")
            # 初始设置列宽
            self.list_treeview.column("index", minwidth=45, width=45, stretch=False, anchor=tk.W)
            self.list_treeview.column("value", minwidth=80, width=80, stretch=False, anchor=tk.W)
        else:
            # For Tuple[E1, E2, ...], fixed length
            self.list_treeview = ttk.Treeview(self.tree_frame, columns=("index", "type", "value"), show="headings")
            self.list_treeview.heading("index", text="Index")
            self.list_treeview.heading("type", text="Type")
            self.list_treeview.heading("value", text="Value")
            # 初始设置列宽
            self.list_treeview.column("index", minwidth=45, width=45, stretch=False, anchor=tk.W)
            self.list_treeview.column("type", minwidth=80, width=80, stretch=False, anchor=tk.W)
            self.list_treeview.column("value", minwidth=80, width=80, stretch=False, anchor=tk.W)
        
        # 绑定Treeview的Configure事件，当Treeview大小改变时调整列宽
        def adjust_tree_columns(event: Optional[tk.Event] = None):
            # 获取Treeview的实际宽度
            tree_width = self.list_treeview.winfo_width()
            
            # 确保tree_width有有效值
            if tree_width > 0:
                # 计算各列的宽度，预留一些空间给滚动条和边框
                available_width = tree_width - 2  # 减去滚动条和边框的宽度
                
                if is_variable_length:
                    # 2列的情况：Index和Value
                    index_ratio = 0.2  # 20%
                    # 计算各列宽度
                    index_width = max(45, int(available_width * index_ratio))
                    value_width = max(80, available_width - index_width)
                    
                    # 确保总和不超过可用宽度
                    total_width = index_width + value_width
                    if total_width > available_width:
                        # 调整value列宽度
                        value_width = available_width - index_width
                    
                    # 设置列宽
                    self.list_treeview.column("index", minwidth=45, width=index_width, stretch=False, anchor=tk.W)
                    self.list_treeview.column("value", minwidth=80, width=value_width, stretch=False, anchor=tk.W)
                else:
                    # 3列的情况：Index, Type和Value
                    index_ratio = 0.2  # 20%
                    type_ratio = 0.3       # 30%
                    # 计算各列宽度
                    index_width = max(45, int(available_width * index_ratio))
                    type_width = max(80, int(available_width * type_ratio))
                    value_width = max(80, available_width - index_width - type_width)
                    
                    # 确保总和不超过可用宽度
                    total_width = index_width + type_width + value_width
                    if total_width > available_width:
                        # 调整value列宽度
                        value_width = available_width - index_width - type_width
                    
                    # 设置列宽
                    self.list_treeview.column("index", minwidth=45, width=index_width, stretch=False, anchor=tk.W)
                    self.list_treeview.column("type", minwidth=80, width=type_width, stretch=False, anchor=tk.W)
                    self.list_treeview.column("value", minwidth=80, width=value_width, stretch=False, anchor=tk.W)
        
        # 绑定事件
        self.list_treeview.bind("<Configure>", adjust_tree_columns, add="+")

        # Add vertical scrollbar
        self.tree_frame_vscrollbar = ttk.Scrollbar(
            self.tree_frame,
            orient=tk.VERTICAL,
            command=self.list_treeview.yview
        )
        self.list_treeview.configure(yscrollcommand=self.tree_frame_vscrollbar.set)
        self.tree_frame_vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add horizontal scrollbar with custom command to increase step size
        def custom_xview(*args):
            # args[0] 是 'moveto' 或 'scroll'
            if args[0] == 'scroll':
                # args[1] 是滚动单位数，args[2] 是单位类型 ('units' 或 'pages')
                # 增加滚动步长
                if args[2] == 'units':
                    # 对于单位滚动，增加步长
                    self.list_treeview.xview_scroll(int(args[1]) * 10, 'units')
                else:
                    # 对于页滚动，保持默认行为
                    self.list_treeview.xview_scroll(int(args[1]), 'pages')
            else:
                # 对于 moveto，保持默认行为
                self.list_treeview.xview_moveto(args[1])

        self.tree_frame_hscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL, command=custom_xview)
        self.list_treeview.configure(xscrollcommand=self.tree_frame_hscrollbar.set)
        self.tree_frame_hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.list_treeview.pack(fill=tk.BOTH, expand=True)

        # Populate Treeview with initial values
        if self.editor_value:
            for i, item in enumerate(self.editor_value):
                if is_variable_length:
                    self.list_treeview.insert("", tk.END, values=(i, repr(item)))
                else:
                    # For fixed length tuple, show type information
                    type_name = "Any"
                    if i < len(self.item_maintainer_cls_list):
                        type_args = get_args(self.attribute_type)
                        if i < len(type_args):
                            type_name = self.item_maintainer_cls_list[i].get_simplest_type_name_static(
                                target_type=type_args[i])
                    self.list_treeview.insert("", tk.END, values=(i, type_name, repr(item)))

        # Add a blank row at the end for variable length tuple
        if is_variable_length:
            self.list_treeview.insert("", tk.END, values=(TupleMaintainer.NEW_ITEM_INDICATOR, ""))

        # 跟踪当前选中的项目
        self.current_selected_item = None

        # 绑定Treeview选择事件
        self.list_treeview.bind("<<TreeviewSelect>>", self._on_treeview_select)

        # 绑定Add按钮点击事件
        self.add_button.config(command=self._add_item)

        # 绑定Remove按钮点击事件
        self.remove_button.config(command=self._on_remove_button_click)

        # 绑定Up按钮点击事件
        self.up_button.config(command=self._move_item_up)

        # 绑定Down按钮点击事件
        self.down_button.config(command=self._move_item_down)

        # 绑定Edit按钮点击事件
        if self.view_mode == "Packed":
            self.edit_button.config(command=self._edit_item)

        # 为Treeview绑定Delete键
        self.list_treeview.bind("<Delete>", self._on_delete_key)

        # 为Treeview绑定Backspace键
        self.list_treeview.bind("<BackSpace>", self._on_backspace_key)

        # 确保Treeview可以获得焦点
        self.list_treeview.config(takefocus=True)

        self.list_treeview.bind("<Double-1>", self._on_tree_double_click)

        # Bind mouse wheel to horizontal scroll with increased sensitivity
        def on_mouse_wheel(event: Optional[tk.Event] = None):
            # 检测Shift键是否被按下（使用更可靠的方式）
            shift_pressed = event.state & 0x0001 != 0

            if not shift_pressed:  # No modifier key
                # Vertical scroll (default behavior)
                self.list_treeview.yview_scroll(-1 * (event.delta // 120), "units")
            else:  # Shift key pressed
                # Horizontal scroll with increased sensitivity
                self.list_treeview.xview_scroll(-1 * (event.delta // 10), "units")

        self.list_treeview.bind("<MouseWheel>", on_mouse_wheel)

        # Disable add/remove buttons for fixed length tuple
        type_args = get_args(self.attribute_type)
        if not (len(type_args) == 2 and type_args[1] == Ellipsis):
            self.add_button.config(state=tk.DISABLED)
            self.remove_button.config(state=tk.DISABLED)
            self.up_button.config(state=tk.DISABLED)
            self.down_button.config(state=tk.DISABLED)

        return self.editor

    # 处理Treeview选择事件
    def _on_treeview_select(self, event: Optional[tk.Event] = None):
        selected_items = self.list_treeview.selection()
        if selected_items:
            self.current_selected_item: str = selected_items[0]

            type_args = get_args(self.attribute_type)
            is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

            if is_variable_length:
                # 启用Add按钮
                self.add_button.config(state=tk.NORMAL)
                # 启用Edit按钮
                if self.view_mode == "Packed":
                    self.edit_button.config(state=tk.NORMAL)
                # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
                item_values = self.list_treeview.item(self.current_selected_item, "values")
                if item_values and item_values[0] != TupleMaintainer.NEW_ITEM_INDICATOR:
                    self.remove_button.config(state=tk.NORMAL)
                    if self.view_mode == "Packed":
                        self.edit_button.config(state=tk.NORMAL)
                    # 检查Up按钮是否可用
                    if self.list_treeview.prev(self.current_selected_item):
                        self.up_button.config(state=tk.NORMAL)
                    else:
                        self.up_button.config(state=tk.DISABLED)
                    # 检查Down按钮是否可用
                    next_item: str = self.list_treeview.next(self.current_selected_item)
                    if next_item:
                        next_values: Any = self.list_treeview.item(next_item, "values")
                        if next_values and next_values[0] != TupleMaintainer.NEW_ITEM_INDICATOR:
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
                # For fixed length tuple, only enable edit button
                if self.view_mode == "Packed":
                    self.edit_button.config(state=tk.NORMAL)
        else:
            self.current_selected_item = None
            type_args = get_args(self.attribute_type)
            is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

            if is_variable_length:
                self.add_button.config(state=tk.DISABLED)
                self.remove_button.config(state=tk.DISABLED)
                self.up_button.config(state=tk.DISABLED)
                self.down_button.config(state=tk.DISABLED)

            if self.view_mode == "Packed":
                self.edit_button.config(state=tk.DISABLED)

        if self.view_mode == "Standalone":
            self._update_item_inspector()

    def _on_tuple_content_change(self, new_value: Any) -> None:
        """Handle value change"""
        # Assuming new_value is mutable
        is_valid, validated_value = self.editor_validate(new_value)
        if is_valid:
            # editor_value shall be already modified by methods like _remove_item
            self.editor_value = validated_value
            # Transferring a copy of editor_value, always assuming editor_value as immutable
            self.on_value_change(copy.deepcopy(self.editor_value))

    def _remove_item(self, select_prev: bool = False):
        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        if not is_variable_length:
            return  # Fixed length tuple, cannot remove items

        if self.current_selected_item is None:
            return

        # 获取选中项的索引
        index: int = self.list_treeview.index(self.current_selected_item)
        items: Tuple[str, ...] = self.list_treeview.get_children()

        # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
        if index == len(items) - 1:
            return

        assert 0 <= index < len(self.editor_value), f"Index {index} out of range"

        # 从原始值中移除该项
        list_value = list(self.editor_value)
        list_value.pop(index)
        self.editor_value = tuple(list_value)

        # 保存当前选中项的前一个项目
        # item = [Index, Value] or [Index, Type, Value]
        prev_item: Optional[str] = self.list_treeview.prev(self.current_selected_item)

        # 直接删除选中的项目
        self.list_treeview.delete(self.current_selected_item)

        # 更新剩余项目的索引
        for idx in range(index + 1, len(items) - 1):
            # item = [Index, Value] or [Index, Type, Value]
            item_vals = self.list_treeview.item(items[idx], "values")
            if len(item_vals) == 2:
                # [Index, Value]
                self.list_treeview.item(items[idx], values=(idx - 1, item_vals[1]))
            else:
                # [Index, Type, Value]
                self.list_treeview.item(items[idx], values=(idx - 1, item_vals[1], item_vals[2]))

        # 根据操作类型设置新的选中项
        if select_prev and prev_item:
            # Backspace: 选择上一项
            self.list_treeview.selection_set(prev_item)
            self.current_selected_item = prev_item
            # 确保选中项在视野范围内
            self.list_treeview.see(prev_item)
        else:
            # Delete: 保持选中项位置不变
            # 尝试选择当前位置的项目（如果存在）
            current_items: Tuple[str, ...] = self.list_treeview.get_children()
            if index < len(current_items):
                new_selected: str = current_items[index]
                self.list_treeview.selection_set(new_selected)
                self.current_selected_item = new_selected
                # 确保选中项在视野范围内
                self.list_treeview.see(new_selected)
                # 检查新选中的是否是最后一行（NEW_ITEM_INDICATOR）
                if index != len(current_items) - 1:
                    self.remove_button.config(state=tk.NORMAL)
                else:
                    self.remove_button.config(state=tk.DISABLED)
            else:
                # 如果索引超出范围，重置选择状态
                self.current_selected_item = None
                self.remove_button.config(state=tk.DISABLED)

        # 调用_on_tuple_content_change传递更新
        self._on_tuple_content_change(self.editor_value)

    def _move_item_up(self):
        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        if not is_variable_length:
            return  # Fixed length tuple, cannot reorder items

        if not self.current_selected_item:
            return

        # 获取选中项的索引
        index: int = self.list_treeview.index(self.current_selected_item)
        items: Tuple[str, ...] = self.list_treeview.get_children()

        # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
        if index == len(items) - 1:
            return

        assert 0 < index < len(self.editor_value), f"Index {index} out of range"

        # 获取选中项的前一个项目
        prev_item: str = self.list_treeview.prev(self.current_selected_item)
        if not prev_item:
            return  # 已经是第一个项目

        # 交换元组中的项目
        list_value = list(self.editor_value)
        list_value[index], list_value[index - 1] = list_value[index - 1], list_value[index]
        self.editor_value = tuple(list_value)

        # 更新交换项目的内容
        if len(self.list_treeview.item(items[index - 1], "values")) == 2:
            # [Index, Value]
            self.list_treeview.item(items[index - 1], values=(index - 1, repr(self.editor_value[index - 1])))
            self.list_treeview.item(items[index], values=(index, repr(self.editor_value[index])))
        else:
            # [Index, Type, Value]
            type_args = get_args(self.attribute_type)
            type_name_prev = "Any"
            type_name_current = "Any"
            if index - 1 < len(self.item_maintainer_cls_list):
                if index - 1 < len(type_args):
                    type_name_prev = self.item_maintainer_cls_list[index - 1].get_simplest_type_name_static(
                        target_type=type_args[index - 1])
            if index < len(self.item_maintainer_cls_list):
                if index < len(type_args):
                    type_name_current = self.item_maintainer_cls_list[index].get_simplest_type_name_static(
                        target_type=type_args[index])
            self.list_treeview.item(items[index - 1],
                                    values=(index - 1, type_name_prev, repr(self.editor_value[index - 1])))
            self.list_treeview.item(items[index], values=(index, type_name_current, repr(self.editor_value[index])))

        # 选择移动后的项目
        # 找到新位置的项目（索引为index-1）
        new_selected: str = items[index - 1]
        self.list_treeview.selection_set(new_selected)
        self.current_selected_item = new_selected
        # 确保选中项在视野范围内
        self.list_treeview.see(new_selected)

        # 更新按钮状态
        self._on_treeview_select()

        # 保持焦点在Treeview上
        self.list_treeview.focus_set()
        self.list_treeview.focus(new_selected)

        # 调用_on_tuple_content_change传递更新
        self._on_tuple_content_change(self.editor_value)

    def _move_item_down(self):
        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        if not is_variable_length:
            return  # Fixed length tuple, cannot reorder items

        if not self.current_selected_item:
            return

        # 获取选中项的索引
        index: int = self.list_treeview.index(self.current_selected_item)
        items: Tuple[str, ...] = self.list_treeview.get_children()

        # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
        if index == len(items) - 1:
            return

        assert 0 <= index < len(self.editor_value) - 1, f"Index {index} out of range"

        # 获取选中项的后一个项目
        next_item: str = self.list_treeview.next(self.current_selected_item)
        if not next_item:
            return  # 已经是最后一个项目

        # 交换元组中的项目
        list_value = list(self.editor_value)
        list_value[index], list_value[index + 1] = list_value[index + 1], list_value[index]
        self.editor_value = tuple(list_value)

        # 更新交换项目的内容
        if len(self.list_treeview.item(items[index], "values")) == 2:
            # [Index, Value]
            self.list_treeview.item(items[index], values=(index, repr(self.editor_value[index])))
            self.list_treeview.item(items[index + 1], values=(index + 1, repr(self.editor_value[index + 1])))
        else:
            # [Index, Type, Value]
            type_args = get_args(self.attribute_type)
            type_name_current = "Any"
            type_name_next = "Any"
            if index < len(self.item_maintainer_cls_list):
                if index < len(type_args):
                    type_name_current = self.item_maintainer_cls_list[index].get_simplest_type_name_static(
                        target_type=type_args[index])
            if index + 1 < len(self.item_maintainer_cls_list):
                if index + 1 < len(type_args):
                    type_name_next = self.item_maintainer_cls_list[index + 1].get_simplest_type_name_static(
                        target_type=type_args[index + 1]
                    )
            self.list_treeview.item(items[index], values=(index, type_name_current, repr(self.editor_value[index])))
            self.list_treeview.item(
                items[index + 1],
                values=(index + 1, type_name_next, repr(self.editor_value[index + 1]))
            )

        # 选择移动后的项目
        # 找到新位置的项目（索引为index+1）
        new_selected: str = items[index + 1]
        self.list_treeview.selection_set(new_selected)
        self.current_selected_item = new_selected
        # 确保选中项在视野范围内
        self.list_treeview.see(new_selected)

        # 更新按钮状态
        self._on_treeview_select()

        # 保持焦点在Treeview上
        self.list_treeview.focus_set()
        self.list_treeview.focus(new_selected)

        # 调用_on_tuple_content_change传递更新
        self._on_tuple_content_change(self.editor_value)

    # 绑定Remove按钮点击事件
    def _on_remove_button_click(self):
        self._remove_item(select_prev=False)
        # 将焦点设置回Treeview，确保按键依然能被接收
        self.list_treeview.focus_set()
        # 如果有选中项，确保选中项被高亮
        if self.list_treeview.selection():
            self.list_treeview.selection_set(self.list_treeview.selection())

    # 绑定Delete键
    def _on_delete_key(self, event: Optional[tk.Event] = None):
        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        if not is_variable_length:
            return "break"  # Fixed length tuple, cannot remove items

        selected_items = self.list_treeview.selection()
        if selected_items:
            item_values = self.list_treeview.item(selected_items[0], "values")
            if item_values and item_values[0] != TupleMaintainer.NEW_ITEM_INDICATOR:
                self._remove_item(select_prev=False)
        return "break"  # 阻止事件继续传播

    # 绑定Backspace键
    def _on_backspace_key(self, event: Optional[tk.Event] = None):
        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        if not is_variable_length:
            return "break"  # Fixed length tuple, cannot remove items

        selected_items = self.list_treeview.selection()
        if selected_items:
            item_values = self.list_treeview.item(selected_items[0], "values")
            if item_values and item_values[0] != TupleMaintainer.NEW_ITEM_INDICATOR:
                self._remove_item(select_prev=True)
        return "break"  # 阻止事件继续传播

    # 绑定双击事件，双击项目打开编辑界面
    def _on_tree_double_click(self, event: Optional[tk.Event] = None):
        # 获取点击的项
        item: str = self.list_treeview.identify_row(event.y)
        # 只有双击到具体项时才执行编辑操作，双击标题不触发
        if item:
            # 模拟点击Edit按钮
            self._edit_item()

    # 共用的编辑窗口创建方法
    def _create_popup_inspector_window(
            self,
            title: str,
            item_attribute_name: str,
            item_attribute_type: Type,
            item_attribute_value: Any
    ):
        # 创建编辑窗口
        self.popup_top_level = tk.Toplevel(self.editor)
        self.popup_top_level.title(title)
        self.popup_top_level.resizable(True, True)
        # 设置焦点到子窗口
        self.popup_top_level.focus_set()

        # 添加inspector LabelFrame
        self.popup_inspector_frame = ttk.Frame(self.popup_top_level)

        self.popup_inspector_frame.pack(fill=tk.BOTH, expand=True)

        # 配置inspector_frame的grid布局
        self.popup_inspector_frame.grid_rowconfigure(0, weight=1)
        self.popup_inspector_frame.grid_rowconfigure(1, weight=0)  # 横向滚动条固定高度
        self.popup_inspector_frame.grid_columnconfigure(0, weight=1)
        self.popup_inspector_frame.grid_columnconfigure(1, weight=0)  # 纵向滚动条固定宽度

        # 创建带滚动条的容器
        self.popup_canvas = tk.Canvas(self.popup_inspector_frame, highlightthickness=0)
        self.popup_canvas_vscrollbar = ttk.Scrollbar(
            self.popup_inspector_frame,
            orient="vertical",
            command=self.popup_canvas.yview
        )
        self.popup_canvas_hscrollbar = ttk.Scrollbar(
            self.popup_inspector_frame,
            orient="horizontal",
            command=self.popup_canvas.xview
        )
        self.popup_canvas.configure(
            yscrollcommand=self.popup_canvas_vscrollbar.set,
            xscrollcommand=self.popup_canvas_hscrollbar.set
        )

        # 布局 - 使用grid布局
        self.popup_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.popup_canvas_vscrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.popup_canvas_hscrollbar.grid(row=1, column=0, sticky=tk.EW)

        # 创建内部框架，用于容纳所有Inspector内容
        self.popup_canvas_inner_frame = ttk.Frame(self.popup_canvas, padding=5)
        # 将内部框架添加到canvas
        self.popup_canvas_inner_frame_id = self.popup_canvas.create_window(
            (0, 0), window=self.popup_canvas_inner_frame, anchor=tk.NW
        )

        # 绑定事件，当内部框架大小改变时更新canvas的滚动区域
        self.popup_canvas_inner_frame.bind("<Configure>", self._on_popup_canvas_inner_frame_configure)

        # 绑定事件，当canvas大小改变时调整内部框架的宽度
        self.popup_canvas.bind("<Configure>", self._on_popup_canvas_configure)

        # 结果变量
        self.popup_wnd_result = {'value': item_attribute_value, 'confirmed': False}

        # 定义值变化的回调函数
        def on_popup_editor_value_change(new_val: Any):
            self.popup_wnd_result['value'] = new_val

        # 使用 create_inspector 创建编辑控件，popup_canvas_inner_frame 作为父控件
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        item_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(item_attribute_type)
        self.item_maintainer = item_maintainer_cls(
            item_attribute_name,
            item_attribute_type,
            item_attribute_value,
            self.logger
        )
        self.item_maintainer.config_view("Standalone")
        wnd_size = self.item_maintainer.default_standalone_window_size()
        self.popup_top_level.geometry(f"{wnd_size[0]}x{wnd_size[1]}")
        self.popup_inspector_content = self.item_maintainer.create_inspector(
            self.popup_canvas_inner_frame,
            on_popup_editor_value_change
        )
        self.popup_inspector_content.pack(fill=tk.BOTH, expand=True)

        # 创建一个容器来居中按钮
        self.popup_wnd_button_container = ttk.Frame(self.popup_top_level)
        self.popup_wnd_button_container.pack(side=tk.TOP, anchor=tk.CENTER)

        # 添加Confirm按钮（放在前面）
        self.popup_confirm_button = ttk.Button(
            self.popup_wnd_button_container,
            text="Confirm",
            command=self._on_popup_confirm
        )
        self.popup_confirm_button.pack(side=tk.LEFT, padx=10, pady=(0, 10))

        # 添加Cancel按钮
        self.popup_cancel_button = ttk.Button(
            self.popup_wnd_button_container,
            text="Cancel",
            command=self._on_popup_cancel
        )
        self.popup_cancel_button.pack(side=tk.LEFT, padx=10, pady=(0, 10))

        # 为窗口绑定按键事件
        hotkey_confirm, hotkey_cancel = self.item_maintainer.shall_hotkey_confirm_cancel()
        if hotkey_confirm:
            self.popup_top_level.bind("<Return>", self._on_popup_return_key)
        if hotkey_cancel:
            self.popup_top_level.bind("<Escape>", self._on_popup_escape_key)

        # 阻塞主窗口
        self.popup_top_level.transient(self.editor.winfo_toplevel())
        self.popup_top_level.grab_set()
        self.editor.winfo_toplevel().wait_window(self.popup_top_level)
        self.editor.winfo_toplevel().grab_set()

        return self.popup_wnd_result

    def _on_popup_canvas_inner_frame_configure(self, event: Optional[tk.Event] = None) -> None:
        self.popup_canvas.configure(scrollregion=self.popup_canvas.bbox("all"))

    def _on_popup_canvas_configure(self, event: Optional[tk.Event] = None) -> None:
        """当canvas大小改变时调整内部框架的宽度和高度"""
        # 获取canvas的宽度和高度，留出滚动条的空间
        canvas_width = event.width - 5  # 留出滚动条的宽度空间
        canvas_height = event.height - 5  # 留出滚动条的高度空间

        # 获取内部框架的最小宽度和高度
        min_width = self.popup_canvas_inner_frame.winfo_reqwidth()
        min_height = self.popup_canvas_inner_frame.winfo_reqheight()

        # 计算内部框架的实际宽度和高度
        # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
        # 否则使用内部控件的最小宽度和高度
        actual_width = max(canvas_width, min_width)
        actual_height = max(canvas_height, min_height)
        self.popup_canvas.itemconfig(self.popup_canvas_inner_frame_id, width=event.width)

        # 设置内部框架的宽度和高度
        self.popup_canvas.itemconfig(self.popup_canvas_inner_frame_id, width=actual_width, height=actual_height)

        # 更新canvas的滚动区域
        self.popup_canvas.configure(scrollregion=self.popup_canvas.bbox("all"))

    # 确认按钮回调
    def _on_popup_confirm(self):
        self.popup_wnd_result['confirmed'] = True
        self.popup_top_level.destroy()

    # 取消按钮回调
    def _on_popup_cancel(self):
        self.popup_wnd_result['confirmed'] = False
        self.popup_top_level.destroy()

    # 绑定弹出编辑窗口的回车键和Esc键
    def _on_popup_return_key(self, event: Optional[tk.Event] = None):
        self._on_popup_confirm()
        return "break"  # 阻止事件继续传播

    def _on_popup_escape_key(self, event: Optional[tk.Event] = None):
        self._on_popup_cancel()
        return "break"  # 阻止事件继续传播

    # 实现编辑元组元素功能
    def _edit_item(self):
        if self.current_selected_item is None:
            return

        # 获取选中项的索引
        index: int = self.list_treeview.index(self.current_selected_item)
        items: Tuple[str, ...] = self.list_treeview.get_children()

        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
        if is_variable_length and index == len(items) - 1:
            # 当焦点在<New Item>上时，调用add_item函数，等同于添加新元素
            self._add_item()
            return

        if self.view_mode != "Packed":
            return

        assert 0 <= index < len(self.editor_value), f"Index {index} out of range"

        # 获取选中项的当前值
        current_value: Any = self.editor_value[index]

        # 构建属性名称：{元组属性名称}[{此元素的下标序号}]
        item_attribute_name: str = f"{self.attribute_name}[{index}]"

        # 确定元素类型
        if is_variable_length:
            # Tuple[E, ...]
            item_attribute_type = type_args[0]
        else:
            # Tuple[E1, E2, ...]
            if index < len(type_args):
                item_attribute_type = type_args[index]
            else:
                item_attribute_type = Any

        # 调用共用方法创建编辑窗口
        # self.popup_wnd_result will store status
        self._create_popup_inspector_window(
            f"Edit {self.attribute_name}[{index}]",
            item_attribute_name,
            item_attribute_type,
            current_value
        )

        # 如果用户确认，更新值
        if self.popup_wnd_result['confirmed']:
            list_value = list(self.editor_value)
            list_value[index] = self.popup_wnd_result['value']
            self.editor_value = tuple(list_value)

            # 更新Treeview中的显示
            if is_variable_length:
                self.list_treeview.item(items[index], values=(index, repr(self.editor_value[index])))
            else:
                # For fixed length tuple, show type information
                type_name = "Any"
                if index < len(self.item_maintainer_cls_list):
                    if index < len(type_args):
                        type_name = self.item_maintainer_cls_list[index].get_simplest_type_name_static(
                            target_type=type_args[index])
                self.list_treeview.item(items[index], values=(index, type_name, repr(self.editor_value[index])))

            # 确保选中项在视野范围内
            self.list_treeview.see(self.current_selected_item)
            # 设置焦点回到Treeview并选中当前编辑的项
            self.list_treeview.focus_set()
            self.list_treeview.selection_set(self.current_selected_item)

            # 调用_on_tuple_content_change传递更新
            self._on_tuple_content_change(self.editor_value)

    # 实现添加元组元素功能
    def _add_item(self):
        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        if not is_variable_length:
            return  # Fixed length tuple, cannot add items

        if not self.current_selected_item:
            return

        # 获取选中项的索引
        index: int = self.list_treeview.index(self.current_selected_item)
        items: Tuple[str, ...] = self.list_treeview.get_children()

        assert 0 <= index <= len(self.editor_value), f"Index {index} out of range"

        # 获取默认值
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        item_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(type_args[0])
        default_value = item_maintainer_cls.get_default_value_static(target_type=type_args[0])

        # 构建属性名称：{元组属性名称}[{此元素的下标序号}]
        item_attribute_name = f"{self.attribute_name}[{index}]"

        # 调用共用方法创建编辑窗口
        self._create_popup_inspector_window(
            f"Edit New {self.attribute_name}[{index}]",
            item_attribute_name,
            type_args[0],
            default_value
        )

        # 如果用户确认，添加新值
        if self.popup_wnd_result['confirmed']:
            # 在指定位置插入新元素
            list_value = list(self.editor_value)
            list_value.insert(index, self.popup_wnd_result['value'])
            self.editor_value = tuple(list_value)

            # 从index开始重新填充Treeview
            for idx in range(index, len(self.editor_value)):
                self.list_treeview.item(items[idx], values=(idx, repr(self.editor_value[idx])))

            # 添加空白行
            self.list_treeview.insert("", tk.END, values=(TupleMaintainer.NEW_ITEM_INDICATOR, ""))

            # 选择新添加的项目
            items = self.list_treeview.get_children()
            assert index < len(items) - 1, f"Index {index} out of range"

            self.current_selected_item = items[index]
            # 确保新添加的项目在视野范围内
            self.list_treeview.see(self.current_selected_item)
            # 设置焦点回到Treeview并选中新插入的项
            self.list_treeview.focus_set()
            self.list_treeview.selection_set(self.current_selected_item)

            # 调用_on_tuple_content_change传递更新
            self._on_tuple_content_change(self.editor_value)

    def _update_item_inspector(self) -> None:
        """更新弹出编辑窗口内部的Inspector面板，显示属性信息和编辑控件"""
        if self.view_mode != "Standalone":
            return

        # 清空内部框架
        for widget in self.item_inspector_inner_frame.winfo_children():
            widget.destroy()

        if self.current_selected_item is None:
            return

        # 获取选中项的索引
        index: int = self.list_treeview.index(self.current_selected_item)
        items: Tuple[str, ...] = self.list_treeview.get_children()

        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        # 检查选中的是否是最后一行（NEW_ITEM_INDICATOR）
        if is_variable_length and index == len(items) - 1:
            return

        assert 0 <= index < len(self.editor_value), f"Index {index} out of range"

        # 获取选中项的当前值
        current_value: Any = self.editor_value[index]

        # 构建属性名称：{元组属性名称}[{此元素的下标序号}]
        item_attribute_name: str = f"{self.attribute_name}[{index}]"

        # 确定元素类型
        if is_variable_length:
            # Tuple[E, ...]
            item_attribute_type = type_args[0]
        else:
            # Tuple[E1, E2, ...]
            if index < len(type_args):
                item_attribute_type = type_args[index]
            else:
                item_attribute_type = Any

        # 使用 create_inspector 创建编辑控件，popup_canvas_inner_frame 作为父控件
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        item_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(item_attribute_type)
        self.item_maintainer = item_maintainer_cls(
            item_attribute_name,
            item_attribute_type,
            current_value,
            self.logger
        )
        self.item_maintainer.config_view("Packed")
        self.item_inspector = self.item_maintainer.create_inspector(
            self.item_inspector_inner_frame,
            self._on_item_value_change
        )
        self.item_inspector.pack(fill=tk.BOTH, expand=True)

        # 更新内部框架的宽高规格
        # 获取canvas的宽度和高度，留出滚动条的空间
        canvas_width: int = self.item_inspector_canvas.winfo_width() - 5  # 留出滚动条的宽度空间
        canvas_height: int = self.item_inspector_canvas.winfo_height() - 5  # 留出滚动条的高度空间

        # 强制更新布局，确保获取准确的所需规格
        self.item_inspector_inner_frame.update_idletasks()

        # 获取内部框架的最小宽度和高度
        min_width: int = self.item_inspector_inner_frame.winfo_reqwidth()
        min_height: int = self.item_inspector_inner_frame.winfo_reqheight()

        # 计算内部框架的实际宽度和高度
        # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
        # 否则使用内部控件的最小宽度和高度
        actual_width: int = max(canvas_width, min_width)
        actual_height: int = max(canvas_height, min_height)

        # 设置内部框架的宽度和高度
        self.item_inspector_canvas.itemconfig(
            self.item_inspector_inner_frame_id,
            width=actual_width,
            height=actual_height
        )

        # 更新canvas的滚动区域
        self.item_inspector_canvas.configure(scrollregion=self.item_inspector_canvas.bbox("all"))

    def _on_item_value_change(self, new_value: Any) -> None:
        """处理元素值变化"""
        assert self.item_maintainer is not None
        assert self.current_selected_item is not None

        # Detect currently selected item
        # 获取选中项的索引
        index: int = self.list_treeview.index(self.current_selected_item)
        items: Tuple[str, ...] = self.list_treeview.get_children()

        list_value = list(self.editor_value)
        list_value[index] = new_value
        self.editor_value = tuple(list_value)
        self.item_maintainer.confirm_editor_change()

        type_args = get_args(self.attribute_type)
        is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis

        # 更新Treeview中的显示
        if is_variable_length:
            self.list_treeview.item(items[index], values=(index, repr(self.editor_value[index])))
        else:
            # For fixed length tuple, show type information
            type_name = "Any"
            if index < len(self.item_maintainer_cls_list):
                if index < len(type_args):
                    type_name = self.item_maintainer_cls_list[index].get_simplest_type_name_static(
                        target_type=type_args[index])
            self.list_treeview.item(items[index], values=(index, type_name, repr(self.editor_value[index])))

        # 确保选中项在视野范围内
        self.list_treeview.see(self.current_selected_item)

        # 调用_on_tuple_content_change传递更新
        self._on_tuple_content_change(self.editor_value)

        # 记录日志
        self.log_message(f"Updated attribute '{self.attribute_name}[{index}]' to {repr(new_value)}")

    @override
    def editor_enable(self):
        if self.editor is not None:
            # 清空Treeview
            for item in self.list_treeview.get_children():
                self.list_treeview.delete(item)
            type_args = get_args(self.attribute_type)
            is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis
            # 重新填充Treeview
            if self.attribute_value:
                for i, item in enumerate(self.attribute_value):
                    if is_variable_length:
                        self.list_treeview.insert("", tk.END, values=(i, repr(item)))
                    else:
                        # For fixed length tuple, show type information
                        type_name = "Any"
                        if i < len(self.item_maintainer_cls_list):
                            if i < len(type_args):
                                type_name = self.item_maintainer_cls_list[i].get_simplest_type_name_static(
                                    target_type=type_args[i])
                        self.list_treeview.insert("", tk.END, values=(i, type_name, repr(item)))
            # 添加空白行（仅对可变长度元组）
            if is_variable_length:
                self.list_treeview.insert("", tk.END, values=(TupleMaintainer.NEW_ITEM_INDICATOR, ""))
            for widget in self.buttons_frame.winfo_children():
                if isinstance(widget, ttk.Button):
                    # For fixed length tuple, only enable edit button
                    if not is_variable_length and widget != self.edit_button:
                        widget.config(state='disabled')
                    else:
                        widget.config(state='normal')
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None:
            # 清空Treeview
            for item in self.list_treeview.get_children():
                self.list_treeview.delete(item)
            for widget in self.buttons_frame.winfo_children():
                if isinstance(widget, ttk.Button):
                    widget.config(state='disabled')
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            # 清空Treeview
            for item in self.list_treeview.get_children():
                self.list_treeview.delete(item)
            type_args = get_args(self.attribute_type)
            is_variable_length = len(type_args) == 2 and type_args[1] == Ellipsis
            # 重新填充Treeview
            if new_value:
                for i, item in enumerate(new_value):
                    if is_variable_length:
                        self.list_treeview.insert("", tk.END, values=(i, repr(item)))
                    else:
                        # For fixed length tuple, show type information
                        type_name = "Any"
                        if i < len(self.item_maintainer_cls_list):
                            if i < len(type_args):
                                type_name = self.item_maintainer_cls_list[i].get_simplest_type_name_static(
                                    target_type=type_args[i])
                        self.list_treeview.insert("", tk.END, values=(i, type_name, repr(item)))
            # 添加空白行（仅对可变长度元组）
            if is_variable_length:
                self.list_treeview.insert("", tk.END, values=(TupleMaintainer.NEW_ITEM_INDICATOR, ""))
        super().editor_set_value(new_value)

    @override
    def config_view(self, view_mode: Literal["Standalone", "Packed"], *args, **kwargs):
        self.view_mode = view_mode

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        if isinstance(input_value, tuple):
            return True, input_value
        return False, None

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        sim_type: Type = simplify_type(target_type)
        # Assuming attribute_type is simplified
        origin: Type = get_origin(sim_type)
        return origin in {tuple, Tuple}

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = Tuple[Any]) -> bool:
        sim_type: Type = simplify_type(target_type)
        if not TupleMaintainer.is_type_compatible_static(sim_type):
            return False

        # Value is not a tuple, never compatible
        if not isinstance(value, tuple):
            return False

        # Empty tuple () is always compatible
        if len(value) == 0:
            return True

        type_args: Tuple[Any, ...] = get_args(sim_type)

        # tuple[()], Tuple[()]
        if len(type_args) == 0 or len(type_args) == 1 and type_args[0] == tuple():
            return False

        # Tuple[E, Ellipsis(...)]
        if len(type_args) == 2 and type_args[1] == Ellipsis:
            from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
            item_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(type_args[0])

            # Unsupported element type, never compatible
            if issubclass(item_maintainer_cls, UnsupportedMaintainer):
                return False

            # Check compatibility for each item
            for item in value:
                if not item_maintainer_cls.is_value_compatible_static(item, type_args[0]):
                    return False
            return True

        # Tuple[E1, E2, ...]
        # Check if value length matches type length
        if len(value) != len(type_args):
            return False

        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        # Check compatibility for each item
        for i, item in enumerate(value):
            maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(type_args[i])
            if issubclass(maintainer_cls, UnsupportedMaintainer):
                return False
            if not maintainer_cls.is_value_compatible_static(item, type_args[i]):
                return False
        return True

    @staticmethod
    @override
    def get_default_value_static(target_type: Type, *args, **kwargs) -> Any:
        sim_type: Type = simplify_type(target_type)
        if not TupleMaintainer.is_type_compatible_static(sim_type):
            return None
        return tuple()

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return simplify_type(target_type)

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, *args, **kwargs) -> str:
        sim_type: Type = simplify_type(target_type)
        if not TupleMaintainer.is_type_compatible_static(sim_type):
            return ""

        type_args: Tuple[Any, ...] = get_args(sim_type)

        # tuple[()], Tuple[()]
        if len(type_args) == 0 or len(type_args) == 1 and type_args[0] == tuple():
            return "Tuple[()]"

        # Tuple[E, ...]
        if len(type_args) == 2 and type_args[1] == Ellipsis:
            from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
            item_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(type_args[0])
            subtype_str: str = item_maintainer_cls.get_simplest_type_name_static(
                target_type=type_args[0],
                *args, **kwargs
            )
            return f"Tuple[{subtype_str}, ...]"

        # Tuple[E1, E2, ...]
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        subtype_strs = []
        for arg_type in type_args:
            item_maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(arg_type)
            subtype_str = item_maintainer_cls.get_simplest_type_name_static(
                target_type=arg_type,
                *args, **kwargs
            )
            subtype_strs.append(subtype_str)
        return f"Tuple[{', '.join(subtype_strs)}]"
