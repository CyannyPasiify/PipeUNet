import os
import sys
import datetime
import traceback
import dataclasses
import importlib.util
from typing import Dict, List, Optional, Any, Type, Tuple, cast
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, font
from dataclasses import fields, Field
import argparse

from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.unsupported_maintainer import UnsupportedMaintainer
from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer
# 导入类型维护器
from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory

# 从foo.py中导入相关类
spec = importlib.util.spec_from_file_location("foo", "d:\\CBIB\\Storages\\DevelopmentSoftwares\\Trae\\PipeUNet\\foo.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
from foo import ParserABC


class YAMLConfigCLI:
    # Singleton instance
    _instance = None

    def __new__(cls, root: Optional[tk.Tk] = None, debug_level: int = 3) -> 'YAMLConfigCLI':
        """Create and return the singleton instance"""
        if cls._instance is None:
            cls._instance = super(YAMLConfigCLI, cls).__new__(cls)
        return cls._instance

    def __init__(self, root: Optional[tk.Tk] = None, debug_level: int = 3):
        """Initialize the YAML Config CLI"""
        # Only initialize once
        if hasattr(self, 'initialized'): return
        if root is not None:
            self.root: Optional[tk.Tk] = root
            self.root.title("YAML Config CLI")
            self.root.geometry("1366x768")
        else:
            self.root: Optional[tk.Tk] = tk.Tk()

        # 当前Parser实例
        self.current_parser: Optional[Any] = None
        self.current_parser_fields: Optional[Dict[str, Field[Any]]] = None

        # 日志级别
        self.debug_level: int = debug_level

        # Mark as initialized
        self.initialized: bool = True

        # Parser类字典
        self.parser_classes_dict: Dict[str, Optional[Type]] = {}

        # 配置主窗口的grid布局
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # 主框架
        self.main_frame: ttk.Frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # 配置main_frame使用PanedWindow实现上下可拖动调整
        self.main_paned: ttk.PanedWindow = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # 顶部内容区域 - 左右分栏，使用PanedWindow实现可拖动调整
        self.top_frame: ttk.PanedWindow = ttk.PanedWindow(self.main_paned, orient=tk.HORIZONTAL)
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        # 初始化分割位置50%
        def on_pane_map(event: Optional[tk.Event] = None):
            # 控件真正渲染完成了 → 此时宽度 100% 准确
            w = self.top_frame.winfo_width()
            self.top_frame.sashpos(0, w // 2)

            # 只执行一次，执行完解绑，避免多次触发
            self.top_frame.unbind("<Map>")

        self.top_frame.bind("<Map>", on_pane_map)

        # 左侧区域
        self.left_frame: ttk.Frame = ttk.Frame(self.top_frame)
        self.top_frame.add(self.left_frame, weight=1)

        # 右侧区域
        self.right_frame: ttk.Frame = ttk.Frame(self.top_frame)
        self.top_frame.add(self.right_frame, weight=1)

        # 左侧区域内容
        # 第一行：Parser下拉选项
        self.parser_frame: ttk.Frame = ttk.Frame(self.left_frame)
        self.parser_frame.pack(fill=tk.X, pady=(0, 5))

        # 配置grid布局，确保整行拉伸
        self.parser_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.parser_frame, text="Parser:", width=10).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.parser_var: tk.StringVar = tk.StringVar()
        self.parser_combobox: ttk.Combobox = ttk.Combobox(
            self.parser_frame,
            textvariable=self.parser_var,
            state="readonly"
        )
        self.parser_combobox.grid(row=0, column=1, sticky=tk.W + tk.E, pady=5, padx=5)

        self.export_example_btn: ttk.Button = ttk.Button(
            self.parser_frame,
            text="Export Example YAML",
            state=tk.DISABLED,
            command=self.export_example_yaml, width=22
        )
        self.export_example_btn.grid(row=0, column=2, pady=5, padx=5)

        # 第二行：Config File
        self.config_frame: ttk.Frame = ttk.Frame(self.left_frame)
        self.config_frame.pack(fill=tk.X, pady=(0, 5))

        # 配置grid布局，确保整行拉伸
        self.config_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.config_frame, text="Config File:", width=10).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.config_var: tk.StringVar = tk.StringVar()
        self.config_entry: ttk.Entry = ttk.Entry(self.config_frame, textvariable=self.config_var)
        self.config_entry.grid(row=0, column=1, sticky=tk.W + tk.E, pady=5, padx=5)

        self.browse_btn: ttk.Button = ttk.Button(
            self.config_frame,
            text="Browse",
            command=self.browse_config_file,
            width=22
        )
        self.browse_btn.grid(row=0, column=2, pady=5, padx=5)

        # 第三行：Load Config和Save Config按钮
        self.load_frame: ttk.Frame = ttk.Frame(self.left_frame)
        self.load_frame.pack(fill=tk.X, pady=(0, 10))

        # 配置grid布局
        self.load_frame.grid_columnconfigure(0, weight=1)
        self.load_frame.grid_columnconfigure(1, weight=1)

        self.load_config_btn: ttk.Button = ttk.Button(
            self.load_frame,
            text="Load Config",
            state=tk.DISABLED,
            command=self.load_config
        )
        self.load_config_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W + tk.E)

        self.save_config_btn: ttk.Button = ttk.Button(
            self.load_frame,
            text="Save Config",
            state=tk.DISABLED,
            command=self.save_config
        )
        self.save_config_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W + tk.E)

        # Attributes面板 - 占据剩余所有空间
        self.attributes_frame: ttk.LabelFrame = ttk.LabelFrame(self.left_frame, text="Attributes")
        self.attributes_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

        # 创建可交互的表格
        self.table_frame: ttk.Frame = ttk.Frame(self.attributes_frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 配置table_frame的grid布局
        self.table_frame.grid_rowconfigure(0, weight=1)
        self.table_frame.grid_columnconfigure(0, weight=1)

        # 创建Treeview组件作为可交互表格
        self.tree: ttk.Treeview = ttk.Treeview(
            self.table_frame,
            columns=("attribute", "type", "value"),
            show="headings"
        )
        # Add tag for unsupported types (bold red text)
        italic_font: font.Font = font.nametofont("TkDefaultFont").copy()
        italic_font.configure(slant="italic")
        self.tree.tag_configure('unsupported', font=italic_font, foreground='red')
        self.tree.grid(row=0, column=0, sticky=tk.NSEW)

        # 设置列标题
        self.tree.heading("attribute", text="Attribute")
        self.tree.heading("type", text="Type")
        self.tree.heading("value", text="Value")

        # 设置列宽（允许用户拖动调整）
        self.tree.column("attribute", minwidth=70, width=150, stretch=False, anchor=tk.W)
        self.tree.column("type", minwidth=70, width=180, stretch=False, anchor=tk.W)
        self.tree.column("value", minwidth=70, width=310, stretch=False, anchor=tk.W)

        # 添加垂直滚动条
        vsb: ttk.Scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        self.tree.configure(yscrollcommand=vsb.set)

        # 添加水平滚动条
        # 创建一个自定义的水平滚动条命令，增加步长
        def custom_xview(*args: Any) -> None:
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

        hsb: ttk.Scrollbar = ttk.Scrollbar(self.table_frame, orient="horizontal", command=custom_xview)
        hsb.grid(row=1, column=0, sticky=tk.EW)
        self.tree.configure(xscrollcommand=hsb.set)

        # 绑定鼠标滚轮事件，增加水平滚动条的灵敏度
        def on_mouse_wheel(event: Optional[tk.Event] = None) -> None:
            # 检测Shift键是否被按下（使用更可靠的方式）
            shift_pressed = event.state & 0x0001 != 0

            if not shift_pressed:  # No modifier key
                # Vertical scroll (default behavior)
                self.tree.yview_scroll(-1 * (event.delta // 120), "units")
            else:  # Shift key pressed
                # 尝试使用更大的滚动量
                self.tree.xview_scroll(-1 * (event.delta // 10), "units")

        self.tree.bind("<MouseWheel>", on_mouse_wheel)

        # 设置样式，使背景为白色
        style: ttk.Style = ttk.Style()
        style.configure("Treeview", background="white", fieldbackground="white")
        style.configure("Treeview.Heading", background="#f0f0f0", font=("Arial", 10, "bold"))
        style.map("Treeview",
                  background=[("selected", "#347083"), ("!selected", "white")],
                  foreground=[("selected", "white")])

        # 右侧Inspector面板 - 占据所有空间
        self.inspector_frame: ttk.LabelFrame = ttk.LabelFrame(self.right_frame, text="Inspector")
        self.inspector_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self.inspector_frame.grid_rowconfigure(0, weight=1)
        self.inspector_frame.grid_rowconfigure(1, weight=0)  # 横向滚动条固定高度
        self.inspector_frame.grid_columnconfigure(0, weight=1)
        self.inspector_frame.grid_columnconfigure(1, weight=0)  # 纵向滚动条固定宽度

        # 创建带滚动条的容器
        self.inspector_canvas: tk.Canvas = tk.Canvas(self.inspector_frame, highlightthickness=0)
        self.inspector_vscrollbar: ttk.Scrollbar = ttk.Scrollbar(
            self.inspector_frame,
            orient="vertical",
            command=self.inspector_canvas.yview
        )
        self.inspector_hscrollbar: ttk.Scrollbar = ttk.Scrollbar(
            self.inspector_frame,
            orient="horizontal",
            command=self.inspector_canvas.xview
        )
        self.inspector_canvas.configure(
            yscrollcommand=self.inspector_vscrollbar.set,
            xscrollcommand=self.inspector_hscrollbar.set
        )

        # 布局 - 使用grid布局
        self.inspector_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.inspector_vscrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.inspector_hscrollbar.grid(row=1, column=0, sticky=tk.EW)

        # 创建内部框架，用于容纳所有Inspector内容
        self.inspector_inner_frame: ttk.Frame = ttk.Frame(self.inspector_canvas, padding=10)
        # 将内部框架添加到canvas
        self.inspector_inner_frame_id: int = self.inspector_canvas.create_window(
            (0, 0),
            window=self.inspector_inner_frame,
            anchor=tk.NW
        )

        # 绑定事件，当内部框架大小改变时更新canvas的滚动区域
        self.inspector_inner_frame.bind("<Configure>", self._on_inspector_inner_configure)
        # 绑定事件，当canvas大小改变时调整内部框架的宽度
        self.inspector_canvas.bind("<Configure>", self._on_inspector_canvas_configure)

        # 初始为空
        self.current_attribute: Optional[str] = None
        self.current_maintainer: Optional[BaseMaintainer] = None

        # 日志区域 - 占据全宽
        self.log_frame: ttk.LabelFrame = ttk.LabelFrame(self.main_paned, text="Log")
        self.log_frame.grid_rowconfigure(0, weight=1)

        self.log_text: scrolledtext.ScrolledText = scrolledtext.ScrolledText(
            self.log_frame,
            wrap=tk.WORD, bg="black", fg="white", font=('Consolas', 10),
            height=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        # 配置日志文本的标签和颜色
        self.log_text.tag_configure("error", foreground="red", font=('Consolas', 10, 'bold'))
        self.log_text.tag_configure("warn", foreground="yellow")
        self.log_text.tag_configure("info", foreground="white")
        self.log_text.tag_configure("debug", foreground="cyan")
        self.log_text.tag_configure("trace", foreground="grey")
        self.log_text.tag_configure("special", foreground="pink")

        # 添加顶部区域和日志区域到主PanedWindow
        self.main_paned.add(self.top_frame, weight=7)
        self.main_paned.add(self.log_frame, weight=3)

        # 绑定事件
        self.parser_var.trace_add("write", self._on_parser_change)
        self.config_var.trace_add("write", self.on_config_change)

        # 绑定TreeView选择事件
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # 初始化Parser选项
        self.init_parser_options()

    def init_attributes_table(self) -> None:
        """初始化Attributes表格"""
        # 清除Treeview中的所有项
        for item in self.tree.get_children():
            self.tree.delete(item)

    def init_parser_options(self) -> None:
        """初始化Parser下拉选项"""
        # 获取所有继承ParserABC的类
        parser_classes: List[Tuple[str, Type]] = []
        for name, obj in foo.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, ParserABC) and obj != ParserABC:
                full_path = f"foo.{obj.__name__}"
                display_text = f"{obj.__name__} ({full_path})"
                parser_classes.append((display_text, obj))

        # 添加空白选项
        self.parser_classes_dict = {"": None}
        for text, cls in parser_classes:
            self.parser_classes_dict[text] = cls

        self.parser_combobox['values'] = list(self.parser_classes_dict.keys())
        # 初始保持选中为空白
        self.parser_var.set("")

    def _on_parser_change(self, *args: Any) -> None:
        """当Parser选择发生变化时"""
        selected_parser: str = self.parser_var.get()

        # 清除之前的Parser实例
        self.current_parser = None

        # 清空Attributes内容显示
        self.init_attributes_table()

        # 清空Inspector面板
        self.clear_inspector()

        if selected_parser:
            parser_cls: Optional[Type] = self.parser_classes_dict[selected_parser]
            if parser_cls:
                # 实例化新的Parser对象
                try:
                    self.current_parser = parser_cls()
                    parser_fields: Tuple[Field[Any], ...] = fields(self.current_parser)
                    self.current_parser_fields = {fid.name: fid for fid in parser_fields}
                    # 在Attributes面板表格中显示此Parser的成员属性信息
                    self.display_parser_attributes()
                except Exception as e:
                    error_msg = f"Error creating parser instance: {str(e)}\n"

                    # 根据日志级别添加不同详细程度的信息
                    if self.debug_level >= 2:  # WARN级别及以上
                        error_msg += f"Exception type: {type(e).__name__}\n"

                    if self.debug_level >= 4:  # DEBUG级别及以上
                        error_msg += f"Stack trace:\n{traceback.format_exc()}"

                    if self.debug_level >= 5:  # TRACE级别
                        error_msg += f"Python version: {sys.version}\n"
                        error_msg += f"Parser class: {parser_cls.__name__}\n"

                    self.log_message(error_msg, level="error")

            # 检查是否实现了dump_example_to_yaml方法
            if parser_cls and hasattr(parser_cls, 'dump_example_to_yaml'):
                self.export_example_btn.config(state=tk.NORMAL)
            else:
                self.export_example_btn.config(state=tk.DISABLED)
            # 检查Load Config按钮状态
            self.check_load_config_button()
        else:
            # 选中空白选项时，禁用所有相关按钮
            self.export_example_btn.config(state=tk.DISABLED)
            self.check_load_config_button()

    def display_parser_attributes(self) -> None:
        """在Attributes面板表格中显示Parser的成员属性信息"""
        if not self.current_parser:
            return

        # 清空Treeview中的所有项
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            # 检查是否是dataclass
            if hasattr(self.current_parser, '__dataclass_fields__'):
                parser_fields = fields(self.current_parser)
                for field in parser_fields:
                    field_name: str = field.name
                    field_type: Type = field.type
                    field_value: Any = getattr(self.current_parser, field_name, "")
                    self.log_message(f"Attribute: {field_name}, Type: {field_type}", level="debug")

                    # 使用维护器获取标准类型名称
                    maintainer: BaseMaintainer = MaintainerFactory.get_maintainer_supported_type(
                        field_name, field_type, field_value, logger=self
                    )
                    field_type_name: str = maintainer.get_simplest_type_name()

                    # 检查值是否与类型兼容
                    if not maintainer.is_value_compatible():
                        # 使用维护器的默认值
                        default_value: Any = maintainer.get_default_value()
                        # 更新Parser实例的属性值
                        setattr(self.current_parser, field_name, default_value)
                        # 记录警告日志
                        self.log_message(
                            f"Attribute '{field_name}' value '{field_value}' is not compatible with type '{field_type_name}'. Reset to default: {default_value}",
                            level="warn"
                        )
                        field_value = default_value

                    # 添加到Treeview，使用!r确保无歧义显示
                    # 为Unsupported类型添加特殊样式
                    tags = ('unsupported',) if type(maintainer) is UnsupportedMaintainer else ()
                    self.tree.insert("", tk.END, values=(field_name, field_type_name, repr(field_value)), tags=tags)
            else:
                # 对于非dataclass，输出警告
                self.log_message(
                    f"Parser {self.current_parser.__class__.__name__} is not a dataclass. Attributes will not be displayed.",
                    level="warn")
        except Exception as e:
            error_msg = f"Error displaying parser attributes: {str(e)}\n"

            # 根据日志级别添加不同详细程度的信息
            if self.debug_level >= 2:  # WARN级别及以上
                error_msg += f"Exception type: {type(e).__name__}\n"

            if self.debug_level >= 4:  # DEBUG级别及以上
                error_msg += f"Stack trace:\n{traceback.format_exc()}"

            if self.debug_level >= 5:  # TRACE级别
                error_msg += f"Python version: {sys.version}\n"
                if hasattr(self, 'current_parser'):
                    error_msg += f"Current parser: {self.current_parser.__class__.__name__}\n"

            self.log_message(error_msg, level="error")

    def on_tree_select(self, event: Optional[tk.Event] = None) -> None:
        """当TreeView选择发生变化时"""
        selected_items = self.tree.selection()
        if not selected_items:
            self.clear_inspector()
            return

        item = selected_items[0]
        values = self.tree.item(item, "values")
        if not values or len(values) < 3:
            self.clear_inspector()
            return

        attribute_name: str = values[0]
        self._update_inspector(attribute_name)

    def on_config_change(self, *args: Any) -> None:
        """当Config File路径变化时"""
        self.check_load_config_button()

    def check_load_config_button(self) -> None:
        """检查Load Config和Save Config按钮是否可用"""
        selected_parser: str = self.parser_var.get()
        config_path: str = self.config_var.get()

        # 确保选中了有效的parser且文件存在
        if selected_parser and self.parser_classes_dict[selected_parser] and config_path and os.path.exists(
                config_path):
            self.load_config_btn.config(state=tk.NORMAL)
        else:
            self.load_config_btn.config(state=tk.DISABLED)

        # Save Config按钮只要有parser实例就可用
        if self.current_parser:
            self.save_config_btn.config(state=tk.NORMAL)
        else:
            self.save_config_btn.config(state=tk.DISABLED)

    def export_example_yaml(self) -> None:
        """导出示例YAML文件"""
        selected_parser: str = self.parser_var.get()
        if not selected_parser:
            self.log_message("Error: No parser selected", level="error")
            return

        parser_cls: Optional[Type] = self.parser_classes_dict[selected_parser]
        if not parser_cls:
            self.log_message("Error: No parser selected", level="error")
            return

        if not hasattr(parser_cls, 'dump_example_to_yaml'):
            self.log_message("Error: Selected parser does not have dump_example_to_yaml method", level="error")
            return

        # 打开文件保存对话框
        file_path: Optional[str] = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # 调用dump_example_to_yaml方法
                parser_instance = parser_cls()
                # 设置allow_unicode=True以保持中文等Unicode字符原样
                parser_instance.dump_example_to_yaml(file_path, allow_unicode=True)
                self.log_message(f"Example YAML exported to: {file_path}")
            except Exception as e:
                error_msg = f"Error exporting example YAML: {str(e)}\n"

                # 根据日志级别添加不同详细程度的信息
                if self.debug_level >= 2:  # WARN级别及以上
                    error_msg += f"Exception type: {type(e).__name__}\n"

                if self.debug_level >= 4:  # DEBUG级别及以上
                    error_msg += f"Stack trace:\n{traceback.format_exc()}"

                if self.debug_level >= 5:  # TRACE级别
                    error_msg += f"Python version: {sys.version}\n"
                    error_msg += f"Parser class: {parser_cls.__name__}\n"
                    error_msg += f"Export path: {file_path}\n"

                self.log_message(error_msg, level="error")

    def browse_config_file(self) -> None:
        """浏览配置文件"""
        file_path: Optional[str] = filedialog.askopenfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )

        if file_path:
            self.config_var.set(file_path)
            self.log_message(f"Selected config file: {file_path}")

    def load_config(self) -> None:
        """加载配置文件"""
        selected_parser: str = self.parser_var.get()
        config_path: str = self.config_var.get()

        if not selected_parser:
            self.log_message("Error: No parser selected", level="error")
            return

        parser_cls: Optional[Type] = self.parser_classes_dict[selected_parser]
        if not parser_cls:
            self.log_message("Error: No parser selected", level="error")
            return

        if not config_path or not os.path.exists(config_path):
            self.log_message("Error: Config file does not exist", level="error")
            return

        try:
            # 实例化新的Parser对象
            parser_instance = parser_cls()

            # 标志：是否发生了加载异常
            load_error: bool = False

            # 尝试加载配置
            try:
                parser_instance.from_yaml(config_path)
            except Exception as e:
                # 如果加载失败，记录错误但继续执行
                load_error = True
                error_msg = f"Error loading config file: {str(e)}\n"
                error_msg += "Will try to load valid attributes only."

                # 根据日志级别添加不同详细程度的信息
                if self.debug_level >= 2:  # WARN级别及以上
                    error_msg += f"\nException type: {type(e).__name__}"

                if self.debug_level >= 4:  # DEBUG级别及以上
                    error_msg += f"\nStack trace:\n{traceback.format_exc()}"

                self.log_message(error_msg, level="warn")

            # 类型校验并修复不兼容的属性
            type_errors: List[str] = self._validate_and_fix_parser_types(parser_instance)
            if type_errors:
                for error in type_errors:
                    self.log_message(error, level="error")

            # 更新当前Parser实例
            self.current_parser = parser_instance
            parser_fields: Tuple[Field[Any], ...] = fields(self.current_parser)
            self.current_parser_fields = {fid.name: fid for fid in parser_fields}

            # 更新Attributes面板中显示的属性信息
            self.display_parser_attributes()

            # 更新Save Config按钮状态
            self.check_load_config_button()

            if load_error:
                if type_errors:
                    self.log_message(
                        f"Config loaded from {config_path} with errors and some type mismatches (reset to defaults)",
                        level="warn"
                    )
                else:
                    self.log_message(
                        f"Config loaded from {config_path} with errors but no type mismatches",
                        level="warn"
                    )
            elif type_errors:
                self.log_message(f"Config loaded from {config_path} with some type mismatches (reset to defaults)")
            else:
                self.log_message(f"Config loaded successfully from: {config_path}")
        except Exception as e:
            error_msg = f"Error loading config: {str(e)}\n"

            # 根据日志级别添加不同详细程度的信息
            if self.debug_level >= 2:  # WARN级别及以上
                error_msg += f"Exception type: {type(e).__name__}\n"

            if self.debug_level >= 4:  # DEBUG级别及以上
                error_msg += f"Stack trace:\n{traceback.format_exc()}"

            if self.debug_level >= 5:  # TRACE级别
                error_msg += f"Python version: {sys.version}\n"
                error_msg += f"Current working directory: {os.getcwd()}\n"
                error_msg += f"Config file path: {config_path}\n"

            self.log_message(error_msg, level="error")

    def save_config(self) -> None:
        """保存配置文件"""
        if not self.current_parser:
            self.log_message("Error: No parser instance", level="error")
            return

        # 打开文件保存对话框
        file_path: Optional[str] = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # 调用to_yaml方法保存配置
                if hasattr(self.current_parser, 'to_yaml'):
                    # 设置allow_unicode=True以保持中文等Unicode字符原样
                    self.current_parser.to_yaml(file_path, allow_unicode=True)
                    self.log_message(f"Config saved successfully to: {file_path}")
                else:
                    self.log_message("Error: Selected parser does not have to_yaml method", level="error")
            except Exception as e:
                error_msg = f"Error saving config: {str(e)}\n"

                # 根据日志级别添加不同详细程度的信息
                if self.debug_level >= 2:  # WARN级别及以上
                    error_msg += f"Exception type: {type(e).__name__}\n"

                if self.debug_level >= 4:  # DEBUG级别及以上
                    error_msg += f"Stack trace:\n{traceback.format_exc()}"

                if self.debug_level >= 5:  # TRACE级别
                    error_msg += f"Python version: {sys.version}\n"
                    error_msg += f"Current working directory: {os.getcwd()}\n"
                    error_msg += f"Save file path: {file_path}\n"

                self.log_message(error_msg, level="error")

    def _validate_and_fix_parser_types(self, parser_instance: Any) -> List[str]:
        """校验Parser实例的属性类型是否匹配，不兼容的属性重置为默认值
        
        Returns:
            类型错误列表，如果没有错误则返回空列表
        """
        errors: List[str] = []

        # 检查是否是dataclass
        if not hasattr(parser_instance, '__dataclass_fields__'):
            return errors

        parser_fields = fields(parser_instance)

        for field in parser_fields:
            field_name: str = field.name
            field_type: Type = field.type
            field_value: Any = getattr(parser_instance, field_name, None)

            # 获取类型维护器
            maintainer: BaseMaintainer = MaintainerFactory.get_maintainer_supported_type(
                field_name, field_type, field_value, logger=self
            )

            # 检查类型是否兼容
            if not maintainer.is_value_compatible():
                # 确定默认值
                if field.default is not dataclasses.MISSING:
                    # 有默认值
                    default_value: Any = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    # 有默认工厂函数
                    default_value: Any = field.default_factory()
                else:
                    # 没有默认值，使用维护器的默认值
                    default_value: Any = maintainer.get_default_value()

                # 设置默认值
                setattr(parser_instance, field_name, default_value)

                # 记录错误信息
                error_msg = (f"Type mismatch for '{field_name}': expected {maintainer.get_simplest_type_name()}, "
                             f"got {type(field_value).__name__} (reset to default: {default_value})")
                errors.append(error_msg)

        return errors

    def clear_inspector(self) -> None:
        """清空Inspector面板"""
        # 清空内部框架
        for widget in self.inspector_inner_frame.winfo_children():
            widget.destroy()
        self.current_attribute = None

    def _on_inspector_inner_configure(self, event: Optional[tk.Event] = None) -> None:
        """当内部框架大小改变时更新canvas的滚动区域"""
        # 更新canvas的滚动区域
        self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))

    def _on_inspector_canvas_configure(self, event: Optional[tk.Event] = None) -> None:
        """当canvas大小改变时调整内部框架的宽度和高度"""
        # 获取canvas的宽度和高度，留出滚动条的空间
        canvas_width: int = event.width - 5  # 留出滚动条的宽度空间
        canvas_height: int = event.height - 5  # 留出滚动条的高度空间

        # 获取内部框架的最小宽度和高度
        min_width: int = self.inspector_inner_frame.winfo_reqwidth()
        min_height: int = self.inspector_inner_frame.winfo_reqheight()

        # 计算内部框架的实际宽度和高度
        # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
        # 否则使用内部控件的最小宽度和高度
        actual_width: int = max(canvas_width, min_width)
        actual_height: int = max(canvas_height, min_height)

        # 设置内部框架的宽度和高度
        self.inspector_canvas.itemconfig(self.inspector_inner_frame_id, width=actual_width, height=actual_height)

        # 更新canvas的滚动区域
        self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))

    def _update_inspector(self, attribute_name: str) -> None:
        """更新Inspector面板，显示属性信息和编辑控件"""
        if not self.current_parser:
            self.clear_inspector()
            return

        # 清空内部框架
        for widget in self.inspector_inner_frame.winfo_children():
            widget.destroy()

        self.current_attribute = attribute_name

        try:
            # 检查是否是dataclass
            if hasattr(self.current_parser, '__dataclass_fields__'):
                # 获取属性值
                attribute_value: Any = getattr(self.current_parser, self.current_attribute, None)
                if self.current_parser_fields is None:
                    parser_fields: Tuple[Field[Any], ...] = fields(self.current_parser)
                    self.current_parser_fields = {fid.name: fid for fid in parser_fields}

                # 获取字段信息
                current_field: Optional[Field[Any]] = self.current_parser_fields.get(self.current_attribute, None)

                if current_field:
                    # 获取类型维护器
                    self.current_maintainer = MaintainerFactory.get_maintainer_supported_type(
                        attribute_name, current_field.type, attribute_value, self
                    )
                else:
                    # 非字段属性
                    self.current_maintainer = MaintainerFactory.get_maintainer_supported_value(
                        attribute_name, attribute_value, self
                    )

                # 渲染整个Inspector面板内容
                self.current_maintainer.config_view(view_mode="Packed")
                attribute_inspector: ttk.Widget = self.current_maintainer.create_inspector(
                    self.inspector_inner_frame, self._on_attribute_change
                )
                attribute_inspector.pack(fill=tk.BOTH, expand=True)
            else:
                # 对于非dataclass，输出警告
                self.log_message(
                    f"Parser {self.current_parser.__class__.__name__} is not a dataclass. "
                    f"Inspector will not be displayed.",
                    level="warn"
                )

        except Exception as e:
            error_msg = f"Error updating inspector: {str(e)}\n"

            # 根据日志级别添加不同详细程度的信息
            if self.debug_level >= 2:  # WARN级别及以上
                error_msg += f"Exception type: {type(e).__name__}\n"

            if self.debug_level >= 4:  # DEBUG级别及以上
                error_msg += f"Stack trace:\n{traceback.format_exc()}"

            if self.debug_level >= 5:  # TRACE级别
                error_msg += f"Python version: {sys.version}\n"
                if hasattr(self, 'current_parser'):
                    error_msg += f"Current parser: {self.current_parser.__class__.__name__}\n"
                if hasattr(self, 'current_attribute'):
                    error_msg += f"Current attribute: {self.current_attribute}\n"

            self.log_message(error_msg, level="error")

        # 更新内部框架的宽高规格
        # 获取canvas的宽度和高度，留出滚动条的空间
        canvas_width: int = self.inspector_canvas.winfo_width() - 5  # 留出滚动条的宽度空间
        canvas_height: int = self.inspector_canvas.winfo_height() - 5  # 留出滚动条的高度空间

        # 强制更新布局，确保获取准确的所需规格
        self.inspector_inner_frame.update_idletasks()
        
        # 获取内部框架的最小宽度和高度
        min_width: int = self.inspector_inner_frame.winfo_reqwidth()
        min_height: int = self.inspector_inner_frame.winfo_reqheight()

        # 计算内部框架的实际宽度和高度
        # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
        # 否则使用内部控件的最小宽度和高度
        actual_width: int = max(canvas_width, min_width)
        actual_height: int = max(canvas_height, min_height)

        # 设置内部框架的宽度和高度
        self.inspector_canvas.itemconfig(self.inspector_inner_frame_id, width=actual_width, height=actual_height)

        # 更新canvas的滚动区域
        self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))

    def _on_attribute_change(self, new_value: Any) -> None:
        """处理属性值变化"""
        assert self.current_parser is not None

        # Detect currently selected item
        selected_items = self.tree.selection()
        target_item: Optional[Any] = None
        if selected_items:
            target_item = selected_items[0]
            values: Tuple = cast(Tuple, cast(object, self.tree.item(target_item, "values")))
            if not values or len(values) != 3 or values[0] != self.current_attribute:
                target_item = None
        try:
            # 检查是否是dataclass
            if hasattr(self.current_parser, '__dataclass_fields__'):
                # 更新属性值
                setattr(self.current_parser, self.current_attribute, new_value)
                if self.current_maintainer:
                    self.current_maintainer.confirm_editor_change()

                # 直接更新TreeView中对应项的值，而不是重新显示所有属性
                if target_item is None:
                    for item in self.tree.get_children():
                        item_values = self.tree.item(item, "values")
                        if item_values and item_values[0] == self.current_attribute:
                            target_item = item
                if target_item is None:
                    self.log_message(
                        f"Attribute {self.current_attribute} item could not be found.",
                        level="error"
                    )
                    return

                # 获取类型维护器以获取标准类型名称
                if self.current_parser_fields is None:
                    parser_fields: Tuple[Field[Any], ...] = fields(self.current_parser)
                    self.current_parser_fields = {fid.name: fid for fid in parser_fields}
                # 获取字段信息
                maintainer: Optional[BaseMaintainer] = self.current_maintainer
                if maintainer is None:
                    current_field: Optional[Field[Any]] = self.current_parser_fields.get(self.current_attribute, None)

                    if current_field:
                        attr_type: Type = current_field.type
                    else:
                        attr_type: Type = type(new_value)

                    maintainer = MaintainerFactory.get_maintainer_supported_type(
                        self.current_attribute,
                        attr_type,
                        new_value,
                        None
                    )
                # 更新TreeView项
                # 为Unsupported类型添加特殊样式
                tags = ('unsupported',) if type(maintainer) is UnsupportedMaintainer else ()
                self.tree.item(
                    target_item,
                    values=(self.current_attribute, maintainer.get_simplest_type_name(), repr(new_value)),
                    tags=tags
                )
                # 记录日志
                self.log_message(f"Updated attribute '{self.current_attribute}' to {repr(new_value)}")
            else:
                # 对于非dataclass，输出警告
                self.log_message(
                    f"Parser {self.current_parser.__class__.__name__} is not a dataclass. Attribute changes are not supported.",
                    level="warn")
        except Exception as e:
            error_msg = f"Error updating attribute '{self.current_attribute}': {str(e)}\n"

            # 根据日志级别添加不同详细程度的信息
            if self.debug_level >= 2:  # WARN级别及以上
                error_msg += f"Exception type: {type(e).__name__}\n"

            if self.debug_level >= 4:  # DEBUG级别及以上
                error_msg += f"Stack trace:\n{traceback.format_exc()}"

            if self.debug_level >= 5:  # TRACE级别
                error_msg += f"Python version: {sys.version}\n"
                if hasattr(self, 'current_parser'):
                    error_msg += f"Current parser: {self.current_parser.__class__.__name__}\n"
                error_msg += f"Current attribute: {self.current_attribute}\n"
                error_msg += f"New value: {repr(new_value)}\n"

            self.log_message(error_msg, level="error")

    @staticmethod
    def log_message(message: str, level: str = "info") -> None:
        """记录日志信息"""
        # 日志级别映射
        level_map: Dict[str, int] = {
            "error": 1,
            "warn": 2,
            "info": 3,
            "debug": 4,
            "trace": 5
        }

        # 获取当前消息的日志级别
        current_level: int = level_map.get(level.lower(), 3)

        # 检查是否需要打印日志
        debug_level: int = 5  # 默认最高日志级别
        if YAMLConfigCLI._instance is not None:
            debug_level = YAMLConfigCLI._instance.debug_level

        if debug_level == 0:
            return

        # 获取当前时间，精确到毫秒
        current_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        tagged_message: str = f"{current_time} [{level.upper()}] {message}"

        # 只有当当前消息的日志级别小于等于设置的debug_level时才打印
        if current_level <= debug_level:
            # 无论是否有实例，都打印到控制台
            print(tagged_message)

            # 如果有实例且log_text存在，同步到Log框
            if YAMLConfigCLI._instance is not None:
                instance = YAMLConfigCLI._instance
                if hasattr(instance, 'log_text') and instance.log_text is not None:
                    # 清除旧内容，保持日志区域简洁
                    if instance.log_text.index('end-1c') != '1.0':
                        instance.log_text.insert(tk.END, "\n")

                    # 根据级别设置颜色
                    if level in level_map:
                        instance.log_text.insert(tk.END, tagged_message, level)
                    else:
                        # 例外紫色信息
                        instance.log_text.insert(tk.END, tagged_message, "special")

                    # 滚动到底部
                    instance.log_text.see(tk.END)


if __name__ == "__main__":
    # 解析命令行参数
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="YAML Config CLI")
    parser.add_argument('-d', '--debug_level', type=int, default=5, choices=range(6),
                        help='Debug level (0-5): 0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE')
    args = parser.parse_args()

    root: tk.Tk = tk.Tk()
    app: YAMLConfigCLI = YAMLConfigCLI(root, debug_level=args.debug_level)
    app.log_message('Logging initialized', level="info")
    app.log_message('Will print error messages', level="error")
    app.log_message('Will print warning messages', level="warn")
    app.log_message('Will print info messages', level="info")
    app.log_message('Will print debug messages', level="debug")
    app.log_message('Will print trace messages', level="trace")
    app.log_message('Will print special messages', level="special")

    root.mainloop()
