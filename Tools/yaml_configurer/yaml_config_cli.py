import os
import sys
import typing
import traceback
import dataclasses
from pathlib import Path
import importlib.util
from dataclasses import fields
from typing import Dict, List, Optional, Any
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import inspect
from dataclasses import fields
import argparse

# 导入类型维护器
from type_maintainer.maintainer_factory import MaintainerFactory

# 从foo.py中导入相关类
spec = importlib.util.spec_from_file_location("foo", "d:\\CBIB\\Storages\\DevelopmentSoftwares\\Trae\\PipeUNet\\foo.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
from foo import ParserABC


class YAMLConfigCLI:
    # Singleton instance
    _instance = None
    
    def __new__(cls, root=None, debug_level=3):
        """Create and return the singleton instance"""
        if cls._instance is None:
            cls._instance = super(YAMLConfigCLI, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, root=None, debug_level=3):
        """Initialize the YAML Config CLI"""
        
        # Only initialize once
        if not hasattr(self, 'initialized'):
            if root is not None:
                self.root = root
                self.root.title("YAML Config CLI")
                self.root.geometry("1366x768")
            else:
                self.root = None

            # 当前Parser实例
            self.current_parser = None

            # 日志级别
            self.debug_level = debug_level

            # 日志级别映射
            self.log_levels = {
                0: "OFF",
                1: "ERROR",
                2: "WARN",
                3: "INFO",
                4: "DEBUG",
                5: "TRACE"
            }
            
            # Mark as initialized
            self.initialized = True

            # 配置主窗口的grid布局
            if root is not None:
                root.grid_rowconfigure(0, weight=1)
                root.grid_columnconfigure(0, weight=1)

                # 主框架
                self.main_frame = ttk.Frame(root, padding="10")
                self.main_frame.grid(row=0, column=0, sticky=tk.NSEW)

                # 配置main_frame使用PanedWindow实现上下可拖动调整
                self.main_paned = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
                self.main_paned.pack(fill=tk.BOTH, expand=True)

                # 顶部内容区域 - 左右分栏，使用PanedWindow实现可拖动调整
                self.top_frame = ttk.PanedWindow(self.main_paned, orient=tk.HORIZONTAL)

                # 左侧区域
                self.left_frame = ttk.Frame(self.top_frame)
                self.top_frame.add(self.left_frame, weight=1)

                # 右侧区域
                self.right_frame = ttk.Frame(self.top_frame)
                self.top_frame.add(self.right_frame, weight=1)

                # 左侧区域内容
                # 第一行：Parser下拉选项
                self.parser_frame = ttk.Frame(self.left_frame)
                self.parser_frame.pack(fill=tk.X, pady=(0, 5))

                # 配置grid布局，确保整行拉伸
                self.parser_frame.grid_columnconfigure(1, weight=1)

                ttk.Label(self.parser_frame, text="Parser:", width=10).grid(row=0, column=0, sticky=tk.W, pady=5)
                self.parser_var = tk.StringVar()
                self.parser_combobox = ttk.Combobox(self.parser_frame, textvariable=self.parser_var, state="readonly")
                self.parser_combobox.grid(row=0, column=1, sticky=tk.W + tk.E, pady=5, padx=5)

                self.export_example_btn = ttk.Button(self.parser_frame, text="Export Example YAML", state=tk.DISABLED,
                                                     command=self.export_example_yaml, width=22)
                self.export_example_btn.grid(row=0, column=2, pady=5, padx=5)

                # 第二行：Config File
                self.config_frame = ttk.Frame(self.left_frame)
                self.config_frame.pack(fill=tk.X, pady=(0, 5))

                # 配置grid布局，确保整行拉伸
                self.config_frame.grid_columnconfigure(1, weight=1)

                ttk.Label(self.config_frame, text="Config File:", width=10).grid(row=0, column=0, sticky=tk.W, pady=5)
                self.config_var = tk.StringVar()
                self.config_entry = ttk.Entry(self.config_frame, textvariable=self.config_var)
                self.config_entry.grid(row=0, column=1, sticky=tk.W + tk.E, pady=5, padx=5)

                self.browse_btn = ttk.Button(self.config_frame, text="Browse", command=self.browse_config_file, width=22)
                self.browse_btn.grid(row=0, column=2, pady=5, padx=5)

                # 第三行：Load Config和Save Config按钮
                self.load_frame = ttk.Frame(self.left_frame)
                self.load_frame.pack(fill=tk.X, pady=(0, 10))

                # 配置grid布局
                self.load_frame.grid_columnconfigure(0, weight=1)
                self.load_frame.grid_columnconfigure(1, weight=1)

                self.load_config_btn = ttk.Button(self.load_frame, text="Load Config", state=tk.DISABLED,
                                              command=self.load_config)
                self.load_config_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W+tk.E)

                self.save_config_btn = ttk.Button(self.load_frame, text="Save Config", state=tk.DISABLED,
                                              command=self.save_config)
                self.save_config_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)

                # Attributes面板 - 占据剩余所有空间
                self.attributes_frame = ttk.LabelFrame(self.left_frame, text="Attributes")
                self.attributes_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

                # 创建可交互的表格
                self.table_frame = ttk.Frame(self.attributes_frame)
                self.table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # 配置table_frame的grid布局
                self.table_frame.grid_rowconfigure(0, weight=1)
                self.table_frame.grid_columnconfigure(0, weight=1)

                # 创建Treeview组件作为可交互表格
                self.tree = ttk.Treeview(self.table_frame, columns=("attribute", "type", "value"), show="headings")
                self.tree.grid(row=0, column=0, sticky=tk.NSEW)

                # 设置列标题
                self.tree.heading("attribute", text="Attribute")
                self.tree.heading("type", text="Type")
                self.tree.heading("value", text="Value")

                # 设置列宽（允许用户拖动调整）
                self.tree.column("attribute", minwidth=80, width=150, stretch=False, anchor=tk.W)
                self.tree.column("type", minwidth=80, width=150, stretch=False, anchor=tk.W)
                self.tree.column("value", minwidth=80, width=150, stretch=False, anchor=tk.W)

                # 添加垂直滚动条
                vsb = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
                vsb.grid(row=0, column=1, sticky=tk.NS)
                self.tree.configure(yscrollcommand=vsb.set)

                # 添加水平滚动条
                hsb = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.tree.xview)
                hsb.grid(row=1, column=0, sticky=tk.EW)
                self.tree.configure(xscrollcommand=hsb.set)

                # 设置样式，使背景为白色
                style = ttk.Style()
                style.configure("Treeview", background="white", fieldbackground="white")
                style.configure("Treeview.Heading", background="#f0f0f0", font=("Arial", 10, "bold"))
                style.map("Treeview", background=[("selected", "#347083"), ("!selected", "white")], foreground=[("selected", "white")])

                # 右侧Inspector面板 - 占据所有空间
                self.inspector_frame = ttk.LabelFrame(self.right_frame, text="Inspector")
                self.inspector_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
                self.inspector_frame.grid_rowconfigure(0, weight=1)
                self.inspector_frame.grid_rowconfigure(1, weight=0)  # 横向滚动条固定高度
                self.inspector_frame.grid_columnconfigure(0, weight=1)
                self.inspector_frame.grid_columnconfigure(1, weight=0)  # 纵向滚动条固定宽度

                # 创建带滚动条的容器
                self.inspector_canvas = tk.Canvas(self.inspector_frame)
                self.inspector_scrollbar = ttk.Scrollbar(self.inspector_frame, orient="vertical", command=self.inspector_canvas.yview)
                self.inspector_hscrollbar = ttk.Scrollbar(self.inspector_frame, orient="horizontal", command=self.inspector_canvas.xview)
                self.inspector_canvas.configure(yscrollcommand=self.inspector_scrollbar.set, xscrollcommand=self.inspector_hscrollbar.set)

                # 布局 - 使用grid布局
                self.inspector_canvas.grid(row=0, column=0, sticky=tk.NSEW)
                self.inspector_scrollbar.grid(row=0, column=1, sticky=tk.NS)
                self.inspector_hscrollbar.grid(row=1, column=0, sticky=tk.EW)

                # 创建内部框架，用于容纳所有Inspector内容
                self.inspector_inner_frame = ttk.Frame(self.inspector_canvas, padding=10)
                # 将内部框架添加到canvas
                self.inspector_inner_frame_id = self.inspector_canvas.create_window((0, 0), window=self.inspector_inner_frame, anchor=tk.NW)

                # 绑定事件，当内部框架大小改变时更新canvas的滚动区域
                self.inspector_inner_frame.bind("<Configure>", self.on_inspector_inner_configure)
                # 绑定事件，当canvas大小改变时调整内部框架的宽度
                self.inspector_canvas.bind("<Configure>", self.on_inspector_canvas_configure)

                # 初始为空
                self.current_attribute = None

                # 日志区域 - 占据全宽
                self.log_frame = ttk.LabelFrame(self.main_paned, text="Log")
                self.log_frame.grid_rowconfigure(0, weight=1)

                self.log_text = scrolledtext.ScrolledText(
                    self.log_frame,
                    wrap=tk.WORD, bg="black", fg="white", font=('Consolas', 10),
                    height=10
                )
                self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

                # 添加顶部区域和日志区域到主PanedWindow
                self.main_paned.add(self.top_frame, weight=7)
                self.main_paned.add(self.log_frame, weight=3)

                # 初始化Parser选项
                self.init_parser_options()

                # 绑定事件
                self.parser_var.trace_add("write", self.on_parser_change)
                self.config_var.trace_add("write", self.on_config_change)

                # 绑定TreeView选择事件
                self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def init_attributes_table(self):
        """初始化Attributes表格"""
        # 清除Treeview中的所有项
        for item in self.tree.get_children():
            self.tree.delete(item)

    def init_parser_options(self):
        """初始化Parser下拉选项"""
        # 获取所有继承ParserABC的类
        parser_classes = []
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

    def on_parser_change(self, *args):
        """当Parser选择发生变化时"""
        selected_parser = self.parser_var.get()

        # 清除之前的Parser实例
        self.current_parser = None

        # 清空Attributes内容显示
        self.init_attributes_table()

        # 清空Inspector面板
        self.clear_inspector()

        if selected_parser:
            parser_cls = self.parser_classes_dict[selected_parser]
            if parser_cls:
                # 实例化新的Parser对象
                try:
                    self.current_parser = parser_cls()
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

    def display_parser_attributes(self):
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
                    field_name = field.name
                    field_type = field.type
                    field_value = getattr(self.current_parser, field_name, "")
                    self.log_message(f"Attribute: {field_name}, Type: {field_type}", level="debug")

                    # 使用维护器获取标准类型名称
                    maintainer = MaintainerFactory.get_maintainer(field_type)
                    field_type_name = maintainer.get_expected_type_name()

                    # 检查值是否与类型兼容
                    if not maintainer.is_compatible(field_value):
                        # 使用维护器的默认值
                        default_value = maintainer.get_default_value()
                        # 更新Parser实例的属性值
                        setattr(self.current_parser, field_name, default_value)
                        # 记录警告日志
                        self.log_message(f"Warning: Attribute '{field_name}' value '{field_value}' is not compatible with type '{field_type_name}'. Reset to default: {default_value}", level="warning")
                        field_value = default_value

                    # 添加到Treeview，使用!r确保无歧义显示
                    self.tree.insert("", tk.END, values=(field_name, field_type_name, repr(field_value)))
            else:
                # 对于非dataclass，获取所有属性
                attributes = inspect.getmembers(self.current_parser, lambda a: not inspect.isroutine(a))
                for name, value in attributes:
                    if not name.startswith('_'):
                        try:
                            # 使用维护器获取标准类型名称
                            attr_type = type(value)
                            maintainer = MaintainerFactory.get_maintainer(attr_type)
                            attr_type_name = maintainer.get_expected_type_name()
                            
                            # 检查值是否与类型兼容
                            if not maintainer.is_compatible(value):
                                # 使用维护器的默认值
                                default_value = maintainer.get_default_value()
                                # 更新Parser实例的属性值
                                setattr(self.current_parser, name, default_value)
                                # 记录警告日志
                                self.log_message(f"Warning: Attribute '{name}' value '{value}' is not compatible with type '{attr_type_name}'. Reset to default: {default_value}", level="warning")
                                value = default_value
                            
                            # 添加到Treeview，使用!r确保无歧义显示
                            self.tree.insert("", tk.END, values=(name, attr_type_name, repr(value)))
                        except Exception as e:
                            # 只在DEBUG级别及以上记录这个异常
                            if self.debug_level >= 4:
                                self.log_message(f"Error processing attribute {name}: {str(e)}", level="debug")
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

    def on_tree_select(self, event):
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

        attribute_name = values[0]
        self.update_inspector(attribute_name)

    def on_config_change(self, *args):
        """当Config File路径变化时"""
        self.check_load_config_button()

    def check_load_config_button(self):
        """检查Load Config和Save Config按钮是否可用"""
        selected_parser = self.parser_var.get()
        config_path = self.config_var.get()

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

    def export_example_yaml(self):
        """导出示例YAML文件"""
        selected_parser = self.parser_var.get()
        if not selected_parser:
            self.log_message("Error: No parser selected", level="error")
            return

        parser_cls = self.parser_classes_dict[selected_parser]
        if not parser_cls:
            self.log_message("Error: No parser selected", level="error")
            return

        if not hasattr(parser_cls, 'dump_example_to_yaml'):
            self.log_message("Error: Selected parser does not have dump_example_to_yaml method", level="error")
            return

        # 打开文件保存对话框
        file_path = filedialog.asksaveasfilename(
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

    def browse_config_file(self):
        """浏览配置文件"""
        file_path = filedialog.askopenfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )

        if file_path:
            self.config_var.set(file_path)
            self.log_message(f"Selected config file: {file_path}")

    def load_config(self):
        """加载配置文件"""
        selected_parser = self.parser_var.get()
        config_path = self.config_var.get()

        if not selected_parser:
            self.log_message("Error: No parser selected", level="error")
            return

        parser_cls = self.parser_classes_dict[selected_parser]
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
            load_error = False

            # 尝试加载配置
            try:
                parser_instance.from_yaml(config_path)
            except Exception as e:
                # 如果加载失败，记录错误但继续执行
                load_error = True
                error_msg = f"Warning: Error loading config file: {str(e)}\n"
                error_msg += "Will try to load valid attributes only."

                # 根据日志级别添加不同详细程度的信息
                if self.debug_level >= 2:  # WARN级别及以上
                    error_msg += f"\nException type: {type(e).__name__}"

                if self.debug_level >= 4:  # DEBUG级别及以上
                    error_msg += f"\nStack trace:\n{traceback.format_exc()}"

                self.log_message(error_msg, level="warning")

            # 类型校验并修复不兼容的属性
            type_errors = self._validate_and_fix_parser_types(parser_instance)
            if type_errors:
                for error in type_errors:
                    self.log_message(error, level="error")

            # 更新当前Parser实例
            self.current_parser = parser_instance

            # 更新Attributes面板中显示的属性信息
            self.display_parser_attributes()

            # 更新Save Config按钮状态
            self.check_load_config_button()

            if load_error:
                if type_errors:
                    self.log_message(f"Config loaded from {config_path} with errors and some type mismatches (reset to defaults)", level="warning")
                else:
                    self.log_message(f"Config loaded from {config_path} with errors but no type mismatches", level="warning")
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

    def save_config(self):
        """保存配置文件"""
        if not self.current_parser:
            self.log_message("Error: No parser instance", level="error")
            return

        # 打开文件保存对话框
        file_path = filedialog.asksaveasfilename(
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

    def _validate_and_fix_parser_types(self, parser_instance) -> list:
        """校验Parser实例的属性类型是否匹配，不兼容的属性重置为默认值
        
        Returns:
            类型错误列表，如果没有错误则返回空列表
        """
        errors = []

        # 检查是否是dataclass
        if not hasattr(parser_instance, '__dataclass_fields__'):
            return errors

        parser_fields = fields(parser_instance)

        for field in parser_fields:
            field_name = field.name
            field_type = field.type
            field_value = getattr(parser_instance, field_name, None)

            # 获取类型维护器
            maintainer = MaintainerFactory.get_maintainer(field_type)

            # 检查类型是否兼容
            if not maintainer.is_compatible(field_value):
                # 确定默认值
                if field.default is not dataclasses.MISSING:
                    # 有默认值
                    default_value = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    # 有默认工厂函数
                    default_value = field.default_factory()
                else:
                    # 没有默认值，使用维护器的默认值
                    default_value = maintainer.get_default_value()

                # 设置默认值
                setattr(parser_instance, field_name, default_value)

                # 记录错误信息
                error_msg = f"Type mismatch for '{field_name}': expected {maintainer.get_expected_type_name()}, got {type(field_value).__name__} (reset to default: {default_value})"
                errors.append(error_msg)

        return errors

    def _get_expected_type_name(self, field_type) -> str:
        """获取预期类型的名称"""
        # 处理Optional类型
        origin = typing.get_origin(field_type)
        args = typing.get_args(field_type)

        if origin is typing.Union and type(None) in args:
            # Optional[T] 的情况
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                return self._get_type_name(non_none_args[0])
            return str(field_type)

        return self._get_type_name(field_type)

    def _get_type_name(self, type_obj) -> str:
        """获取类型对象的名称"""
        if hasattr(type_obj, '__name__'):
            return type_obj.__name__
        elif hasattr(type_obj, '__origin__'):
            origin = type_obj.__origin__
            args = typing.get_args(type_obj)
            if args:
                args_str = ', '.join(self._get_type_name(arg) for arg in args)
                return f"{self._get_type_name(origin)}[{args_str}]"
            return self._get_type_name(origin)
        return str(type_obj)

    def _check_type_match(self, value, expected_type) -> bool:
        """检查值是否与预期类型匹配"""
        # 处理Optional类型
        origin = typing.get_origin(expected_type)
        args = typing.get_args(expected_type)

        if origin is typing.Union and type(None) in args:
            # Optional[T] 的情况，允许None
            if value is None:
                return True
            # 检查非None值是否匹配第一个非None类型
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                return self._check_type_match(value, non_none_args[0])
            return True

        # 处理泛型类型（如 list[str], dict[str, int]）
        if origin is not None:
            # 检查是否是预期的容器类型
            if origin is list:
                if not isinstance(value, list):
                    return False
                # 如果有类型参数，检查列表元素
                if args:
                    element_type = args[0]
                    return all(self._check_type_match(item, element_type) for item in value)
                return True
            elif origin is dict:
                if not isinstance(value, dict):
                    return False
                # 如果有类型参数，检查字典键值
                if len(args) >= 2:
                    key_type, value_type = args[0], args[1]
                    return all(
                        self._check_type_match(k, key_type) and self._check_type_match(v, value_type)
                        for k, v in value.items()
                    )
                return True
            else:
                # 其他泛型类型，只检查是否为该类型的实例
                return isinstance(value, origin)

        # 基本类型检查
        return isinstance(value, expected_type)

    def clear_inspector(self):
        """清空Inspector面板"""
        # 清空内部框架
        for widget in self.inspector_inner_frame.winfo_children():
            widget.destroy()
        self.current_attribute = None
    
    def on_inspector_inner_configure(self, event):
        """当内部框架大小改变时更新canvas的滚动区域"""
        # 更新canvas的滚动区域
        self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))
    
    def on_inspector_canvas_configure(self, event):
        """当canvas大小改变时调整内部框架的宽度和高度"""
        # 获取canvas的宽度和高度，留出滚动条的空间
        canvas_width = event.width - 5  # 留出滚动条的宽度空间
        canvas_height = event.height - 5  # 留出滚动条的高度空间
        
        # 获取内部框架的最小宽度和高度
        min_width = self.inspector_inner_frame.winfo_reqwidth()
        min_height = self.inspector_inner_frame.winfo_reqheight()
        
        # 计算内部框架的实际宽度和高度
        # 如果canvas窗口更大，则伸长到占满canvas窗口的大小
        # 否则使用内部控件的最小宽度和高度
        actual_width = max(canvas_width, min_width)
        actual_height = max(canvas_height, min_height)
        
        # 设置内部框架的宽度和高度
        self.inspector_canvas.itemconfig(self.inspector_inner_frame_id, width=actual_width, height=actual_height)
        
        # 更新canvas的滚动区域
        self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))

    def update_inspector(self, attribute_name):
        """更新Inspector面板，显示属性信息和编辑控件"""
        if not self.current_parser:
            self.clear_inspector()
            return

        # 清空内部框架
        for widget in self.inspector_inner_frame.winfo_children():
            widget.destroy()

        self.current_attribute = attribute_name

        try:
            # 获取属性值
            attribute_value = getattr(self.current_parser, attribute_name, None)

            # 检查是否是dataclass
            if hasattr(self.current_parser, '__dataclass_fields__'):
                # 获取字段信息
                parser_fields = fields(self.current_parser)
                field = None
                for f in parser_fields:
                    if f.name == attribute_name:
                        field = f
                        break

                if field:
                    # 获取类型维护器
                    maintainer = MaintainerFactory.get_maintainer(field.type)
                else:
                    # 非字段属性
                    attr_type = type(attribute_value)
                    maintainer = MaintainerFactory.get_maintainer(attr_type)
            else:
                # 非dataclass，获取类型维护器
                attr_type = type(attribute_value)
                maintainer = MaintainerFactory.get_maintainer(attr_type)

            # 渲染整个Inspector面板内容
            control_frame = maintainer.render_control(self.inspector_inner_frame, attribute_name, attribute_value, self.on_attribute_change)
            control_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

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

    def on_attribute_change(self, new_value):
        """处理属性值变化"""
        if not self.current_parser or not self.current_attribute:
            return

        try:
            # 更新属性值
            setattr(self.current_parser, self.current_attribute, new_value)
            # 直接更新TreeView中对应项的值，而不是重新显示所有属性
            for item in self.tree.get_children():
                item_values = self.tree.item(item, "values")
                if item_values and item_values[0] == self.current_attribute:
                    # 获取更新后的值和类型
                    updated_value = getattr(self.current_parser, self.current_attribute, None)
                    # 获取类型维护器以获取标准类型名称
                    if hasattr(self.current_parser, '__dataclass_fields__'):
                        from dataclasses import fields
                        parser_fields = fields(self.current_parser)
                        field = None
                        for f in parser_fields:
                            if f.name == self.current_attribute:
                                field = f
                                break
                        if field:
                            from Tools.yaml_configurer.type_maintainer.maintainer_factory import MaintainerFactory
                            maintainer = MaintainerFactory.get_maintainer(field.type)
                            attr_type_name = maintainer.get_expected_type_name()
                        else:
                            attr_type_name = type(updated_value).__name__
                    else:
                        attr_type_name = type(updated_value).__name__
                    # 更新TreeView项
                    self.tree.item(item, values=(self.current_attribute, attr_type_name, repr(updated_value)))
                    break
            # 记录日志
            self.log_message(f"Updated attribute '{self.current_attribute}' to {repr(new_value)}")
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
    def log_message(message: str, level: str = "info"):
        """记录日志信息"""
        # 日志级别映射
        level_map = {
            "error": 1,
            "warning": 2,
            "info": 3,
            "debug": 4,
            "trace": 5
        }

        # 获取当前消息的日志级别
        current_level = level_map.get(level.lower(), 3)

        # 检查是否需要打印日志
        debug_level = 5  # 默认最高日志级别
        if YAMLConfigCLI._instance is not None:
            debug_level = YAMLConfigCLI._instance.debug_level
        
        if debug_level == 0:
            return

        # 只有当当前消息的日志级别小于等于设置的debug_level时才打印
        if current_level <= debug_level:
            # 无论是否有实例，都打印到控制台
            print(f"[{level.upper()}] {message}")
            
            # 如果有实例且log_text存在，同步到Log框
            if YAMLConfigCLI._instance is not None:
                instance = YAMLConfigCLI._instance
                if hasattr(instance, 'log_text') and instance.log_text is not None:
                    # 清除旧内容，保持日志区域简洁
                    if instance.log_text.index('end-1c') != '1.0':
                        instance.log_text.insert(tk.END, "\n")

                    # 根据级别设置颜色
                    if level == "error":
                        # 红色错误信息
                        instance.log_text.insert(tk.END, message, "error")
                    elif level == "warning":
                        # 黄色警告信息
                        instance.log_text.insert(tk.END, message, "warning")
                    else:
                        # 白色普通信息
                        instance.log_text.insert(tk.END, message)

                    # 滚动到底部
                    instance.log_text.see(tk.END)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YAML Config CLI")
    parser.add_argument('-d', '--debug_level', type=int, default=5, choices=range(6),
                        help='Debug level (0-5): 0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE')
    args = parser.parse_args()

    root = tk.Tk()
    app = YAMLConfigCLI(root, debug_level=args.debug_level)

    # 配置日志文本的标签
    app.log_text.tag_configure("error", foreground="red", font=("Consolas", 10, "bold"))
    app.log_text.tag_configure("warning", foreground="yellow", font=("Consolas", 10, "bold"))

    root.mainloop()
