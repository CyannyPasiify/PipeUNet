import copy
from dataclasses import fields, is_dataclass
from typing import Any, Tuple, Type, Callable, Optional, List, Literal, Dict, get_origin, get_args
from typing_extensions import override
import tkinter as tk
from tkinter import ttk
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.unsupported_maintainer import UnsupportedMaintainer
from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer
from Tools.YamlConfigurer.Maintainer.container_maintainer import ContainerMaintainer


class DataclassMaintainer(ContainerMaintainer):
    """Dataclass type Maintainer"""

    @classmethod
    @override
    def default_standalone_window_size(cls: Type) -> Tuple[int, int]:
        return 1236, 600

    @classmethod
    @override
    def shall_hotkey_confirm_cancel(cls: Type) -> Tuple[bool, bool]:
        return False, False

    def __init__(
            self,
            attribute_name: str = "",
            attribute_type: Type = Any,
            attribute_value: Any = None,
            logger: Any = None
    ):
        super().__init__(attribute_name, attribute_type, attribute_value, logger)
        self.current_dataclass_type: Optional[Type] = None
        self.dataclass_subtypes: List[Type] = []
        self.map_subtype_name: Dict[str, Type] = {}
        
        if self.is_type_compatible():
            from Tools.YamlConfigurer.configurations import Configurations
            for maintainer_cls in Configurations.maintainer_collection:
                if hasattr(maintainer_cls, 'is_type_compatible_static'):
                    try:
                        if maintainer_cls.is_type_compatible_static(self.attribute_type):
                            pass
                    except:
                        pass
        
        self.view_mode: Literal["Standalone", "Packed"] = "Standalone"
        self.item_inspector_inner_frame_id: Optional[int] = None
        self.popup_canvas_inner_frame_id: Optional[int] = None
        self.popup_wnd_result: Optional[Dict[str, Any]] = None
        self.item_maintainer: Optional[BaseMaintainer] = None
        self.current_selected_item: Optional[str] = None
        
        self.editor_label_frame: Optional[ttk.LabelFrame] = None
        self.type_combobox_frame: Optional[ttk.Frame] = None
        self.type_string_var: Optional[tk.StringVar] = None
        self.type_combobox: Optional[ttk.Combobox] = None
        self.tree_frame: Optional[ttk.Frame] = None
        self.list_treeview: Optional[ttk.Treeview] = None
        self.tree_frame_vscrollbar: Optional[ttk.Scrollbar] = None
        self.tree_frame_hscrollbar: Optional[ttk.Scrollbar] = None
        
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
        return is_dataclass(self.attribute_type)

    @override
    def is_value_compatible(self) -> bool:
        if not self.is_type_compatible():
            return False
        if self.attribute_value is None:
            return True
        return isinstance(self.attribute_value, self.attribute_type)

    @override
    def get_default_value(self, *args, **kwargs) -> Any:
        if not self.is_type_compatible():
            return None
        try:
            return self.attribute_type()
        except:
            return None

    @override
    def get_simplest_type(self, *args, **kwargs) -> Type:
        return self.attribute_type

    @override
    def get_simplest_type_name(self, *args, **kwargs) -> str:
        if not self.is_type_compatible():
            return ""
        return self.attribute_type.__name__

    def _collect_dataclass_subtypes(self):
        from Tools.YamlConfigurer.configurations import Configurations
        self.dataclass_subtypes = []
        self.map_subtype_name = {}
        
        for maintainer_cls in Configurations.maintainer_collection:
            if hasattr(maintainer_cls, '__orig_bases__'):
                for base in maintainer_cls.__orig_bases__:
                    if get_origin(base) is BaseMaintainer:
                        arg = get_args(base)[0]
                        if is_dataclass(arg) and issubclass(arg, self.attribute_type):
                            self.dataclass_subtypes.append(arg)
                            self.map_subtype_name[arg.__name__] = arg

    @override
    def create_inspector(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        if self.inspector is not None:
            self.inspector.destroy()

        self.inspector = ttk.Frame(parent)
        self.inspector.pack(fill=tk.BOTH, expand=True)

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
            self.double_paned_wnd = ttk.PanedWindow(self.inspector, orient=tk.HORIZONTAL)
            self.double_paned_wnd.pack(fill=tk.BOTH, expand=True)

            def on_pane_map(event: Optional[tk.Event] = None):
                w = self.double_paned_wnd.winfo_width()
                self.double_paned_wnd.sashpos(0, w // 2)
                self.double_paned_wnd.unbind("<Map>")

            self.double_paned_wnd.bind("<Map>", on_pane_map)

            self.main_inspector_left_frame = ttk.Frame(self.double_paned_wnd)
            self.double_paned_wnd.add(self.main_inspector_left_frame, weight=1)

            self.item_inspector_right_frame = ttk.Frame(self.double_paned_wnd)
            self.double_paned_wnd.add(self.item_inspector_right_frame, weight=1)

            self._create_attribute_type_display(self.main_inspector_left_frame)

            if not self.can_edit():
                return self.inspector

            self.editor_label_frame = ttk.LabelFrame(self.main_inspector_left_frame, text="Dataclass")
            self.editor_label_frame.pack(
                anchor=tk.N,
                fill=tk.BOTH,
                expand=True,
                padx=(5, 0),
                pady=(5, 0)
            )

            self.item_inspector_label_frame = ttk.LabelFrame(
                self.item_inspector_right_frame,
                text="Inspector"
            )
            self.item_inspector_label_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            self.item_inspector_label_frame.grid_rowconfigure(0, weight=1)
            self.item_inspector_label_frame.grid_rowconfigure(1, weight=0)
            self.item_inspector_label_frame.grid_columnconfigure(0, weight=1)
            self.item_inspector_label_frame.grid_columnconfigure(1, weight=0)

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

            self.item_inspector_canvas.grid(row=0, column=0, sticky=tk.NSEW)
            self.item_inspector_vscrollbar.grid(row=0, column=1, sticky=tk.NS)
            self.item_inspector_hscrollbar.grid(row=1, column=0, sticky=tk.EW)

            self.item_inspector_inner_frame = ttk.Frame(self.item_inspector_canvas, padding=10)
            self.item_inspector_inner_frame_id = self.item_inspector_canvas.create_window(
                (0, 0),
                window=self.item_inspector_inner_frame,
                anchor=tk.NW
            )

            self.item_inspector_inner_frame.bind("<Configure>", self._on_item_inspector_inner_frame_configure)
            self.item_inspector_canvas.bind("<Configure>", self._on_item_inspector_canvas_configure)

            self.editor = self.create_editor(self.editor_label_frame, on_value_change)
            self.editor.pack(fill=tk.BOTH, expand=True)
        else:
            raise NotImplementedError(f"{self.view_mode} is not implemented")

        return self.inspector

    def _create_attribute_type_display(self, parent: ttk.Widget):
        self.attribute_frame = ttk.Frame(parent)
        self.attribute_frame.pack(anchor=tk.W, padx=10, pady=5)
        self.attribute_title_label = ttk.Label(self.attribute_frame, text="Attribute:")
        self.attribute_title_label.pack(side=tk.LEFT)
        self.attribute_content_label = ttk.Label(self.attribute_frame, text=f"{self.attribute_name}")
        self.attribute_content_label.pack(side=tk.LEFT)

        self.type_frame = ttk.Frame(parent)
        self.type_frame.pack(anchor=tk.W, padx=10, pady=5)
        self.type_title_label = ttk.Label(self.type_frame, text="Type:")
        self.type_title_label.pack(side=tk.LEFT)
        self.type_content_label = ttk.Label(self.type_frame, text=f"{self.get_simplest_type_name()}")
        self.type_content_label.pack(side=tk.LEFT)

    def _on_item_inspector_inner_frame_configure(self, event: Optional[tk.Event] = None) -> None:
        self.item_inspector_canvas.configure(scrollregion=self.item_inspector_canvas.bbox("all"))

    def _on_item_inspector_canvas_configure(self, event: Optional[tk.Event] = None) -> None:
        canvas_width: int = event.width - 5
        canvas_height: int = event.height - 5

        min_width: int = self.item_inspector_inner_frame.winfo_reqwidth()
        min_height: int = self.item_inspector_inner_frame.winfo_reqheight()

        actual_width: int = max(canvas_width, min_width)
        actual_height: int = max(canvas_height, min_height)

        self.item_inspector_canvas.itemconfig(
            self.item_inspector_inner_frame_id,
            width=actual_width,
            height=actual_height
        )

        self.item_inspector_canvas.configure(scrollregion=self.item_inspector_canvas.bbox("all"))

    @override
    def create_editor(
            self,
            parent: ttk.Widget,
            on_value_change: Optional[Callable[[Any], None]] = None,
    ) -> ttk.Widget:
        super().create_editor(parent, on_value_change)

        self._collect_dataclass_subtypes()

        self.type_combobox_frame = ttk.Frame(self.editor)
        self.type_combobox_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)

        self.type_string_var = tk.StringVar(value="")
        
        type_names: List[str] = list(self.map_subtype_name.keys())
        if type_names:
            default_type_name = self.attribute_type.__name__
            if default_type_name in type_names:
                self.type_string_var.set(default_type_name)
            else:
                self.type_string_var.set(type_names[0])
        else:
            self.type_string_var.set(self.attribute_type.__name__)

        self.type_label = ttk.Label(self.type_combobox_frame, text="Type:")
        self.type_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.type_combobox = ttk.Combobox(
            self.type_combobox_frame,
            textvariable=self.type_string_var,
            values=type_names if type_names else [self.attribute_type.__name__],
            width=20,
            state="readonly" if self.can_edit() else "disabled"
        )
        self.type_combobox.pack(side=tk.LEFT, padx=5, pady=5)

        def on_type_change(event: Optional[tk.Event] = None):
            selected_type_name: str = self.type_string_var.get()
            if selected_type_name in self.map_subtype_name:
                self.current_dataclass_type = self.map_subtype_name[selected_type_name]
            else:
                self.current_dataclass_type = self.attribute_type
            
            self.editor_value = self.current_dataclass_type()
            self._update_treeview()
            self.on_value_change(copy.deepcopy(self.editor_value))

        self.type_combobox.bind("<<ComboboxSelected>>", on_type_change)

        self.tree_frame = ttk.Frame(self.editor)
        self.tree_frame.pack(anchor=tk.W, padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.list_treeview = ttk.Treeview(self.tree_frame, columns=("name", "type", "value"), show="headings")
        self.list_treeview.heading("name", text="Name")
        self.list_treeview.heading("type", text="Type")
        self.list_treeview.heading("value", text="Value")

        self.list_treeview.column("name", minwidth=80, width=120, stretch=False, anchor=tk.W)
        self.list_treeview.column("type", minwidth=80, width=150, stretch=False, anchor=tk.W)
        self.list_treeview.column("value", minwidth=100, width=200, stretch=False, anchor=tk.W)

        def adjust_tree_columns(event: Optional[tk.Event] = None):
            tree_width = self.list_treeview.winfo_width()
            if tree_width > 0:
                available_width = tree_width - 2
                name_ratio = 0.2
                type_ratio = 0.3
                
                name_width = max(80, int(available_width * name_ratio))
                type_width = max(80, int(available_width * type_ratio))
                value_width = max(100, available_width - name_width - type_width)
                
                total_width = name_width + type_width + value_width
                if total_width > available_width:
                    value_width = available_width - name_width - type_width
                
                self.list_treeview.column("name", minwidth=80, width=name_width, stretch=False, anchor=tk.W)
                self.list_treeview.column("type", minwidth=80, width=type_width, stretch=False, anchor=tk.W)
                self.list_treeview.column("value", minwidth=100, width=value_width, stretch=False, anchor=tk.W)

        self.list_treeview.bind("<Configure>", adjust_tree_columns, add="+")

        self.tree_frame_vscrollbar = ttk.Scrollbar(
            self.tree_frame,
            orient=tk.VERTICAL,
            command=self.list_treeview.yview
        )
        self.list_treeview.configure(yscrollcommand=self.tree_frame_vscrollbar.set)
        self.tree_frame_vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def custom_xview(*args):
            if args[0] == 'scroll':
                if args[2] == 'units':
                    self.list_treeview.xview_scroll(int(args[1]) * 10, 'units')
                else:
                    self.list_treeview.xview_scroll(int(args[1]), 'pages')
            else:
                self.list_treeview.xview_moveto(args[1])

        self.tree_frame_hscrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL, command=custom_xview)
        self.list_treeview.configure(xscrollcommand=self.tree_frame_hscrollbar.set)
        self.tree_frame_hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.list_treeview.pack(fill=tk.BOTH, expand=True)

        if self.current_dataclass_type is None:
            self.current_dataclass_type = self.attribute_type
        
        if self.editor_value is None:
            self.editor_value = self.current_dataclass_type()
        
        self._update_treeview()

        self.current_selected_item = None
        self.list_treeview.bind("<<TreeviewSelect>>", self._on_treeview_select)
        self.list_treeview.bind("<Double-1>", self._on_tree_double_click)

        def on_mouse_wheel(event: Optional[tk.Event] = None):
            shift_pressed = event.state & 0x0001 != 0
            if not shift_pressed:
                self.list_treeview.yview_scroll(-1 * (event.delta // 120), "units")
            else:
                self.list_treeview.xview_scroll(-1 * (event.delta // 10), "units")

        self.list_treeview.bind("<MouseWheel>", on_mouse_wheel)
        self.list_treeview.config(takefocus=True)

        return self.editor

    def _update_treeview(self):
        self.list_treeview.delete(*self.list_treeview.get_children())
        
        if self.editor_value is None:
            return
        
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        
        dataclass_fields = fields(self.editor_value)
        for field in dataclass_fields:
            field_value = getattr(self.editor_value, field.name)
            field_type = field.type
            
            maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(field_type)
            type_name = maintainer_cls.get_simplest_type_name_static(target_type=field_type)
            
            value_repr = repr(field_value)
            if len(value_repr) > 50:
                value_repr = value_repr[:50] + "..."
            
            self.list_treeview.insert("", tk.END, values=(field.name, type_name, value_repr))

    def _on_treeview_select(self, event: Optional[tk.Event] = None):
        selected_items = self.list_treeview.selection()
        if selected_items:
            self.current_selected_item = selected_items[0]
        else:
            self.current_selected_item = None
        
        if self.view_mode == "Standalone":
            self._update_item_inspector()

    def _update_item_inspector(self):
        for widget in self.item_inspector_inner_frame.winfo_children():
            widget.destroy()
        
        if self.current_selected_item is None or self.editor_value is None:
            return
        
        item_values = self.list_treeview.item(self.current_selected_item, "values")
        if not item_values or len(item_values) < 1:
            return
        
        field_name = item_values[0]
        
        dataclass_fields = fields(self.editor_value)
        field_type = None
        field_value = None
        
        for field in dataclass_fields:
            if field.name == field_name:
                field_type = field.type
                field_value = getattr(self.editor_value, field.name)
                break
        
        if field_type is None:
            return
        
        from Tools.YamlConfigurer.maintainer_factory import MaintainerFactory
        
        maintainer_cls = MaintainerFactory.get_maintainer_cls_supported_type(field_type)
        self.item_maintainer = maintainer_cls(field_name, field_type, field_value, self.logger)
        self.item_maintainer.config_view("Packed")
        
        def on_value_change(new_value: Any):
            setattr(self.editor_value, field_name, new_value)
            self._update_treeview()
            self.on_value_change(copy.deepcopy(self.editor_value))
        
        inspector = self.item_maintainer.create_inspector(self.item_inspector_inner_frame, on_value_change)
        inspector.pack(fill=tk.BOTH, expand=True)

    def _on_dataclass_content_change(self, new_value: Any) -> None:
        is_valid, validated_value = self.editor_validate(new_value)
        if is_valid:
            self.editor_value = validated_value
            self.on_value_change(copy.deepcopy(self.editor_value))

    def _on_tree_double_click(self, event: Optional[tk.Event] = None):
        item = self.list_treeview.identify_row(event.y)
        if item:
            self._edit_item()

    def _create_popup_inspector_window(
            self,
            title: str,
            item_attribute_name: str,
            item_attribute_type: Type,
            item_attribute_value: Any
    ):
        self.popup_top_level = tk.Toplevel(self.editor)
        self.popup_top_level.title(title)
        self.popup_top_level.resizable(True, True)
        self.popup_top_level.focus_set()

        self.popup_inspector_frame = ttk.Frame(self.popup_top_level)
        self.popup_inspector_frame.pack(fill=tk.BOTH, expand=True)

        self.popup_inspector_frame.grid_rowconfigure(0, weight=1)
        self.popup_inspector_frame.grid_rowconfigure(1, weight=0)
        self.popup_inspector_frame.grid_columnconfigure(0, weight=1)
        self.popup_inspector_frame.grid_columnconfigure(1, weight=0)

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

        self.popup_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.popup_canvas_vscrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.popup_canvas_hscrollbar.grid(row=1, column=0, sticky=tk.EW)

        self.popup_canvas_inner_frame = ttk.Frame(self.popup_canvas, padding=5)
        self.popup_canvas_inner_frame_id = self.popup_canvas.create_window(
            (0, 0), window=self.popup_canvas_inner_frame, anchor=tk.NW
        )

        self.popup_canvas_inner_frame.bind("<Configure>", self._on_popup_canvas_inner_frame_configure)
        self.popup_canvas.bind("<Configure>", self._on_popup_canvas_configure)

        self.popup_wnd_result = {'value': item_attribute_value, 'confirmed': False}

        def on_popup_editor_value_change(new_val: Any):
            self.popup_wnd_result['value'] = new_val

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

        self.popup_wnd_button_container = ttk.Frame(self.popup_top_level)
        self.popup_wnd_button_container.pack(side=tk.TOP, anchor=tk.CENTER)

        self.popup_confirm_button = ttk.Button(
            self.popup_wnd_button_container,
            text="Confirm",
            command=self._on_popup_confirm
        )
        self.popup_confirm_button.pack(side=tk.LEFT, padx=10, pady=(0, 10))

        self.popup_cancel_button = ttk.Button(
            self.popup_wnd_button_container,
            text="Cancel",
            command=self._on_popup_cancel
        )
        self.popup_cancel_button.pack(side=tk.LEFT, padx=10, pady=(0, 10))

        hotkey_confirm, hotkey_cancel = self.item_maintainer.shall_hotkey_confirm_cancel()
        if hotkey_confirm:
            self.popup_top_level.bind("<Return>", self._on_popup_return_key)
        if hotkey_cancel:
            self.popup_top_level.bind("<Escape>", self._on_popup_escape_key)

        self.popup_top_level.transient(self.editor.winfo_toplevel())
        self.popup_top_level.grab_set()
        self.editor.winfo_toplevel().wait_window(self.popup_top_level)

        return self.popup_wnd_result

    def _on_popup_canvas_inner_frame_configure(self, event: Optional[tk.Event] = None) -> None:
        self.popup_canvas.configure(scrollregion=self.popup_canvas.bbox("all"))

    def _on_popup_canvas_configure(self, event: Optional[tk.Event] = None) -> None:
        canvas_width = event.width - 5
        canvas_height = event.height - 5

        min_width = self.popup_canvas_inner_frame.winfo_reqwidth()
        min_height = self.popup_canvas_inner_frame.winfo_reqheight()

        actual_width = max(canvas_width, min_width)
        actual_height = max(canvas_height, min_height)

        self.popup_canvas.itemconfig(self.popup_canvas_inner_frame_id, width=actual_width, height=actual_height)
        self.popup_canvas.configure(scrollregion=self.popup_canvas.bbox("all"))

    def _on_popup_confirm(self):
        self.popup_wnd_result['confirmed'] = True
        self.popup_top_level.destroy()

    def _on_popup_cancel(self):
        self.popup_wnd_result['confirmed'] = False
        self.popup_top_level.destroy()

    def _on_popup_return_key(self, event: Optional[tk.Event] = None):
        self._on_popup_confirm()
        return "break"

    def _on_popup_escape_key(self, event: Optional[tk.Event] = None):
        self._on_popup_cancel()
        return "break"

    def _edit_item(self):
        if self.current_selected_item is None or self.editor_value is None:
            return

        item_values = self.list_treeview.item(self.current_selected_item, "values")
        if not item_values or len(item_values) < 1:
            return

        field_name = item_values[0]
        field_type = None
        field_value = None

        dataclass_fields = fields(self.editor_value)
        for field in dataclass_fields:
            if field.name == field_name:
                field_type = field.type
                field_value = getattr(self.editor_value, field.name)
                break

        if field_type is None:
            return

        result = self._create_popup_inspector_window(
            title=f"Edit {field_name}",
            item_attribute_name=field_name,
            item_attribute_type=field_type,
            item_attribute_value=field_value
        )

        if result['confirmed']:
            new_value = result['value']
            setattr(self.editor_value, field_name, new_value)
            self._update_treeview()
            self._on_dataclass_content_change(self.editor_value)

    @override
    def editor_enable(self):
        if self.editor is not None and self.type_combobox is not None:
            self.type_combobox.config(state="readonly")
            if self.current_maintainer is not None:
                self.current_maintainer.editor_enable()
        super().editor_enable()

    @override
    def editor_disable(self):
        if self.editor is not None and self.type_combobox is not None:
            self.type_combobox.config(state="disabled")
            if self.current_maintainer is not None:
                self.current_maintainer.editor_disable()
        super().editor_disable()

    @override
    def editor_set_value(self, new_value: Any):
        if self.editor is not None:
            self.editor_value = new_value
            self._update_treeview()
        super().editor_set_value(new_value)

    @override
    def config_view(self, view_mode: Literal["Standalone", "Packed"], *args, **kwargs):
        self.view_mode = view_mode

    @override
    def editor_validate(self, input_value: Any) -> Tuple[bool, Any]:
        return True, input_value

    @staticmethod
    @override
    def is_type_compatible_static(target_type: Type) -> bool:
        return is_dataclass(target_type)

    @staticmethod
    @override
    def is_value_compatible_static(value: Any, target_type: Type = Any) -> bool:
        if not DataclassMaintainer.is_type_compatible_static(target_type):
            return False
        if value is None:
            return True
        return isinstance(value, target_type)

    @staticmethod
    @override
    def get_default_value_static(target_type: Type, *args, **kwargs) -> Any:
        if not DataclassMaintainer.is_type_compatible_static(target_type):
            return None
        try:
            return target_type()
        except:
            return None

    @staticmethod
    @override
    def get_simplest_type_static(target_type: Type, *args, **kwargs) -> Type:
        return target_type

    @staticmethod
    @override
    def get_simplest_type_name_static(target_type: Type, *args, **kwargs) -> str:
        if not DataclassMaintainer.is_type_compatible_static(target_type):
            return ""
        return target_type.__name__