# -*- coding: utf-8 -*-
import os
import sys
import importlib
import inspect
import subprocess
import threading
import logging
from pathlib import Path
from typing import Dict, Type, List, Tuple, Any, Optional
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from Launcher.Parser.parser_ABC import ParserABC
from Launcher.launcher_ABC import LauncherABC


class TkinterLogHandler(logging.Handler):
    """Custom logging handler that outputs to Tkinter Text widget"""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    
    def emit(self, record):
        """Emit a log record to the text widget"""
        try:
            msg = self.format(record)
            
            # Check if message starts with task prefix
            if msg.startswith("Task "):
                # Find the colon after task ID
                colon_pos = msg.find(":")
                if colon_pos != -1:
                    # Insert task prefix in bold yellow
                    self.text_widget.config(state=tk.NORMAL)
                    self.text_widget.insert(tk.END, msg[:colon_pos + 1], "task")
                    
                    # Check if the message contains error keywords
                    content = msg[colon_pos + 1:]
                    
                    # Check if the content contains error keywords
                    error_keywords = ['error', 'exception', 'failed', 'failure']
                    has_error_keyword = any(keyword in content.lower() for keyword in error_keywords)
                    
                    if has_error_keyword:
                        # If content contains error keywords, highlight individual keywords
                        self._insert_with_error_highlight(content)
                    else:
                        # Otherwise, insert as normal
                        self.text_widget.insert(tk.END, content + "\n", "normal")
                    
                    self.text_widget.config(state=tk.DISABLED)
                    self.text_widget.see(tk.END)
                else:
                    self.text_widget.config(state=tk.NORMAL)
                    self.text_widget.insert(tk.END, msg + "\n", "task")
                    self.text_widget.config(state=tk.DISABLED)
                    self.text_widget.see(tk.END)
            else:
                # Check if it's an error/exception/failure message
                error_keywords = ['error', 'exception', 'failed', 'failure']
                has_error_keyword = any(keyword in msg.lower() for keyword in error_keywords)
                
                if has_error_keyword:
                    # If message contains error keywords, display entire message in red bold
                    self.text_widget.config(state=tk.NORMAL)
                    self.text_widget.insert(tk.END, msg + "\n", "error")
                    self.text_widget.config(state=tk.DISABLED)
                    self.text_widget.see(tk.END)
                else:
                    # Otherwise, display as normal
                    self.text_widget.config(state=tk.NORMAL)
                    self.text_widget.insert(tk.END, msg + "\n", "normal")
                    self.text_widget.config(state=tk.DISABLED)
                    self.text_widget.see(tk.END)
                    
        except Exception:
            self.handleError(record)
    
    def _insert_with_error_highlight(self, text):
        """Insert text with error keywords highlighted in red bold"""
        error_keywords = ['error', 'exception', 'failed', 'failure']
        start = 0
        
        while start < len(text):
            # Find the next error keyword
            next_keyword_pos = len(text)
            next_keyword = None
            
            for keyword in error_keywords:
                pos = text.lower().find(keyword, start)
                if pos != -1 and pos < next_keyword_pos:
                    next_keyword_pos = pos
                    next_keyword = keyword
            
            if next_keyword is None:
                # No more keywords, insert remaining text
                self.text_widget.insert(tk.END, text[start:] + "\n", "normal")
                break
            else:
                # Insert text before the keyword
                if next_keyword_pos > start:
                    self.text_widget.insert(tk.END, text[start:next_keyword_pos], "normal")
                
                # Insert the keyword in red bold
                keyword_length = len(next_keyword)
                self.text_widget.insert(tk.END, text[next_keyword_pos:next_keyword_pos + keyword_length], "error")
                
                start = next_keyword_pos + keyword_length


def import_class(module_path: str, class_name: str) -> Type:
    """Dynamically import a class from a module"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def scan_parser_classes() -> List[Tuple[str, str, Type[ParserABC]]]:
    """Scan all ParserABC subclasses in Launcher.Parser"""
    parser_classes = []
    parser_dir = Path(__file__).parent / 'Parser'
    
    for file in parser_dir.glob('parser_*.py'):
        if file.name == 'parser_ABC.py':
            continue
        
        module_name = f'Launcher.Parser.{file.stem}'
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, ParserABC) and 
                    obj != ParserABC):
                    parser_classes.append((module_name, name, obj))
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
    
    return parser_classes


def scan_launcher_classes() -> List[Tuple[str, str, Type[LauncherABC]]]:
    """Scan all LauncherABC subclasses in Launcher"""
    launcher_classes = []
    launcher_dir = Path(__file__).parent
    
    for file in launcher_dir.glob('launcher_*.py'):
        if file.name == 'launcher_ABC.py' or file.name == 'launcher_cmd.py' or file.name == 'launcher_cli.py':
            continue
        
        module_name = f'Launcher.{file.stem}'
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, LauncherABC) and 
                    obj != LauncherABC):
                    launcher_classes.append((module_name, name, obj))
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
    
    return launcher_classes


def get_launcher_routines(launcher_class: Type[LauncherABC]) -> List[str]:
    """Get callable public methods (excluding magic methods) from launcher class"""
    routines = []
    for name, method in inspect.getmembers(launcher_class, inspect.isfunction):
        if (not name.startswith('__') and 
            name != 'fit' and name != 'finetune' and 
            name != 'validation' and name != 'test' and 
            name != 'predict'):
            routines.append(name)
    
    # Add standard routines from LauncherABC
    standard_routines = ['fit', 'finetune', 'validation', 'test', 'predict']
    for routine in standard_routines:
        if hasattr(launcher_class, routine):
            routines.append(routine)
    
    return sorted(routines)


class LauncherCLI:
    def __init__(self, root):
        self.root = root
        self.root.title("PipeUNet Launcher CLI")
        self.root.geometry("700x450")
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Scan classes
        self.parser_classes = scan_parser_classes()
        self.launcher_classes = scan_launcher_classes()
        
        # Store selected classes
        self.selected_parser = None
        self.selected_launcher = None
        
        # Task ID counter
        self.task_counter = 0
        
        # Create GUI components
        self.create_widgets()
    
    def apply_dark_theme(self):
        """Apply medium gray theme to the application"""
        # Configure root window
        self.root.configure(bg="#d1d5db")
        
        # Configure ttk style with system theme
        style = ttk.Style()
        # Use system default theme instead of hardcoding
        
        # Medium gray theme colors
        bg_color = "#d1d5db"          # Medium gray background
        card_bg = "#e5e7eb"           # Light gray card background
        surface_bg = "#f3f4f6"        # Very light gray surface
        fg_color = "#111827"          # Dark text
        accent_color = "#9ca3af"      # Light gray button background
        button_text = "#000000"       # Black text for buttons
        success_color = "#10b981"      # Emerald green for success
        warning_color = "#f59e0b"      # Amber for warning
        error_color = "#ef4444"        # Red for error
        
        # Configure styles with modern design
        style.configure("TFrame", background=bg_color)
        
        # Labels with modern font
        style.configure("TLabel", background=bg_color, foreground=fg_color, 
                       font=("Segoe UI", 10, "bold"))
        
        # Entries with modern design
        style.configure("TEntry", fieldbackground=card_bg, foreground=fg_color, 
                       background=card_bg, font=("Segoe UI", 9), padding=8, 
                       borderwidth=1, relief="flat")
        
        # Buttons with medium gray style - light gray background, black text
        style.configure("TButton", background=accent_color, foreground=button_text, 
                       font=("Segoe UI", 9, "bold"), padding=(12, 8), borderwidth=0,
                       relief="flat")
        style.map("TButton",
                 background=[("active", "#6b7280"), ("pressed", "#4b5563"), 
                            ("disabled", "#9ca3af")],
                 foreground=[("active", button_text), ("pressed", button_text), 
                            ("disabled", "#6b7280")])
        
        # Combobox with modern design
        style.configure("TCombobox", fieldbackground=card_bg, foreground="#000000", 
                       background=card_bg, font=("Segoe UI", 9), padding=8, 
                       borderwidth=1, relief="flat")
        style.map("TCombobox",
                 fieldbackground=[("readonly", card_bg), ("active", card_bg)],
                 background=[("readonly", card_bg), ("active", card_bg)],
                 foreground=[("readonly", "#000000"), ("active", "#000000")])
        
        # Configure combobox dropdown menu style
        style.configure("TComboboxPopdownFrame", background=card_bg)
        style.configure("TComboboxListbox", background=card_bg, foreground="#000000",
                       font=("Segoe UI", 9))
        style.map("TComboboxListboxItem",
                 background=[("selected", accent_color)],
                 foreground=[("selected", "#000000")])
        
        # Configure Treeview styles with modern design
        style.configure("Treeview", background=surface_bg, foreground="#000000", 
                       fieldbackground=surface_bg, font=("Segoe UI", 9),
                       borderwidth=1, relief="flat")
        style.configure("Treeview.Heading", background=card_bg, foreground="#000000",
                       font=("Segoe UI", 9, "bold"), padding=8)
        style.map("Treeview",
                 background=[("selected", accent_color)],
                 foreground=[("selected", "#000000")])
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Split main frame into two parts: left (controls) and right (task list)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, rowspan=9, padx=(0, 10), sticky=tk.NSEW)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, rowspan=9, sticky=tk.NSEW)
        
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Configure left frame grid
        left_frame.grid_columnconfigure(1, weight=1)
        
        # Config File Selection
        ttk.Label(left_frame, text="Config File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.config_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.config_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(left_frame, text="Browse", command=self.browse_config).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.config_var.trace_add('write', lambda *args: self.validate_run_button())
        
        # Parser Selection
        ttk.Label(left_frame, text="Parser:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.parser_var = tk.StringVar()
        self.parser_combo = ttk.Combobox(left_frame, textvariable=self.parser_var, state="readonly")
        parser_options = [f"{name} ({module}.{name})" for module, name, _ in self.parser_classes]
        self.parser_combo['values'] = parser_options
        self.parser_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.parser_combo.bind("<<ComboboxSelected>>", lambda e: (self.on_parser_selected(e), self._deselect_combobox_text(self.parser_combo)))
        
        # Export Example YAML Button
        self.export_btn = ttk.Button(left_frame, text="Export Example YAML", command=self.export_example_yaml, state=tk.DISABLED)
        self.export_btn.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Launcher Selection
        ttk.Label(left_frame, text="Launcher:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.launcher_var = tk.StringVar()
        self.launcher_combo = ttk.Combobox(left_frame, textvariable=self.launcher_var, state="readonly")
        launcher_options = [f"{name} ({module}.{name})" for module, name, _ in self.launcher_classes]
        self.launcher_combo['values'] = launcher_options
        self.launcher_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.launcher_combo.bind("<<ComboboxSelected>>", lambda e: (self.on_launcher_selected(e), self._deselect_combobox_text(self.launcher_combo)))
        
        # Routine Selection
        ttk.Label(left_frame, text="Routine:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.routine_var = tk.StringVar()
        self.routine_combo = ttk.Combobox(left_frame, textvariable=self.routine_var, state="readonly")
        self.routine_combo.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        self.routine_combo.bind("<<ComboboxSelected>>", lambda e: self._deselect_combobox_text(self.routine_combo))
        self.routine_var.trace_add('write', lambda *args: self.validate_run_button())
        
        # Checkpoint Selection
        ttk.Label(left_frame, text="Checkpoint:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.checkpoint_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.checkpoint_var).grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(left_frame, text="Browse", command=self.browse_checkpoint).grid(row=4, column=2, padx=5, pady=5, sticky=tk.W)
        self.checkpoint_var.trace_add('write', lambda *args: self.validate_run_button())
        
        # Run Log Selection
        ttk.Label(left_frame, text="Run Log:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.run_log_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.run_log_var).grid(row=5, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(left_frame, text="Browse", command=self.browse_run_log).grid(row=5, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Run Button
        self.run_btn = ttk.Button(left_frame, text="Run", command=self.run_launcher, state=tk.DISABLED)
        self.run_btn.grid(row=6, column=0, columnspan=3, pady=20, sticky=tk.EW)
        
        # Log Text Area
        ttk.Label(left_frame, text="Log:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.log_text = tk.Text(left_frame, wrap=tk.WORD, state=tk.DISABLED, 
                              bg="#1e293b", fg="#ffffff", insertbackground="#3b82f6",
                              font=("Segoe UI", 9))
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky=tk.NSEW)
        scrollbar.grid(row=8, column=3, sticky=tk.NS)
        left_frame.grid_rowconfigure(8, weight=1)
        
        # Configure text tags for styling with dark log background
        self.log_text.tag_configure("task", foreground="#f59e0b", font=("Segoe UI", 9, "bold"))
        self.log_text.tag_configure("normal", foreground="#ffffff", font=("Segoe UI", 9))
        self.log_text.tag_configure("error", foreground="#ef4444", font=("Segoe UI", 9, "bold"))
        
        # Configure logging
        self.setup_logging()
        
        # Task List Table on the right
        ttk.Label(right_frame, text="Task Status", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, pady=5)
        
        # Create treeview for task list
        self.task_tree = ttk.Treeview(right_frame, columns=("task", "experiment", "status"), show="headings")
        
        # Configure columns with width adjustment
        self.task_tree.heading("task", text="Task")
        self.task_tree.heading("experiment", text="Experiment")
        self.task_tree.heading("status", text="Status")
        
        # Configure columns with width adjustment for adaptive layout
        self.task_tree.column("task", width=50, anchor=tk.CENTER, stretch=tk.NO)
        self.task_tree.column("experiment", width=200, anchor=tk.W, stretch=tk.YES)
        self.task_tree.column("status", width=80, anchor=tk.CENTER, stretch=tk.NO)
        

        
        # Add scrollbar
        task_scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=task_scrollbar.set)
        
        self.task_tree.grid(row=1, column=0, sticky=tk.NSEW)
        task_scrollbar.grid(row=1, column=1, sticky=tk.NS)
        
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Store task information
        self.tasks = {}
        
        # Configure tags for status colors
        self.task_tree.tag_configure("running", foreground="#f59e0b")
        self.task_tree.tag_configure("failed", foreground="#ef4444")
        self.task_tree.tag_configure("finished", foreground="#10b981")
    
    def browse_config(self):
        # Set initial directory for config files
        config_dir = Path(os.getcwd()) / 'Launcher' / 'Configs'
        initial_dir = str(config_dir) if config_dir.exists() else os.getcwd()
        
        file_path = filedialog.askopenfilename(
            title="Select Config File",
            initialdir=initial_dir,
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if file_path:
            self.config_var.set(file_path)
    
    def browse_checkpoint(self):
        file_path = filedialog.askopenfilename(
            title="Select Checkpoint File",
            initialdir=os.getcwd(),
            filetypes=[("Checkpoint files", "*.pt"), ("Checkpoint files", "*.pth"), ("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
        )
        if file_path:
            self.checkpoint_var.set(file_path)
    
    def browse_run_log(self):
        file_path = filedialog.asksaveasfilename(
            title="Select Run Log File",
            initialdir=os.getcwd(),
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.run_log_var.set(file_path)
    
    def on_parser_selected(self, event):
        parser_display = self.parser_var.get()
        if parser_display:
            # Extract class path from display string (format: "ClassName (Module.ClassName)")
            class_path = parser_display.split('(')[1].rstrip(')')
            module_name, class_name = class_path.rsplit('.', 1)
            
            for mod_name, cls_name, cls in self.parser_classes:
                if mod_name == module_name and cls_name == class_name:
                    self.selected_parser = cls
                    # Check if parser has dump_example_to_yaml method
                    if hasattr(cls, 'dump_example_to_yaml'):
                        self.export_btn.config(state=tk.NORMAL)
                    else:
                        self.export_btn.config(state=tk.DISABLED)
                    break
        self.validate_run_button()
    
    def on_launcher_selected(self, event):
        launcher_display = self.launcher_var.get()
        if launcher_display:
            # Extract class path from display string (format: "ClassName (Module.ClassName)")
            class_path = launcher_display.split('(')[1].rstrip(')')
            module_name, class_name = class_path.rsplit('.', 1)
            
            for mod_name, cls_name, cls in self.launcher_classes:
                if mod_name == module_name and cls_name == class_name:
                    self.selected_launcher = cls
                    # Update routine options
                    routines = get_launcher_routines(cls)
                    self.routine_combo['values'] = routines
                    if routines:
                        self.routine_var.set(routines[0])
                    break
        self.validate_run_button()
    
    def export_example_yaml(self):
        if self.selected_parser:
            # Set initial directory based on parser type
            parser_dir = Path(os.getcwd()) / 'Launcher' / 'Parser'
            initial_dir = str(parser_dir) if parser_dir.exists() else os.getcwd()
            
            file_path = filedialog.asksaveasfilename(
                title="Save Example YAML",
                initialdir=initial_dir,
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    self.selected_parser.dump_example_to_yaml(file_path)
                    self.log_message(f"Example YAML exported to: {file_path}")
                except Exception as e:
                    self.log_message(f"Failed to export example YAML: {str(e)}")
    
    def validate_run_button(self):
        """Validate all required parameters and enable/disable Run button"""
        config_path = self.config_var.get()
        parser_display = self.parser_var.get()
        launcher_display = self.launcher_var.get()
        routine = self.routine_var.get()
        checkpoint = self.checkpoint_var.get()
        
        # Check if all required fields are filled
        if not (config_path and parser_display and launcher_display and routine):
            self.run_btn.config(state=tk.DISABLED)
            return
        
        # Check if config file exists
        if not os.path.isfile(config_path):
            self.run_btn.config(state=tk.DISABLED)
            return
        
        # Check if checkpoint is required for the selected routine
        if self.selected_launcher and routine:
            # Get the routine method
            try:
                routine_method = getattr(self.selected_launcher, routine)
                # Check if checkpoint is a required parameter (no default value)
                import inspect
                sig = inspect.signature(routine_method)
                params = list(sig.parameters.values())
                
                # Check if checkpoint parameter exists and has no default value
                for param in params:
                    if param.name == 'checkpoint':
                        if param.default is inspect.Parameter.empty:
                            # checkpoint is required
                            if not checkpoint or not os.path.isfile(checkpoint):
                                self.run_btn.config(state=tk.DISABLED)
                                return
                        break
            except (AttributeError, ValueError):
                # Method not found or error getting signature
                self.run_btn.config(state=tk.DISABLED)
                return
        
        # All validation passed
        self.run_btn.config(state=tk.NORMAL)
    
    def setup_logging(self):
        """Configure logging to output to the text widget"""
        # Create logger
        self.logger = logging.getLogger('LauncherCLI')
        self.logger.setLevel(logging.INFO)
        
        # Create and configure custom handler
        handler = TkinterLogHandler(self.log_text)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
    
    def log_message(self, message):
        """Add message to log text area using logging"""
        self.logger.info(message)
    
    def run_launcher(self):
        config_path = self.config_var.get()
        parser_display = self.parser_var.get()
        launcher_display = self.launcher_var.get()
        routine = self.routine_var.get()
        checkpoint = self.checkpoint_var.get()
        run_log_path = self.run_log_var.get()
        
        # Extract class paths from display strings
        parser_class_path = parser_display.split('(')[1].rstrip(')')
        launcher_class_path = launcher_display.split('(')[1].rstrip(')')
        
        # Get experiment information from launcher instance
        experiment_info = self.get_experiment_info(launcher_class_path, config_path)
        experiment_name = experiment_info.get('experiment_name', 'Unknown')
        experiment_version = experiment_info.get('experiment_version', '0')
        experiment_id = f"{experiment_name}-{experiment_version}"
        
        # Assign task ID
        self.task_counter += 1
        task_id = self.task_counter
        
        # Add task to the table
        self.add_task_to_table(task_id, experiment_id, "Running")
        
        self.log_message(f"Task {task_id}: Running with config: {config_path}")
        self.log_message(f"Task {task_id}: Parser: {parser_class_path}")
        self.log_message(f"Task {task_id}: Launcher: {launcher_class_path}")
        self.log_message(f"Task {task_id}: Routine: {routine}")
        if checkpoint:
            self.log_message(f"Task {task_id}: Checkpoint: {checkpoint}")
        if run_log_path:
            self.log_message(f"Task {task_id}: Run log will be saved to: {run_log_path}")
        
        # Build command to run launcher_cmd.py
        cmd = [
            sys.executable, "Launcher/launcher_cmd.py",
            "-c", config_path,
            "-p", parser_class_path,
            "-u", launcher_class_path,
            "-r", routine
        ]
        
        if checkpoint:
            cmd.extend(["-ckpt", checkpoint])
        
        self.log_message(f"Task {task_id}: Command: {' '.join(cmd)}")
        self.log_message(f"Task {task_id}: Started")
        
        # Run command in a separate thread (non-blocking)
        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, cmd, run_log_path)
        )
        thread.daemon = True
        thread.start()
    
    def get_experiment_info(self, launcher_class_path, config_path):
        """Get experiment_name and experiment_version from launcher instance"""
        try:
            # Parse the config file
            module_name, class_name = launcher_class_path.rsplit('.', 1)
            launcher_class = import_class(module_name, class_name)
            
            # Find the parser class
            parser_class = None
            for module, name, cls in self.parser_classes:
                if f"{module}.{name}" in self.parser_var.get():
                    parser_class = cls
                    break
            
            if parser_class:
                # Create parser instance and load config
                parser = parser_class()
                parser.from_yaml(config_path)
                launcher_args = parser.to_dict()
                
                # Try to get experiment info from launcher_args first
                experiment_info = {}
                
                # Check if experiment_name is in launcher_args
                if 'experiment_name' in launcher_args:
                    experiment_info['experiment_name'] = launcher_args['experiment_name']
                # Check if experiment_version is in launcher_args
                if 'experiment_version' in launcher_args:
                    experiment_info['experiment_version'] = launcher_args['experiment_version']
                
                # If experiment_info is empty, return <UND>
                if not experiment_info:
                    return {'experiment_name': '<UND>', 'experiment_version': ''}
                
                return experiment_info
        except Exception as e:
            self.log_message(f"Error getting experiment info: {str(e)}")
        
        return {'experiment_name': '<UND>', 'experiment_version': ''}
    
    def add_task_to_table(self, task_id, experiment_id, status):
        """Add task to the task table"""
        # Determine tag and display text based on status
        if status == "Running":
            tag = "running"
            display_status = "· Running"
        elif status == "Failed":
            tag = "failed"
            display_status = "· Failed"
        else:
            tag = "finished"
            display_status = "· Completed"
        
        # Add task to treeview
        item_id = self.task_tree.insert("", tk.END, values=(task_id, experiment_id, display_status), tags=(tag,))
        
        # Store task information
        self.tasks[task_id] = {
            'item_id': item_id,
            'experiment_id': experiment_id,
            'status': status
        }
    
    def update_task_status(self, task_id, status):
        """Update task status in the table"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            item_id = task_info['item_id']
            experiment_id = task_info['experiment_id']
            
            # Determine tag and display text based on status
            if status == "Running":
                tag = "running"
                display_status = "· Running"
            elif status == "Failed":
                tag = "failed"
                display_status = "· Failed"
            else:
                tag = "finished"
                display_status = "· Completed"
            
            # Update treeview item
            self.task_tree.item(item_id, values=(task_id, experiment_id, display_status), tags=(tag,))
            
            # Update stored information
            self.tasks[task_id]['status'] = status
    
    def _deselect_combobox_text(self, combobox):
        """Deselect text in combobox after selection"""
        try:
            # Use after to ensure the text is deselected after selection
            combobox.after(10, lambda: combobox.selection_clear())
        except:
            pass
    
    def _run_task(self, task_id: int, cmd: List[str], run_log_path: str):
        """Run task in a separate thread"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Open log file if specified
            log_file = None
            if run_log_path:
                log_file = open(run_log_path, 'w', encoding='utf-8')
            
            # Read output line by line
            for line in process.stdout:
                # Only write to log file, don't display in GUI
                if log_file:
                    log_file.write(line)
                    log_file.flush()
            
            # Wait for process to complete
            process.wait()
            
            # Close log file
            if log_file:
                log_file.close()
            
            # Update GUI with result (must run in main thread)
            if process.returncode == 0:
                self.root.after(0, lambda: self.log_message(f"Task {task_id}: Completed successfully with return code: {process.returncode}"))
                self.root.after(0, lambda: self.update_task_status(task_id, "Finished"))
            else:
                self.root.after(0, lambda: self.log_message(f"Task {task_id}: Failed with return code: {process.returncode}"))
                self.root.after(0, lambda: self.update_task_status(task_id, "Failed"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Task {task_id}: Error: {str(e)}"))
            self.root.after(0, lambda: self.update_task_status(task_id, "Failed"))


def main():
    root = tk.Tk()
    app = LauncherCLI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
