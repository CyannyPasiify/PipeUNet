# -*- coding: utf-8 -*-
import importlib
from pprint import pprint
from typing import cast, Dict, Any, Type, Tuple, Callable
from Launcher.Parser.parser_ABC import ParserABC
from Launcher.launcher_ABC import LauncherABC


def import_class(module_path: str, class_name: str) -> Type:
    """Dynamically import a class from a module"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def parse_class_path(class_path: str) -> Tuple[str, str]:
    """Parse class path to module path and class name"""
    parts = class_path.split('.')
    module_path = '.'.join(parts[:-1])
    class_name = parts[-1]
    return module_path, class_name


def main() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description='Dynamic launcher for PipeUNet')

    parser.add_argument('-c', '--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('-p', '--parser', type=str, required=True,
                        help='Parser class path (e.g., Launcher.Parser.parser_segmentation_default.ParserSegmentationDefault)')
    parser.add_argument('-u', '--launcher', type=str, required=True,
                        help='Launcher class path (e.g., Launcher.launcher_segmentation_default.LauncherSegmentationDefault)')
    parser.add_argument('-r', '--routine', type=str, required=True, help='Routine method name to execute')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help='Checkpoint file path')

    args = parser.parse_args()

    # Parse parser class path
    parser_module_path: str
    parser_class_name: str
    parser_module_path, parser_class_name = parse_class_path(args.parser)
    print(f'Parser module fetched: {parser_module_path}.{parser_class_name}')

    # Parse launcher class path
    launcher_module_path: str
    launcher_class_name: str
    launcher_module_path, launcher_class_name = parse_class_path(args.launcher)
    print(f'Launcher module fetched: {launcher_module_path}.{launcher_class_name}')

    # Import parser class
    parser_class: Type[ParserABC] = cast(Type[ParserABC], import_class(parser_module_path, parser_class_name))

    # Import launcher class
    launcher_class: Type[LauncherABC] = cast(Type[LauncherABC], import_class(launcher_module_path, launcher_class_name))

    # Parse YAML configuration
    parser_instance: ParserABC = parser_class()
    parser_instance.from_yaml(args.config)
    parsed_kwarg_dict: Dict[str, Any] = parser_instance.to_dict()
    print(f'YAML config parsed with keys:')
    pprint(list(parsed_kwarg_dict.keys()))

    # Convert to dictionary for launcher
    launcher_kwargs: Dict[str, Any] = parsed_kwarg_dict
    print(f'Launcher initialized with args:')
    pprint(launcher_kwargs, sort_dicts=False)

    # Instantiate launcher
    launcher_instance: LauncherABC = launcher_class(**launcher_kwargs)

    # Get the routine method
    routine_method: Callable = getattr(launcher_instance, args.routine)

    try:
        # Execute routine with checkpoint
        result: Any = routine_method(checkpoint=args.checkpoint)

        print(f"Routine '{args.routine}' executed successfully")
        return result
    except Exception as e:
        print(f"Error executing routine '{args.routine}': {str(e)}")
        raise


if __name__ == "__main__":
    main()
