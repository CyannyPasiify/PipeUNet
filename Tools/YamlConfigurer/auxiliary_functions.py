from dataclasses import dataclass
from typing import Union, Optional, List, Type, Tuple, Dict, Set, Any, Iterable, Collection, \
    get_origin, get_args
import sys

"""
Wrapper Types:
    Type
    · Type
    · Type[T]
    · Type[Union[T1, T2, ...]]
    Union
    · Union = Union[Any]
    · Union[T1, T2, ...]
    · Union[T1, T2, ... (Contains None)]
    · Union[T1, T2, ... (Contains Type[*])]
    Optional
    · Optional = Optional[Any]
    · Optional[T]
    · Optional[Union[T1, T2, ...]]
"""

test_types: List[Type] = [
    None,
    Type,
    Type[Any],
    type(None),
    Type[int],
    Type[Union[int, float, str]],
    Union,
    Union[int, float, str],
    Union[Any, float, str],
    Union[None, int, float, str],
    Union[Type[int], Type[Union[float, str]]],
    Optional,
    Optional[Any],
    Optional[int],
    Optional[Union[int]],
    Optional[Union[int, str]],
    Optional[Optional[int]]
]


@dataclass
class TestClass:
    m0: None = None
    m1: Type = int
    m2: Type[Any] = int
    m3: Union[None] = None
    m4: Type[int] = 1
    m5: Type[Union[int, float, str]] = 1
    m6: Union = 1
    m7: Union[int, float, str] = 1
    m8: Union[Any, float, str] = 1
    m9: Union[None, int, float, str] = 1
    m10: Union[Type[int], Type[Union[float, None]], str, None, int] = int
    m11: Optional[None] = 1
    m12: Optional[Any] = 1
    m13: Optional[int] = 1
    m14: Optional[Union[int]] = 1
    m15: Optional[Union[int, str]] = 1
    m16: Optional[Optional[int]] = 1
    m17: Optional[Union[List[Union[int, str]], str, Tuple[Union[Type[int], Type[Union[float, str]]]]]] = 1
    m18: Optional[Union[Iterable[Union[int, str]], str, Collection[Union[Type[int], Type[Union[float, str]]]]]] = 1
    m19: Optional[Type] = 1


def simplify_type(target_type: Type) -> Any:
    """Simplify type by removing duplicates and handling nested types

    Args:
        target_type: Type to simplify

    Returns:
        Simplified type
    """
    if target_type in {None, type(None)}:
        return type(None)

    if target_type in {Union, Optional}:
        return Any

    if target_type == Type:
        return Type[Any]

    if target_type == type:
        return type

    map_to_typing: Dict[Type, Type[Union[List, Tuple, Set, Dict]]] = {
        list: List,
        tuple: Tuple,
        set: Set,
        dict: Dict
    }
    if target_type in map_to_typing:
        if target_type == dict:
            return map_to_typing[target_type][Any, Any]
        if target_type == tuple:
            return map_to_typing[target_type][Any, ...]
        return map_to_typing[target_type][Any]

    # Optional will be Union, no bother processing Optional
    origin: Type = get_origin(target_type)
    type_args: Tuple[Any, ...] = get_args(target_type)

    # Union, Union[T], Union[T, ...], Union[None, ...], Union[Type, ...]
    if origin is Union:
        # Simplify types and remove duplicates
        unique_args: List = []
        seen: Set = set()
        for arg in type_args:
            # There shan't be origin only types, and no Union|Optional types with [Any] cast
            simplified_arg = simplify_type(arg)
            if simplified_arg not in seen:
                seen.add(simplified_arg)
                unique_args.append(simplified_arg)

        # Extract subtypes from Union elems
        reserved_args: List = []
        seen_non_type: Set = set()
        reserved_type_args: List = []
        seen_type: Set = set()
        for union_elem in unique_args:
            union_elem_origin: Type = get_origin(union_elem)
            # Extract types in Union
            if union_elem_origin is Union:
                # Format: Union[T, ...]
                union_args: Tuple[Any, ...] = get_args(union_elem)  # Assume args are all simplest
                assert len(union_args) > 1, f"{union_elem} must have >2 arguments"
                for arg in union_args:
                    # Insert unseen types
                    if arg not in seen_non_type:
                        seen_non_type.add(arg)
                        reserved_args.append(arg)
            elif union_elem_origin in {type, Type}:
                # Type[Union[T, ...]], Type[T], Type[Any]
                # Union[T, ...], T, Any
                type_args: Tuple[Any, ...] = get_args(union_elem)  # Assume args are all simplest
                assert len(type_args) == 1, f"{union_elem} must have only 1 argument"
                # Union[T, ...], T, Any
                only_arg: Type = type_args[0]
                # Union, T, Any
                only_arg_origin: Type = get_origin(only_arg)
                # Union[T, ...]
                if only_arg_origin is Union:
                    # T, ...
                    only_arg_args: Tuple[Any, ...] = get_args(only_arg)
                    for arg in only_arg_args:
                        # Insert unseen types
                        if arg not in seen_type:
                            seen_type.add(arg)
                            reserved_type_args.append(arg)
                # T, Any
                else:
                    if only_arg not in seen_type:
                        seen_type.add(only_arg)
                        reserved_type_args.append(only_arg)
            else:  # Non-wrapper type
                if union_elem not in seen_non_type:
                    seen_non_type.add(union_elem)
                    reserved_args.append(union_elem)

        if len(reserved_type_args) > 0:
            reserved_args.append(Type[Union[tuple(reserved_type_args)]])

        if len(reserved_args) == 0 or Any in reserved_args:
            return Any

        if len(reserved_args) == 1:
            return reserved_args[0]

        if type(None) in reserved_args and len(reserved_args) == 2:
            no_none_reserved_arg: Type = reserved_args[0] if reserved_args[1] is type(None) else reserved_args[1]
            return Optional[no_none_reserved_arg]

        return Union[tuple(reserved_args)]

    # Type[T], origin=Type
    elif origin in {type, Type}:
        assert len(type_args) == 1, f"{target_type} must have 1 argument"
        # T can be Type[T], Union[T]
        only_arg: Type = simplify_type(type_args[0])
        only_arg_origin: Type = get_origin(only_arg)
        # Type[T], only_arg_origin=Type
        if only_arg_origin in {type, Type}:
            return Type[type]
        # Union[T], only_arg_origin=Union
        elif only_arg_origin is Union:
            # Format: Union[T, ..., Type[T]]
            # Type arg in Union shall be reduced to type
            union_args: Tuple[Any, ...] = get_args(only_arg)  # Assume args are all simplest
            reduced_union_args: List[Any] = [type if get_origin(t) in {type, Type} else t for t in union_args]
            return Type[Union[tuple(reduced_union_args)]]

        return Type[only_arg]

    elif origin in [list, tuple, set, dict]:  # May be python Container types, such as list[int]
        origin_type: Type[Union[List, Tuple, Set, Dict]] = map_to_typing[origin] if origin in map_to_typing else origin

        if len(type_args) == 1 and type_args[0] == ():
            return origin_type[()]

        # Simply simplify all args
        sim_args: List = [simplify_type(t) for t in type_args]
        return origin_type[tuple(sim_args)]

    elif origin is not None:  # May be typing Container types, such as List[int]
        try:
            if len(type_args) == 1 and type_args[0] == ():
                return origin[()]
            # Simply simplify all args
            sim_args: List = [simplify_type(t) for t in type_args]
            return origin[tuple(sim_args)] if sim_args else origin
        except Exception as e:
            print(e)

    return target_type


def all_available_types() -> list[Type[Any]]:
    from Tools.YamlConfigurer.configurations import Configurations

    types_list = []

    # 内置类型 (int, str, list, dict 等)
    for type_name in dir(__builtins__):
        contained_obj = getattr(__builtins__, type_name, None)
        try:
            if isinstance(contained_obj, type):
                types_list.append(contained_obj)
        except:
            pass

    # 当前模块定义的所有类
    current_module = sys.modules[__name__]
    for type_name, contained_obj in vars(current_module).items():
        try:
            if isinstance(contained_obj, type):
                types_list.append(contained_obj)
        except:
            pass

    # 已加载模块中公开的类型
    for module_name, module in sys.modules.items():
        if module is None:
            continue
        try:
            for type_name, contained_obj in vars(module).items():
                if isinstance(contained_obj, type) and contained_obj not in types_list:
                    types_list.append(contained_obj)
        except:
            continue

    # 去重
    unique_types = list(set(types_list))

    # 计算每个类型的优先级
    maintainer_collection = Configurations.maintainer_collection
    type_priorities = {}

    for t in unique_types:
        priority = len(maintainer_collection)  # 默认优先级
        for i, maintainer_cls in enumerate(maintainer_collection):
            try:
                if hasattr(maintainer_cls, 'is_type_compatible_static') and maintainer_cls.is_type_compatible_static(t):
                    priority = i  # 找到第一个兼容的维护器，使用其索引作为优先级
                    break
            except:
                pass
        type_priorities[t] = priority

    # 按优先级排序，优先级越小越靠前，相同优先级内按类型名称字母表顺序排序
    sorted_types = sorted(unique_types, key=lambda x: (type_priorities[x], str(x)))

    return sorted_types


if __name__ == "__main__":
    import torch

    print(torch.IntType is torch.int)
