import torch
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Union, List, Tuple, get_origin, get_args, Type, Callable, Sequence, Literal
from Launcher.Parser.parser_ABC import ParserABC
from monai.utils.enums import BlendMode, PytorchPadMode, NumpyPadMode, InterpolateMode
import numpy as np

Dtype = Union[str, np.dtype, Type[Union[str, complex, bool, object]]]
DtypeArg = Union[Dtype, Dict[str, Dtype]]

from Operator import ConfigOperatorHookStepBase, ConfigOperatorHookStepDisplayDictKeys


def default_list_int():
    return [0, 1, 2]


def default_list_str():
    return ['0', '1', '2']


def default_list_any():
    return ['Hammer', 80, 0.0]


def default_dict_int():
    return {
        'car': 2,
        'house': 1,
        'gold': 50
    }


def default_dict_list_int():
    return {
        'car': [1, 2, 3],
        'house': [10, 20],
        'gold': [88, 888]
    }


def default_dict_tuple_int_float_str():
    return {
        'car': (1, 2.0, 'bence'),
        'house': (1, -1.0, 'Lake-wise'),
        'gold': (1, 99.0, '99% Pure'),
    }


def default_list_nested():
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def default_list_tuple_nested():
    return [
        ('year', 2026, [True, False, False]),
        ('month', 4, [False, True, False]),
        ('day', 23, [False, True]),
        ('hour', 10, [True])
    ]


@dataclass
class ParserRealBeings(ParserABC):
    # dty: Optional[DtypeArg] = "Good"
    # torch_d: torch.dtype = torch.int
    # na_1: None = None
    # bool_1: bool = True
    # bool_2: bool = 1
    # int_1: int = 1
    # int_2: int = 1.0
    # float_1: float = 1.0
    # float_2: float = "1.0"
    # str_1: str = 'Hello'
    # str_2: str = 1
    # tp_1: Type = type
    # tp_2: Type[None] = type(None)
    # tp_3: Type[int] = float
    # tp_4: Type[Type[int]] = type
    # tp_5: Type[Union[int, Type[int], Type[float], ParserABC]] = type
    # tp_6: Type[Optional[int]] = int
    # tp_7: Type[Optional[Type[int]]] = int
    # opt_1: Optional[int] = 1
    # opt_2: Optional[float] = 2.0
    # opt_3: Optional[Union[int, float]] = 2.0
    # opt_4: Optional[None] = 9
    # opt_5: Optional[Type] = int
    # opt_6: Optional[str] = 'int'
    # un_1: Union[None, int] = None
    # un_2: Union[float, int, Type] = 1
    # un_3: Union[str, int, float] = "1.0"
    un_4: Union[int, int, int] = 1.0
    un_5: Union[int, Union[float, str]] = None
    # any_1: Any = 1
    # lt_1: Literal[1, "b", "c"] = 1
    # lt_2: Literal[int, Union[str], "c"] = 1
    # dict_int_1: Dict[str, int] = field(default_factory=default_dict_int)
    # dict_list_int_2: Dict[str, List[int]] = field(default_factory=default_dict_list_int)
    # dict_tuple_int_float_str_3: Dict[str, Tuple[int, float, str]] = \
    #     field(default_factory=default_dict_tuple_int_float_str)
    # enum_1: BlendMode = BlendMode.GAUSSIAN
    # enum_2: PytorchPadMode = PytorchPadMode.CIRCULAR
    # enum_3: NumpyPadMode = NumpyPadMode.EDGE
    # enum_4: InterpolateMode = InterpolateMode.NEAREST_EXACT
    # empty_tuple: Tuple[()] = ()
    # int_1_tuple: Tuple[int] = (1,)
    # int_2_tuple: Tuple[int, float, str] = (1, 2.0, '3s')
    # int_many_tuple: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    # any_many_tuple: Tuple = (1, '2', 3.0)
    # tup_list_int_bool: Tuple[List[int], List[bool]] = ([1, 2, 3], [True, False, False, True])
    # list_tuple_str_int_list_bool: List[Tuple[str, int, List[bool]]] = field(default_factory=default_list_tuple_nested)
    # int_list: List[int] = field(default_factory=default_list_int)
    # str_list: List[str] = field(default_factory=default_list_str)
    # any_list: List[Any] = field(default_factory=default_list_any)
    # nested_list: List[List[int]] = field(default_factory=default_list_nested)
    opt_hs1: ConfigOperatorHookStepBase = ConfigOperatorHookStepDisplayDictKeys()
    list_opt_hs2: List[ConfigOperatorHookStepBase] = field(default_factory=list)

@dataclass
class ParserBeings(ParserABC):
    age: Optional[int] = None
    grows: Optional[float] = None
    greeting: Optional[str] = None

    # def __post_init__(self):
    #     fields_info = fields(self)
    #     for field in fields_info:
    #         print(field.type)


@dataclass
class ParserHuman(ParserBeings):
    sex: Optional[Optional[bool]] = None
    ability: Optional[str] = None


@dataclass
class ParserPerson(ParserHuman):
    name: Optional[str] = None
    family: Optional[list[str]] = None
    talent: Optional[dict[str, int]] = None
    climax: Optional[dict[str, list[int]]] = None


def tup_test(value: tuple, on_change: Callable):
    def change():
        print(value)
        ls = list(value)
        ls.pop(0)
        ls_tup = tuple(ls)
        on_change(ls_tup)

    return change


class footup:
    def __init__(self):
        self.tup = (1, 2, 3, 4)

        def on_change(new_val: tuple):
            self.tup = new_val

        self.change = tup_test(self.tup, on_change)

    def run(self):
        self.change()
        self.change()
        self.change()


if __name__ == '__main__':
    tp = Literal[1, "b", "c"]
    tp_origin = get_origin(tp)
    tp_args = get_args(tp)
    print(tp_origin)
    print(tp_args)
