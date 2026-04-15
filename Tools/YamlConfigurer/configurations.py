from typing import List, Type, Tuple, Dict, Set, Any, Sequence, Union
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.none_maintainer import NoneMaintainer
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.bool_maintainer import BoolMaintainer
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.int_maintainer import IntMaintainer
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.float_maintainer import FloatMaintainer
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.str_maintainer import StrMaintainer
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.type_maintainer import TypeMaintainer
from Tools.YamlConfigurer.Maintainer.WrapperMaintainer.optional_maintainer import OptionalMaintainer
from Tools.YamlConfigurer.Maintainer.WrapperMaintainer.union_maintainer import UnionMaintainer
from Tools.YamlConfigurer.Maintainer.WrapperMaintainer.any_maintainer import AnyMaintainer
from Tools.YamlConfigurer.Maintainer.ContainerMaintainer.list_maintainer import ListMaintainer
from Tools.YamlConfigurer.Maintainer.PrimitiveMaintainer.unsupported_maintainer import UnsupportedMaintainer
from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer


class Configurations:
    # The sequence order is critical, in generic ascending manner
    maintainer_collection: List[Type[BaseMaintainer]] = [
        NoneMaintainer,
        BoolMaintainer,
        IntMaintainer,
        FloatMaintainer,
        StrMaintainer,
        TypeMaintainer,
        OptionalMaintainer,
        UnionMaintainer,
        AnyMaintainer,
        ListMaintainer,
        UnsupportedMaintainer,
    ]


if __name__ == '__main__':
    pass
