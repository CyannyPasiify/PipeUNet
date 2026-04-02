from Tools.yaml_configurer.type_maintainer.int_maintainer import IntMaintainer
from Tools.yaml_configurer.type_maintainer.float_maintainer import FloatMaintainer
from Tools.yaml_configurer.type_maintainer.str_maintainer import StrMaintainer
from Tools.yaml_configurer.type_maintainer.bool_maintainer import BoolMaintainer
from Tools.yaml_configurer.type_maintainer.none_maintainer import NoneMaintainer
from Tools.yaml_configurer.type_maintainer.any_maintainer import AnyMaintainer
from Tools.yaml_configurer.type_maintainer.optional_wrapper_maintainer import OptionalWrapperMaintainer
from Tools.yaml_configurer.type_maintainer.union_wrapper_maintainer import UnionWrapperMaintainer
from Tools.yaml_configurer.type_maintainer.base_maintainer import TypeMaintainer, ValueTypeMaintainer
from typing import Any, Type, Optional, Union, get_origin, get_args
from typing_inspection.typing_objects import NoneType

class MaintainerFactory:
    """Type maintainer factory"""

    @staticmethod
    def _simplify_type(expected_type: Type) -> Type:
        """Simplify type by removing duplicates and handling nested types
        
        Args:
            expected_type: Type to simplify
        
        Returns:
            Simplified type
        """
        # Check if it's None type
        if expected_type is None:
            return NoneType

        # Check if it's already Any type
        if expected_type is Any:
            return Any

        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Check if Union type has no arguments or contains Any
        if origin is Union:
            # If no arguments or any argument is Any, return Any
            if not args or Any in args:
                return Any

            # Remove duplicates
            unique_args = []
            seen = set()
            for arg in args:
                simplified_arg = MaintainerFactory._simplify_type(arg)
                # If any simplified arg is Any, return Any
                if simplified_arg is Any:
                    return Any
                if simplified_arg not in seen:
                    seen.add(simplified_arg)
                    unique_args.append(simplified_arg)

            # Handle single type case
            if len(unique_args) == 1:
                return unique_args[0]

            # Handle Optional case
            if type(None) in unique_args:
                non_none_args = [arg for arg in unique_args if arg is not type(None)]
                if non_none_args:
                    simplified_non_none = MaintainerFactory._simplify_type(Union[tuple(non_none_args)])

                    # If simplified non-none type is Any, return Any
                    if simplified_non_none is Any:
                        return Any
                    return Optional[simplified_non_none]

            return Union[tuple(unique_args)]

        # Check if Optional type has no arguments or contains Any
        elif origin is Optional:
            # If no arguments or any argument is Any, return Any
            if not args or Any in args:
                return Any

            simplified_args = [MaintainerFactory._simplify_type(arg) for arg in args]
            # If any simplified arg is Any, return Any
            if Any in simplified_args:
                return Any

            return Optional[tuple(simplified_args)]

        return expected_type

    @staticmethod
    def get_maintainer(expected_type: Type) -> TypeMaintainer:
        """Get corresponding maintainer based on type
        
        Args:
            expected_type: Expected type
        
        Returns:
            Corresponding type maintainer
        """
        # Simplify type first
        simplified_type = MaintainerFactory._simplify_type(expected_type)

        origin = get_origin(simplified_type)
        args = get_args(simplified_type)
        
        # Handle Optional type
        if origin is Union and type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                if len(non_none_args) == 1:
                    # Only use OptionalWrapperMaintainer if there's exactly one non-None type
                    inner_maintainer = MaintainerFactory.get_maintainer(non_none_args[0])
                    assert isinstance(inner_maintainer, ValueTypeMaintainer)
                    return OptionalWrapperMaintainer(inner_maintainer)
                else:
                    # Use UnionWrapperMaintainer for multiple non-None types
                    maintainers = [MaintainerFactory.get_maintainer(arg) for arg in non_none_args]
                    # Add None maintainer to the union
                    maintainers.append(NoneMaintainer())
                    return UnionWrapperMaintainer(maintainers)

        # Handle Union type
        elif origin is Union:
            maintainers = [MaintainerFactory.get_maintainer(arg) for arg in args]
            return UnionWrapperMaintainer(maintainers)

        # Handle basic types
        elif simplified_type is int:
            return IntMaintainer()
        elif simplified_type is float:
            return FloatMaintainer()
        elif simplified_type is str:
            return StrMaintainer()
        elif simplified_type is bool:
            return BoolMaintainer()
        elif simplified_type is type(None):
            return NoneMaintainer()
        elif simplified_type is Any:
            return AnyMaintainer()

        # Default to Any maintainer
        return AnyMaintainer()
