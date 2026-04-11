from Tools.YamlConfigurer.configurations import (
    Configurations,
    BaseMaintainer,
)
from typing import Any, Type, Optional, Union, List, Tuple, get_origin, get_args, Dict


class MaintainerFactory:
    """Type Maintainer factory"""

    @staticmethod
    def get_maintainer_supported_type_value(
            attribute_name: str,
            attribute_type: Type,
            attribute_value: Any,
            logger: Any = None
    ) -> BaseMaintainer:
        """Get supported Maintainer based on type and value
        
        Args:
            attribute_name: Name of the attribute
            attribute_type: Expected type
            attribute_value: Initial value
            logger: Logger instance for logging
        
        Returns:
            Corresponding type Maintainer
        """
        # Traverse all available Maintainers
        available_maintainers: List[Type[BaseMaintainer]] = Configurations.maintainer_collection
        valid_maintainer_cls: Optional[Type[BaseMaintainer]] = None
        maintainer: Type[BaseMaintainer]
        for maintainer in available_maintainers:
            if (maintainer.is_type_compatible_static(attribute_type)
                    and maintainer.is_value_compatible_static(attribute_value, attribute_type)):
                valid_maintainer_cls = maintainer
                break

        if valid_maintainer_cls is not None:
            valid_maintainer_cls: Type[BaseMaintainer]
            return valid_maintainer_cls(attribute_name, attribute_type, attribute_value, logger)

        raise TypeError(f"{attribute_type} is not supported")

    @staticmethod
    def get_maintainer_supported_type(
            attribute_name: str,
            attribute_type: Type,
            attribute_value: Any = None,
            logger: Any = None
    ) -> BaseMaintainer:
        """Get supported Maintainer based on type

        Args:
            attribute_name: Name of the attribute
            attribute_type: Expected type
            attribute_value: Initial value
            logger: Logger instance for logging

        Returns:
            Corresponding type Maintainer
        """
        # Traverse all available Maintainers
        available_maintainers: List[Type[BaseMaintainer]] = Configurations.maintainer_collection
        valid_maintainer_cls: Optional[Type[BaseMaintainer]] = None
        maintainer: Type[BaseMaintainer]
        for maintainer in available_maintainers:
            if maintainer.is_type_compatible_static(attribute_type):
                valid_maintainer_cls = maintainer
                break

        if valid_maintainer_cls is not None:
            valid_maintainer_cls: Type[BaseMaintainer]
            return valid_maintainer_cls(attribute_name, attribute_type, attribute_value, logger)

        raise TypeError(f"{attribute_type} is not supported")

    @staticmethod
    def get_maintainer_supported_value(
            attribute_name: str,
            attribute_value: Any,
            logger: Any = None
    ) -> BaseMaintainer:
        """Get supported Maintainer based on value

        Args:
            attribute_name: Name of the attribute
            attribute_value: Initial value
            logger: Logger instance for logging

        Returns:
            Corresponding type Maintainer
        """
        attribute_type: Type = type(attribute_value)
        return MaintainerFactory.get_maintainer_supported_type_value(
            attribute_name,
            attribute_type,
            attribute_value,
            logger
        )

    @staticmethod
    def get_maintainer_cls_supported_type_value(
            attribute_type: Type,
            attribute_value: Any
    ) -> Type[BaseMaintainer]:
        """Get supported Maintainer class based on type and value

        Args:
            attribute_type: Expected type
            attribute_value: Initial value

        Returns:
            Corresponding type Maintainer class
        """
        # Traverse all available Maintainers
        available_maintainers: List[Type[BaseMaintainer]] = Configurations.maintainer_collection
        valid_maintainer_cls: Optional[Type[BaseMaintainer]] = None
        maintainer: Type[BaseMaintainer]
        for maintainer in available_maintainers:
            if (maintainer.is_type_compatible_static(attribute_type)
                    and maintainer.is_value_compatible_static(attribute_value, attribute_type)):
                valid_maintainer_cls = maintainer
                break

        if valid_maintainer_cls is not None:
            return valid_maintainer_cls

        raise TypeError(f"{attribute_type} is not supported")

    @staticmethod
    def get_maintainer_cls_supported_type(
            attribute_type: Type
    ) -> Type[BaseMaintainer]:
        """Get supported Maintainer class based on type

        Args:
            attribute_type: Expected type

        Returns:
            Corresponding type Maintainer class
        """
        # Traverse all available Maintainers
        available_maintainers: List[Type[BaseMaintainer]] = Configurations.maintainer_collection
        valid_maintainer_cls: Optional[Type[BaseMaintainer]] = None
        maintainer: Type[BaseMaintainer]
        for maintainer in available_maintainers:
            if maintainer.is_type_compatible_static(attribute_type):
                valid_maintainer_cls = maintainer
                break

        if valid_maintainer_cls is not None:
            return valid_maintainer_cls

        raise TypeError(f"{attribute_type} is not supported")

    @staticmethod
    def get_maintainer_cls_supported_value(
            attribute_value: Any
    ) -> Type[BaseMaintainer]:
        """Get supported Maintainer class based on value

        Args:
            attribute_value: Initial value

        Returns:
            Corresponding type Maintainer class
        """
        attribute_type: Type = type(attribute_value)
        return MaintainerFactory.get_maintainer_cls_supported_type_value(
            attribute_type,
            attribute_value
        )

    @staticmethod
    def get_simplest_type_name(
            attribute_type: Type,
            params_get_name: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get expected type name for a given type
        
        Args:
            attribute_type: Type to get name for
        
        Returns:
            Expected type name
        """
        if params_get_name is None:
            params_get_name = {}

        # Traverse all available Maintainers
        available_maintainers: List[Type[BaseMaintainer]] = Configurations.maintainer_collection
        valid_maintainer_cls: Optional[Type[BaseMaintainer]] = None
        maintainer: Type[BaseMaintainer]
        for maintainer in available_maintainers:
            if maintainer.is_type_compatible_static(attribute_type):
                valid_maintainer_cls = maintainer
                break

        if valid_maintainer_cls is not None:
            valid_maintainer_cls: Type[BaseMaintainer]
            return valid_maintainer_cls.get_simplest_type_name_static(target_type=attribute_type, **params_get_name)

        raise TypeError(f"{attribute_type} is not supported")
