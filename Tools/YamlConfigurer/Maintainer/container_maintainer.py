from Tools.YamlConfigurer.Maintainer.base_maintainer import BaseMaintainer

"""
    Maintainer必须具备以下功能：
    · 记录属性名称、属性类型、属性值
    · 提供此类型的默认值
    · 判断目标值类型是否与自身兼容
    · 判断目标类型是否与自身兼容
    · 获取预期的类型名称（化简后的）
    · 创建监视器布局
    · 创建编辑器布局
"""

class ContainerMaintainer(BaseMaintainer):
    """Base class for container type maintainers
    
    Container type maintainers handle container types like List, Tuple, Dict.
    """
    pass
