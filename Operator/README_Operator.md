# Operator

**Operator（算子）** 负责定义各种零杂的可配置和持久化的算子类，用于打印日志、在损失函数或指标计算中提供自定义的附加预处理和后处理。

**Operator（算子）** 部分包含1个预设代码文件：

- [`operator_configurer`](operator_configurer.py)：**关键代码**。定义了`ConfigOperatorBase`**算子配置包装器**基类。算子配置包装器是支持延迟计算的可调用对象。

  | 算子配置包装器         | 功能                                                         |
  | ---------------------- | ------------------------------------------------------------ |
  | ConfigOperatorIdentity | 等价算子。此算子不执行任何操作，将输入值直接输出。一般作为默认值使用。 |

- [`operator_configurer_hook_step`](operator_configurer_hook_step.py)：**关键代码**。定义了派生自`ConfigOperatorBase`的`ConfigOperatorHookStepBase`**单步钩子算子配置包装器**基类。此类算子主要在LightningModule[主模组](../Module/README_Module.md)的单步例程方法中调用，用于导出预测结果和可视化。

  | 算子配置包装器                                            | 功能                                                         |
  | --------------------------------------------------------- | ------------------------------------------------------------ |
  | ConfigOperatorHookStepDisplayDictKeys                     | 单步返回键显示算子。以模型单步例程的汇总结果字典为输入，打印全部键名。 |
  | ConfigOperatorHookStepExportMulticlassPredWithMaskResults | 单步多分类带蒙版预测结果导出算子。主要用于导出验证（validation）、测试（test）例程中每个样本的预测结果以及相关的体积图像、蒙版文件。 |
  | ConfigOperatorHookStepExportMulticlassPredOnlyResults     | 单步多分类无蒙版预测结果导出算子。主要用于导出预测（predict）例程中每个样本的预测结果以及相关的体积图像文件。 |

- [`operator_configurer_tensor_process`](operator_configurer_tensor_process.py)：**关键代码**。定义了派生自`ConfigOperatorBase`的`ConfigOperatorTensorProcessBase`**张量处理算子配置包装器**基类。此类算子主要用于处理单个张量Tensor，常作为计算损失或指标的预处理或后处理例程使用。

  | 算子配置包装器                             | 功能                                                         |
  | ------------------------------------------ | ------------------------------------------------------------ |
  | ConfigOperatorTensorProcessIdentity        | 等价算子。此算子不执行任何操作，将输入值直接输出。一般作为默认值使用。 |
  | ConfigOperatorTensorProcessMonaiAsDiscrete | Monai AsDiscrete包装算子。用于对连续预测结果的离散化，可转换One-Hot编码以及执行Argmax、阈值处理。一些指标的计算依赖于AsDiscrete预处理。 |
  | ConfigOperatorTensorProcessTorchSoftmax    | PyTorch Softmax包装算子。用于对指定维度应用Softmax总和归一化。 |

- [`operator_configurer_tensor_remap`](operator_configurer_tensor_remap.py)：**关键代码**。定义了派生自`ConfigOperatorBase`的`ConfigOperatorTensorRemapBase`**张量映射算子配置包装器**基类。此类算子主要用于将单个张量Tensor拆分为多个张量以便用于日志打印。

  | 算子配置包装器                     | 功能                                                         |
  | ---------------------------------- | ------------------------------------------------------------ |
  | ConfigOperatorTensorRemapConfMat   | 多分类混淆矩阵映射算子。负责将形如(C, C)的多分类混淆矩阵张量转换为包含C²个单值的字典以便于打印日志。 |
  | ConfigOperatorTensorRemapClassWise | 逐类映射算子。负责将形如(C)的逐类指标张量转换为包含C个单值的字典以便于打印日志。忽略背景时类别数量变为C-1。 |

## 使用指南

**算子配置包装器**本身是可调用对象，其`__call__`方法内部定义了计算流程，对于包装算子将调用算子实例完成计算。可通过`get_operator`方法获取内部算子实例进行直接调用，如果包装器本身就是算子，则将直接返回自身。

```python
# Operator/operator_*.py
def get_operator(self) -> Any:
    self._assert_init_essentials()
    return self.operator  # 如果是算子包装器
    return self  # 如果自身就是算子
```

**算子配置包装器**的典型使用方式如下：以`ConfigOperatorTensorProcessTorchSoftmax`为例。

```python
# 实例化ConfigOperatorBase
config_operator = ConfigOperatorTensorProcessTorchSoftmax()
# 设置配置项
config_operator.dim = 1
# 在step步骤中调用
for step:
    ...
    # 在计算指标前对预测值进行总和归一化处理
    post_pred = config_operator(pred)
    ...
    # 计算具体指标
    auroc = metric_func(post_pred, gt)
```

## 自定义指南

进行**自定义**时可参考现有的**算子配置包装器**。算子配置包装器应总是从`ConfigOperatorBase`派生。

**请注意**算子配置包装器的用途和定义都比较松散，推荐为每种特定用途的算子单独派生基类以便于类型管理，这将有利于类型验证和提高参数配置友好性。可参考[`operator_configurer_hook_step`](operator_configurer_hook_step.py)、[`operator_configurer_tensor_process`](operator_configurer_tensor_process.py)及[`operator_configurer_tensor_remap`](operator_configurer_tensor_remap.py)。

### ConfigOperatorBase

**算子配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **拟包装器**：采用比较松散的类型约束，既可以定义为算子也可以定义为算子包装器。如果是包装器，则使用成员引用算子实例（例如`torch.nn.Softmax`算子）。上层可通过包装器方法获取内部保存的算子实例（如果自身就是算子，则直接返回自身）。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员以及可能的算子实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类**`__init__`**的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`__call__`**：调用方法。执行计算例程或在内部调用算子实例。
- **`get_operator`**：获取包装器内部所包装的算子实例。如果自身就是算子，则直接返回自身。

在定义算子配置包装器前需先完成**算子**的定义，或者直接将算子逻辑写在类内，亦可利用现有算子直接封装：

- [转换 — MONAI 框架](https://docs.monai.org.cn/en/stable/transforms.html#asdiscrete)
- [torch.nn — PyTorch documentation](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-other)

**请注意 算子配置包装器**是当前唯一的即可作为算子定义又可作为算子包装器定义的类型，这主要是考虑到算子用途的多样性、偶然性和轻量性，如果对于一些使用简单lambda表达式都能实现的计算例程仍要对算子和算子包装器进行完整定义，这会让事情变得过于麻烦，因此允许直接将**算子配置包装器**定义为算子本身，从而可以在实现上减少一层封装。但是由此带来的不规范性是需要代码维护者仔细斟酌处理的。

