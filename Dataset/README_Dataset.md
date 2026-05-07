# Dataset

**Dataset（数据集）**模组负责定义例程控制和表现相关的自定义功能，通过例程中的钩子进行调用。常用回调功能包括早停策略介入、模型检查点持久化、模型预览和进度条渲染。包装自Lightning。

**Dataset（数据集）**部分包含2个预设代码文件：

- `dataset_configurer`：**关键代码**。定义了`ConfigDatasetBase`**数据集配置包装器**基类，以及来自Monai的3个常用预设数据集类的配置包装器。这些数据集能够按照一定策略加速数据的载入和处理流程，完成样本索引和变换（需要绑定Transform）操作。

| 数据集配置包装器        | 功能                                                         |
| ----------------------- | ------------------------------------------------------------ |
| ConfigDatasetCache      | 缓存数据集。模拟缓存方式，在内存中保存一定数量的已完成所有非随机变换后的数据样本，以提高后续重复取用数据的效率。 |
| ConfigDatasetPersistent | 持久化数据集。利用文件系统，在外存中持久化保存已完成所有非随机变换后的数据样本，以提高后续重复取用数据的效率。 |
| ConfigDatasetLMDB       | LMDB持久化数据集。采用LMDB协议，持久化保存已完成所有非随机变换后的数据样本，以提高后续重复取用数据的效率。 |

- `dataset_manifest_retriever_configurer`：**关键代码**。定义了`ConfigDatasetManifestRetrieverBase`**数据清单检索配置包装器**基类，以及1个默认预设。

| 数据清单检索配置包装器                            | 功能                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| ConfigDatasetManifestRetrieverSegmentationDefault | 此包装器主要负责载入和解析数据集随附的清单文件，从中提取所需字段和样本索引信息，转换绝对路径，完成样本文件的并组（如果存在多序列，多类蒙版），以及绑定Dataset和Transform并构造Monai的Dataset实例。 |

## 快速测试

可以使用`dataset_manifest_retriever_configurer`的主例程和本项目提供的示例样本（样本在`Samples`目录中）进行快速测试或调试以观察执行细节。示例程序测试了多次数据加载的可复现性。请从项目根启动主例程以确保相对路径的正确性。

## 使用指南

数据集配置包装器的使用并无特殊性，按照签名传递恰当的缓存率或持久化保存目录，绑定一个Transform实例，即可按照一般Dataset使用。由数据集配置包装器内部的Dataset实例提供样本路径索引，通过Transform完成单个样本的加载和预处理。

数据清单检索配置包装器`ConfigDatasetManifestRetrieverSegmentationDefault`是PipeUNet特设的模组，它负责将外部清单文件内容转换到Dataset所需的索引格式。其主要目的是**隔离数据存档处理时的存储协议以及Dataset的内存记录格式协议**。二者经由数据清单检索配置包装器转换中介，能够提高两端代码的可扩展性。预设的数据清单检索配置包装器参数描述见下表。

| 属性                     | 描述                                                         | 示例                                                         |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| root_dir                 | 数据集根目录。用于映射清单文件中的相对路径。                 | Samples                                                      |
| manifest_file            | 清单文件路径。要求使用Excel格式（.xlsx），清单文件内部应当包含1个工作表，每行一个样本，记录了此样本相关文件的路径。 | Samples/split01_TJ/split01_TJ_train.xlsx                     |
| column_dtype_map         | 列值类型映射。用于指定载入表格时某些特定列的数值类型，只在必要时使用。 | {'ID': str}<br />要求以str类型载入ID列值，避免意外转换为整数。 |
| column_key_map           | 列名-内部键映射。此映射表以列名为键，内部键为值。用于筛选所需列，并将列名重命名为内部键。 | {'volume': 'volume_0',<br /> 'mask_00_Bg': mask_0',<br /> 'mask_01_EsoROI': 'mask_1'} |
| column_key_relative_path | 相对路径列内部键列表。这个列表包含一系列内部键，这些键所对应的表项将通过`root_dir`转换为绝对路径。 | ['volume_0', 'mask_0', 'mask_1']<br />表示这些内部键对应的列记录的是相对路径，需要转换为绝对路径。 |
| column_group_map         | 列组映射。此映射表以组名为键，内部键列表为值。一般用于分别汇总3D图像序列路径和多类蒙版路径为图像组和蒙版组以便通过Transform按通道优先模式合并载入。 | {'volume': ['volume_0'],<br /> 'mask': ['mask_0', 'mask_1']}<br />volume_0属于volume组，而mask_0和mask_1都属于mask组，因此mask在载入时会变为2通道One-Hot蒙版。 |

通过`ConfigDatasetManifestRetrieverSegmentationDefault`获取Monai Dataset实例可通过调用`get_assembled_dataset`方法实现，需传入所选择的`ConfigDatasetBase`**数据集配置包装器**和变换算子（一般是Transform变换类实例或变换配置包装器实例）。

```python
# Dataset/dataset_manifest_retriever_configurer.py
def get_assembled_dataset(
    self,
    dataset: ConfigDatasetBase,
    transform: Union[Sequence[Callable], Callable] = ConfigOperatorIdentity()
) -> ConfigDatasetBase:
    """
    Create a MONAI dataset from the manifest
    Args:
    dataset: Wrapped MONAI dataset (before init_essentials) to use
    transform: Transform pipe to process the data
    Returns:
    Wrapped MONAI dataset instance
    """
    self._assert_init_essentials()
    dataset.init_essentials(self.manifest, transform)
    return dataset
```

## 自定义指南

进行**自定义**时可参考现有的**数据集配置包装器**和**数据清单检索配置包装器**。数据集配置包装器应总是从`ConfigDatasetBase`派生。数据清单检索配置包装器应总是从`ConfigDatasetManifestRetrieverBase`派生。预设包装器已经能够应对绝大多数场景，并且具有更好的配套支持，建议尽量使用预设以充分利用Monai的加速特性。

### ConfigDatasetBase

**数据集配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用Monai Dataset实例。上层可通过包装器方法获取内部保存的数据集实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和数据集实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类__init__的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_dataset`**：获取包装器内部所包装的数据集实例。

`dataset_configurer`已经对最常用的3个Monai Dataset进行了包装，可查阅以下文档以了解细节。

- [数据 — MONAI 框架](https://docs.monai.org.cn/en/stable/data.html)

### ConfigDatasetManifestRetrieverBase

**数据清单检索配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用检索器实例，检索器可以是Pandas DataFrame、列表等容器类型。提供一个组装工厂方法创建**数据集配置包装器**实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类__init__的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_assembled_dataset`**：工厂方法，传入一个**数据集配置包装器**实例和一个变换算子（例如，Transform变换类实例或变换配置包装器实例），检索器将变换算子注册到数据集中，执行数据集初始化并返回数据集配置包装器实例。
