# DataModule

**DataModule（数据模型）** 负责调度数据集和变换模组协同工作，针对多种实验例程（训练、验证等）创建不同的数据装载器。负责配置参数和运行状态的记录和恢复，尤其是变换模组的随机状态。扩展自定义于Lightning。

**DataModule（数据模型）** 部分包含2个预设代码文件：

- [`data_module_configurer`](data_module_configurer.py)：**关键代码**。定义了`ConfigDataModuleBase`**数据模型配置包装器**基类，其中预设了1个用于分割任务的`ConfigDataModuleSegmentationDefault`数据模型配置包装器。数据模型配置包装器本身只负责保存初始化配置和实例化数据模型，并提供内部数据模型实例的获取方法。

  | 数据模型配置包装器                  | 功能                                                         |
  | ----------------------------------- | ------------------------------------------------------------ |
  | ConfigDataModuleSegmentationDefault | 此包装器主要负责保存配置和延迟实例化数据模型，而并无实际功能，关键功能都在数据模型本体中定义。 |

- [`data_module_segmentation_default`](data_module_segmentation_default.py)：**关键代码**。定义了1个用于分割任务的`DataModuleSegmentationDefault`数据模型预设，此数据模型派生自`Lightning.LightningDataModule`并重写了其中与数据准备、初始化、数据加载器实例化以及状态管理相关的接口。此外，还定义了用于表示数据模型各例程（训练、验证、测试、预测）参数的数据类`DataModuleSegmentationDefaultInitArgs`。

## 快速测试

可以使用[`data_module_segmentation_default`](data_module_segmentation_default.py)的主例程和本项目提供的示例样本（样本在[`Samples`](../Samples)目录中）进行快速测试或调试以观察执行细节。示例程序提供3项测试，包括在训练和测试例程中使用数据模型，以及数据模型可复现性验证。请从项目根启动主例程以确保相对路径的正确性。

## 使用指南

**数据模型配置包装器**本身并无实际功能，应从中获取`DataModuleSegmentationDefault`数据模型实例进行使用。`DataModuleSegmentationDefault`数据模型派生自`Lightning.LightningDataModule`。它使用至多4套`DataModuleSegmentationDefaultInitArgs`用于配置各例程（训练、验证、测试、预测）的数据加载功能，此数据类的属性描述见下表，每个例程的配置可以不同。

| 属性                  | 描述                                                         |
| --------------------- | ------------------------------------------------------------ |
| config_retriever      | 数据清单检索配置包装器`ConfigDatasetManifestRetrieverSegmentationDefault`实例。数据模型通过此包装器获取数据集实例。 |
| config_dataset        | 数据集配置包装器`ConfigDatasetBase`实例。此实例是用于提供给数据清单检索配置包装器构建数据集实例的。 |
| config_transform      | 变换配置包装器`ConfigTransformBase`实例。此实例是用于提供给数据清单检索配置包装器构建数据集实例的。 |
| batch_size            | 批量规格。数据加载器（Dataloader）实例化所需参数，表示在每个批次中载入的样本数量（B）。注意如果使用了产生多个切片（P）的变换，则实际合批后的批量规格会变为B×P。一般在除训练以外的例程中总是设置`batch_size=1`。 |
| shuffle               | 是否在随机混洗后再执行索引采样。设置为`True`可在每个 epoch 重新打乱数据。 |
| num_workers           | 用于数据加载的子进程数量。`0` 表示数据将在主进程中加载。     |
| pin_memory            | 如果为 `True`，数据加载器将在返回前将张量复制到设备/CUDA 的固定内存（pinned memory）中。 |
| drop_last             | 如果数据集大小不能被批次大小整除，设置为`True`可丢弃最后一个不完整的批次。如果为 `False` 且数据集大小不能被整除，则最后一个批次会更小。 |
| persistent_workers    | 如果为 `True`，数据加载器在数据集被消耗一次后不会关闭工作进程。这可以保持工作进程的 Dataset 实例处于活跃状态。数据加载的子进程（workers）不会被重新创建，但这会破坏中断恢复时的可复现性。 |
| generator_random_seed | 用于初始化[torch.Generator](https://docs.pytorch.ac.cn/docs/stable/generated/torch.Generator.html#torch.Generator)的随机数种子，此 RNG（随机数生成器）将用于索引采样，提供可复现性支持。 |

`DataModuleSegmentationDefault`主要方法的功能描述见下表。

| 方法               | 功能                                                         |
| ------------------ | ------------------------------------------------------------ |
| prepare_data       | 一般在此方法中对数据集进行准备，例如下载、校验路径等。由于预设使用的是已经准备好的数据集，因此在此方法中仅校验路径。 |
| setup              | 此方法接收`stage`参数用于指定目标例程[fit, val, test, predict]，根据选中的例程选用恰当的配置参数完成数据集的实例化，并准备用于实例化数据加载器的参数。 |
| train_dataloader   | 构建训练例程的数据加载器。适用于fit例程。                    |
| val_dataloader     | 构建验证例程的数据加载器。适用于fit、val例程。               |
| test_dataloader    | 构建测试例程的数据加载器。适用于test例程。                   |
| predict_dataloader | 构建预测例程的数据加载器。适用于predict例程。                |
| state_dict         | 返回数据模型的状态参数字典，通常包括变换的RNG状态、数据加载器的状态。此字典应当包含所有需要保存至检查点的参数。 |
| load_state_dict    | 与state_dict相对的，从字典中载入数据模型的状态。             |

上表所述方法仅是`lightning.LightningDataModule`的一部分，完整的使用指南请参阅[LightningDataModule — PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)。

通过`ConfigDataModuleSegmentationDefault`获取`DataModuleSegmentationDefault`实例可通过调用`ConfigDataModuleBase`基类方法`get_data_module`实现。

```python
# DataModule/data_module_configurer.py
def get_data_module(self) -> L.LightningDataModule:
    self._assert_init_essentials()
    return self.data_module
```

**数据模型配置包装器**的典型使用方式如下：以`DataModuleSegmentationDefault`为例。

```python
# 实例化ConfigDataModuleBase
config_data_module = DataModuleSegmentationDefault()
# 设置配置项，以准备用于fit例程为例，准备训练集和验证集
train_init_args = SomeArgs()
val_init_args = SomeArgs()
...
# 实例化内部DataModule
data_module = config_data_module.get_data_module()
# 进行数据准备
data_module.prepare_data()
data_module.setup(stage='fit')
# 实例化和获取加载器
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

## 自定义指南

进行**自定义**时可参考现有的**数据模型配置包装器**，即`ConfigDataModuleSegmentationDefault`。数据模型配置包装器应总是从`ConfigDataModuleBase`派生。

### ConfigDataModuleBase

**数据模型配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`Lightning.LightningDataModule`实例。上层可通过包装器方法获取内部保存的数据模型实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和数据集实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类 **`__init__`** 的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_data_module`**：获取包装器内部所包装的数据模型实例。

在定义数据模型配置包装器前需先完成`Lightning.LightningDataModule`的定义，请参阅：

- [LightningDataModule — PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
