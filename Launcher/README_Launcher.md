# Launcher

**Launcher（启动器）** 负责解析命令行或YAML配置文件参数，对各模组执行初始化并启动目标例程。包括配置参数解析器。

**Launcher（启动器）** 模组的目录结构包括两级，根目录下包含启动器预设。另设`Parser`子目录，其中包含参数解析器预设：

- [`launcher_ABC`](launcher_ABC.py)：**关键代码**。定义了`LauncherABC`启动器基类，包含以下方法签名。其中包含了各种例程的入口方法定义，包括训练（fit）、微调（finetune）、验证（validation）、测试（test）、预测（predict）。

  | 方法                                 | 功能                                                         |
  | ------------------------------------ | ------------------------------------------------------------ |
  | is_ready                             | 判断是否已初始化。                                           |
  | _assert_init_essentials              | 校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。 |
  | init_essentials                      | 初始化逻辑。它相当于常规类**`__init__`**的功能，但只在必要时才需要初始化。 |
  | fit/finetune/validation/test/predict | 5个功能例程。派生类必须实现这5个接口，从而支持管线的训练、微调、验证、测试、预测用途。 |
  | detect                               | 侦测功能例程。侦测运行时使用，可自行定义侦测功能，例如根据显存确定批量大小，确定数据集长度等。 |
  | debug                                | 调试功能例程。调试运行时使用，可自行定义调试功能。           |
  
- [`launcher_segmentation_default`](launcher_segmentation_default.py)：**关键代码**。定义了用于3D图像分割任务的`LauncherSegmentationDefault`启动器预设，以及主程序入口。`LauncherSegmentationDefault`数据类的属性描述见下表。该预设实现了`LauncherABC`声明的全部功能例程接口，通过调用`ConfigTrainerSegmentationDefault`训练器的对应例程方法实现。关于主程序的使用方法详见[使用指南](#launcher_segmentation_default%20主程序)。

  | 属性               | 描述                                                         |
  | ------------------ | ------------------------------------------------------------ |
  | config_trainer     | `ConfigTrainerBase`数据类实例。定义训练器配置。              |
  | config_data_module | `ConfigDataModuleBase`数据类实例。定义数据模型配置，包括数据集、数据清单检索器、数据加载器参数。 |
  | config_ltn_module  | `ConfigLightningModuleBase`数据类实例。定义主模型配置，包括网络和单步例程控制参数。 |
  
- [`launcher_cmd`](launcher_cmd.py)：**关键代码**。定义了通过YAML配置文件启动管线的主程序。此例程要求用户提供一套YAML配置文件、一个解析器、一个启动器（例如`LauncherSegmentationDefault`）和目标例程名称（例如`fit`）从而启动管线。其采用的YAML配置文件允许对管线全部参数进行自定义，比启动器预设中主程序的可配置粒度更细。配置文件内容经解析器处理后传递给启动器以执行例程。关于该主程序的使用方法详见[使用指南](#launcher_cmd%20主程序)。

- [`Parser/parser_ABC`](Parser/parser_ABC.py)：**关键代码**。定义了YAML配置文件解析器基类，主要负责YAML文件的反序列化以及内存属性数据记录的序列化和持久化保存。解析器自身也是一个数据类。其中定义了以下基类方法。

  | 方法                 | 功能                                                         |
  | -------------------- | ------------------------------------------------------------ |
  | to_yaml              | 数据类序列化和持久化YAML文件保存。将内存中的数据类实例序列化保存到外存YAML文件。 |
  | from_yaml            | YAML文件读取和反序列化。将外存中的YAML文件反序列化为内存中的数据类实例。 |
  | to_dict              | 将数据类序列化为字典。                                       |
  | dump_example_to_yaml | 静态方法。基于解析器自身的属性字段，导出一个作为示例的YAML文件。可选实现。 |

- [`Parser/parser_segmentation_default`](Parser/parser_segmentation_default.py)：**关键代码**。定义了一个适配`LauncherSegmentationDefault`启动器的专用解析器`ParserSegmentationDefault`。其中还定义了用于解析`torch.dtype`类型值的`YamlDumperSegmentationDefault`和`YamlLoaderSegmentationDefault`。

## 使用指南

本节对[`launcher_segmentation_default`](launcher_segmentation_default.py)和[`launcher_cmd`](launcher_cmd.py)主程序的使用提供初步指导。

### [launcher_segmentation_default](launcher_segmentation_default.py) 主程序

此主程序使用Argparse提供海量命令行参数选项。

*<u>**对海量参数的配置感到苦恼？**</u>*

这或许是做深度学习研究必须付出的代价，我们发现即便只是配置一个相当基础的3D-UNet全套管线，其配置文件就可达1272行，这意味着有1000+的超参数需要配置，其中不乏各种名称、键、路径和日志参数。对此情形暂时也没有更好的办法，PipeUNet所能提供的帮助是，从海量的超参数中择出相对关键和经常需要调整的超参数，以便通过命令行参数选项进行配置。此外，PipeUNet所提供的YAML图形化配置工具或许也能助您一臂之力，它可以提供一些属性提示和类型提示以帮助您更快地完成超参数配置。

此主程序可以指定一个**谓词`routine`**，此谓词用于指示目标例程，一般可指定为`fit`, `finetune`, `validation`, `test`, `predict`其中之一，对于支持更多例程的启动器，也可以指定例如`detect`, `debug`等自定义例程。此谓词必须作为第一个参数进行指定。

在谓词之后可指定以下**固定例程参数**。

| 参数                           | 类型             | 描述                                                         |
| ------------------------------ | ---------------- | ------------------------------------------------------------ |
| -r, --experiment_root_dir      | str              | 实验根目录。                                                 |
| -e, --experiment_name          | str              | 实验名称。                                                   |
| -v, --experiment_version       | str              | 实验版本号。                                                 |
| --accelerator                  | str              | 加速器选项。通常可选择`cpu`或`gpu`。                         |
| --devices                      | int \| list[int] | 设备号。对于`cpu`加速器，指定一个整数表示所使用的子进程数；对于`gpu`加速器，指定一个整数列表表示所使用的设备编号集合。 |
| --deterministic                | str              | 确定性选项。可选值为`none`, `warn`, `true`, `false`。设置为`none`或者`false`时，允许采用非确定性算子；设置为`true`时，将强制要求全部使用确定性算子，如果某个算子没有确定性版本，则产生异常；设置为`warn`时，将总是优先尝试使用确定性算子，如果某个算子没有确定性版本，则产生警告并转而使用非确定性算子。 |
| --wandb_project                | str              | WandB项目名称。                                              |
| --dataset_root_dir             | str              | 数据集根目录。用于处理数据样本索引，相对路径转换。           |
| --dataset_manifest_file        | str              | 数据集清单文件路径。该清单将由`ConfigDatasetManifestRetrieverBase`数据清单检索配置包装器负责处理并用于构建数据集加载对象。 |
| --volume_keys                  | list[str]        | 体积序列键列表。用于指定`dataset_manifest_file`中的哪些列包含需要使用的体积图像索引，一般每列对应于一种序列。如果需要使用多序列，例如CT+MR，T1+T2+FLAIR，可指定多个。 |
| --mask_keys                    | list[str]        | 蒙版键列表。用于指定`dataset_manifest_file`中的哪些列包含需要使用的蒙版索引。一般每列对应于一个二值蒙版，每个蒙版标记一个具体类别的前景区域。如果是多分类或多标签任务，例如多器官分割，可指定多个。 |
| --cache_dir                    | str              | 缓存目录。如果使用外存缓存数据集，则将缓存文件保存于此目录。如未指定，默认缓存目录会自动构造为`{experiment_root_dir}/{experiment_name}/{experiment_version}/cache`，否则请指定一个具体路径，绝对或相对路径均可。 |
| --roi_size                     | (int, int, int)  | 分片规格。在不同例程中此参数有不同含义：在`fit`, `finetune`例程中，此参数表示训练时采用的随机切片规格；在`validation`, `test`, `predict`例程中，此参数表示滑动窗口分片规格。 |
| --num_workers                  | int              | 数据加载子进程数。用于Dataloader数据加载的加速。             |
| --batch_size                   | int              | 批量大小。用于指定当前例程所采用的合批大小。在`fit`, `finetune`例程中，此参数仅表示训练时所采用的批量大小，在嵌入的验证环节中所采用的批量大小由例程子参数指定。 |
| --num_sequence, --num_modality | int              | 序列数量。网络结构参数，用于指定输入通道数。                 |
| --num_classes                  | int              | 类别数量。网络结构参数，用于指定输出通道数。                 |

对于每个谓词`routine`所对应的例程，还需指定**例程特定参数**。每个例程可能有不同的特定参数，见下表。例程参数列项目中标注了参数所归属的例程。

| 例程参数                                                     | 类型             | 描述                                                         |
| ------------------------------------------------------------ | ---------------- | ------------------------------------------------------------ |
| `fit` `finetune` `validation` `test` `predict` -ckpt, --resume_checkpoint | str              | 检查点文件路径。在`fit`例程中用于中断恢复；在其他例程中用于初始化模型参数。 |
| `fit` `finetune` --epochs                                    | int              | 训练周期数。                                                 |
| `fit` `finetune` --accumulate_grad_batches                   | int              | 梯度累积批次数。每当累积批次数达到预设目标才执行一次反向传播。 |
| `fit` `finetune` --early_stopping                            | int              | 是否启用早停策略。不指定时不启用早停策略，指定整数表示在损失改善前最多等待的周期数。 |
| `fit` `finetune` --crop_per_sample                           | int              | 训练数据变换随机切片时每个样本产生的切片数。                 |
| `fit` `finetune` --val_dataset_root_dir                      | str              | 嵌入验证环节的验证集根目录。用于处理数据样本索引，相对路径转换。 |
| `fit` `finetune` --val_dataset_manifest_file                 | str              | 嵌入验证环节的验证集清单文件路径。该清单将由`ConfigDatasetManifestRetrieverBase`数据清单检索配置包装器负责处理并用于构建数据集加载对象。 |
| `fit` `finetune` --val_volume_keys                           | list[str]        | 嵌入验证环节的体积序列键列表。用于指定`val_dataset_manifest_file`中的哪些列包含需要使用的体积图像索引，一般每列对应于一种序列。如果需要使用多序列，例如CT+MR，T1+T2+FLAIR，可指定多个。 |
| `fit` `finetune` --val_mask_keys                             | list[str]        | 嵌入验证环节的蒙版键列表。用于指定`val_dataset_manifest_file`中的哪些列包含需要使用的蒙版索引。一般每列对应于一个二值蒙版，每个蒙版标记一个具体类别的前景区域。如果是多分类或多标签任务，例如多器官分割，可指定多个。 |
| `fit` `finetune` --val_cache_dir                             | str              | 嵌入验证环节的验证集样本缓存目录。如果使用外存缓存数据集，则将缓存文件保存于此目录。如未指定，默认缓存目录会自动构造为`{experiment_root_dir}/{experiment_name}/{experiment_version}/cache`，否则请指定一个具体路径，绝对或相对路径均可。 |
| `fit` `finetune` --val_batch_size                            | int              | 批量大小。用于指定当前例程所采用的合批大小。在`fit`, `finetune`例程中，此参数仅表示训练时所采用的批量大小，在嵌入的验证环节中所采用的批量大小由例程子参数指定。 |
| `fit` `finetune` --max_lr                                    | float            | 最大学习率。预设OneCycleLR学习率调度器参数。决定训练途中的最高学习率。 |
| `fit` `finetune` --steps_per_epoch                           | int              | 每个周期中的步数。预设OneCycleLR学习率调度器参数。           |
| `fit` `finetune` --final_div_factor                          | float            | 最终除法系数。预设OneCycleLR学习率调度器参数。决定训练结束时的学习率。 |
| `fit` `finetune` --val_roi_size                              | (int, int, int)  | 嵌入验证环节的验证集样本分片规格。即滑动窗口分片规格。       |
| `fit` `finetune` `validation` `test` `predict` --sw_batch_size | int              | 滑动窗口推断每次处理的批量大小。即每次合批处理时的窗口分片数量，值越大空间开销越高，但可以提速。当`routine=fit, finetune`时，表示嵌入验证环节的批量大小。 |
| `fit` `finetune` `validation` `test` `predict` --overlap     | float            | 滑动窗口间的重叠率。决定滑动窗口的步长，例如0.5表示步长为分片规格的50%。当`routine=fit, finetune`时，表示嵌入验证环节的重叠率。 |
| `fit` `finetune` --export_val_results                        | -                | 是否导出验证环节中每个样本的预测结果。这是一个开关选项，不带参数。 |
| `validation` `test` `predict` --export_results               | -                | 是否导出每个样本的预测结果。这是一个开关选项，不带参数。     |
| `fit` `finetune` `validation` `test` `predict` --export_root_dir | str              | 结果导出根目录。如未指定，默认缓存目录会自动构造为`{experiment_root_dir}/{experiment_name}/{experiment_version}/{hook_dir}/exported_pred_results`，其中当`routine=fit, finetune`时，`hook_dir=hook_{routine}_val`，否则`hook_dir=hook_{routine}`。如果需要保存在具体目录下，则请指定一个具体路径，绝对或相对路径均可。 |
| `fit` `finetune` `validation` `test` `predict` --id_keys     | list[str]        | 标识符键组。用于指定`val_dataset_manifest_file`（`routine=fit, finetune`时）/ `dataset_manifest_file`中的哪些列构成样本索引ID，将用于保存文件名称的生成。 |
| `fit` `finetune` `validation` `test` `predict` --combined_mask_key | str              | 组合蒙版键。用于指定`val_dataset_manifest_file`（`routine=fit, finetune`时）/ `dataset_manifest_file`中的哪一列是多值组合蒙版的路径索引项。由于此启动器针对的是多分类任务，因此有且只有唯一的多值蒙版。 |
| `fit` `finetune` `validation` `test` `predict` --save_option | list[str]        | 结果保存选项。`volume`, `mask`, `pred`, `diff`的任意组合，用于指定是否要导出特定类型的结果。指定`volume`将导出样本原始的体积图像；`mask`将导出参照蒙版图像，包括多值蒙版和每个类别的二值蒙版；`pred`将导出预测结果，包括Softmax归一化后的logits图像以及离散化后生成的每个类别的预测二值蒙版；`diff`将导出每个类别的参照二值蒙版与预测二值蒙版的混淆区域蒙版（蒙版值对应关系：0-TN, 1-FP, 2-TP, 3-FN）。 |
| `finetune` --map_location                                    | list[(str, str)] | 设备映射表。在载入模型权重用于微调时，加速设备的映射。一般而言，如果要使用与原模型加速设备不同的新设备执行微调训练，可设置此项为[(原设备, 'cpu')]（例如[('cuda', 'cpu')]），而后训练器会负责将映射到cpu上的模型载入到新的加速设备。 |

**启动器**的典型使用方式如下：以`LauncherSegmentationDefault`为例。

```python
# 获取命令行参数
args = parser.parse_args()
# 实例化启动器并完成参数配置
config_launcher = LauncherSegmentationDefault(
    config_trainer=...,
    config_data_module=...,
    config_ltn_module=...
)
# 根据谓词启动对应的例程
if args.routine == 'fit':
    config_launcher.fit(args.resume_checkpoint)
elif args.routine == 'finetune':
    map_location: Dict[int, int] = {item[0]: item[1] for item in args.map_location}
    config_launcher.finetune(args.init_checkpoint, finetune_map_location=map_location)
elif args.routine == 'validation':
    config_launcher.validation(args.init_checkpoint)
elif args.routine == 'test':
    config_launcher.test(args.init_checkpoint)
elif args.routine == 'predict':
    config_launcher.predict(args.init_checkpoint)
```

### [launcher_cmd](launcher_cmd.py) 主程序

此主程序使用Argparse提供命令行参数选项。

`launcher_cmd`主程序要比`launcher_segmentation_default `主程序简单得多，并且具有通用性。这是因为`launcher_cmd`从YAML配置文件中读取海量参数，从而无需定义复杂和特定化的命令行参数。请从项目根启动主例程以确保类索引路径的正确性。其命令行参数见下表。

| 参数                | 类型 | 描述                                                         |
| ------------------- | ---- | ------------------------------------------------------------ |
| -c, --config        | str  | YAML配置文件路径。管线的大多数超参数都从此文件中读取。       |
| -p, --parser        | str  | 解析器类路径。指定一个从项目根开始的解析器类路径，例如`Launcher.Parser.parser_segmentation_default.ParserSegmentationDefault`。此解析器将被实例化用于读取和解析YAML配置文件。 |
| -u, --launcher      | str  | 启动器类路径。指定一个从项目根开始的启动器类路径，例如`Launcher.launcher_segmentation_default.LauncherSegmentationDefault`。此启动器将被实例化用于执行例程，它从解析器的解析结果中获取所需超参数配置。 |
| -r, --routine       | str  | 目标例程。一般是`fit`, `finetune`, `validation`, `test`, `predict`其中之一，对于支持更多例程的启动器，也可以指定例如`detect`, `debug`等自定义例程。 |
| -ckpt, --checkpoint | str  | 检查点文件路径。在`fit`例程中用于中断恢复；在其他例程中用于初始化模型参数。 |

## 自定义指南

进行**自定义**时可参考现有的**启动器**`LauncherSegmentationDefault`和**解析器**`ParserSegmentationDefault`进行实现，如果您不准备使用`launcher_cmd`主程序，则可以不实现解析器。启动器应总是从`LauncherABC`派生，解析器应总是从`ParserABC`派生。

**请注意**启动器和解析器都应当是数据类，并且启动器的属性字段应当完全包含解析器的全部属性字段。一般而言，另二者属性字段完全一致为好。

### LauncherABC

**启动器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类 **`__init__`** 的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`fit/validate/test/predict`**：具体功能例程。

### ParserABC

**解析器**的抽象基类，其中定义了以下基类方法。

| 方法                 | 功能                                                         |
| -------------------- | ------------------------------------------------------------ |
| to_yaml              | 数据类序列化和持久化YAML文件保存。将内存中的数据类实例序列化保存到外存YAML文件。 |
| from_yaml            | YAML文件读取和反序列化。将外存中的YAML文件反序列化为内存中的数据类实例。 |
| to_dict              | 将数据类序列化为字典。                                       |
| dump_example_to_yaml | 静态方法。基于解析器自身的属性字段，导出一个作为示例的YAML文件。可选实现。 |

PyYaml（导入命令`import yaml`）包库已经提供了一套足够完善的Loader, Dumper序列化工具类，可参考[Home · yaml/pyyaml Wiki](https://github.com/yaml/pyyaml/wiki)以了解其使用方式。但对于一些特殊类型（例如`torch.dtype`），其预设的工具类无法执行恰当的处理，此时可能需要自行扩充Loader, Dumper的定义，[Launcher/Parser/parser_segmentation_default.py](Parser/parser_segmentation_default.py)中提供了一个扩充定义示例，可参考`YamlLoaderSegmentationDefault`和`YamlDumperSegmentationDefault`类的实现。

```python
class YamlLoaderSegmentationDefault(yaml.Loader):
    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        Constructor.__init__(self)
        Resolver.__init__(self)

    def construct_torch_dtype(self, suffix: str, node: ScalarNode) -> ScalarNode:
        value: str = self.construct_scalar(node)
        if value:
            raise ConstructorError("while constructing a Torch dtype", node.start_mark,
                                   "expected the empty value, but found %r" % value, node.start_mark)
        if not hasattr(torch, suffix):
            raise ConstructorError("while constructing a Torch dtype", node.start_mark,
                                   "type %r is not supported" % suffix, node.start_mark)
        return getattr(torch, suffix)

YamlLoaderSegmentationDefault.add_multi_constructor(
    'tag:yaml.org,2002:torch/dtype:',
    YamlLoaderSegmentationDefault.construct_torch_dtype  # noqa
)

class YamlDumperSegmentationDefault(yaml.Dumper):
    def __init__(
            self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True
    ):
        Emitter.__init__(
            self, stream, canonical=canonical,
            indent=indent, width=width,
            allow_unicode=allow_unicode, line_break=line_break
        )
        Serializer.__init__(
            self, encoding=encoding,
            explicit_start=explicit_start, explicit_end=explicit_end,
            version=version, tags=tags
        )
        Representer.__init__(
            self, default_style=default_style,
            default_flow_style=default_flow_style, sort_keys=sort_keys
        )
        Resolver.__init__(self)

    def ignore_aliases(self, data):
        return True

    def represent_torch_dtype(self, data: torch.dtype) -> Node:
        name: str = data.__reduce__()
        node: ScalarNode = self.represent_scalar('tag:yaml.org,2002:torch/dtype:' + name, '')
        return node

YamlDumperSegmentationDefault.add_representer(
    torch.dtype,
    YamlDumperSegmentationDefault.represent_torch_dtype  # noqa
)
```

