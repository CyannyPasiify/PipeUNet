# Transform

**Transform（变换）** 模组负责对已载入内存的数据样本进行在线预处理、数据增强变换。也包括部分在线后处理变换。扩展自定义于和包装自Monai。

**Transform（变换）** 部分包含2个预设代码文件：

- [`transform_configurer`](transform_configurer.py)：**关键代码**。定义了`ConfigTransformBase`**变换配置包装器**基类，一个基础性的训练预处理变换配置包装器`ConfigTransformSegmentationDefaultTrain`，以及适用于推断时使用的推断预处理变换配置包装器`ConfigTransformSegmentationDefaultInferencePre`和推断后处理变换配置包装器`ConfigTransformSegmentationDefaultInferencePost`。

  | 变换配置包装器                                    | 功能                                                         |
  | ------------------------------------------------- | ------------------------------------------------------------ |
  | ConfigTransformSegmentationDefaultTrain           | 一个用于训练例程的变换包装器。支持3D图像文件载入、通道合并、间距规范化、随机裁剪、值域归一化功能的简单变换管线。 |
  | ConfigTransformSegmentationSimulateNNUNetAugTrain | 一个用于训练例程的变换包装器。此变换管线模仿自[nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L739)（但不完全等价），提供多种随机空间、值域增强。 |
  | ConfigTransformSegmentationDefaultInferencePre    | 一个用于推断例程预处理的变换包装器。支持3D图像文件载入、通道合并、值域归一化功能的简单变换管线。 |
  | ConfigTransformSegmentationDefaultInferencePost   | 一个用于推断例程后处理的变换包装器。支持3D图像Tensor的参考重采样（用于恢复原规格）、值域逆向映射（用于恢复原值域范围）的简单变换管线。 |


- [`monai_transform_custom`](monai_transform_custom.py)：*<u>辅助代码</u>*。定义了一些扩展自Monai的变换字典包装类以提供Compose便捷性。

  | 自定义Monai变换类       | 功能                                                         |
  | ----------------------- | ------------------------------------------------------------ |
  | RenameItemsd            | 修改变换字典中键的名称。                                     |
  | DuplicateItemsd         | 拷贝字典中的值并分配于新的键。                               |
  | RandCropByLabelClassesd | Monai标签加权随机切片变换类的扩展，追加了一个获取随机状态的方法。 |

*<u>**为什么只提供了非常有限的Transform预设？**</u>*

Transform的定制性通常很强，两项不同的分割任务可能使用完全不同的变换，因此很难说存在某种固定且总是有效的Transform预设。此外，Transform预设有时还依赖于离线预处理和数据集结构协议，PipeUNet无法考虑到那么多的因素，因此只提供一个小而简洁的预设。

## 快速测试

可以使用[`transform_configurer`](transform_configurer.py)的主例程和本项目提供的示例样本（样本在`Samples`目录中）进行快速测试或调试以观察执行细节。请从项目根启动主例程以确保相对路径的正确性。

## 自定义指南

进行**自定义**时可参考现有的**变换配置包装器**，尤其是`ConfigTransformSegmentationDefaultTrain`包含了更丰富的实现参考信息。变换配置包装器应总是从`ConfigTransformBase`派生。

### ConfigTransformBase

**变换配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用已定义的变换算子实例，并负责管理这些算子的状态和调用。对上层伪装成一个可调用算子。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和变换算子实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类 **`__init__`** 的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_composed_transform`**：获取一个Monai复合变换（Compose）实例，这是一个由流程中各种变换顺序排列构成的封装实例，能够支持Monai的一些加速特性。
- **`__call__`**：包装器调用方法。调用所包装的变换执行实际功能。
- **`get_state`**：返回一个能够反映当前变换状态的字典，用于检查点保存。
- **`set_state`**：从字典中载入变换状态，用于状态恢复。

### ConfigTransformSegmentationDefaultTrain

这是一个十分基础的用于分割任务训练例程的**变换配置包装器**。它将实例化以下变换算子：

- **`_tf_load_image(LoadImaged)`**：从文件路径字典读取3D图像。输入形如{'volume': 'path/to/volume', 'mask': 'path/to/mask'}或{'volume': ['path/to/volume_seq_1', 'path/to/volume_seq_2', ...]}（多序列情景）,{'mask': ['path/to/mask_1', 'path/to/mask_2', ...]}（多类情景），输出形如{'volume': 表示3D图像的通道优先MetaTensor, 'mask': 表示蒙版的通道优先MetaTensor}。
- **`_tf_duplicate_items(DuplicateItemsd)`**：拷贝值到新键。其主要目的是保存一个所载入数据内容的原始副本，这个副本不会被后续步骤继续变换。
- **`_tf_spacing(Spacingd)`**：体素间距规格化。按照指定的XYZ间距重采样3D图像和蒙版。常用`1mm³`作为目标规格。
- **`_tf_spatial_pad(SpatialPadd)`**：基于目标规格的空间填充。为了确保后续裁剪步骤输出的统一性，提前进行空间填充确保所有样本在继续处理前的XYZ规格都大于等于裁剪切片规格，对于本就满足裁剪切片规格的轴向不做填充处理。
- **`_tf_scale_intensity_range(ScaleIntensityRanged)`**：体素值最大最小值截断和值域重映射。选择一个值域窗口，对窗口外的体素值截断为窗口最值，然后对体素值进行值域映射，通常将目标窗口设置为`[0,1]`以执行归一化。
- **`_tf_rand_crop_by_label_classes(RandCropByLabelClassesd)`**：标签加权随机切片。按照蒙版中的标签和指定的`ratios`权重在3D图像和蒙版中同步随机裁剪切片，切片大小由`spatial_size`指定。随机裁剪模式为首先按照`ratios`权重在蒙版中随机选择一个带有对应类型标签的体素点作为中心，然后修正中心点确保`spatial_size`规格的切片不超出图像边界，最后执行裁剪。此过程重复`num_samples`次，因此一个图像样本可以产生若干用于训练的样本组，它们将在训练前合批。
- **`_tf_cast_to_type(CastToTyped)`**：将3D图像和蒙版的数据类型变更为指定的`dtype`。通常模型所需的3D图像和One-Hot标签图像的数据类型都是`torch.float`。

这些变换算子通过`_initialize_transforms`完成随机状态的初始化，从而支持可复现性，然后被组装为复合变换`_composed_transform`。复合变换无需手动逐个调用变换算子，并且可以使用Monai的加速特性。

```python
# Transform/transform_configurer.py
@override
def init_essentials(self) -> 'ConfigTransformSegmentationDefaultTrain':
    # Initialize individual transforms
    # ...

    # Build transform dictionary
    self.transform_dict: Dict[str, mT.Transform] = {
        'LoadImaged': self._tf_load_image,
        'DuplicateItemsd': self._tf_duplicate_items,
        'Spacingd': self._tf_spacing,
        'SpatialPadd': self._tf_spatial_pad,
        'ScaleIntensityRanged': self._tf_scale_intensity_range,
        'RandCropByLabelClassesd': self._tf_rand_crop_by_label_classes,
        'CastToTyped': self._tf_cast_to_typeTransform/transform_configurer.py
    }

    # Initialize transforms with random seed
    self._initialize_transforms()

    # Compose transforms into a pipeline
    self._composed_transform: mT.Compose = \
      mT.Compose(list(self.transform_dict.values()))

    return self

def _initialize_transforms(self) -> None:
    """
    Initialize transforms with random seed for reproducibility
    """
    if self.random_seed is None: return
    for name, transform in self.transform_dict.items():
        if hasattr(transform, 'set_random_state'):
            transform_seed: int = self.random_seed + hash(name) % 10000
            random_state: np.random.RandomState = np.random.RandomState(transform_seed)
            transform.set_random_state(state=random_state)
```

作为示例，`ConfigTransformSegmentationDefaultTrain`**属性值描述**见下表。在调用 **`__call__`** 时，将会利用数据类各项属性创建算子实例并执行计算。可以将`volume_key`的值指定为列表以实现多序列输入；将`mask_key`的值指定为列表以实现多类蒙版输入（对于多类分割任务，总是应当提供等同于类别数量的二值蒙版，而非一个合并的多值蒙版）。

| 属性                                             | 描述                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| volume_key                                       | 3D体积图像键，输入字典中应当包含形如{volume_key: 'path/to/volume'或多序列[]}的内容。 |
| mask_key                                         | 蒙版键，输入字典中应当包含形如{mask_key: 'path/to/mask'或多类蒙版[]}的内容。 |
| param_volume_tf_duplicate_items_dup_keys_volume  | 拷贝作为原始副本的3D体积图像键，例如volume_raw。             |
| param_mask_tf_duplicate_items_dup_keys_mask      | 拷贝作为原始副本的蒙版键，例如mask_raw。                     |
| param_tf_spacing_pixdim                          | 图像间距规格化目标间距，通常设置为(1.0, 1.0, 1.0)mm³。       |
| param_tf_spacing_mode_volume                     | 3D体积图像规格化重采样模式，常用线性插值。                   |
| param_tf_spacing_mode_mask                       | 蒙版规格化重采样模式，常用近邻插值。                         |
| param_tf_padding_mode_volume                     | 3D体积图像重采样时的填充模式，常用边缘填充。                 |
| param_tf_padding_mode_mask                       | 蒙版重采样时的填充模式，常用边缘填充。                       |
| param_tf_spatial_pad_spatial_size                | 3D体积图像空间填充目标规格，通常设置为(128, 128, 128)。      |
| param_tf_spatial_pad_mode                        | 3D体积图像空间填充模式，常用边缘填充。                       |
| param_tf_rand_crop_by_label_classes_spatial_size | 标签加权随机切片目标空间规格。必须不大于空间填充目标规格，可同样设置为(128, 128, 128)。 |
| param_tf_rand_crop_by_label_classes_ratios       | 标签加权随机切片类别权重列表。由于一般总是不考虑以背景为中心进行裁剪，因此列表的首元素常设置为0。 |
| param_tf_rand_crop_by_label_classes_num_classes  | 标签加权随机切片类别数，应当与输入通道数以及类别权重列表长度相等。由于使用One-Hot蒙版时类别数可以自行推断，因此也可以填为None，但仍推荐填写为类别数以提供校验。 |
| param_tf_rand_crop_by_label_classes_num_samples  | 标签加权随机切片生成切片数。一个样本可以随机裁剪多个切片构成训练数据。 |
| param_tf_scale_intensity_range_a_min             | 值域重映射原始值域下限，也作为值域截断下限。对于CT而言可设置为-1000（空气密度及以上）。 |
| param_tf_scale_intensity_range_a_max             | 值域重映射原始值域上限，也作为值域截断上限。对于CT而言可设置为1000（覆盖绝大多数组织的密度）。 |
| param_tf_scale_intensity_range_b_min             | 值域重映射目标值域下限，一般设置为0以实现值域归一化。        |
| param_tf_scale_intensity_range_b_max             | 值域重映射目标值域上限，一般设置为1以实现值域归一化。        |
| param_tf_scale_intensity_range_clip              | 值域重映射是否进行值域裁剪。设置为`True`时，将截断到原始值域范围内后再进行映射，否则直接映射。 |
| param_tf_allow_missing_keys                      | 是否允许缺失键。此参数将设置于所有变换算子，设置为`False`可提供校验。 |
| random_seed                                      | 用于初始化随机算子（标签加权随机切片）的种子。用于支持可复现性。 |

### 推断变换配置包装器

用于推断的变换配置包装器与`ConfigTransformSegmentationDefaultTrain`的结构是类似的，并且不需要裁剪，因而更加简单，可自行参考。

其中，推断预处理变换配置包装器`ConfigTransformSegmentationDefaultInferencePre`的调用输入同样是样本路径字典，而推断后处理变换配置包装器`ConfigTransformSegmentationDefaultInferencePost`的调用输入应当是Tensor数据体字典。

## 使用指南

**变换配置包装器**的典型使用方式如下：以`ConfigTransformSegmentationDefaultTrain`为例。

```python
# 实例化ConfigTransformBase
config_transform = ConfigTransformSegmentationDefaultTrain()
# 设置配置项，准备好全部参数
config_transform.volume_key = 'volume'
config_transform.mask_key = 'mask'
...
# 获取变换算子
transform = config_transform.get_composed_transform()
# 注册用于数据集变换或在数据集中调用
loaded_data = ...  # 加载到内存的数据对象
transformed_data = transform(loaded_data)
```
