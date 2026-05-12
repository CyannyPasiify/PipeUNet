# Module

**Module（主模型）** 负责定义网络针对复杂输入数据的调用方式和网络输出的统筹方式。通常用于针对需要分片处理的大型单个样本的推断任务中。包装自Monai。

**Module（主模型）** 部分包含1个预设代码文件：

- [`ltn_module_configurer`](ltn_module_configurer.py)：**关键代码**。定义了`ConfigLightningModuleBase`**主模型配置包装器**基类。其中定义了一个用于3D分割任务的预设`ConfigLightningModuleSegmentationDefault`。

  | 主模型配置包装器                         | 功能                                                         |
  | ---------------------------------------- | ------------------------------------------------------------ |
  | ConfigLightningModuleSegmentationDefault | 此包装器负责实例化`LightningModule`对象，作为其外层包装容器使用。绝大多数功能仍需通过主模型本身进行调用。 |

- [`ltn_module_segmentation_default`](ltn_module_segmentation_default.py)：**关键代码**。定义了`LightningModuleSegmentationDefault`**主模型**类，以及模型所需参数的一系列具有派生关系的数据类。

  | 数据类或主模型                     | 功能                                                         |
  | ---------------------------------- | ------------------------------------------------------------ |
  | LightningModuleSegmentationDefault | 分割主模型。派生自`lightning.LightningModule`，并重写了其所需的一系列钩子方法。主模型可通过数据类的形式接收神经网络初始化参数以及训练、验证、测试、预测各阶段的特定参数，并执行相应功能。 |
  | NamedInitArgs                      | 带名参数基类。提供名称`name`和描述信息`description_info`2个描述性字段，可作为ID使用。 |
  | NamedNetworkInitArgs               | 带名神经网络参数。包含一个`ConfigNetworkBase`网络配置包装器类型的属性。 |
  | NamedLossInitArgs                  | 带名损失参数。包含一个`ConfigLossBase`损失配置包装器类型的属性，后处理算子属性，以及若干日志和统计相关属性。后处理算子主要用于将损失变换到适合打印日志和呈现的形式，不影响损失结果本身，特别适用于需分别打印复合损失每个分量结果的情况，但这需要损失函数算子本身提供支持（返回包含多项损失分量的Tensor）。 |
  | NamedOptimizerInitArgs             | 带名优化器参数。包含一个`ConfigOptimizerBase`优化器配置包装器类型的属性。 |
  | LRSchedulerLightningConfig         | 学习率优化器的Lightning配置项。定义学习率优化器的更新频率选项、监视指标等属性。 |
  | NamedLRSchedulerInitArgs           | 带名学习率调度器参数。包含一个`ConfigLRSchedulerBase`学习率调度器配置包装器类型的属性，以及Lightning所需的`LRSchedulerLightningConfig`配置属性。 |
  | NamedMetricInitArgs                | 带名指标参数。包含一个`ConfigMetricBase`指标配置包装器类型的属性，预测值预处理算子、参照值预处理算子和后处理算子属性，以及若干日志和统计相关属性。预处理算子主要用于预测值和参照值的规格、值域对齐以及施加图像变换，有些指标依赖于后处理算子才能正确打印，例如包含非标量结果的混淆矩阵指标。 |
  | ModuleStepAdditionArgs             | 单步例程附加参数基类。用于派生定义训练、验证、测试、预测例程所需的附加参数。包含一个`ConfigInfererBase`推断器配置包装器类型的属性，带名指标参数`NamedMetricInitArgs`的列表，以及[单步钩子算子配置包装器](../Operator/README_Operator.md)`ConfigOperatorHookStepBase`的列表（此类包装器负责在例如`training_step`等例程方法中执行自定义功能，例如保存结果和可视化）。 |
  | ModuleStepWithLossAdditionArgs     | 带有损失的单步例程附加参数基类。增加了`NamedLossInitArgs`类型的属性以及后处理损失键名属性。一般在除了预测例程以外的例程中都会使用损失函数，键名用于标识已经过后处理算子处理的损失结果。 |
  | ModuleTrainingStepAdditionArgs     | 训练单步例程附加参数。包含一些在训练例程中所需的参数，包含用于优化器和学习率调度器配置的`NamedOptimizerInitArgs`和`NamedLRSchedulerInitArgs`属性以及一些键名属性。 |
  | ModuleValidationStepAdditionArgs   | 验证单步例程附加参数。包含一些在验证例程中所需的参数，主要是3D体积对象、蒙版、预测值相关的一些键名属性。 |
  | ModuleTestStepAdditionArgs         | 测试单步例程附加参数。包含一些在测试例程中所需的参数，主要是3D体积对象、蒙版、预测值相关的一些键名属性。 |
  | ModulePredictStepAdditionArgs      | 预测单步例程附加参数。包含一些在预测例程中所需的参数，主要是3D体积对象、预测值相关的一些键名属性。 |

## 快速测试

可以使用[`ltn_module_segmentation_default`](ltn_module_segmentation_default.py)的主例程进行快速测试或调试以观察执行细节。示例程序对训练、验证、测试、预测4个单步例程都进行了运行测试，并打印其返回结果。

## 使用指南

**主模型配置包装器**本身并无实际功能，应从中获取`LightningModule`模型实例进行使用。它使用1套网络配置`NamedNetworkInitArgs`以及至多4套`ModuleStepAdditionArgs`单步例程附加参数用于配置各例程（训练、验证、测试、预测）的单步计算和打印功能，此数据类的属性描述见下表，每个例程的配置可以不同。

| 属性                                 | 描述                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| network_init_args                    | 带名神经网络参数`NamedNetworkInitArgs`实例。定义用于实例化神经网络模型所需的结构参数。 |
| module_training_step_addition_args   | 训练单步例程附加参数`ModuleTrainingStepAdditionArgs`实例。主模型不用于训练例程时可为空。 |
| module_validation_step_addition_args | 验证单步例程附加参数`ModuleValidationStepAdditionArgs`实例。主模型不用于验证例程时可为空。请注意，在实际训练中通常会在训练周期中穿插验证例程，此时也需要配置此参数项。 |
| module_test_step_addition_args       | 测试单步例程附加参数`ModuleTestStepAdditionArgs`实例。主模型不用于测试例程时可为空。 |
| module_predict_step_addition_args    | 预测单步例程附加参数`ModulePredictStepAdditionArgs`实例。主模型不用于预测例程时可为空。 |

**主模型配置包装器**的典型使用方式如下：以`ConfigLightningModuleSegmentationDefault`为例。

```python
# 实例化ConfigLightningModuleBase
config_module = ConfigLightningModuleSegmentationDefault()
# 设置配置项（例如需要使用训练和验证单步例程）
# NamedNetworkInitArgs配置
config_module.network_init_args = ...
# ModuleTrainingStepAdditionArgs配置
config_module.module_training_step_addition_args = ...
# ModuleValidationStepAdditionArgs配置
config_module.module_validation_step_addition_args = ...
# 获取主模型
ltn_module = config_module.get_ltn_module()
# 将主模型注册到训练器并启动训练例程
trainer.fit(model=ltn_module, datamodule=...)
```

## 自定义指南

进行**自定义**时可参考现有的**主模型配置包装器**`ConfigLightningModuleSegmentationDefault`以及`lightning.LightningModule`**主模型类**。主模型配置包装器应总是从`ConfigLightningModuleBase`派生。

### ConfigLightningModuleBase

**主模型配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`lightning.LightningModule`实例。上层可通过包装器方法获取内部保存的主模型实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和推断器实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类**`__init__`**的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_ltn_module`**：获取包装器内部所包装的主模型实例。

在定义主模型配置包装器前需先完成**主模型**的定义，在此对预设的`LightningModuleSegmentationDefault`所实现的方法进行简要说明，见下表。

`DataModuleSegmentationDefault`主要方法的功能描述见下表。

| 方法                 | 功能                                                         |
| -------------------- | ------------------------------------------------------------ |
| to                   | 将一些用于追溯记录的Tensor以及`torch.nn.Module`算子迁移到指定设备。如果`torch.nn.Module`对象已经在`__init__`中直接定义为成员，则无需在此方法中显式迁移设备，因为基类方法已经能够处理这种情况；如果`torch.nn.Module`对象被包含于容器中，例如`List[torch.nn.Module]`，则需要显式迁移。 |
| forward              | 在内部调用神经网络的前向过程获取预测值。                     |
| training_step        | 定义训练的单步例程。应当包含获取数据、使用推断器执行前向过程、计算损失、计算各类指标、调用自定义钩子方法、返回损失以及其他结果。 |
| validation_step      | 定义验证的单步例程。应当包含获取数据、使用推断器（通常是滑动窗口推断器）执行前向过程、计算损失、计算各类指标、调用自定义钩子方法、返回损失以及其他结果。 |
| test_step            | 定义测试的单步例程。应当包含获取数据、使用推断器（通常是滑动窗口推断器）执行前向过程、计算损失、计算各类指标、调用自定义钩子方法、返回损失以及其他结果。 |
| predict_step         | 定义预测的单步例程。应当包含获取数据、使用推断器（通常是滑动窗口推断器）执行前向过程、计算各类指标、调用自定义钩子方法、返回结果（不强制要求返回损失）。特别的，预测例程可以不使用参照蒙版，因为只需计算预测结果即可，而无需进行评估。 |
| configure_optimizers | 返回配置好的优化器和学习率调度器以便用于训练例程。           |

上表所述方法仅是`lightning.LightningModule`的一部分，完整的使用指南请参阅指南：

- [Train a model (basic) — PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html)
- [Level 2: Add a validation and test set — PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/levels/basic_level_2.html)