# Trainer

**Trainer（训练器）** 负责定义多种例程管线。称作训练器只是一种命名传统，实际上是例程管理器（Routine Manager）。其中定义了针对训练、验证、测试、预测例程的可编程管线。包装自Lightning。

**Trainer（训练器）** 部分包含1个预设代码文件：

- [`trainer_configurer`](trainer_configurer.py)：**关键代码**。定义了`ConfigTrainerBase`**训练器配置包装器**基类，以及用于分割任务的训练器预设`ConfigTrainerSegmentationDefault`。其中还定义了一系列用作训练器参数类型的数据类。

  | 训练器配置包装器                 | 功能                                                         |
  | -------------------------------- | ------------------------------------------------------------ |
  | ConfigTrainerSegmentationDefault | 分割任务训练器预设。负责根据配置参数实例化训练器、并实例化和注册回调、日志器，定义了fit、validate、test、predict 4个功能例程。 |

  | 数据类           | 功能                                                         |
  | ---------------- | ------------------------------------------------------------ |
  | TrainerInitArgs  | 训练器参数。包括平台控制参数（加速设备、精度、并行策略）、例程控制参数（周期数、验证频率）、梯度控制参数（梯度累积批次、梯度裁剪）、日志控制参数（日志频率、进度条、模型概览、启用检查点）、可复现性控制参数以及调试控制参数。 |
  | CallbackInitArgs | 回调参数。包括在[回调子模组](../Callback/README_Callback.md)中预设的各种回调配置包装器类型的定义字段，其中`ConfigCallbackModelCheckpoint`检查点采用列表容器，允许配置多个。 |
  | LoggerInitArgs   | 日志器参数。包含一系列开关变量，用于控制是否启用各种日志器。 |

## 使用指南

**训练器配置包装器**是训练器的容器，同时也对内部训练器实例的主要功能例程（fit、validate、test、predict）进行了包装。获取实例可通过调用基类方法`get_trainer`实现。

```python
# Trainer/trainer_configurer.py
def get_trainer(self) -> Trainer:
    self._assert_init_essentials()
    return self.trainer
```

**训练器配置包装器**的典型使用方式如下：以`ConfigTrainerSegmentationDefault`为例。

```python
# 实例化ConfigTrainerBase
config_trainer = ConfigTrainerSegmentationDefault()
# 设置配置项
config_trainer.experiment_root_dir = "Experiments"
config_trainer.experiment_name = "pipeunet"
config_trainer.experiment_version = "v0"
config_trainer.trainer_init_args = TrainerInitArgs(accelerator='gpu', devices=[1], ...)
config_trainer.callback_init_args = CallbackInitArgs(...)
config_trainer.logger_init_args = LoggerInitArgs(...)
# 以上配置将确定目标保存目录"{save_dir}/{name}/{version}"
config_trainer.prefix = ""  # 添加到指标键名的前缀
config_trainer.flush_logs_every_n_steps = 100  # 每多少步日志写入一次文件...
# 注册模型到训练器
model = SomeModule()
datamodule = SomeDataModule()
# 启动例程（以fit为例）
config_trainer.fit(model, datamodule)
# 也可以获取训练器实例直接控制
trainer = config_trainer.get_trainer()
trainer.fit(model, datamodule)
```

## 自定义指南

进行**自定义**时可参考现有的**训练器配置包装器**。训练器配置包装器应总是从`ConfigTrainerBase`派生。

### ConfigTrainerBase

**训练器配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`lightning.pytorch.trainer.Trainer`实例。上层可通过包装器方法获取内部保存的训练器实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和训练器实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类 **`__init__`** 的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_trainer`**：获取包装器内部所包装的训练器实例。
- **`fit/validate/test/predict`**：具体功能例程。

在定义训练器配置包装器前需先完成**训练器**的定义，亦可利用现有训练器直接封装：

- [Trainer — PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html)