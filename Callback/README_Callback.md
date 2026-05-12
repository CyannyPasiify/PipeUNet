# Callback

**Callback（回调）** 模组负责定义例程控制和表现相关的自定义功能，通过例程中的钩子进行调用。常用回调功能包括早停策略介入、模型检查点持久化、模型预览和进度条渲染。包装自Lightning。

**Callback（回调）** 部分包含1个预设代码文件：

- [`callback_configurer`](callback_configurer.py)：**关键代码**。定义了`ConfigCallbackBase`**回调配置包装器**基类，以及来自Lightning的8个常用预设回调的配置包装器。

  | 回调配置包装器                    | 功能                                                         |
  | --------------------------------- | ------------------------------------------------------------ |
  | ConfigCallbackDeviceStatsMonitor  | 监控CPU、GPU设备参数并按一定频率打印日志。                   |
  | ConfigCallbackEarlyStopping       | 早停策略介入，可配置等待耐心、指标改善阈值等参数。           |
  | ConfigCallbackLearningRateMonitor | 监控优化器动态参数，包括学习率和动量等信息并按一定频率打印日志。 |
  | ConfigCallbackModelCheckpoint     | **最常用回调**。用于监控指标变化并生成模型检查点文件，具有根据配置自动维护保存数量（Top-K）等功能。 |
  | ConfigCallbackModelSummary        | 在例程开始前打印模型概览，可在此观察模型的构成子模块情况以及各模块的参数量。 |
  | ConfigCallbackRichModelSummary    | 在例程开始前打印模型概览，这是一个使用rich包库优化显示效果的版本。 |
  | ConfigCallbackRichProgressBar     | 控制台进度条提示，这是一个使用rich包库优化显示效果的版本。   |
  | ConfigCallbackTQDMProgressBar     | 控制台TQDM进度条提示，可以配置刷新频率和显示位置。           |

## 快速测试

可以使用[`callback_configurer`](callback_configurer.py)的主例程和本项目提供的示例样本（样本在`Samples`目录中）进行快速测试或调试以观察执行细节。命令行参数设置`--callback all`以测试全部回调包装器，或指定要测试的回调包装器名称（请去除共同前缀`ConfigCallback`）。请从项目根启动主例程以确保相对路径的正确性。

## 使用指南

本文档只对最常用回调`ConfigCallbackModelCheckpoint`的使用进行专门说明，其他回调的使用请参见Lightning文档。

`ConfigCallbackModelCheckpoint`封装了Lightning中`ModelCheckpoint`的几项常用参数，其他不常用参数则被隐藏。常用参数的描述见下表。

| 属性           | 描述                                                         |
| -------------- | ------------------------------------------------------------ |
| dirpath        | 保存检查点的目录。可以指定绝对或相对路径，使用相对路径时基准目录由训练器（Trainer）参数`default_root_dir`确定，默认为当前工作目录。 |
| filename       | 设置保存检查点文件名称的格式化文本。默认为'{epoch}-{step}'，亦可指定详细的格式化选项，例如'{epoch}-{val_loss:.2f}-{other_metric:.2f}'。 |
| monitor        | 监控指标名称。保存检查点一般需要以某一项具体指标的高低变化来确认保存时机。 |
| save_top_k     | 控制保存检查点的数量。-1保存所有检查点，为非负数时保存所监控指标最优的k个模型检查点。 |
| mode           | 优势模式。可以设置为min或max，用于指示监控指标是越小越好还是越大越好。 |
| save_last      | 是否总是保存一个`last.ckpt`副本，其内容指向最新保存的检查点。 |
| every_n_epochs | 保存检查点的周期间隔，即每个多少个epoch进行一次检查点保存逻辑。训练器（Trainer）参数`check_val_every_n_epoch`会对此产生连带影响，只有当epoch数为二者公倍数时才触发检查点保存逻辑。通常设置为1。 |

**回调配置包装器**需注册到Lightning**训练器**中进行使用：以`ConfigCallbackModelCheckpoint`为例。

```python
# 实例化ConfigCallbackBase
config_ckpt = ConfigCallbackModelCheckpoint()
# 设置配置项
config_ckpt.dirpath = 'my_train/v0'  # 检查点保存至实验目录下的'my_train/v0'子目录
config_ckpt.filename = '{epoch}-{val_loss:.2f}'  # 保存文件名，形如'10-0.15.ckpt'
config_ckpt.monitor = 'val_loss'  # 监控'val_loss'损失
config_ckpt.mode = 'min'  # 损失越小越好
...
# 获取检查点回调实例
callback_ckpt = config_ckpt.get_callback_hooker()
# 注册到Lightning训练器以便在例程中调用
trainer = lightning.pytorch.trainer.trainer.Trainer(callbacks=[callback_ckpt])
```

## 自定义指南

进行**自定义**时可参考现有的**回调配置包装器**。回调配置包装器应总是从`ConfigCallbackBase`派生。

### ConfigCallbackBase

**回调配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用已定义的回调实例。上层可通过包装器方法获取内部保存的回调实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和回调实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类__init__的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_callback_hooker`**：获取包装器内部所包装的回调实例。

`transform_configurer`已经对Lightning的所有回调进行了包装，且回调通常不需要进行自定义。如确实需要自定义回调类，请参见以下Lightning文档进行自定义，然后再进行包装。

- [Customize the progress bar](https://lightning.ai/docs/pytorch/stable/common/progress_bar.html)
- [Customize checkpointing behavior (intermediate)](https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html)

