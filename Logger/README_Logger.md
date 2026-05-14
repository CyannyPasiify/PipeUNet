# Logger

**Logger（日志器）** 负责逐步骤持久化主模型状态记录，提供日志信息的结构化和可视化。例如CSV和Tensorboard日志器。包装自Lightning。

**Logger（日志器）** 部分包含1个预设代码文件：

- [`logger_configurer`](logger_configurer.py)：**关键代码**。定义了`ConfigLoggerBase`**日志器配置包装器**基类，以及若干常用日志器的包装器预设，包括`ConfigLoggerCSV`、`ConfigLoggerTensorBoard`和`ConfigLoggerWandb`。

  | 日志器配置包装器        | 功能                      |
  | ----------------------- | ------------------------- |
  | ConfigLoggerCSV         | CSV表格日志器。           |
  | ConfigLoggerTensorBoard | TensorBoard仪表盘日志器。 |
  | ConfigLoggerWandb       | WandB在线仪表盘日志器。   |

## 快速测试

可以使用[`logger_configurer`](logger_configurer.py)的主例程进行快速测试或调试以观察执行细节。示例程序将构建一些示例指标记录和绘图对所有日志器配置包装器进行测试，指标类型涵盖文本、布尔、整数、实数、列表、Numpy数值对象。

## 使用指南

**日志器配置包装器**是日志器的容器，实际功能由内部日志器实例提供支持。获取实例可通过调用基类方法`get_logger`实现。

```python
# Logger/logger_configurer.py
def get_logger(self, *args, **kwargs) -> loggers.Logger:
    self._assert_init_essentials(*args, **kwargs)
    return self.logger
```

**日志器配置包装器**的典型使用方式如下：以`ConfigLoggerCSV`为例。

```python
# 实例化ConfigLoggerBase
config_logger = ConfigLoggerCSV()
# 设置配置项
config_logger.save_dir = "Experiments"
config_logger.name = "csv_logs"
config_logger.version = "v0"
# 以上配置将确定目标保存目录"{save_dir}/{name}/{version}"
config_logger.prefix = ""  # 添加到指标键名的前缀
config_logger.flush_logs_every_n_steps = 100  # 每多少步日志写入一次文件...
# 获取日志器实例
logger = config_logger.get_logger()
# 注册到Lightning训练器以便在例程中调用
trainer = lightning.pytorch.trainer.trainer.Trainer(logger=[logger])
```

## 自定义指南

进行**自定义**时可参考现有的**日志器配置包装器**。日志器配置包装器应总是从`ConfigLoggerBase`派生。

### ConfigLoggerBase

**日志器配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`lightning.pytorch.loggers.Logger`实例。上层可通过包装器方法获取内部保存的日志器实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和日志器实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类 **`__init__`** 的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_logger`**：获取包装器内部所包装的日志器实例。

在定义日志器配置包装器前需先完成**日志器**的定义，亦可利用现有日志器直接封装：

- [Logging — PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)