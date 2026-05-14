# LRScheduler

**LRScheduler（学习率调度器）** 负责定义学习率调度器。包装自PyTorch。

**LRScheduler（学习率调度器）** 部分包含1个预设代码文件：

- [`lrscheduler_configurer`](lrscheduler_configurer.py)：**关键代码**。定义了`ConfigLRSchedulerBase`**学习率调度器配置包装器**基类，以及若干常用学习率调度器的包装器预设，包括`ConfigOptimizerSGD`和`ConfigOptimizerAdamW`。

  | 学习率调度器包装器                           | 功能                                                         |
  | -------------------------------------------- | ------------------------------------------------------------ |
  | ConfigLRSchedulerLinear                      | 线性学习率调度器。在两个指定的学习率缩放因子之间线性过渡，以优化器学习率为缩放基准。具有稳定的变化率。 |
  | ConfigLRSchedulerCosineAnnealing             | 余弦退火学习率调度器。以优化器学习率为缩放基准，以余弦函数正半周期曲线为缩放因子函数进行衰减，具有衰减率慢-快-慢的特点。 |
  | ConfigLRSchedulerCosineAnnealingWarmRestarts | 余弦退火热重启学习率调度器。采用余弦退火方式调度，但执行多次重启，可对重启的幅值和余弦函数周期进行控制。 |
  | ConfigLRSchedulerOneCycle                    | OneCycle学习率调度器。其参数覆盖优化器学习率。首先经过一个学习率逐步上升的预热过程，然后开始衰减，学习率具有先升后降的特点。 |
  | ConfigLRSchedulerReduceLROnPlateau           | 平台期自适应学习率衰减学习率调度器。以优化器学习率为缩放基准，此调度器需要监控损失，并根据损失的变化自适应决定是否衰减学习率，一般工作模式为：当在N个周期内，监控指标下降幅度不超过TH时，将学习率衰减为之前的R倍。 |

## 快速测试

可以使用[`lrscheduler_configurer`](lrscheduler_configurer.py)的主例程进行快速测试或调试以观察执行细节。示例程序对学习率调度器包装器的一个完整迭代周期进行模拟（对于ReduceLROnPlateau，还会构建损失结果），并绘制学习率变化曲线图以供观察。

## 使用指南

**学习率调度器包装器**是学习率调度器的容器，实际功能由内部学习率调度器实例提供支持。获取实例可通过调用基类方法`get_lr_scheduler`实现。

```python
# Optimizer/lrscheduler_configurer.py
def get_lr_scheduler(
    self,
    optimizer: torch.optim.Optimizer
) -> Optional[lr_scheduler.LRScheduler]:
    if optimizer is None:
        if self.is_ready():
            return self.lr_scheduler
        else:
            return None
    self._assert_init_essentials(optimizer)
    return self.lr_scheduler
```

**学习率调度器包装器**的典型使用方式如下：以`ConfigLRSchedulerLinear`为例。

```python
# 实例化ConfigLRSchedulerBase
config_lr_scheduler = ConfigLRSchedulerLinear()
# 设置配置项
config_lr_scheduler.start_factor = 0.01
config_lr_scheduler.end_factor = 0.0001
...
# 注册模型参数和优化器
model = SomeNetwork(...)
optimizer = SomeOptimizer(model.parameters())
lr_scheduler = config_lr_scheduler.get_lr_scheduler(optimizer)
# 在step/epoch中调用
for step | epoch:
    ...
    # 在完成前向传播和损失计算之后
    optimizer.step()
    lr_scheduler.step()
```

## 自定义指南

进行**自定义**时可参考现有的**学习率调度器包装器**。学习率调度器包装器应总是从`ConfigOptimizerBase`派生。

### ConfigLRSchedulerBase

**学习率调度器包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`torch.optim.lr_scheduler.LRScheduler`实例。上层可通过包装器方法获取内部保存的学习率调度器实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和学习率调度器实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类 **`__init__`** 的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_lr_scheduler`**：获取包装器内部所包装的学习率调度器实例。

在定义学习率调度器包装器前需先完成**学习率调度器**的定义，亦可利用现有算子直接封装：

- [torch.optim.lr_scheduler — PyTorch documentation](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)