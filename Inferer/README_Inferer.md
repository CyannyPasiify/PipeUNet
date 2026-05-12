# Inferer

**Inferer（推断器）** 负责定义网络针对复杂输入数据的调用方式和网络输出的统筹方式。通常用于针对需要分片处理的大型单个样本的推断任务中。包装自Monai。

**Inferer（推断器）** 部分包含1个预设代码文件：

- [`inferer_configurer`](inferer_configurer.py)：**关键代码**。定义了`ConfigInfererBase`**推断器配置包装器**基类。其中提供一个不带任何特殊处理的简单推断器`ConfigInfererSimple`，以及用于大规格3D医学图像的滑动窗口推断器`ConfigInfererSlidingWindow`和一个支持多尺度推断的`ConfigInfererMainWithAuxSlidingWindow`推断器，这些推断器均包装自Monai Inferer。

  | 推断器配置包装器                      | 功能                                                         |
  | ------------------------------------- | ------------------------------------------------------------ |
  | ConfigInfererSimple                   | 简单推断器。此推断器不执行任何特殊处理，直接调用网络的前向过程。 |
  | ConfigInfererSlidingWindow            | 滑动窗口推断器。此推断器执行特殊的预处理和后处理，预处理部分按照一定重叠比例对原图执行滑动窗口分片，然后调用网络的前向过程对每个分片进行预测，最后按照一定加权策略对预测分片进行镶嵌后处理。 |
  | ConfigInfererMainWithAuxSlidingWindow | 多尺度滑动窗口推断器。对于能够产生多尺度预测结果的模型，此推断器对每个尺度结果均按照既定重叠比例进行后处理镶嵌，从而产生多尺度的全图预测结果。 |

## 使用指南

**配置包装器**是推断器的容器，实际功能由内部推断器实例提供支持。获取实例可通过调用基类方法`get_inferer_operator`实现。

```python
# Inferer/inferer_configurer.py
def get_inferer_operator(self, *args, **kwargs) -> mI.Inferer:
    self._assert_init_essentials(*args, **kwargs)
    return self.inferer
```

**推断器配置包装器**的典型使用方式如下：以`ConfigInfererSlidingWindow`为例。

```python
# 实例化ConfigInfererBase
config_inferer = ConfigInfererSlidingWindow()
# 设置配置项
config_inferer.roi_size = (128, 128, 128)
config_inferer.overlap = 0.25
...
# 获取推断器
inferer = config_inferer.get_inferer_operator()
# 在step步骤中调用，多用于验证、测试、预测例程
for step:
    ...
    # 此时的数据对象batch应当包含全图，而非局部切片
    # forward前向过程由inferer代理调用
    pred = inferer(batch)
    # 此后可继续计算损失和指标
    ...
```

## 自定义指南

进行**自定义**时可参考现有的**推断器配置包装器**。推断器配置包装器应总是从`ConfigInfererBase`派生。

### ConfigInfererBase

**推断器配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`monai.losses._Loss`实例。上层可通过包装器方法获取内部保存的推断器实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和推断器实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类**`__init__`**的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`__call__`**：调用方法。内部调用推断器实例，对全图输入自动完成预设的分片和镶嵌流程。
- **`get_inferer_operator`**：获取包装器内部所包装的推断器实例。

在定义推断器配置包装器前需先完成**推断器**的定义，亦可利用现有推断器直接封装：

- [推理方法 — MONAI 框架](https://docs.monai.org.cn/en/stable/inferers.html)
