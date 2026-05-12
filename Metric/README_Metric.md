# Metric

**Metric（指标）** 负责定义指标函数计算流，以及执行指标计算。包装自TorchMetrics和Monai。

**Metric（指标）** 部分包含1个预设代码文件：

- [`metric_configurer`](metric_configurer.py)：**关键代码**。定义了`ConfigMetricBase`**指标配置包装器**基类，指标配置包装器主要包装自TorchMetrics和Monai Metric，由于2个包库的实现方式差异较大，因此又分别定义了`ConfigMetricTorch`**TorchMetrics指标配置包装器**基类以及`ConfigMetricMonai`**Monai Metric指标配置包装器**基类，其中TorchMetrics指标计算器均派生自`torch.nn.Module`，而Monai Metrics指标计算器则无此继承关系。
  
  - **TorchMetrics指标配置包装器**：来自TorchMetrics的指标计算器都具有`plot`制图方法，为了拓展其功能，从`ConfigMetricTorch`基类单独派生`ConfigMetricTorchScalar`作为所有标量指标包装器的基类，它重写了`plot`制图方法。TorchMetrics包含几乎所有常见的分类指标，包括Acc、AUROC、AP、F1-Score、Prec、Recall、Spec、ROC、PR-Curve、Confusion Matrix，并对每种指标提供二分类（Binary）、多分类（Multiclass）、多标签（Multilabel）3个版本的指标计算器。预设的TorchMetrics指标配置包装器详见下表。代码中有更详细的IO注释，TorchMetrics官方文档请参阅[TorchMetrics in PyTorch Lightning — PyTorch-Metrics documentation](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)。
  
    | TorchMetrics指标配置包装器<br />task={Binary\|Multiclass\|Multilabel} | 功能                                                         |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | ConfigMetric{task}StatScores                                 | 状态得分。每类包含5个指标，即[tp, fp, tn, fn, sup]，sup=tp+fn为支持度。 |
    | ConfigMetric{task}Accuracy                                   | 正确率。衡量整体预测正确样本占总体的比例，(tp+fn)/(tp+fp+tn+fn)。 |
    | ConfigMetric{task}AUROC                                      | 受试者曲线下面积。                                           |
    | ConfigMetric{task}AveragePrecision                           | PR曲线下面积。                                               |
    | ConfigMetric{task}F1Score                                    | F1分数。精确率和召回率的调和平均值，2/(1/prec+1/recall)。    |
    | ConfigMetric{task}Precision                                  | 精确率。预测正例中的正确率，tp/(tp+fp)。                     |
    | ConfigMetric{task}Recall                                     | 召回率/真正例率/敏感性。实际正例中的正确率，tp/(tp+fn)。     |
    | ConfigMetric{task}Specificity                                | 特异度。实际负例中的正确率，tn/(tn+fp)。                     |
    | ConfigMetric{task}ROC                                        | 受试者曲线。基于TPR=tp/(tp+fn)和FPR=fp/(fp+tn)指标绘制。     |
    | ConfigMetric{task}PrecisionRecallCurve                       | PR曲线。基于Prec和Recall指标绘制。                           |
    | ConfigMetric{task}ConfusionMatrix                            | 混淆矩阵。二分类和多分类版本中，矩阵规模为类别数C×C；多标签版本中，则包含类别数C个2×2矩阵。 |
  
  - **Monai Metrics指标配置包装器**：Monai提供了绝大多数医学图像分割评估相关的指标计算器，包括Dice系数、类别加权Generalized Dice系数、IoU、Hausdorff距离、平均表面距离、标准化表面Dice系数，以及一些自监督相关指标，例如MSE、MAE、RMSE、PSNR、SSIM、Multi Scale SSIM。特别的，Monai Metrics需要显式调用`aggregate`方法才能获得归约结果。预设的Monai Metrics指标配置包装器详见下表。代码中有更详细的IO注释，Monai Metrics官方文档请参阅[指标 — MONAI 框架](https://docs.monai.org.cn/en/stable/metrics.html)。
  
    | Monai Metrics指标配置包装器                            | 功能                                                         |
    | ------------------------------------------------------ | ------------------------------------------------------------ |
    | ConfigMetricDiceScore                                  | Dice系数，体重叠度指标，一般认为等同于样本平均F1分数，对每个样本的每个类别计算2tp/(2tp+fp+fn)并求平均值，然后再求所有样本的平均值。 |
    | ConfigMetricGeneralizedDiceScore                       | 类别加权Dice系数，体重叠度指标。对每个样本的每个类别计算2tp/(2tp+fp+fn)，以每个类别在这个样本中的实际支持数量的倒数为权重加权平均计算Dice系数，用于提升对少数类的关注度，然后再求所有样本的平均值。 |
    | ConfigMetricMeanIoU                                    | 平均交并比，体重叠度指标，与Dice系数类似，但计算式变更为tp/(tp+fp+fn)。 |
    | ConfigMetricHausdorffDistance                          | 豪斯多夫距离，边界距离指标，即预测和参照蒙版表面点对距离的最大值（或某分位数）。 |
    | ConfigMetricSurfaceDistance                            | 平均表面距离，边界距离指标，即预测和参照蒙版表面点对距离的均值。 |
    | ConfigMetricNormalizedSurfaceDiceScore                 | 标准化表面Dice系数，边界重叠度指标，衡量表面点的重叠度。     |
    | ConfigMetricMeanSquaredError                           | 均方误差。预测和参照蒙版差值平方和。                         |
    | ConfigMetricMeanAbsoluteError                          | 绝对误差。预测和参照蒙版差值绝对值之和。                     |
    | ConfigMetricRootMeanSquaredError                       | 均方根误差。均方误差的平方根。                               |
    | ConfigMetricPeakSignalToNoiseRatio                     | 峰值信噪比。用于评价图像和图像之间的相似性，而非用于蒙版。下同。 |
    | ConfigMetricStructuralSimilarityIndexMeasure           | 结构相似性。                                                 |
    | ConfigMetricMultiScaleStructuralSimilarityIndexMeasure | 多尺度结构相似性。                                           |

  - **效能指标配置包装器**：效能指标是指与预测精度（一般称作性能指标）无关而关乎计算效率、能量消耗的指标。此类指标从`ConfigMetricEfficiency`基类派生，在使用时一般需要进行双时点调用，即在计算开始前调用一次，在完成计算后再调用一次。此类指标计算器通常需要自行定义。当前提供了一个用于计算每秒处理体素数量的效能指标预设。
  
    | 效能指标配置包装器                   | 功能                   |
    | ------------------------------------ | ---------------------- |
    | ConfigMetricVoxelProcessingPerSecond | 计算每秒处理体素数量。 |
  

## 快速测试

可以使用`Metric/test_metrics`目录下的[`test_monai_metrics.py`](test_metrics/test_monai_metrics.py)和[`test_wrapped_torchmetrics.py`](test_metrics/test_wrapped_torchmetrics.py)中的主例程进行快速测试或调试以观察执行细节。`test_monai_metrics`能够打印Monai Metrics在不同`reduction`模式下的输出形状，`test_wrapped_torchmetrics`则将包装器结果和直接调用TorchMetrics指标计算器的结果进行比较，同时显示指标制图结果。

## 使用指南

**指标配置包装器**本身是可调用对象，其`__call__`方法内部将调用损失函数算子实例完成计算，也可以通过`get_metric_operator`方法获取内部指标计算器实例进行直接调用。包装器预设的使用方式是指标计算器只对单个传入样本计算指标，而不保存其状态，因此总是在完成指标计算后`reset`；如果需要启用指标计算器的自动累积功能以便累积多个样本的计算结果并在最后归约，请通过`get_metric_operator`获取内部指标计算器实例以维持其累积状态。

TorchMetrics的指标包含2类可设置归约模式：

- `average`：可选值范围[micro, macro, weighted, none]。有些指标的可选值范围可能更窄。

  - `micro`：微平均。将所有类别的统计量累积起来，然后直接计算指标。受主要类和类别支持样本数量差异影响大。
  - `macro`：宏平均。先分别求每个类别的指标，然后求所有类别指标的平均。公平对待每类样本。

  - `weighted`：类别加权宏平均。先分别求每个类别的指标，然后按照一定权重系数求所有类别指标的加权平均。
  - `none`：无。分别求每个类别的指标，不执行归约。

- `multidim_average`：可选值范围[global, samplewise]。

  - `global`：忽略额外维度（例如Batch），将所有样本并入一个集合内计算指标。
  - `samplewise`：按照额外维度划分计算集合，在每个计算集合内计算指标。

Monai Metrics的指标可设置`reduction`归约模式：可选值范围[none, mean, sum, mean_batch, sum_batch, mean_channel, sum_channel]。

- `none`：不执行归约。对每个样本的每个通道（类别）都分别计算一个指标。
- `mean`：先在样本内求通道（类别）平均，然后求样本平均。
- `sum`：先在样本内求通道（类别）和，然后求样本和。
- `mean_batch`：对每个通道（类别）的指标分别求样本平均，但不求通道平均。
- `sum_batch`：对每个通道（类别）的指标分别求样本和，但不求通道和。
- `mean_channel`：对每个样本的指标分别求通道（类别）平均，但不求样本平均。
- `sum_channel`：对每个样本的指标分别求通道（类别）和，但不求样本和。

关于指标计算器的IO规格、归约模式等更多使用指南请参阅以下文档（其他具体指标文档可自行查阅）：

- [F-1 Score — PyTorch-Metrics documentation](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)

- [Dice指标 — MONAI 框架](https://docs.monai.org.cn/en/stable/metrics.html#monai.metrics.DiceMetric)

通过`ConfigMetricBase`获取**指标计算器**实例可通过调用基类方法`get_metric_operator`实现。对于派生自`torch.nn.Module`的指标计算器（TorchMetrics），**指标配置包装器**也提供了`to`方法以便迁移算子至指定设备，此方法对Monai Metrics不起任何作用。

```python
# Loss/loss_configurer.py
@dataclass
class ConfigMetricTorch(ConfigMetricBase, metaclass=ABCMeta):
    def to(self, *args, **kwargs) -> 'ConfigMetricBase':
        self._assert_init_essentials()
        return self

    def get_metric_operator(self, *args, **kwargs) -> SupportedMetric:
        self._assert_init_essentials(*args, **kwargs)
        return self.metric

@dataclass
class ConfigMetricTorch(ConfigMetricBase, metaclass=ABCMeta):
    def to(self, *args, **kwargs) -> 'ConfigMetricBase':
        self._assert_init_essentials()
        self.metric.to(*args, **kwargs)
        return self
```

**指标配置包装器**的典型使用方式如下：以`ConfigMetricDiceScore`为例。

```python
# 实例化ConfigMetricBase
config_metric = ConfigMetricDiceScore()
# 设置配置项
config_metric.include_background = False
config_metric.num_classes = 3
...
# 迁移设备（只对TorchMetrics的算子生效）
config_metric.to(device='cuda')
# [可选] 获取损失函数算子
metric_operator = config_metric.get_metric_operator()
# 在step步骤中调用
for step:
    ...
    # 在完成前向传播之后
    # 使用包装器__call__，内部自动归约和重置
    metric = config_metric(pred, gt)
    # [可选] 或者调用损失函数算子__call__，并自行处理归约和重置
    metric = metric_operator(pred, gt)
    # metric.aggregate()  # 归约，其中TorchMetrics指标在单步时无需显式归约
    # metric.reset()  # 重置状态
    ...
# [可选] 对于TorchMetrics指标，可调用制图方法
config_metric.plot(...)
```

## 自定义指南

进行**自定义**时可参考现有的**指标配置包装器**。指标配置包装器应总是从`ConfigMetricBase`派生，并实现延迟初始化的必要接口，除此之外其他功能方法均可自行定义。

### ConfigMetricBase

**指标配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`torchmetrics.metric.Metric`或`monai.metric.Metric`实例。上层可通过包装器方法获取内部保存的指标计算器实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和指标计算器实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类**`__init__`**的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`__call__`**：调用方法。内部调用指标计算器实例，基于输入和参照值计算指标。
- **`to`**：转换设备和数据类型方法。内部调用`torch.nn.Module`（如果从此基类派生）的`to`方法将指标计算器实例迁移至指定设备和转换为指定数据类型。一般只用于迁移设备。
- **`get_metric_operator`**：获取包装器内部所包装的损失算子实例。

在定义指标配置包装器前需先完成**指标计算器**的定义，亦可利用现有指标计算器直接封装，关于预设指标计算器和自定义方法请参阅：

- [All TorchMetrics — PyTorch-Metrics documentation](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html)

- [Implementing a Metric — PyTorch-Metrics documentation](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html)

- [指标 — MONAI 框架](https://docs.monai.org.cn/en/stable/metrics.html)


### ConfigMetricTorch

对于带有制图支持的指标计算器（例如`torchmetrics.metric.Metric`，包装于`ConfigMetricTorch`以及`ConfigMetricTorchScalar`），还可以额外实现**`plot`**方法以便更加精细地或自定义地控制制图过程。

如果指标计算器派生自`torch.nn.Module`，还应重写**`to`**方法以迁移指标计算器到指定设备。
