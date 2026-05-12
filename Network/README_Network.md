# Network

**Network（网络）** 负责定义神经网络架构和前馈计算流。扩展自定义于PyTorch。

**Network（网络）** 部分包含3个预设代码文件：

- [`network_configurer`](network_configurer.py)：定义了`ConfigNetworkBase`**网络配置包装器**基类，以及1个基础的4阶段-5尺度-3D-UNet包装器预设`ConfigNetworkUNet`。

  | 网络配置包装器    | 功能                                |
  | ----------------- | ----------------------------------- |
  | ConfigNetworkUNet | 配置和构建一个4阶段-5尺度-3D-UNet。 |

- [`module_block`](module_block.py)：*<u>辅助代码</u>*。定义了一些描述性接口和派生自`torch.nn.Module`的常用复合模块。

  | 接口或模块    | 功能                                                         |
  | ------------- | ------------------------------------------------------------ |
  | IODescriptive | 抽象基类。提供一个用于描述IO规格的字符串描述方法`io_description`。 |
  | ConvNormAct   | 根据所提供的**卷积层-标准化层-激活层**实例，构建一个顺序连接的复合模块。 |
  | ConvBNReLU    | 一个用于快速配置**标准卷积层-BatchNormalization-ReLU**复合模块的快捷类。 |
  | Concat        | 一个包装为模块的通道拼接算子。                               |

- [`module_unet`](module_unet.py)：**关键代码**。定义3D-UNet主干模型以及用于构造3D-UNet的若干子模块，均派生自`torch.nn.Module`。`module_unet`代码中提供了更详细的图文注释，如有需要请参阅[module_unet.py](module_unet.py)。

  | 模块                                      | 功能                                                         |
  | ----------------------------------------- | ------------------------------------------------------------ |
  | UNet                                      | 3D-UNet最上级复合模块。                                      |
  | UNetFocuser                               | 最前端的特征嵌入模块，预编码器。负责扩张通道数量。由Conv-Norm-Act模块堆叠而成。 |
  | UNetEncoderPriorBank                      | 编码器先验池。负责为编码器多阶段准备先验嵌入特征。*<u>此模块在预设中只是占位符。</u>* |
  | UNetEncoderPriorBankInjector              | 编码器先验池中用于各阶段先验嵌入的投影器。*<u>此模块在预设中只是占位符。</u>* |
  | UNetEncoder                               | 编码器主干模块。                                             |
  | UNetEncoderPrimaryExtractor               | 编码器初级特征提取器。负责提取小范围细粒度的初级特征，顺序通过阶段模块和下采样模块。预设中由2个阶段构成。 |
  | UNetEncoderPrimaryExtractorStage          | 编码器初级特征提取器阶段模块。负责提取当前尺度特征的主体，由Conv-Norm-Act模块堆叠而成。 |
  | UNetEncoderPrimaryExtractorDownsample     | 编码器初级特征提取器下采样模块。负责下采样当前尺度特征，产生更大尺度特征。 |
  | UNetEncoderAdvancedExtractor              | 编码器高级特征提取器。负责提取大范围粗粒度的高级特征，顺序通过阶段模块和下采样模块。预设中由2个阶段构成。 |
  | UNetEncoderAdvancedExtractorStage         | 编码器高级特征提取器阶段模块。负责提取当前尺度特征的主体，由Conv-Norm-Act模块堆叠而成。 |
  | UNetEncoderAdvancedExtractorDownsample    | 编码器高级特征提取器下采样模块。负责下采样当前尺度特征，产生更大尺度特征。 |
  | UNetRepeater                              | 中继器。负责在编码器和解码器之间转换和传递特征。包含桥接层和瓶颈层，桥接层一般在每个阶段单独设立，可以是跳跃连接，而瓶颈层一般专门定义为用于处理最大尺度（高级）特征的模块。也有一些模型采用混合尺度的中继器，此时桥接层和瓶颈层没有明确边界。 |
  | UNetRepeaterBridge                        | 中继器桥接层。预设中使用跳跃连接。                           |
  | UNetRepeaterBottleneck                    | 中继器瓶颈层。预设中使用Conv-Norm-Act模块堆叠而成。          |
  | UNetDecoderPriorBank                      | 解码器先验池。负责为解码器多阶段准备先验嵌入特征。*<u>此模块在预设中只是占位符。</u>* |
  | UNetDecoderPriorBankInjector              | 解码器先验池中用于各阶段先验嵌入的投影器。*<u>此模块在预设中只是占位符。</u>* |
  | UNetDecoder                               | 解码器主干模块。按照逐步聚合特征模式编排。                   |
  | UNetDecoderAdvancedAggregator             | 解码器高级特征聚合器。负责聚合大范围粗粒度的高级特征，由上采样、融合门户和阶段模块构成。预设中由2个阶段构成。 |
  | UNetDecoderAdvancedAggregatorUpsample     | 解码器高级特征聚合器上采样模块。负责上采样下级大尺度特征，产生更小尺度特征。 |
  | UNetDecoderAdvancedAggregatorFusionPortal | 解码器高级特征聚合器融合门户。负责预处理来自下级的大尺度特征，使其能够与当前尺度特征对齐，以便进行聚合。 |
  | UNetDecoderAdvancedAggregatorStage        | 解码器高级特征聚合器阶段模块。负责聚合当前尺度特征（一部分特征来自桥接层，另一部分特征来自于经融合门户处理后的下级大尺度特征）的主体，由Conv-Norm-Act模块堆叠而成。 |
  | UNetDecoderPrimaryAggregator              | 解码器初级特征聚合器。负责聚合小范围细粒度的初级特征，由上采样、融合门户和阶段模块构成。预设中由2个阶段构成。 |
  | UNetDecoderPrimaryAggregatorUpsample      | 解码器初级特征聚合器上采样模块。负责上采样下级大尺度特征，产生更小尺度特征。 |
  | UNetDecoderPrimaryAggregatorFusionPortal  | 解码器初级特征聚合器融合门户。负责预处理来自下级的大尺度特征，使其能够与当前尺度特征对齐，以便进行聚合。 |
  | UNetDecoderPrimaryAggregatorStage         | 解码器初级特征聚合器阶段模块。负责聚合当前尺度特征（一部分特征来自桥接层，另一部分特征来自于经融合门户处理后的下级大尺度特征）的主体，由Conv-Norm-Act模块堆叠而成。 |
  | UNetAuxiliaryClassifier                   | 辅助分类器。连接在每个解码器阶段模块之后的逐点分类器，用于产生各尺度的类别预测值谱。 |
  | UNetDistributor                           | 末端的特征分配模块。负责将聚合特征图的空间规格恢复为原始输入规格或预测目标规格，也负责压缩聚合特征维度。由Conv-Norm-Act模块堆叠而成。 |
  | UNetClassifier                            | 主分类器。网络最末端的逐点分类器，用于产生目标规格的类别预测值谱。 |

## 快速测试

[`module_unet`](module_unet.py)的主例程提供了预设3D-UNet各主要模块的IO规格测试，可以使用此例程观察模型各模块的输入输出规格变化。

## 使用指南

**网络配置包装器**是神经网络实例的容器，实际功能由内部神经网络实例提供支持。获取实例可通过调用基类方法`get_network_module`实现。

```python
# Network/network_configurer.py
def get_network_module(self, *args, **kwargs) -> Module:
    self._assert_init_essentials(*args, **kwargs)
    return self.network
```

**网络配置包装器**的典型使用方式如下：以`ConfigNetworkUNet`为例。

```python
# 实例化ConfigNetworkBase
config_network = ConfigNetworkUNet()
# 设置配置项
config_network.focuser_in_channels = 1
...
config_network.classifier_out_channels = 2
# 获取网络
model = config_network.get_network_module()
# 在step步骤中调用
for step:
    # 前向传播
    pred = model(batch)
    ...
```

## 自定义指南

进行**自定义**时可参考现有的**网络配置包装器**。网络配置包装器应总是从`ConfigNetworkBase`派生。但是，更关键的步骤在于定义网络结构本身，网络配置包装器只需要汇总和记录网络所需的全部结构参数即可。

### ConfigNetworkBase

**网络配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`torch.nn.Module`实例。上层可通过包装器方法获取内部保存的网络实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和网络实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类**`__init__`**的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_network_module`**：获取包装器内部所包装的网络实例。

在定义网络配置包装器前需先完成**网络**的定义，请参阅：

- [torch.nn — PyTorch documentation](https://docs.pytorch.org/docs/stable/nn.html)