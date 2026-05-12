# Loss

**Loss（损失）** 负责定义损失函数计算流，以及计算可微分形式的损失。包装自Monai。

**Loss（损失）** 部分包含1个预设代码文件：

- [`loss_configurer`](loss_configurer.py)：**关键代码**。定义了`ConfigLossBase`**损失配置包装器**基类和`ConfigLossDeepSupervision`**深监督损失配置包装器**基类。其中提供用于医学图像常见分类、分割、自监督任务的多个损失配置包装器预设，包括Focal Loss、Dice Loss、DiceCE Loss以及Deep Supervision DiceCE Loss，这些损失函数均包装自Monai Loss。

  | 损失配置包装器                                | 功能                                                         |
  | --------------------------------------------- | ------------------------------------------------------------ |
  | ConfigLossFocal                               | 聚焦损失，交叉熵变体。可调节类别惩罚力度和聚焦难分样本。     |
  | ConfigLossTversky                             | Tversky损失。可调节假阳性和假阴性惩罚力度。                  |
  | ConfigLossContrastive                         | 归一化温度缩放交叉熵损失‌。最大化正样本对的相似度，同时最小化与其他所有样本（负样本）的相似度。温度超参数越高对错误的惩罚力度越大。 |
  | ConfigLossDice                                | 用于分割的Dice损失。                                         |
  | ConfigLossMaskedDice                          | 基于蒙版的Dice损失，允许传递一个蒙版来决定计算范围。         |
  | ConfigLossDeepSupervisionDice                 | 多尺度深监督Dice损失。                                       |
  | ConfigLossGeneralizedDice                     | 类别加权Dice损失。以每个类别实际样本量的倒数为权重，用于提升对少数类的关注度。 |
  | ConfigLossDeepSupervisionGeneralizedDice      | 多尺度深监督类别加权Dice损失。                               |
  | ConfigLossDiceCE                              | Dice损失和交叉熵混合损失。                                   |
  | ConfigLossDeepSupervisionDiceCE               | 多尺度深监督Dice损失和交叉熵混合损失。                       |
  | ConfigLossDiceFocal                           | Dice损失和聚焦损失混合损失。                                 |
  | ConfigLossDeepSupervisionDiceFocal            | 多尺度深监督Dice损失和聚焦损失混合损失。                     |
  | ConfigLossGeneralizedDiceFocal                | 类别加权Dice损失和聚焦损失混合损失。                         |
  | ConfigLossDeepSupervisionGeneralizedDiceFocal | 多尺度深监督类别加权Dice损失和聚焦损失混合损失。             |
  | ConfigLossHausdorffDT                         | 豪斯多夫距离损失。这是一种从边界距离度量的分割准确性损失。   |
  | ConfigLossSSIM                                | 结构相似性损失。一般用于自监督任务用于评估恢复图像与原图的相似性。 |
  | ConfigLossPerceptual                          | 感知损失。采用表征模型提取特征，在特征空间中评估恢复图像与原图的相似性。 |

## 快速测试

可以使用[`loss_configurer`](loss_configurer.py)的主例程进行快速测试或调试以观察执行细节。示例程序构建随机输入对所有损失配置包装器进行测试，打印每个损失函数的输入张量、对照张量和损失结果张量的形状。

## 使用指南

**损失配置包装器**本身是可调用对象，其`__call__`方法内部将调用损失函数算子实例完成计算，也可以通过`get_loss_operator`方法获取内部损失函数算子实例进行直接调用。

在所有预设的**损失配置包装器**中，分割任务相关的损失函数支持最为全面，基本上可归纳为以下类型的组合。

| 使用分类损失 | 使用重叠度损失        | 使用深监督     |
| ------------ | --------------------- | -------------- |
| 不使用       | 不使用                | 不使用         |
| 交叉熵CE     | Dice                  | 多尺度深监督DS |
| 聚焦Focal    | 类平衡GeneralizedDice |                |

值得注意的是如果使用深监督，请确保网络模型确实能够生成用以支持深监督的多尺度预测结果，并确保网络输出与深监督损失函数所需输入规格匹配。这需要专门进行人工检查。

相较于以上在分割任务中最常用的混合损失形式，Tversky Loss和Contrastive Loss则是通用的分类损失，亦可适配分割任务（视作逐点分类）。HausdorffDT Loss则是一个边界距离度量损失，它属于分割视角下区别于重叠度度量的另一类评估模式。SSIM Loss和Perceptual Loss更多见于自监督和预训练任务。

损失配置包装器预设主要基于Monai Loss实现，关于预设损失函数的更多使用指南请参阅[损失函数 — MONAI 框架](https://docs.monai.org.cn/en/stable/losses.html)。

获取**损失函数**实例可通过调用基类方法`get_loss_operator`实现。由于损失函数类通常派生自`torch.nn.Module`，**损失配置包装器**也提供了`to`方法以便迁移算子至指定设备。

```python
# Loss/loss_configurer.py
def to(self, device: torch.device, dtype: torch.dtype) -> 'ConfigLossBase':
    self._assert_init_essentials()
    self.loss.to(device=device, dtype=dtype)
    return self

def get_loss_operator(self, *args, **kwargs) -> nn.Module:
    self._assert_init_essentials(*args, **kwargs)
    return self.loss
```

**损失配置包装器**的典型使用方式如下：以`ConfigLossDice`为例。

```python
# 实例化ConfigLossBase
config_loss = ConfigLossDice()
# 设置配置项
config_loss.softmax = True
config_loss.weight = [0.1, 0.9]
...
# 迁移设备
config_loss.to(device='cuda')
# 获取损失函数算子
loss_operator = config_loss.get_loss_operator()
# 在step步骤中调用
for step:
    ...
    # 在完成前向传播之后
    loss = loss_operator(pred, gt)
    ...
    loss.backward()
```

## 自定义指南

进行**自定义**时可参考现有的**损失配置包装器**。损失配置包装器应总是从`ConfigLossBase`派生，深监督的损失配置包装器亦可从`ConfigLossDeepSupervision`基类派生。

### ConfigLossBase

**损失配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`monai.losses._Loss`实例。上层可通过包装器方法获取内部保存的损失算子实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和损失算子实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类**`__init__`**的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`__call__`**：调用方法。内部调用损失算子实例，基于输入和参照值计算损失。
- **`to`**：转换设备和数据类型方法。内部调用`torch.nn.Module`的`to`方法将损失算子实例迁移至指定设备和转换为指定数据类型。一般只用于迁移设备。
- **`get_loss_operator`**：获取包装器内部所包装的损失算子实例。

在定义损失配置包装器前需先完成**损失算子**的定义，亦可利用现有算子直接封装：

- [损失函数 — MONAI 框架](https://docs.monai.org.cn/en/stable/losses.html)

- [torch.nn — PyTorch documentation](https://docs.pytorch.org/docs/stable/nn.html#loss-functions)
