# Optimizer

**Optimizer（优化器）** 负责定义网络参数优化器。包装自PyTorch。

**Optimizer（优化器）** 部分包含1个预设代码文件：

- [`optimizer_configurer`](optimizer_configurer.py)：**关键代码**。定义了`ConfigOptimizerBase`**优化器配置包装器**基类，以及若干常用优化器的包装器预设，包括`ConfigOptimizerSGD`和`ConfigOptimizerAdamW`。

  | 优化器配置包装器     | 功能          |
  | -------------------- | ------------- |
  | ConfigOptimizerSGD   | SGD优化器。   |
  | ConfigOptimizerAdamW | AdamW优化器。 |

## 快速测试

可以使用[`optimizer_configurer`](optimizer_configurer.py)的主例程进行快速测试或调试以观察执行细节。示例程序构建随机输入对所有优化器配置包装器进行测试，优化器将被迭代一定步数，并比较重复实验时的可复现性。

## 使用指南

**优化器配置包装器**是优化器的容器，实际功能由内部优化器实例提供支持。获取实例可通过调用基类方法`get_optimizer`实现。

```python
# Optimizer/optimizer_configurer.py
def get_optimizer(
    self,
    params: Optional[Iterable[torch.nn.parameter.Parameter]] = None
) -> Optional[optim.Optimizer]:
    if params is None:
        if self.is_ready():
            return self.optimizer
        else:
            return None
    self._assert_init_essentials(params)
    return self.optimizer
```

**优化器配置包装器**的典型使用方式如下：以`ConfigOptimizerSGD`为例。

```python
# 实例化ConfigOptimizerBase
config_optimizer = ConfigOptimizerSGD()
# 设置配置项
config_optimizer.lr = 1e-4
...
# 注册模型参数
model = SomeNetwork(...)
optimizer = config_optimizer.get_optimizer(model.parameters())
# 在step步骤中调用
for step:
    ...
    # 在完成前向传播和损失计算之后
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 自定义指南

进行**自定义**时可参考现有的**优化器配置包装器**。优化器配置包装器应总是从`ConfigOptimizerBase`派生。

### ConfigOptimizerBase

**优化器配置包装器**的抽象基类，其设计理念如下：

- **数据类**：这是一个可序列化的数据类，其成员可公开访问，可仅作为结构化参数传递而不执行任何功能。
- **包装器**：这是一个包装器，使用成员引用`torch.optim.Optimizer`实例。上层可通过包装器方法获取内部保存的优化器实例。
- **延迟计算**：只在显式初始化或调用功能性方法时才创建所需成员和优化器实例。

基类声明以下方法：

- **`is_ready`**：判断这个类是否已被初始化过。
- **`init_essentials`**：初始化逻辑。它相当于常规类 **`__init__`** 的功能，但只在必要时才需要初始化。
- **`_assert_init_essentials`**：校验执行方法。确保执行过初始化，一般在功能性方法中首先调用。
- **`get_optimizer`**：获取包装器内部所包装的优化器实例。

在定义优化器配置包装器前需先完成**优化器**的定义，亦可利用现有算子直接封装：

- [torch.optim — PyTorch documentation](https://docs.pytorch.org/docs/stable/optim.html)