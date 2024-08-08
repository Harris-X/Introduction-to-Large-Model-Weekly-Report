# 数据集定义与加载

## 一、定义数据集

### 1.1 直接加载内置数据集

飞桨框架在 [paddle.vision.datasets](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html#api) 和 [paddle.text](https://www.paddlepaddle.org.cn/documentation/docs/zh//api/paddle/text/Overview_cn.html#api) 目录下内置了一些经典数据集可直接调用，通过以下代码可查看飞桨框架中的内置数据集。

```python
import paddle

print("计算机视觉（CV）相关数据集：", paddle.vision.datasets.__all__)
print("自然语言处理（NLP）相关数据集：", paddle.text.__all__)
```

以 [MNIST](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/MNIST_cn.html) 数据集为例，加载内置数据集的代码示例如下所示。

```python
from paddle.vision.transforms import Normalize

# 定义图像归一化处理方法，这里的CHW指图像格式需为 [C通道数，H图像高度，W图像宽度]
transform = Normalize(mean=[127.5], std=[127.5], data_format="CHW")
# 下载数据集并初始化 DataSet
train_dataset = paddle.vision.datasets.MNIST(mode="train", transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode="test", transform=transform)
print(
    "train images: ", len(train_dataset), ", test images: ", len(test_dataset)
)

```

完成数据集初始化之后，可以使用下面的代码直接对数据集进行迭代读取。

```python
from matplotlib import pyplot as plt

for data in train_dataset:
    image, label = data
    print("shape of image: ", image.shape)
    plt.title(str(label))
    plt.imshow(image[0])
    break
```

&nbsp;

### 1.2 使用 paddle.io.Dataset 自定义数据集

在实际的场景中，一般需要使用自有的数据来定义数据集，这时可以通过 [paddle.io.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Dataset_cn.html#dataset) 基类来实现自定义数据集。

可构建一个子类继承自 `paddle.io.Dataset` ，并且实现下面的三个函数：

1. `__init__`：完成数据集初始化操作，将磁盘中的样本文件路径和对应标签映射到一个列表中。

2. `__getitem__`：定义指定索引（index）时如何获取样本数据，最终返回对应 index 的单条数据（样本数据、对应的标签）。

3. `__len__`：返回数据集的样本总数。

```python
import os
import cv2
import numpy as np
from paddle.io import Dataset
from paddle.vision.transforms import Normalize


class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """

    def __init__(self, data_dir, label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super().__init__()
        self.data_list = []
        with open(label_path, encoding="utf-8") as f:
            for line in f.readlines():
                image_path, label = line.strip().split("\t")
                image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image_path, label = self.data_list[index]
        # 读取灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = image.astype("float32")
        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


# 定义图像归一化处理方法，这里的CHW指图像格式需为 [C通道数，H图像高度，W图像宽度]
transform = Normalize(mean=[127.5], std=[127.5], data_format="CHW")
# 打印数据集样本数
train_custom_dataset = MyDataset(
    "mnist/train", "mnist/train/label.txt", transform
)
test_custom_dataset = MyDataset("mnist/val", "mnist/val/label.txt", transform)
print(
    "train_custom_dataset images: ",
    len(train_custom_dataset),
    "test_custom_dataset images: ",
    len(test_custom_dataset),
)
```

&nbsp;

## 二、迭代读取数据集

### 2.1 使用 paddle.io.DataLoader 定义数据读取器

通过前面介绍的直接迭代读取 Dataset 的方式虽然可实现对数据集的访问，但是这种访问方式只能单线程进行并且还需要手动分批次（batch）。在飞桨框架中，推荐使用 [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader) API 对数据集进行多进程的读取，并且可自动完成划分 batch 的工作。

```python
# 定义并初始化数据读取器
train_loader = paddle.io.DataLoader(
    train_custom_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

# 调用 DataLoader 迭代读取数据
for batch_id, data in enumerate(train_loader()):
    images, labels = data
    print(
        "batch_id: {}, 训练数据shape: {}, 标签数据shape: {}".format(
            batch_id, images.shape, labels.shape
        )
    )
    break
```

定义好数据读取器之后，便可用 for 循环方便地迭代读取批次数据，用于模型训练了。值得注意的是，如果使用高层 API 的 [paddle.Model.fit](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#fit-train-data-none-eval-data-none-batch-size-1-epochs-1-eval-freq-1-log-freq-10-save-dir-none-save-freq-1-verbose-2-drop-last-false-shuffle-true-num-workers-0-callbacks-none) 读取数据集进行训练，则只需定义数据集 Dataset 即可，不需要再单独定义 DataLoader，因为 paddle.Model.fit 中实际已经封装了一部分 DataLoader 的功能。

&nbsp;

# 数据预处理

## 一、paddle.vision.transforms 介绍

- 单个使用

```python
from paddle.vision.transforms import Resize

# 定义一个待使用的数据处理方法，这里定义了一个调整图像大小的方法
transform = Resize(size=28)
```

- 多个组合使用

这种使用模式下，需要先定义好每个数据处理方法，然后用`Compose` 进行组合。

```python
from paddle.vision.transforms import Compose, RandomRotation

# 定义待使用的数据处理方法，这里包括随机旋转、改变图片大小两个组合处理
transform = Compose([RandomRotation(10), Resize(size=32)])
```

&nbsp;

## 二、在数据集中应用数据预处理操作

### 2.1 在框架内置数据集中应用

前面已定义好数据处理的方法，在加载内置数据集时，将其传递给 `transform` 字段即可。

```python
# 通过 transform 字段传递定义好的数据处理方法，即可完成对框架内置数据集的增强
train_dataset = paddle.vision.datasets.MNIST(mode="train", transform=transform)
```

&nbsp;

### 2.2 在自定义的数据集中应用

对于自定义的数据集，可以在数据集中将定义好的数据处理方法传入 `__init__` 函数，将其定义为自定义数据集类的一个属性，然后在 `__getitem__` 中将其应用到图像上，如下述代码所示：

```python
import os
import cv2
import numpy as np
from paddle.io import Dataset


class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """

    def __init__(self, data_dir, label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super().__init__()
        self.data_list = []
        with open(label_path, encoding="utf-8") as f:
            for line in f.readlines():
                image_path, label = line.strip().split("\t")
                image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])
        # 2. 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        image_path, label = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype("float32")
        # 3. 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        label = int(label)
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


# 1. 定义随机旋转和改变图片大小的数据处理方法
transform = Compose([RandomRotation(10), Resize(size=32)])

custom_dataset = MyDataset("mnist/train", "mnist/train/label.txt", transform)
```

&nbsp;

# 模型组网

飞桨框架提供了多种模型组网方式，本文介绍如下几种常见用法：

- 直接使用内置模型

- 使用 [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sequential_cn.html#sequential) 组网

- 使用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#layer) 组网

另外飞桨框架提供了 [paddle.summary](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/summary_cn.html#summary) 函数方便查看网络结构、每层的输入输出 shape 和参数信息。

## 一、直接使用内置模型

飞桨框架目前在 [paddle.vision.models](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html#about-models) 下内置了计算机视觉领域的一些经典模型，适合完成一些简单的深度学习任务。

```python
飞桨框架内置模型： ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext50_64x4d', 'resnext101_32x4d', 'resnext101_64x4d', 'resnext152_32x4d', 'resnext152_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2', 'mobilenet_v2', 'MobileNetV3Small', 'MobileNetV3Large', 'mobilenet_v3_small', 'mobilenet_v3_large', 'LeNet', 'DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenet264', 'AlexNet', 'alexnet', 'InceptionV3', 'inception_v3', 'SqueezeNet', 'squeezenet1_0', 'squeezenet1_1', 'GoogLeNet', 'googlenet', 'ShuffleNetV2', 'shufflenet_v2_x0_25', 'shufflenet_v2_x0_33', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenet_v2_swish']
```

以 LeNet 模型为例，可通过如下代码组网：

```python
# 模型组网并初始化网络
lenet = paddle.vision.models.LeNet(num_classes=10)

# 可视化模型组网结构和参数
paddle.summary(lenet, (1, 1, 28, 28))
```

&nbsp;

## 二、Paddle.nn 介绍

飞桨提供继承类（class）的方式构建网络，并提供了几个基类，如：[paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sequential_cn.html#sequential)、 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#layer) 等，构建一个继承基类的子类，并在子类中添加层（layer，如卷积层、全连接层等）可实现网络的构建，不同基类对应不同的组网方式。

- **使用 [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sequential_cn.html#sequential) 组网**：构建顺序的线性网络结构（如 LeNet、AlexNet 和 VGG）时，可以选择该方式。相比于 Layer 方式 ，Sequential 方式可以用更少的代码完成线性网络的构建。

- **使用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#layer) 组网（推荐）**：构建一些比较复杂的网络结构时，可以选择该方式。相比于 Sequential 方式，Layer 方式可以更灵活地组建各种网络结构。Sequential 方式搭建的网络也可以作为子网加入 Layer 方式的组网中。

&nbsp;

## 三、使用 paddle.nn.Sequential 组网

构建顺序的线性网络结构时，可以选择该方式，只需要按模型的结构顺序，一层一层加到 [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sequential_cn.html#sequential) 子类中即可。

参照 LeNet 模型结构，构建该网络结构的代码如下：

```python
from paddle import nn

# 使用 paddle.nn.Sequential 构建 LeNet 模型
lenet_Sequential = nn.Sequential(
    nn.Conv2D(1, 6, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Conv2D(6, 16, 5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Flatten(),
    nn.Linear(400, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 10),
)
# 可视化模型组网结构和参数
paddle.summary(lenet_Sequential, (1, 1, 28, 28))
```

使用 Sequential 组网时，会自动按照层次堆叠顺序完成网络的前向计算过程，简略了定义前向计算函数的代码。由于 Sequential 组网只能完成简单的线性结构模型，所以对于需要进行**分支判断**的模型需要使用 paddle.nn.Layer 组网方式实现。

&nbsp;

## 四、使用 paddle.nn.Layer 组网

构建一些比较复杂的网络结构时，可以选择该方式，组网包括三个步骤：

1. 创建一个继承自 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#layer) 的类；

2. 在类的构造函数 `__init__` 中定义组网用到的神经网络层（layer）；

3. 在类的前向计算函数 `forward` 中使用定义好的 layer 执行前向计算。

仍然以 LeNet 模型为例，使用 paddle.nn.Layer 组网的代码如下：

```python
# 使用 Subclass 方式构建 LeNet 模型
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # 构建 features 子网，用于对输入图像进行特征提取
        self.features = nn.Sequential(
            nn.Conv2D(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
        )
        # 构建 linear 子网，用于分类
        if num_classes > 0:
            self.linear = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(84, num_classes),
            )

    # 执行前向计算
    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.linear(x)
        return x


lenet_SubClass = LeNet()

# 可视化模型组网结构和参数
params_info = paddle.summary(lenet_SubClass, (1, 1, 28, 28))
print(params_info)
```

&nbsp;

# 模型训练、评估与推理

飞桨框架提供了两种训练、评估与推理的方法：

- **使用飞桨高层 API**：先用 [paddle.Model](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html) 对模型进行封装，然后通过 [Model.fit](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#fit-train-data-none-eval-data-none-batch-size-1-epochs-1-eval-freq-1-log-freq-10-save-dir-none-save-freq-1-verbose-2-drop-last-false-shuffle-true-num-workers-0-callbacks-none-accumulate-grad-batches-1-num-iters-none) 、 [Model.evaluate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#evaluate-eval-data-batch-size-1-log-freq-10-verbose-2-num-workers-0-callbacks-none-num-iters-none) 、 [Model.predict](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#predict-test-data-batch-size-1-num-workers-0-stack-outputs-false-verbose-1-callbacks-none) 等完成模型的训练、评估与推理。该方式代码量少，适合快速上手。

- **使用飞桨基础 API**：提供了损失函数、优化器、评价指标、更新参数、反向传播等基础组件的实现，可以更灵活地应用到模型训练、评估与推理任务中，当然也可以很方便地自定义一些组件用于相关任务中。

高层 API 如 `Model.fit` 、 `Model.evaluate` 、 `Model.predict` 等都可以通过基础 API 实现。

## 一、训练前准备

### 1.1 （可选）指定训练的硬件

模型训练时，需要用到 CPU、 GPU 等计算处理器资源，由于飞桨框架的安装包是区分处理器类型的，默认情况下飞桨框架会根据所安装的版本自动选择对应硬件，比如安装的 GPU 版本的飞桨，则**自动使用** GPU 训练模型，无需手动指定。因此一般情况下，无需执行此步骤。

但是如果安装的 GPU 版本的飞桨框架，想切换到 CPU 上训练，则可通过 [paddle.device.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device) 修改。如果本机有多个 GPU 卡，也可以通过该 API 选择指定的卡进行训练，不指定的情况下则**默认使用 'gpu:0'**。

```python
import paddle

# 指定在 CPU 上训练
paddle.device.set_device("cpu")

# 指定在 GPU 第 0 号卡上训练
# paddle.device.set_device('gpu:0')
```

需要注意的是，使用 `paddle.device.set_device` 时，只能使用 `CUDA_VISIBLE_DEVICES` 设置范围内的显卡，例如可以设置`export CUDA_VISIBLE_DEVICES=0,1,2` 和 `paddle.device.set_device('gpu:0')`，但是设置 `export CUDA_VISIBLE_DEVICES=1` 和 `paddle.device.set_device('gpu:0')` 时会冲突报错。

&nbsp;

### 1.2 准备训练用的数据集和模型

模型训练前，需要先完成数据集的加载和模型组网，以 MNIST 手写数字识别任务为例，代码示例如下：

```python
from paddle.vision.transforms import Normalize

transform = Normalize(mean=[127.5], std=[127.5], data_format="CHW")
# 加载 MNIST 训练集和测试集
train_dataset = paddle.vision.datasets.MNIST(mode="train", transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode="test", transform=transform)

# 模型组网，构建并初始化一个模型 mnist
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(1, -1),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10),
)
```

&nbsp;

## 二、使用 paddle.Model 高层 API 训练、评估与推理

以手写数字识别任务为例，使用高层 API 进行模型训练、评估与推理的步骤如下：

### 2.1 使用 paddle.Model 封装模型

使用高层 API 训练模型前，可使用 [paddle.Model](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html) 将模型封装为一个实例，方便后续进行训练、评估与推理。代码如下：

```python
# 封装模型为一个 model 实例，便于进行后续的训练、评估和推理
model = paddle.Model(mnist)
```

&nbsp;

### 2.2 使用 Model.prepare 配置训练准备参数

用 `paddle.Model` 完成模型的封装后，需通过 [Model.prepare](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#prepare-optimizer-none-loss-none-metrics-none-amp-configs-none) 进行训练前的配置准备工作，包括设置优化算法、Loss 计算方法、评价指标计算方法：

```python
# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(
    optimizer=paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters()
    ),
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy(),
)
```

示例中使用 [Adam](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Adam_cn.html#adam) 优化器，设置优化器的学习率 `learning_rate=0.001`，并传入封装好的全部模型参数 `model.parameters` 用于后续更新；使用交叉熵损失函数 [CrossEntropyLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/CrossEntropyLoss_cn.html#crossentropyloss) 用于分类任务评估；使用分类任务常用的准确率指标 [Accuracy](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/metric/accuracy_cn.html#accuracy) 计算模型在训练集上的精度。

&nbsp;

### 2.3 使用 Model.fit 训练模型

做好模型训练的前期准备工作后，调用 [Model.fit](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#fit-train-data-none-eval-data-none-batch-size-1-epochs-1-eval-freq-1-log-freq-10-save-dir-none-save-freq-1-verbose-2-drop-last-false-shuffle-true-num-workers-0-callbacks-none-accumulate-grad-batches-1-num-iters-none) 接口来启动训练。 训练过程采用二层循环嵌套方式：内层循环完成整个数据集的一次遍历，采用分批次方式；外层循环根据设置的训练轮次完成数据集的多次遍历。因此需要指定至少三个关键参数：**训练数据集，训练轮次和每批次大小**。

除此之外，还可以设置样本乱序（`shuffle`）、丢弃不完整的批次样本（`drop_last`）、同步/异步读取数据（`num_workers`） 等参数，另外可通过 `Callback` 参数传入回调函数，在模型训练的各个阶段进行一些自定义操作，比如收集训练过程中的一些数据和参数。

```python
# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_dataset, epochs=5, batch_size=64, verbose=1)
```

示例中传入数据集 `train_dataset` 进行迭代训练，共遍历 5 轮（`epochs=5`），每轮迭代中分批次取数据训练，每批次 64 个样本（`batch_size=64`）。

&nbsp;

### 2.4 使用 Model.evaluate 评估模型

训练好模型后，可在事先定义好的测试数据集上，使用 [Model.evaluate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#evaluate-eval-data-batch-size-1-log-freq-10-verbose-2-num-workers-0-callbacks-none-num-iters-none) 接口完成模型评估操作，结束后根据在 [Model.prepare](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#prepare-optimizer-none-loss-none-metrics-none-amp-configs-none) 中定义的 `loss` 和 `metric` 计算并返回相关评估结果。

返回格式是一个字典:

- 只包含loss， `{'loss': xxx}`

- 包含loss和一个评估指标， `{'loss': xxx, 'metric name': xxx}`

- 包含loss和多个评估指标， `{'loss': xxx, 'metric name1': xxx, 'metric name2': xxx}`

```python
# 用 evaluate 在测试集上对模型进行验证
eval_result = model.evaluate(test_dataset, verbose=1)
print(eval_result)
```

&nbsp;

### 2.5 使用 Model.predict 执行推理

高层 API 中提供了 [Model.predict](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#predict-test-data-batch-size-1-num-workers-0-stack-outputs-false-verbose-1-callbacks-none) 接口，可对训练好的模型进行推理验证。只需传入待执行推理验证的样本数据，即可计算并返回推理结果。

返回格式是一个列表：

- 模型是单一输出：`[(numpy_ndarray_1, numpy_ndarray_2, …, numpy_ndarray_n)]`

- 模型是多输出：`[(numpy_ndarray_1, numpy_ndarray_2, …, numpy_ndarray_n), (numpy_ndarray_1, numpy_ndarray_2, …, numpy_ndarray_n), …]`

如果模型是单一输出，则输出的形状为 [1, n]，n 表示数据集的样本数。其中每个 numpy_ndarray_n 是对应原始数据经过模型计算后得到的预测结果，类型为 numpy 数组，例如 mnist 分类任务中，每个 numpy_ndarray_n 是长度为 10 的 numpy 数组。

如果模型是多输出，则输出的形状为[m, n]，m 表示标签的种类数，在多标签分类任务中，m 会根据标签的数目而定。

```python
# 用 predict 在测试集上对模型进行推理
test_result = model.predict(test_dataset)
# 由于模型是单一输出，test_result的形状为[1, 10000]，10000是测试数据集的数据量。这里打印第一个数据的结果，这个数组表示每个数字的预测概率
print(len(test_result))
print(test_result[0][0])

# 从测试集中取出一张图片
img, label = test_dataset[0]

# 打印推理结果，这里的argmax函数用于取出预测值中概率最高的一个的下标，作为预测标签
pred_label = test_result[0][0].argmax()
print("true label: {}, pred label: {}".format(label[0], pred_label))
# 使用matplotlib库，可视化图片
from matplotlib import pyplot as plt

plt.imshow(img[0])
```

&nbsp;

## 三、使用基础 API 训练、评估与推理

除了通过高层 API 实现模型的训练、评估与推理，飞桨框架也同样支持通过基础 API。简单来说， `Model.prepare` 、 `Model.fit` 、 `Model.evaluate` 、 `Model.predict` 都是由基础 API 封装而来。

### 3.1 模型训练（拆解 Model.prepare、Model.fit）

飞桨框架通过基础 API 对模型进行训练，对应高层 API 的 `Model.prepare` 与 `Model.fit` ，一般包括如下几个步骤：

1. 加载训练数据集、声明模型、设置模型实例为 `train` 模式

2. 设置优化器、损失函数与各个超参数

3. 设置模型训练的二层循环嵌套，并在内层循环嵌套中设置如下内容：
   
   - 3.1 从数据读取器 DataLoader 获取一批次训练数据
   
   - 3.2 执行一次预测，即经过模型计算获得输入数据的预测值
   
   - 3.3 计算预测值与数据集标签的损失
   
   - 3.4 计算预测值与数据集标签的准确率
   
   - 3.5 将损失进行反向传播
   
   - 3.6 打印模型的轮数、批次、损失值、准确率等信息
   
   - 3.7 执行一次优化器步骤，即按照选择的优化算法，根据当前批次数据的梯度更新传入优化器的参数
   
   - 3.8 将优化器的梯度进行清零

```python
# dataset与mnist的定义与使用高层API的内容一致
# 用 DataLoader 实现数据加载
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 将mnist模型及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。
mnist.train()

# 设置迭代次数
epochs = 5

# 设置优化器
optim = paddle.optimizer.Adam(parameters=mnist.parameters())
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]  # 训练数据
        y_data = data[1]  # 训练数据标签
        predicts = mnist(x_data)  # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, y_data)

        # 计算准确率 等价于 prepare 中metrics的设置
        acc = paddle.metric.accuracy(predicts, y_data)

        # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中
        # 反向传播
        loss.backward()

        if (batch_id + 1) % 900 == 0:
            print(
                "epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                    epoch, batch_id + 1, loss.numpy(), acc.numpy()
                )
            )
        # 更新参数
        optim.step()
        # 梯度清零
        optim.clear_grad()
```

&nbsp;

### 3.2 模型评估（拆解 Model.evaluate）

飞桨框架通过基础 API 对训练好的模型进行评估，对应高层 API 的 `Model.evaluate` 。与模型训练相比，模型评估的流程有如下几点不同之处：

1. 加载的数据从训练数据集改为测试数据集

2. 模型实例从 `train` 模式改为 `eval` 模式

3. 不需要反向传播、优化器参数更新和优化器梯度清零

```python
# 加载测试数据集
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()
# 将该模型及其所有子层设置为预测模式。这只会影响某些模块，如Dropout和BatchNorm
mnist.eval()
# 禁用动态图梯度计算
for batch_id, data in enumerate(test_loader()):
    x_data = data[0]  # 测试数据
    y_data = data[1]  # 测试数据标签
    predicts = mnist(x_data)  # 预测结果

    # 计算损失与精度
    loss = loss_fn(predicts, y_data)
    acc = paddle.metric.accuracy(predicts, y_data)

    # 打印信息
    if (batch_id + 1) % 30 == 0:
        print(
            "batch_id: {}, loss is: {}, acc is: {}".format(
                batch_id + 1, loss.numpy(), acc.numpy()
            )
        )
```

&nbsp;

### 3.3 模型推理（拆解 Model.predict）

飞桨框架通过基础 API 对训练好的模型执行推理，对应高层 API 的 `Model.predict` 。模型的推理过程相对独立，是在模型训练与评估之后单独进行的步骤。只需要执行如下步骤：

1. 加载待执行推理的测试数据，并将模型设置为 `eval` 模式

2. 读取测试数据并获得预测结果

3. 对预测结果进行后处理

```python
# 加载测试数据集
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)
# 将该模型及其所有子层设置为预测模式
mnist.eval()
for batch_id, data in enumerate(test_loader()):
    # 取出测试数据
    x_data = data[0]
    # 获取预测结果
    predicts = mnist(x_data)
print("predict finished")

# 从测试集中取出一组数据
img, label = test_loader().next()

# 执行推理并打印结果
pred_label = mnist(img)[0].argmax()
print(
    "true label: {}, pred label: {}".format(
        label[0].item(), pred_label[0].item()
    )
)
# 可视化图片
from matplotlib import pyplot as plt

plt.imshow(img[0][0])
```

&nbsp;

# 模型保存与加载

## 一、概述

在模型训练过程中，通常会在如下场景中用到模型的保存与加载功能：

- 训练调优场景：
  
  - 模型训练过程中定期保存模型，以便后续对不同时期的模型恢复训练或进行研究；
  
  - 模型训练完毕，需要保存模型方便进行评估测试；
  
  - 载入预训练模型，并对模型进行微调（fine-tune）。

- 推理部署场景：
  
  - 模型训练完毕，在云、边、端不同的硬件环境中部署使用。

&nbsp;

针对以上场景，飞桨框架推荐使用的模型保存与加载基础 API 主要包括：

- [paddle.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/save_cn.html)

- [paddle.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/load_cn.html)

- [paddle.jit.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html)

- [paddle.jit.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/load_cn.html#load)

模型保存与加载高层 API 主要包括：

- [paddle.Model.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#save-path-training-true)

- [paddle.Model.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#load-path-skip-mismatch-false-reset-optimizer-false)

&nbsp;

## 二、用于训练调优场景

### 2.1 使用基础 API

#### 2.1.1 保存动态图模型

参数保存时，先获取目标对象（Layer 或者 Optimzier）的 state_dict，然后将 state_dict 保存至磁盘，同时也可以保存模型训练 checkpoint 的信息，保存的 checkpoint 的对象已在上文示例代码中进行了设置，保存代码如下：

```python
# 创建网络、loss和优化器
layer = LinearNet()
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# 创建用于载入数据的DataLoader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2
)

# 开始训练
train(layer, loader, loss_fn, adam)

# 保存Layer参数
paddle.save(layer.state_dict(), "linear_net.pdparams")
# 保存优化器参数
paddle.save(adam.state_dict(), "adam.pdopt")
# 保存检查点checkpoint信息
paddle.save(final_checkpoint, "final_checkpoint.pkl")
```

&nbsp;

#### 2.1.2 加载动态图模型

参数载入时，先从磁盘载入保存的 state_dict，然后通过 `set_state_dict()`方法将 state_dict 配置到目标对象中。另外载入之前保存的 checkpoint 信息并打印出来，示例如下：

```python
# 载入模型参数、优化器参数和最后一个epoch保存的检查点
layer_state_dict = paddle.load("linear_net.pdparams")
opt_state_dict = paddle.load("adam.pdopt")
final_checkpoint_dict = paddle.load("final_checkpoint.pkl")

# 将load后的参数与模型关联起来
layer.set_state_dict(layer_state_dict)
adam.set_state_dict(opt_state_dict)

# 打印出来之前保存的 checkpoint 信息
print(
    "Loaded Final Checkpoint. Epoch : {}, Loss : {}".format(
        final_checkpoint_dict["epoch"], final_checkpoint_dict["loss"].numpy()
    )
)
```

加载以后就可以继续对动态图模型进行训练调优（fine-tune），或者验证预测效果（predict）。

&nbsp;

### 2.2 使用高层 API

下面结合简单示例，介绍高层 API 模型保存和载入的方法。

#### 2.2.1 保存动态图模型

以下示例完成了一个简单网络的训练和保存动态图模型的过程，示例后介绍保存动态图模型的两种方式：

```python
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision.models import LeNet

model = paddle.Model(LeNet())
optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
model.prepare(optim, paddle.nn.CrossEntropyLoss())

transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
data = paddle.vision.datasets.MNIST(mode="train", transform=transform)

# 方式一：设置训练过程中保存模型
model.fit(data, epochs=1, batch_size=32, save_freq=1)

# 方式二：设置训练后保存模型
model.save("checkpoint/test")  # save for training
```

- 方式一：开启训练时调用的`paddle.Model.fit`函数可自动保存模型，通过它的参数 `save_freq`可以设置保存动态图模型的频率，即多少个 epoch 保存一次模型，默认值是 1。

- 方式二：调用 `paddle.Model.save`API。只需要传入保存的模型文件的前缀，格式如 `dirname/file_prefix` 或者 `file_prefix` ，即可保存训练后的模型参数和优化器参数，保存后的文件后缀名固定为 `.pdparams` 和`.pdopt`。

&nbsp;

#### 2.2.2 加载动态图模型

高层 API 加载动态图模型所需要调用的 API 是 `paddle.Model.load`，从指定的文件中载入模型参数和优化器参数（可选）以继续训练。`paddle.Model.load`需要传入的核心的参数是待加载的模型参数或者优化器参数文件（可选）的前缀（需要保证后缀符合 `.pdparams` 和`.pdopt`）。

假设上面的示例代码已经完成了参数保存过程，下面的例子会加载上面保存的参数以继续训练：

```python
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision.models import LeNet

model = paddle.Model(LeNet())
optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
model.prepare(optim, paddle.nn.CrossEntropyLoss())

transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
data = paddle.vision.datasets.MNIST(mode="train", transform=transform)
# 加载模型参数和优化器参数
model.load("checkpoint/test")
model.fit(data, epochs=1, batch_size=32, save_freq=1)

model.save("checkpoint/test_1")  # save for training
```

&nbsp;

## 三、用于推理部署场景

### 3.1 使用高层 API

高层 API `paddle.Model.save`可支持保存推理使用的模型，此时高层 API 在动态图下实际上是对`paddle.jit.save`的封装，在静态图下是对 `paddle.static.save_inference_model`的封装，会自动将训练好的动态图模型保存为静态图模型。

`paddle.Model.save`的第一个参数需要设置为待保存的模型和参数等文件的前缀名，第二个参数 `training` 表示是否保存动态图模型以继续训练，默认是 True，这里需要设为 False，即保存推理部署所需的参数与文件。接前文高层 API 训练的示例代码，保存推理模型代码示例如下：

```python
model.save("inference_model", False)  # save for inference
```

执行上述代码样例后，会在当前目录下生成三个文件，即代表成功导出可用于推理部署的静态图模型：

```python
inference_model.pdiparams // 存放模型中所有的权重数据
inference_model.pdmodel // 存放模型的网络结构
inference_model.pdiparams.info // 存放和参数状态有关的额外信息
```

&nbsp;

## 四、其他场景

### 4.1 静态图模型的保存与加载

- 若仅需要保存/载入模型的参数用于**训练调优**场景，可以使用 `paddle.save`/`paddle.load` 结合静态图模型 Program 的 state_dict 达成目的。也支持保存整个模型，可以使用 `paddle.save` 将 Program 和state_dict 都保存下来。高层 API 兼容了动态图和静态图，因此`Paddle.Model.save`和`Paddle.Model.load`也兼容了动、静态图的保存和加载。

- 若需保存推理模型用于**模型部署**场景，则可以通过 `paddle.static.save_inference_model`、`paddle.static.load_inference_model`实现。

#### 4.1.1 训练调优场景

结合以下简单示例，介绍参数保存和载入的方法：

```python
import paddle
import paddle.static as static

# 开启静态图模式
paddle.enable_static()

# 创建输入数据和网络
x = paddle.static.data(name="x", shape=[None, 224], dtype="float32")
z = paddle.static.nn.fc(x, 10)

# 设置执行器开始训练
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())
prog = paddle.static.default_main_program()
```

如果只想保存模型的参数，先获取 Program 的 state_dict，然后将 state_dict 保存至磁盘，示例如下：

```python
# 保存模型参数
paddle.save(prog.state_dict(), "temp/model.pdparams")
```

如果想要保存整个静态图模型（含模型结构和参数），除了 state_dict 还需要保存 Program：

```python
# 保存模型结构（program）
paddle.save(prog, "temp/model.pdmodel")
```

模型载入阶段，如果只保存了 state_dict，可以跳过下面此段代码，直接载入 state_dict。如果模型文件中包含 Program 和 state_dict，请先载入 Program，示例如下：

```python
# 载入模型结构（program）
prog = paddle.load("temp/model.pdmodel")
```

参数载入时，先从磁盘载入保存的 state_dict，然后通过 `set_state_dict()`方法配置到 Program 中，示例如下：

```python
# 载入模型参数
state_dict = paddle.load("temp/model.pdparams")
# 将load后的参数与模型program关联起来
prog.set_state_dict(state_dict)
```

&nbsp;

#### 4.1.2 推理部署场景

保存/载入静态图推理模型，可以通过 `paddle.static.save_inference_model`、`paddle.static.load_inference_model`实现。结合以下简单示例，介绍参数保存和载入的方法，示例如下：

```python
import paddle
import numpy as np

# 开启静态图模式
paddle.enable_static()

# 创建输入数据和网络
startup_prog = paddle.static.default_startup_program()
main_prog = paddle.static.default_main_program()
with paddle.static.program_guard(main_prog, startup_prog):
    image = paddle.static.data(name="img", shape=[64, 784])
    w = paddle.create_parameter(shape=[784, 200], dtype="float32")
    b = paddle.create_parameter(shape=[200], dtype="float32")
    hidden_w = paddle.matmul(x=image, y=w)
    hidden_b = paddle.add(hidden_w, b)
# 设置执行器开始训练
exe = paddle.static.Executor(paddle.CPUPlace())
exe.run(startup_prog)
```

静态图导出推理模型需要指定导出路径、输入、输出变量以及执行器。`paddle.static.save_inference_model` 会裁剪 Program 的冗余部分，并导出两个文件： `path_prefix.pdmodel`、`path_prefix.pdiparams` 。示例如下：

```python
# 保存静态图推理模型
path_prefix = "./infer_model"
paddle.static.save_inference_model(path_prefix, [image], [hidden_b], exe)
```

载入静态图推理模型时，输入给 `paddle.static.load_inference_model` 的路径必须与 `save_inference_model` 的一致。示例如下：

```python
# 载入静态图推理模型
[
    inference_program,
    feed_target_names,
    fetch_targets,
] = paddle.static.load_inference_model(path_prefix, exe)
tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32)
results = exe.run(
    inference_program,
    feed={feed_target_names[0]: tensor_img},
    fetch_list=fetch_targets,
)
```
