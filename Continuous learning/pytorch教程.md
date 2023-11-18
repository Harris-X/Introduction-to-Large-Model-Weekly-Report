# pytorch教程

### 1. transforms.Compose()

torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起：

```python
transforms.Compose([
    transforms.CenterCrop(10),
    transforms.ToTensor(),
])
```

transforms中的函数：

- Resize：把给定的图片resize到given size

- Normalize：用均值和标准差归一化张量图像

- ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]

- CenterCrop：在图片的中间区域进行裁剪

- RandomCrop：在一个随机的位置进行裁剪

- FiceCrop：把图像裁剪为四个角和一个中心

- RandomResizedCrop：将PIL图像裁剪成任意大小和纵横比

- ToPILImage：convert a tensor to PIL image

- RandomHorizontalFlip：以0.5的概率水平翻转给定的PIL图像

- RandomVerticalFlip：以0.5的概率竖直翻转给定的PIL图像

- Grayscale：将图像转换为灰度图像

- RandomGrayscale：将图像以一定的概率转换为灰度图像

- ColorJitter：随机改变图像的亮度对比度和饱和度

### 2. view()

- view()相当于reshape、resize，重新调整Tensor的形状。


### 3. clip_grad_norm_

torch.nn.utils.clip_grad_norm_(parameters，max_norm，norm_type=2)

梯度裁剪，解决梯度爆炸问题

- parameters：需要进行梯度裁剪的参数列表。通常是模型的参数列表，即model.parameters()

- max_norm：可以理解为梯度（默认是L2 范数）范数的最大阈值

- norm_type：可以理解为指定范数的类型，比如norm_type=1 表示使用L1 范数，norm_type=2 表示使用L2 范数。

## RNN

### 1. torch.nn.RNN()

在PyTorch中，torch.nn.RNN是一个用于构建RNN模型的类。它通常用于处理序列数据，例如时间序列或文本数据。

在PyTorch中，torch.nn.RNN的基本参数包括：

- `input_size`: 输入数据的特征数量。
- `hidden_size`: 隐藏状态的维度。
- `num_layers`: RNN的层数。
- `nonlinearity`: 用于隐藏状态的激活函数。默认是 'relu'。
- `batch_first`: 如果为 True，则输入数据的形状为 (batch_size, seq_length, input_size)，否则为 (seq_length, batch_size, input_size)。
- `dropout`: 在每层RNN之后丢弃一定比例的单元，以防止过拟合。默认值是 0，即不丢弃任何单元。

torch.nn.RNN的输入数据通常是一个形状为(seq_length, batch_size, input_size)的三维张量。其中，seq_length表示序列的长度，batch_size表示每个序列中样本的数量，input_size表示每个样本的特征数量。

#### 1.1 RNN输入：input、h_0

- input：用于存放输入样本的张量，张量的形状如下：
  
  math:`(L, N, H_{in})` 当`batch_first=False` ，或者：
  
  math:`(N, L, H_{in})` 当``batch_first=True`

- h_0： 用于存放RNN初始的隐藏状态，通常为上一时刻预测时隐层状态的输出，如果没有上一时刻，设置全0。

*N：batch size， 一次可以送个一个batch的数据，batch size描述的可以同时并行输入的序列串的个数。
*L：sequence length，连续多个输入样本，一次性送入RNN网络或foward函数中，RNN会依次输出sequence length批次的输出。sequence length可以串行输入序列的个数。*H_{in} ：input_size、H_{out} ：hidden_size

#### 1.2 RNN输出：output、h_t

- output：RNN网络的输出
- h_n：RNN网络隐层的输出，就是RNN的各个层的**最后一个**隐含状态(时间步)的输出。

**例子：**

```python
class SimpleRNN(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):  
        super(SimpleRNN, self).__init__()  
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2)  
        self.fc = nn.Linear(hidden_size, output_size)  

    def forward(self, x):  
        out, _ = self.rnn(x)  
        out = self.fc(out[:, -1, :])  
        return out 
```

## CNN

### 1. Conv2d

torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

- in_channels    int    输入图像通道数

- out_channels    int    卷积产生的通道数

- kernel_size    (int or tuple)    卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3卷积核

- stride    (int or tuple, optional)   卷积步长，默认为1。可以设为1个int型数或者一个(int, int)型的元组

- padding    (int or tuple, optional)    填充操作，控制padding_mode的数目

- padding_mode    (string, optional)    padding模式，默认为Zero-padding 

- dilation    (int or tuple, optional)    扩张操作：控制kernel点（卷积核点）的间距，默值:1

- groups    (int, optional)    group参数的作用是控制分组卷积，默认不分组，为1组

- bias    (bool, optional)    为真，则在输出中添加一个可学习的偏差。默认：True

**例子：**

```python
self.cnn = nn.Sequential(
            # 3：输入通道数，64：输出通道数，3：卷积核大小，1：步长，1：填充大小
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]
);
```

### 2. MaxPool2d

```python
torch.nn.MaxPool2d(
    kernel_size, 
    stride=None, 
    padding=0, 
    dilation=1, 
    return_indices=False, 
    ceil_mode=False
)
```

- kernel_size (int or tuple)【必选】：max pooling 的窗口大小，当最大池化窗口是方形的时候，只需要一个整数边长即可；最大池化窗口不是方形时，要输入一个元组表 高和宽。

- stride (int or tuple, optional)【可选】：max pooling 的窗口移动的步长。默认值是 kernel_size

- padding (int or tuple, optional)【可选】：输入的每一条边补充0的层数

- dilation (int or tuple, optional)【可选】：一个控制窗口中元素步幅的参数

- return_indices (bool)【可选】：如果等于 True，会返回输出最大值的序号，对于上采样操作会有帮助

- ceil_mode (bool)【可选】：如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

**例子：**

```python
self.cnn = nn.Sequential(
            # 3：输入通道数，64：输出通道数，3：卷积核大小，1：步长，1：填充大小
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]
)
```



## Transfomer

以下是一个简单的运行示例

```python
import torch
import torch.nn as nn
input = torch.LongTensor([[5,2,1,0,0],[1,3,1,4,0]])
import numpy as np
src_vocab_size = 10
d_model = 512
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.emb(x)
word_emb = Embeddings(src_vocab_size,d_model)
word_embr = word_emb(input)
print('word_embr',word_embr.shape)

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
encoder_out = transformer_encoder(word_embr)
print('encoder_out',encoder_out.shape)
```

以下是torch搭建transformer的示例代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return F.log_softmax(output, dim=-1)
```

训练模型

```python
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = TransformerModel(input_dim, hidden_dim, n_layers, n_heads, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        src, trg = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output.view(-1, input_dim), trg.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            src, trg = batch[0].to(device), batch[1].to(device)
            output = model(src)
            loss = criterion(output.view(-1, input_dim), trg.view(-1))
            val_loss += loss.item()
        val_loss /= len(val_loader)
        print("Epoch %d, val_loss %.4f" % (epoch, val_loss))
```

### 实现完整的Transformer

> https://blog.csdn.net/ARPOSPF/article/details/132176825

