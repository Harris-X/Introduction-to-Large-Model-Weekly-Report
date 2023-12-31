# Event-based Vision: A Survey

> https://blog.csdn.net/weixin_47805362/article/details/119568659
>
> `pdf`:<https://pan.baidu.com/s/1FBXXnMuRVxEsrIfPIAo9nA?pwd=7ruq>

## 2 工作原理 PRINCIPLE OF OPERATION OF EVENT CAMERAS

事件相机的输出是一个数字“事件”或“峰值”的可变数据序列，每个事件代表一个特定时间像素亮度(对数强度)的变化。这种编码的灵感来自于生物视觉路径的尖峰特性(第3.3节)。

每个像素在每次发送事件时都会记忆日志强度，并持续监控从这个记忆值开始的足够大的变化(图1a)。当变化超过一个阈值时，相机发送一个事件，该事件由芯片传输x, y位置，时间t，以及变化的1位极性p(即亮度增加(“ON”)或减少(“OFF”))。该事件输出如图1b、1e和1f所示。

<img src="Event-based Vision A Survey.assets/image-20231128113657329.png" alt="image-20231128113657329" style="zoom:80%;" />

> 图1。DAVIS相机[4]概述，包括基于事件的动态视觉传感器(DVS[2])和基于帧的有源像素传感器(APS)，在相同的像素阵列中，每个像素共享相同的光电二极管。(a) DAVIS像素的简化电路图(红色为DVS像素，蓝色为APS像素)。(b)分布式交换机像素的操作原理图，将光转换为事件。(c)-(d) DAVIS芯片和USB摄像头的图片。(e) DAVIS所看到的旋转的黑色圆盘上的白色方块产生了灰度帧和时空事件的螺旋。时空中的事件用不同的颜色编码，从绿色(过去)到红色(现在)。(f)自然场景的框架和叠加事件;帧滞后于低延迟事件(根据极性着色)。图片改编自[4]，[35]。在[36]中可以找到分布式交换机、DAVIS和ATIS像素设计的更深入的比较。
>
> 

**事件摄像机是数据驱动的传感器:它们的输出取决于场景中的运动量或亮度变化。**

**一个像素处的入射光是场景照度和表面反射率的乘积。如果照度近似恒定，则对数强度变化表示反射率变化。这些反射率的变化主要是视场中物体运动的结果。**这就是为什么DVS亮度变化事件对场景照明[2]具有内置的不变性。

### 2.2 （优势）Advantages of Event cameras

- 高时间分辨率:在模拟电路中，亮度变化的监测是快速的，事件的读出是数字的，具有1MHz的时钟，即，事件被检测和时间戳的微秒分辨率。
- 低延迟:每个像素独立工作，不需要等待帧的全局曝光时间:一旦检测到变化，它就被传输。因此，事件摄像机有最小的延迟
- 低功耗:由于事件相机只传输亮度变化，因此去除冗余数据，功耗只用于处理变化的像素
- 高动态范围(HDR)。事件相机的高动态范围(>120dB)明显超过了高质量、基于框架的相机的60dB，使它们能够从月光到白天获取信息

### 2.3 （挑战）Challenges Due To The Novel Sensing Paradigm

1)应对不同的时空输出:事件相机的输出与标准相机的输出有根本的不同:事件是异步的，空间稀疏的，而图像是同步的，密集的。因此，为图像序列设计的基于框架的视觉算法不能直接适用于事件数据。

2)应对不同的光度传感:相对于标准相机提供的灰度信息，每个事件包含二值(增减)亮度变化信息。亮度的变化不仅取决于场景的亮度，还取决于场景和摄像机之间当前和过去的相对运动。

3)处理噪声和动态效应:所有的视觉传感器都有噪声，这是由于光子固有的散点噪声和晶体管电路噪声，它们也有非理想性。这种情况尤其适用于事件相机，其中量化时间对比度的过程是复杂的，并没有被完全描述。另外，事件相机相比传统相机，噪声影响极其严重，且处理相对困难。DVS模块工作时会设定一个阈值，每个像素上感知的光强变化只有超过阈值，才会生成事件点，为了能灵敏捕捉事件，往往阈值都会设置得较小，这便造成了事件相机极易受到噪声影响。在很小的触发阈值下，一些轻微的光子扰动便可能误触发事件，影响事件信息的准确性；而若调大阈值，那事件相机的灵敏度便会下降，丢失一些关键事件点。

因此，新的方法需要重新思考事件数据的时空、光度和随机性。这就提出了以下问题:**从与给定任务相关的事件中提取信息的最佳方法是什么?以及如何对噪声和非理想效果进行建模，以便更好地从事件中提取有意义的信息?**

### 2.4 Event Generation Model

> https://www.youtube.com/playlist?list=PL03Gm3nZjVgUFYUh3v5x8jVonjrGfcal8

> <img src="Event-based Vision A Survey.assets/image-20231128155650361.png" alt="image-20231128155650361" style="zoom:80%;" />
>
> 事件相机有独立的像素能够响应他们光电流对数L=log⁡(I)的变化，I为亮度。在一个没有噪音的场景中，在位置$X_k=(x_k, y_k)^T$处的像素亮度变化为：$ΔL(X_k,t_k)=L(X_k,t_k)−L(X_k,t_k−Δt_k)Δt_k$为该像素上次发生事件到现在的时间间隔。当亮度变化达到时间对比阈值± C时，一个事件$e_k=(X_k, t_k, p_k) $于$t_k$时刻在位置$X_k=(x_k, y_k)^T$处被触发，极性$p_k∈\{+1, −1\}$ 为亮度变化的标志。

事件生成模型(s)

- 原始的（像素级、非线性、无噪声）
- 线性化的（使用亮度恒定性）
- 更真实的情况（包含噪音）
  - 钟形，高斯形。在哪个变量？
  - 混合物模型（“好”和“坏”的测量值）
  - 依赖于更多的变量（如不应期）

<img src="Event-based Vision A Survey.assets/image-20231128160154057.png" alt="image-20231128160154057" style="zoom:80%;" />

针对一个像素，只要前后一个时刻之间的亮度差到达C，则触发一个事件，如果是亮度上升就是ON，亮度下降就是OFF。

当我们关注这个事件时，我们需要关注事件触发的原因，是因为什么触发了?

<img src="Event-based Vision A Survey.assets/image-20231128160446132.png" alt="image-20231128160446132" style="zoom:80%;" />

上述的方程也被称为时间对比度，这些变化可能是由于光线变化（例如LED）

但是，假设持续照明...是什么原因导致事件？

→场景中移动边缘

那么具体来说是如何通过移动边缘使得其触发事件？观察事件的空间模式：

如上图二，灰色表示在最后几毫秒内没有触发任何事件。白色意味着积极事件，黑色代表消极事件。

从上图来看这几乎就像一个边缘检测器，白色黑色区域的事件在它们是图案中边缘的区域周围触发。

而图三显示垂直或者水平边缘，白色和黑色的边缘，显示它们与运动有依赖性

接下来就是具体分析其图像与运动的依赖方式。分析看到的事件如何触发时间对比条件转向更空间的条件

<img src="Event-based Vision A Survey.assets/image-20231128160524768.png" alt="image-20231128160524768" style="zoom: 80%;" />

由上图可知，先是对触发事件的定义，然后在一个很短时间，它这个ΔL可以进行等价计算，然后可以使用恒定亮度假设来写时间导数

- 亮度的增量由亮度的梯度在图像平面上以速度v在位移Δx:=vΔt上移动引起的
- 如果亮度梯度⊥运动→ 没有生成事件
- 如果亮度梯度∥运动→ 生成事件最快

以下是上述俩个特例的例子

<img src="Event-based Vision A Survey.assets/image-20231128162724153.png" alt="image-20231128162724153" style="zoom:80%;" />

总之，对于恒定照明，事件是由移动的边缘引起的。

当我们逐像素累积事件时，我们看到的是边缘移动和生成事件时的痕迹。

（这不是“运动模糊”）

有些方法利用了线性化模型，有些则没有

<img src="Event-based Vision A Survey.assets/image-20231128162850895.png" alt="image-20231128162850895" style="zoom:80%;" />

## **3** 事件处理 EVENT PROCESSING

让我们来观察一下事件流形式，第一章中提到，事件相机输出的事件流是一个四元组的序列：<img src="Event-based Vision A Survey.assets/2021081610292235.png" alt="img" style="zoom:67%;" />，像素点坐标、时间戳、事件极性包含了一个事件的所有信息，其中像素坐标告诉了我们事件发生的位置，时间戳则是指明了事件发生的事件，事件极性表示了事件的性质。目前，处理事件主要分为两类思想：

#### 3.2 Methods for Event Processing

> 这个是先开题讲到事件处理，那么在事件处理之前还需要将事件进行表征然后再来处理

- **首先是单个事件独立处理**，网络直接读取每个事件的信息，针对每个事件的信息作出相应反应，**此时事件可以视为一个个脉冲，来刺激网络产生对应输出。**这种处理方式的优势十分明显，那就是最大程度地利用了事件流极高的时间分辨率，不会丢失任何有用信息，延时也极低，不过显然传统的深度神经网络并不能这样处理四元事件点数据，也并不能针对高频输入实时作出反应。然而这也并不意味着我们不能这样处理事件，相反，**目前研究人员已经能够通过概率滤波（Probabilistic Filters）、脉冲神经网路（Spike Neural Network，简称SNN）这两种方法有效处理单个事件的输入，它们能够结合过去事件提供的信息以及新到来事件的信息，异步产生输出。**笔者认为，这种事件处理的方式尤其是SNN，是最贴近人脑工作模式，在未来很大潜力会成为人工智能领域的主流方法，在解决包括硬件问题在内的一系列问题后，该方法将会是事件处理最科学、最有效的解决方案。
- 接着便是批量处理事件，也即“事件包处理”。上述单个事件处理的方法有一个不得不面对的问题便是噪声，我们在上一篇文章中提到过，噪声对事件相机的影响是极大的，可能微小的扰动便可能造成一个错误的事件发生，而单个处理事件时，网络并没有消除事件噪声的能力，这便对获取的事件的信噪比有极高的要求。**这时事件包处理的方式就显出明显的优势了，我们设定一段时间（类似于普通相机的曝光时间），这段时间内包含个事件，然后将这个事件整合起来进行处理，由于包含一定数量的事件，少许噪声事件的影响便会变小，这也使得事件包处理中数据的信噪比显著提高**；同时，对于批量事件，可以通过压缩维度等方法，生成深度卷积神经网络可以接受的数据输入格式，最大可能地发挥现有CV算法模型的优势，但压缩维度可能也意味着丢失信息，可能会对结果造成不好的影响。

#### 3.1 Event Representations

##### 2D事件帧—Event Frame

​        传统相机拍出的一张图像可以称为一帧，它是一段曝光时间内的平均值，通常类似VGG、Resnet等深度网络接受的输入也都是以帧为单位的2D灰度图像，**因此将事件数据转换为2D帧是我们首先考虑的方式，转为为2D图像的事件数据称为事件帧**。

​    我们选取**一个时间段内的事件组成事件包**，统计这个包中每个像素点发生事件的情况，最终输出一张事件帧，其中每一像素点均有一个值，该值可以表示事件的某些信息。例如将每个像素点所有时间戳的事件的极性叠加，即可得到最简单的事件帧，这样的事件帧可以看作事件包时间段内事件的集合体。

<img src="Event-based Vision A Survey.assets/image-20231128202120114.png" alt="image-20231128202120114" style="zoom:80%;" />

  这样的操作虽然简单，但是也带来了许多问题：**事件流原本的时间戳和极性携带的信息都被丢弃，事件帧仅记录了每个像素点在特定时间段内事件发生的频率，即表示“这个像素点发生了多少事件”，而具体在什么时间发生的，发生的什么事件这些信息，都无法从中提取**。其实读到这一部分的时候笔者就在思考，既然把时间戳信息给压缩了，事件相机高时间分辨率这个最大的优势就被放弃了，那么事件相机又怎么能在一些CV问题中表现得比普通相机出色呢？论文给出了答案，**即利用每个像素点可以储存一定信息的特点，将时间戳信息通过一定形式存储到每个像素点中，亦或是在事件叠加时用一些核变换或计量函数来取代简单的加法计算**。**而针对极性消失的问题，许多研究人员都采用双通道事件帧的数据形式，即正负极性的事件分别生成事件帧，并按照通道拼接起来。**

#####  Time surface (TS)

 **时间平面（Time Surface）**便是解决事件帧丢失事件信息的一种解决方案。这种事件数据的表示方式依然是2D图，只**不过每个像素会对时间戳信息进行记录，例如该像素最后一个事件的时间戳。**在时间平面中，每个像素的强度表征着该位置的“运动痕迹”，因为产生事件的明暗变化往往和运动相联系，这使得这种事件表示形式在涉及到物体运动的CV问题上表现得尤为出色。

由于时间面对场景的边缘和运动方向非常敏感，因此时间面被用于许多涉及运动分析和形状识别的任务。例如，将局部平面拟合到TS上可以得到光流信息[21]，[148]。TSs被用作层次特征提取器的构建块，类似于神经网络，这些特征提取器将来自连续更大时空邻居的信息聚合起来，然后传递给分类器进行识别[109]，[113]。使用基于图像的方法(Harris, FAST)[111]、[114]、[115]或新的基于学习的方法[112]的适配，TSs在角点检测中很受欢迎。

​	事实上，事件帧作为一种直观、简便的事件数据形式，已经在CV领域中的许多研究方向上得到了应用。**即便是压缩了时间戳，只要选取事件包的时间段小于曝光时间，事件相机就依旧具有优于传统相机时间分辨率的优势**，而往往相对于传统相机来说，这个时间段是很小的，且可以根据任务需求进行改变。同时，事件帧处理过程中并未影响任何事件流大动态范围的特性，因此在类似汽车转角预测、物体识别等问题上，比传统相机更高的时间分辨率和动态范围使得基于事件帧进行网络训练的模型表现出更高的精确性。

##### Motion-compensated event image

​	**运动补偿事件图像**:是一种既依赖于事件又依赖于运动假设的表示。**运动补偿的思想是，当边缘在图像平面上移动时，它会触发它所经过的像素上的事件;边缘的运动可以通过将事件扭曲到一个参考时间并最大化其对齐来估计，从而产生扭曲事件(IWE)的清晰图像(即直方图)**[128]。因此，**这种表征(IWE)提出了一种标准来衡量事件对候选运动的拟合程度:扭曲事件产生的边缘越尖锐，拟合越好**[99]。此外，由此产生的运动补偿图像具有直观意义(即导致事件的边缘模式)，并提供了比事件更熟悉的视觉信息表示。**在某种意义上，运动补偿揭示了事件流中边缘的隐藏(“运动不变”)映射。这些图像可能对进一步的处理有用，如特征跟踪[64]，[129]。有运动补偿版本的点集和时间曲面。**

​	运动补偿是一种估计最适合一组事件的运动参数的技术。它具有连续时间翘曲模型，可以利用事件的精细时间分辨率(章节3.1)，因此与传统的基于图像的算法不同。运动补偿可用于估计自我运动、光流、深度、运动分割或特征运动。

##### Reconstructed images

​	**重建图像:通过图像重建获得的亮度图像**(章节4.6)可以解释为比事件帧或TSs更具有运动不变的表示，并用于推理[8]，产生一流的结果。**事件极性可以用两种方式来考虑:分别处理正事件和负事件并合并结果**(例如，使用TSs[109])，或者以共同的表示方式处理它们(例如，亮度增量图像[64])，**在这种情况下，极性通常会在邻近的事件中聚合。事件极性依赖于运动方向，因此它是一个麻烦的任务，应该独立于运动，如对象识别**(为了缓和这一点，训练数据从多个运动方向应该可用)。**对于运动估计任务，极性可能是有用的，特别是检测方向的突然变化**。

##### 3D表示法—Voxel Grid

​        尽管事件帧这种2D数据形式已经能够在一些CV问题上取得出色成果，但我们还是希望能够最大程度利用事件流丰富的时间信息，以取得更加出色的结果。Voxel Grid则是这样一种事件数据的表示方式，我们同样需要**选取一段时间间隔来提取事件包，但并不对其时间戳进行压缩，而是构建空间-时间的三维坐标。可以理解为每个时间戳对应一帧，这一帧的各个像素记录该时间戳发生事件的极性和，如此便可生成一种3D形式的事件数据，我们称之为Voxel**。当然，Voxel是可以作为现有卷积网络的输入的，无论是将时间轴看作是通道输入2D卷积层还是直接输入3D卷积层，只要根据输入Voxel的维度设置好输入层，便可与现有的深度卷积网络衔接。

<img src="Event-based Vision A Survey.assets/20210818000855867.png" alt="img" style="zoom:80%;" />

​     **Voxel虽然完整地保存了事件流所有的时间信息，但是也同样丢失了事件极性，不过这可以采取和双通道事件帧同样的思想，对不同极性的事件生成不同的Voxel再拼接。那既然Voxel这么好**，岂不是可以所有问题都能用这种数据形式去解决吗？显然，Voxel保留所有信息的代价是庞大的数据量，这给数据处理带来了更多的计算成本，同时，密集的时间戳意味着卷积层需要作更多次的计算，这就引入了更多的参数，使得网络十分庞大，这对要求轻量化的工业应用是极为不利的。即便抛开计算成本不谈，Voxel这种根据时间戳直接呈现事件的表示方式，是否真的能给所有卷积网络提供更多的特征、优化学习结果还是个未知数。

​    总之，数据形式并非一成不变，需要我们根据不同的问题进行选择，往往在初次尝试时，可以选取数据量压缩了的事件帧作为事件表示形式（同时也要注意事件包大小的选取—即时间间隔的选取，这不仅影响到数据处理的计算成本，甚至还影响到训练结果的好坏，这在将来笔者分享的论文中会有体现）；当事件帧难以取得理想效果时，我们便可采用Voxel的形式再进行尝试（Voxel其实是个宽泛的概念，**多个Event Frame拼接起来也可以当作是Voxel，因此，Voxel时间戳的单位并非一定是事件相机的最小时间精度，可以小规模的事件包为单位**，减少数据量）。

​    除了介绍的两类事件批量处理的表示形式外，其实针对不同CV任务，还有更多优秀的方法去提取事件流中有价值的信息，笔者不才，仅对这两种简单易懂的方法有所了解与实践，读者可根据自己需求阅读论文第三章部分。

​    最后值得一提的是，Daniel等人在2019年ICCV中提出的一种可端对端学习的事件处理表示方法，能够有效针对不同应用场景，由机器学习得到对应的事件表示方法，这相当于是提供了一种数据转换的框架，是十分有价值的工作，这篇论文笔者在将来也会进行分享（论文名：End-to-End Learning of Representations for Asynchronous Event-Based Data ）

#### **3.3 Biologically Inspired Visual Processing**

​	通过snn处理事件:人工神经元，如leaky - integration和Fire或Adaptive指数，是受哺乳动物视觉皮层神经元启发的计算原语。它们是人工snn的基本构建块。一个神经元从视觉空间的一个小区域(一个接受域)接收输入脉冲(“事件”)，然后修改它的内部状态(膜电位)，当状态超过一个阈值时产生一个输出脉冲(动作电位)。神经元以分层的方式连接，形成一个SNN。**尖峰可以由事件摄像机的像素或SNN的神经元产生**。信息沿着层次结构传播，从事件摄像机像素到SNN的第一层，然后通过更高(更深层)的层。**大多数第一层接收域是基于高斯差(选择中心环绕对比度)，Gabor滤波器(选择有向边缘)，以及它们的组合**。随着信息深入网络，接受域变得越来越复杂。在神经网络中，**由内层执行的计算近似为卷积**。**在人工SNN中，一个常见的方法是假设一个神经元如果没有从前一个SNN层接收到任何输入峰值，就不会产生任何输出峰值。这种假设允许跳过这些神经元的计算**。这种视觉处理的结果几乎与刺激呈现同时进行[156]，这与传统的cnn有很大的不同，后者是在固定的时间间隔内在所有位置同时计算卷积。



​	任务:生物启发模型已被采用为几个低水平的视觉任务。例如，**基于事件的光流可以通过使用时空导向的过滤器[80]、[134]、[157]来估计**，这些过滤器模拟初级视觉皮层的接受场的工作原理[158]、[159]。基于[161]的生物学提议，**同样类型的定向过滤器已被用于实现基于尖峰的选择性注意模型**[160]。**双目视觉的仿生模型，如周期性横向连接和兴奋-抑制神经连接[162]，已被用于解决基于事件的立体对应问题[41]，[163]，[164]，[165]，[166]或控制人形机器人的双目收敛[167]**。视觉皮层也启发了[168]中提出的分层特征提取模型，该模型已在snn中实现，并用于对象识别。这类网络的性能越好，它们就越能从峰值的精确时间中提取信息[169]。早期的神经网络是手工制作的(如Gabor滤波器)[53]，但最近的研究让神经网络通过激发大脑的学习来构建接受域，如Spike-Timing Dependent可塑性(STDP)，从而产生更好的识别率[136]。本研究通过在深度网络中使用更多受计算启发的监督学习类型(如反向传播)来补充，以有效地实现峰值深度卷积网络[143]、[170]、[171]、[172]、[173]。与传统的视觉方法相比，上述方法的优点是延迟低、效率高

​	为了构建小型、高效和反应性的计算系统，昆虫视觉也是基于事件处理的灵感来源。为此，**小型机器人快速、高效的避障和目标获取系统已被开发**，该系统基于DVS输出驱动的神经元模型，该模型对隐现的物体作出响应并触发逃跑反射

​	[157]中的SNN通过突触连接延迟事件和使用神经元作为巧合探测器来**检测运动模式**。

## 4 算法/应用 ALGORITHMS / APPLICATIONS

应用场景

- 机器人、可穿戴设备等实时交互系统
- 高速运动状态下的目标追踪、物体检测、姿态估计
- 深度估计、三维全景成像、光流估计

事件相机的应用场景大概有上面这些，大部分场景都存在照明条件不受控制、对延迟和功耗的要求很高的问题，而事件相机的优点恰恰解决了这些问题，所以，事件相机在处理计算机视觉问题时有着先天优势。



## 6.2 数据集和模拟器 Datasets and Simulators

​	数据集和模拟器是促进采用事件驱动技术和推进其研究的基本工具。它们可以降低成本(目前，活动摄像机要比标准摄像机贵得多)，并可以通过定量基准来监控进度(就像传统的计算机视觉:Middlebury、MPI sinintel、KITTI、EuRoC等数据集的情况)。基于事件的视觉数据集和模拟器的数量正在快速增长。其中几个列在[9]中，按任务排序。广义上，它们可以被分类为目标运动估计或图像重建(回归)任务、目标识别(分类)任务和端到端的人类标记数据，如驾驶[280]。在第一组中，有光流、SLAM、目标跟踪、分割等数据集。第二组包括对象和动作识别的数据集。

​	光流数据集包括[22]，[190]，[281]。由于地面真实光流难以获取，[190]只考虑IMU记录的纯旋转运动时的流，因此数据集缺乏平移(视差)运动引起的流。[22]，[281]中的数据集提供光流作为由摄像机运动和场景深度在图像平面上诱导的运动场(用距离传感器，如RGB-D相机、一对立体视觉或激光雷达测量)。自然地，地面真光流受到噪声和不同传感器校准和校准的不准确性的影响。

​	姿态估计和SLAM的数据集包括[225]、[98]、[197]、[281]、[282]。最流行的一种是在[98]中描述的，它被用于基准视觉里程计和视觉惯性里程计方法[27]，[103]，[104]，[127]，[128]，[129]，[131]。这个数据集也被广泛用于评估角点检测器[111]、[115]和特征跟踪器[64]、[126]。

​	与传统的计算机视觉相比，目前用于识别的数据集规模有限。它们包括一副纸牌(4种类别)、面孔(7种类别)、手写数字(36种类别)、动态场景中的手势(石头、布、剪刀)、汽车等。流行的基于框架的计算机视觉数据集(如MNIST和Caltech101)的神经形态版本是通过类扫视运动获得的[242]，[283]。更新的数据集[17]，[113]，[284]，[285]是在真实场景中获得的(不是基于帧的数据生成的)。这些数据集已经在[15]，[109]，[113]，[137]，[150]，[151]等中被用于基准的基于事件的识别算法。

​	[73]、[286]和[98]中的分布式交换机仿真器基于**理想分布式交换机像素的工作原理**(2)。给定一个虚拟三维场景以及其中移动的DAVIS的轨迹，仿真器生成相应的事件流、强度帧和深度图。该模拟器在[74]中进行了扩展:它使用了自适应渲染方案，更加逼真，包括一个简单的事件噪声模型，并返回估计的光流。**v2e工具**[40]从视频中生成事件，使用现实的非理想噪声分布式像素模型，将建模扩展到低光照条件。目前还没有对现有事件相机的噪声和动态效应进行全面的表征，因此，目前所使用的噪声模型有些过于简化