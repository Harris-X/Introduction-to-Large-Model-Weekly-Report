# A Survey on Model Compression for Large Language Models

> - https://arxiv.org/pdf/2308.07633.pdf
> - https://zhuanlan.zhihu.com/p/652092496

## 2 方法

### 2.1 修剪

剪枝是一种强大的技术，可以通过删除不必要或冗余的组件来减少模型的大小或复杂性[LeCun et al., 1989; 韩等人，2015； Li et al., 2017]。众所周知，有很多冗余参数对模型的性能影响很小甚至没有影响，因此直接剪枝这些冗余参数后模型的性能下降最少。 同时，剪枝可以使模型存储友好[Ardakani et al., 2019]、内存效率[Han et al., 2015； Yang et al., 2017], 计算效率 [Li et al., 2017]。 剪枝可以分为非结构化剪枝 [Zhang et al., 2018; Gordon 等人，2020] 和结构化修剪 [Anwar 等人，2017； Fang 等人，2023]。 **结构化剪枝和非结构化剪枝的主要区别在于剪枝目标和所得的网络结构。 结构化剪枝根据特定规则删除连接或层次结构，同时保留整体网络结构。 另一方面，非结构化剪枝会剪枝各个参数，从而产生不规则的稀疏结构。** 最近的研究工作致力于将LLMs与修剪技术相结合，旨在解决与LLMs相关的巨大规模和计算成本。 在本节中，我们根据这些作品是否采用结构化或非结构化修剪策略对这些作品进行系统分类。

**非结构化修剪** 非结构化修剪通过删除特定参数而不考虑其内部结构来简化 LLM。 **这种方法针对的是LLMs中的个体权重或神经元，通常通过应用阈值将低于其的参数归零**。 然而，该方法忽视了LLM的整体结构，导致模型组成不规则。 这种不规则性需要专门的压缩技术来有效存储和计算修剪后的模型。 **非结构化修剪通常涉及对LLMs进行大量的再培训以重新获得准确性，这对于LLMs来说尤其昂贵。** 该领域的一种创新方法是 SparseGPT [Frantar 和 Alistarh，2023]。它引入了一种不需要重新训练的一次性剪枝策略。 该方法**将剪枝视为广泛的稀疏回归问题，并使用近似稀疏回归求解器对其进行求解**。 SparseGPT 实现了显着的非结构化稀疏性，在 OPT-175B 和 BLOOM-176B 等最大的 GPT 模型上甚至高达 60%，而困惑度的增加最小。 与此相反，Syed 等人。 **提出一种迭代剪枝技术，在剪枝过程中以最少的训练步骤微调模型**。 另一项进步是 LoRAPrune [Zhang 等人，2023a]，**它将参数高效调整 (PEFT) 方法与剪枝相结合，以增强下游任务的性能。 它使用低秩适应 (LoRA) 的值和梯度引入了独特的参数重要性标准** [Hu et al., 2022]。 为了响应 SparseGPT 仍然需要的资源密集型权重更新过程，Wanda [Sun et al., 2023] 提出了一种新的剪枝指标。 Wanda **根据每个权重的大小和相应输入激活的范数的乘积来评估每个权重，并使用小型校准数据集进行近似。** 该度量用于线性层输出内的局部比较，从而能够从 LLMS 中删除较低优先级的权重。

**结构化修剪** **结构化修剪通过删除整个结构组件（例如神经元、通道或层）来简化LLMs**。 这种方法同时针对整组权重，具有降低模型复杂性和内存使用量，同时保持整体 LLM 结构完整的优点。 这个领域的一个例子是 LLM-Pruner [Ma et al., 2023]，它采用通用方法来压缩 LLM，同时保护其多任务解决和语言生成能力。 LLM-Pruner 还解决了LLMs使用的大量训练数据带来的挑战，这可能导致大量的数据传输和训练后模型大小。 为了克服这些挑战，LLM-Pruner 结合了**依赖性检测算法来查明模型中相互依赖的结构。 它还实现了一种有效的重要性估计方法，该方法考虑一阶信息和近似的 Hessian 信息**。 该策略有助于选择最佳的剪枝组，从而改进压缩过程。

### 2.2 知识蒸馏

知识蒸馏（KD）[Hinton 等人，2015； 金和拉什，2016； Tung 和 Mori，2019] 是一种有价值的机器学习技术，旨在提高模型性能和泛化能力。 它通过将知识从复杂的模型（称为教师模型）转移到更简单的模型（称为学生模型）来实现这一点。 **KD背后的核心思想是将教师模型的综合知识转化为更精简、更有效的表示。 在本节中，我们概述了聘请LLMs作为教师的蒸馏方法。** 我们根据这些方法的重点是否是将LLMs的新兴能力（EA）提炼成小语言模型（SLM）来对这些方法进行分类。 因此，我们将这些方法分为两个不同的类别：标准 KD 和基于 EA 的 KD。 为了直观地表示，图 2 提供了LLMs知识蒸馏的简要分类。

<img src="A Survey on Model Compression for Large Language Models.assets/v2-3f0e0a13dbef9f11d56e77b9ce5dcea4_720w.webp" alt="img" style="zoom:80%;" />

**标准KD** **标准 KD 侧重于使学生模型能够学习LLMs拥有的公共知识，例如输出分布和特征信息。** 这种方法类似于普通 KD [Gou et al., 2021; 帕克等人，2019； 赵等人，2022； Liu et al., 2021a]，但**区别在于教师模型是LLMs**。 一个说明性的例子是 MINILLM [Gu et al., 2023]，它深入研究了白盒生成LLMs的 distillaLion。 它观察到最小化前向 Kullback-Leibler 散度 (KLD) 的挑战 - 这可能会导致教师分布中不太可能的区域出现概率过高，从而在自由运行生成过程中产生不可能的样本。 为了解决这个问题，MINILLM 选择最小化反向 KLD。 这种方法可以防止学生高估教师分布中的低概率区域，从而提高生成样本的质量。 相比之下，GKD [Agarwal et al., 2023] 探索了自回归模型的蒸馏，其中白盒生成 LLM 是一个子集。 该方法确定了两个关键问题：训练期间的输出序列与学生在部署期间生成的输出序列之间的分布不匹配，以及模型规格不足，其中学生模型可能缺乏与教师分布相匹配的表达能力。 GKD 通过在训练期间对学生的输出序列进行采样来处理分布不匹配。 它还通过优化反向 KL 等替代散度来解决模型规格不足的问题。

**基于 EA 的 KD** 基于 EA 的 KD 不仅仅转移LLMs的常识，还包括提炼他们独特的新兴能力。 最近的研究 [Wei 等人，2022a； 谢弗等人，2023]； [Zhao et al., 2023] 强调，尽管强调增加模型大小，但与 BERT（330M 参数）和 GPT-2 等较小模型相比，GPT-3（175B 参数）和 PaLM（540B 参数）等 LLM 展示了独特的行为 （1.5B 参数）。 这些LLMs在处理复杂的任务时表现出令人惊讶的能力，称为“应急能力”。 **涌现能力包含几个有趣的方面，包括情境学习 (ICL)** [Dong et al., 2023; Wang 等人，2023b]，**思想链 (CoT)** [Wei 等人，2022b； Wang 等人，2023c； Shi et al., 2023]，**以及指令遵循（IF）**[Ouyang et al., 2022; Brooks 等人，2023]。有关直观概述，请参阅图 3，它提供了基于 EA 的知识蒸馏概念的简明表示。

<img src="A Survey on Model Compression for Large Language Models.assets/v2-88b189fa883e193da36fb0a1f59cfd36_720w.webp" alt="img" style="zoom:80%;" />

**ICL 采用结构化自然语言提示，其中包含任务描述以及可能的一些任务示例作为演示。** 通过这些任务示例，LLMs可以掌握并执行新任务，而无需显式梯度更新。 黄等人的作品。 **引入了 ICL 蒸馏，它将上下文中的小样本学习和语言建模功能从 LLM 转移到 SLM**。 这是通过将上下文学习目标与传统语言建模目标相结合来实现的。 为了实现这一目标，他们在两种小样本学习范式下探索了 **ICL 蒸馏：元上下文调优 (Meta-ICT) 和多任务上下文调优 (Multitask-ICT)**。 **在 Meta-ICT 中，语言模型使用上下文学习目标在不同任务中进行元训练。** 这使其能够通过情境学习来适应看不见的任务，从而扩展其解决问题的能力。 另一方面，**Multitask-ICT 使用 ICL 目标和目标任务中的一些示例对模型进行微调。 随后，它采用上下文学习来对这些任务进行预测**。 比较这两种范式，多任务 ICT 表现出优于元 ICT 的性能。 然而，它在任务适应期间确实需要更多的计算资源，从而使其计算更加密集。

**与 ICL 相比，CoT 采用了不同的方法，它将中间推理步骤（可以导致最终输出）合并到提示中，而不是使用简单的输入输出对。** MT-COT [Li et al., 2022] 旨在**利用LLMs产生的解释来加强小型推理机的训练。** 它利用多任务学习框架使较小的模型具有强大的推理能力以及生成解释的能力。 **Fine-tuneCoT** [Ho et al., 20231] 更进一步，**通过随机抽样从 LLM 生成多个推理解决方案**。 训练数据的增强有助于学生模型的学习过程。 Fu 等人的研究人员。 **确定语言模型的多维能力之间的权衡，并提出微调指令调整模型**。 他们从大型教师模型中**提取 CoT 推理路径**，以提高分布外泛化能力。 谢等人。 **使用LLMs原理作为在多任务框架内训练较小模型的额外指导**。 SOCRATIC CoT [Shridhar et al., 2023] **训练两个精炼模型：问题分解器和子问题求解器。 分解器将原始问题分解为一系列子问题，而子问题求解器负责解决这些子问题**。 DISCO [Chen 等人，2023] 介绍了一种**基于 LLM 的全自动反事实知识蒸馏方法。 它的工程师提示使用LLMs生成短语扰动，然后通过特定于任务的教师模型过滤这些扰动，以提取高质量的反事实数据**。 为了保证基本原理的准确性，SCOTT [Wang et al., 2023a] **采用对比解码，将每个基本原理与答案联系起来**。 它鼓励老师提出相关的理由。 此外，引导学生进行反事实推理，并根据导致不同答案的理由进行预测。

IF 致力于仅基于阅读任务描述来增强语言模型执行新任务的能力，而不依赖于少数样本。 **通过使用一系列以指令表示的任务进行微调，语言模型展示了准确执行以前未见过的指令中描述的任务的能力**。 例如，Lion [Jung et al., 2023] 利用LLMs的适应性来提高学生模型的表现。 它**促使LLMs识别并生成“硬”指令，然后利用这些指令来增强学生模型的能力**。 这种方法利用了LLMs的多功能性来指导学生模型的学习，以解决复杂的指令和任务。

### 2.3 量化

**量化可以分为三种主要方法： 量化感知训练** (QAT) [Tailor et al., 2021; Kim 等人，2022； Ding et al., 2022]，**量化感知微调**（QAF）[Cai et al., 2019； Dong et al., 2019] **和训练后量化 (PTQ)** [Liu et al., 2021b; 内格尔等人，2020； Fang et al., 2020]。这些方法的主要区别在于何时应用量化来压缩模型。 QAT 在模型的训练过程中采用量化，QAF 在预训练模型的微调过程中应用量化，PTQ 在模型完成训练后对其进行量化。 最近的研究工作利用量化来压缩LLMs，取得了令人印象深刻的成果。 这些工作分为上述三种方法：量化感知训练、量化感知微调和训练后量化。 此外，表 1 总结了应用于 LLM 的量化方法。 该表根据 LLM 权重中的位数（精度）将这些工作分为 8 位量化和低位量化。

<img src="A Survey on Model Compression for Large Language Models.assets/v2-326fcb036922026ae3ab42976dec495f_720w.webp" alt="img" style="zoom:80%;" />

**量化感知训练** 在 QAT 中，量化目标被无缝集成到模型的训练过程中。 <u>这种方法使LLMs能够在训练期间适应低精度表示，从而增强其处理量化引起的精度损失的能力。 这种调整的目的是即使在量化过程之后也能保持更高的性能。</u> 例如，LLM-QAT [Liu et al., 2023] 深入研究了获取 LLM 训练数据的挑战。 鉴于收集LLMs训练数据的要求可能很高，LLMs-QAT 提出了一种创新的解决方案。 它利用预训练模型生成的世代来实现无数据蒸馏。 这种方法极大地有助于规避数据收集挑战。 此外，LLM-QAT 更进一步，不仅量化权重和激活，还量化键值 (KV) 缓存。 该策略旨在提高吞吐量并支持更长的序列依赖性。 LLM-QAT 的一个值得注意的成就是它能够提取具有量化权重和仅 4 位的 KV 缓存的大型 LLaMA 模型。 这一突破性的结果证明了产生精确的 4 位量化 LLM 的可行性。

**量化感知微调** QAF 涉及在微调过程中量化 LLM。 <u>主要目标是确保经过微调的 LLM 即使在量化到较低位宽后也能维持其性能。</u> 通过将量化意识集成到微调中，LLMs旨在在模型压缩和保持其性能之间取得平衡。 PEQA [Kim 等人，2023] 和 QLORA [Dettmers 等人，2023a] 都属于量化感知参数高效微调 (PEFT) 技术的范畴 [Liu 等人，2022a； 丁等人，2023； Fu 等人，2023b]。 这些技术专注于促进模型压缩和加速推理。 PEQA 采用双阶段工艺。 在第一阶段，每个全连接层的参数矩阵被量化为低位整数矩阵和标量向量。 在第二阶段，对每个特定下游任务的标量向量进行微调。 QLORA 引入了创新概念，例如新数据类型、双量化和分页优化器。 这些想法旨在节省内存而不影响性能。 QLORA 使大型模型能够在单个 GPU 上进行微调，同时在 Vicuna 基准上实现最先进的结果 [Chiang et al., 2023]。

**训练后量化** PTQ 涉及在LLMs培训阶段完成后量化LLMs的参数。 PTQ 的主要目标是降低 LLM 的存储和计算复杂性，而无需修改 LLM 架构或需要重新训练过程。 <u>PTQ 的主要优点是其实现模型压缩的简单性和效率。</u> 然而，值得注意的是，由于量化过程，PTQ 可能会带来一定程度的精度损失。 这种方法是提高LLMs效率的直接方法，无需进行重大改变或进行大量培训。

在 PTQ 中，某些方法专注于仅量化 LLM 的权重，以提高效率并减少计算需求。 具体来说，LUT-GEMM [Park et al., 2022] 使用仅权重量化和 BCQ 格式优化 **LLM 内的矩阵乘法** [Rastegari et al., 2016]，通过提高计算效率来减少延迟并提高性能。 LLM.int8() [Dettmers et al., 2022] 在 LLM 转换器中采用 8 位量化进行矩阵乘法，有效地将推理期间的 GPU 内存使用量减半，同时保持性能精度。 **该方法采用矢量量化和混合精度分解来处理异常值，以提高效率进行推理。** 值得注意的是，LLM.int8() 可以在具有多达 1750 亿个参数的模型中进行推理，而不会影响性能。 ZeroQuant [Yao et al., 2022] 集成了硬件友好的量化方案、**逐层知识蒸馏和优化的量化支持**，以将基于 Transformer 的模型中的权重和激活精度降低到 INT8，同时对精度影响最小。 GPTQ [Frantar et al., 2022] 承认上述方法对于 8 位权重等低压缩目标效果很好，但在保持更高速率的准确性方面面临挑战。 为了应对这些挑战，GPTQ 提出了一种基于近似二阶信息的新型分层量化技术。 结果是将位宽减少到每个权重 3 或 4 位，与未压缩版本相比，精度损失最小。 Dettmers 和 Zettiemoyer 通过**分析推理缩放定律**，深入研究了LLMs中涉及零样本性能的模型大小和位精度之间的权衡。 他们在各种 LLM 系列中进行的广泛实验表明，**4 位精度几乎是在模型总位数和零样本精度之间实现适当平衡的最佳选择**。 **AWQ [Lin et al., 2023] 发现权重对于 LLM 的性能并不同等重要，仅保护 1% 的显着权重就可以大大减少量化误差**。 基于这一见解，AWQ 通过考虑与较大激活幅度相对应的权重通道的重要性，采用激活感知方法，这在处理重要特征中发挥着关键作用。 **该方法结合了每通道缩放技术来确定最佳缩放因子，在量化所有权重的同时最大限度地减少量化误差**。 OWQ [lee et al., 2023] **对激活异常值如何放大权重量化中的误差**进行了理论分析。 从此分析中汲取见解，OWQ 引入了混合精度量化方案，该方案对易受激活异常值引起的量化影响的权重应用更高的精度。 为了进一步将精确的 LLM 压缩到每个参数 3-4 位，同时保持近乎无损，SPQR [Dettmers et al., 2023b] **识别并隔离异常值权重，以更高的精度存储它们，并将所有其他权重压缩到 3-4 位。**

除了上述仅量化 LLM 权重的工作外，PTQ 中的许多工作都尝试量化 LLM 的权重和激活。 具体来说，SmoothQuant [Xiao et al.,2022]解决了量化激活的挑战，由于异常值的存在，激活通常更加复杂。 观察到不同的标记在其通道上表现出相似的变化，**SmoothQuant 引入了每通道缩放变换，可以有效地平滑幅度，使模型更适合量化**。 认识到LLMs中量化激活的复杂性，RPTQ [Yuan et al., 2023] 揭示了除了异常值的存在之外，不同通道的不均匀范围所带来的挑战。 为了解决这个问题，**RPTQ策略性地将通道排列成簇进行量化，有效地减轻了通道范围的差异。 此外，它将通道重新排序集成到层范数操作和线性层权重中，以最小化相关开销。** Olive [Guo et al., 2023] 进一步**采用异常值-受害者对（OVP）量化，并在本地处理异常值**，硬件开销低，性能增益高，因为它发现**异常值很重要，而旁边的正常值则不那么重要**。 。 异常值抑制+ [Wei et al., 2023] 通过**确认激活中的有害异常值表现出不对称分布（主要集中在特定通道）**，扩展了这种理解，并引入了一种涉及通道级移位和缩放操作的新颖策略，以纠正异常值的不对称呈现 异常值并减轻有问题通道的影响，并定量分析移动和缩放的最佳值，同时考虑异常值的不对称性质和下一层权重引起的量化误差。 ZeroQuant-FP [Wu et al., 2023] 探索**浮点 (FP) 量化的适用性，特别关注 FP8 和 FP4 格式**。 研究表明，**对于 LLM，FP8 激活始终优于其整数对应项 (INT8)，而在权重量化方面，FP4 与 INT4 相比，即使不是更出色，也表现出相当的性能。** 为了解决权重和空腔之间的差异带来的挑战，**ZeroQuant-FP 要求所有缩放因子均为 2 的幂，并将缩放因子限制在单个计算组内。** 值得注意的是，ZeroQuant-FP还**集成了低秩补偿（LoRC）策略**，以进一步增强其量化方法的有效性。



### 2.4 低阶因式分解

低阶分解 [Cheng et al., 2017; 波维等人，2018； Idelbayev 和 Carreira-Perpinan，2020] 是一种模型压缩技术，**旨在通过将给定的权重矩阵分解为两个或多个维度显着降低的较小矩阵来近似给定的权重矩阵。** 低秩分解背后的核心思想涉及将大权重矩阵 W 分解为两个矩阵 U 和 V，使得 W ≈ UV，其中 U 是 m x k 矩阵，V 是 k x n 矩阵，其中 k 远小于 m 和 n. U和V的乘积近似于原始权重矩阵，从而导致参数数量和计算开销的大幅减少。 在LLM研究领域，低秩分解已被广泛采用来有效地微调LLM，例如LORA [Hu et al., 2022]及其变体[Valipour et al., 2023; 张等人，2023b； Chavan 等人，2023]。 与上述工作不同，我们重点关注这些使用低秩分解来压缩 LLM 的工作。 在LLM研究的模型压缩领域，研究人员经常将多种技术与低秩分解相结合，包括剪枝、量化等，例如LoRAPrune [Zhang et al., 2023a]和ZeroQuantFP [Wu et al., 2023] ，在保持性能的同时实现更有效的压缩。 随着该领域研究的继续，将低秩因式分解应用于压缩LLMs方面可能会取得进一步的进展，但似乎仍需要进行持续的探索和实验，以充分利用其对LLMs的潜力。