---
layout: post
title: "Transformer模型优化及加速概览"
subtitle: ""
author: "Roger"
header-img: "/img/NLP/Taxonomy_of_Transformer_models.jpg"
header-mask: 0.4
mathjax: true
tags:
  - NLP
---

> 本文主要翻译总结自[博客](https://chengh.medium.com/evolution-of-fast-and-efficient-transformers-ec0378257994)，内容上做了些许扩充，感兴趣的建议看原文。
## 背景介绍
&emsp;&emsp;Transformer 模型在当今NLP的地位自不必说，在CV、多模态等领域，基于 Transformer 也开始崭露头角。但在实际使用中，往往会因为 Transformer 的模型复杂度为$\rm{O}(N^2)$而限制了其使用场景。为了能让该模型具备处理更长文本序列的能力，从不同的优化角度出发诞生了许多 Transformer 的变种。这里对这些变种的特点做一个总结和概述。

## 模型复杂度
### 模型参数
&emsp;&emsp;Transformer 模型可学习参数主要由 embedding matrix，attention 模块两部分组成。Embedding matrix 主要由词表大小$V$及隐层维度$H$决定，总参数量为$V\times H$。  
&emsp;&emsp;对于 Attention 模块的注意力部分，假设注意力头数为$A$。维度为$H$的输入经过矩阵$Q$，$K$及$V$的变换得到对应的 query, key, value 向量表征，这里一个头的$Q$，$K$，$V$维度为$H * \frac{H}{A}$，将多个头的参数合并（实际中通过reshape来实现），则$Q$，$K$，$V$维度均为$H\times H$。自注意力的输出会经过一个投影层，模块的输出维度不变，所以这部分参数为$H\times H$。所以注意力模块的参数数量为$4\times H\times H=4\times H^2$。  
&emsp;&emsp;Attention部分的输出会经过两层 MLP，输入维度为$H$，中间维度为$4\times H$，则两层 MLP 的总参数量为：$2 \times H\times 4H=8\times H^2$。总的一个 Attention模块的参数为$12\times H^2$。

&emsp;&emsp;假设Transformer的模块层数为$L$，那么模型总的参数量为$V\times H + L\times12\times H^2$

### 计算复杂度
&emsp;&emsp;Transformer 模型的计算/空间复杂度主要由其注意力模块来决定，这也是模型的瓶颈所在。注意力计算公式：
$$
{\rm{Softmax}} (QK^T)V
$$
&emsp;&emsp;根据上式，若序列长度为$N$，由于每个 token 都需要与所有 token 计算相似性，所以光$QK^T$这项做矩阵乘的复杂度为$N\times H\times H\times N$，即为平方复杂度${\rm O}(N^2)$。

## 优化加速方法
&emsp;&emsp;目前有许多针对 Transformer 的优化，用于处理更长的序列，减少内存占用或加速推理。其中四种常见的方法有：  
1. Segment Level recurrence
   1. Transformer-XL
   2. Compressive Transformers
2. Sparse Attention
   1. Sparse Transformer
   2. Longformer
   3. Adaptive Transformer
   4. Big Bird
   5. Reformer
   6. Routing Transformer
3. Approximation
   1. Linformer
   2. Perceiver
   3. Nyströmformer
   4. Linear Transformer
   5. RFA
   6. Performer
4. Inference Acceleration  
   1. Turbo Transformer
   2. Faster Transformer  

&emsp;&emsp;下面详细介绍四种方法及对应的模型变种。

### 1. Segment Level Recurrence
&emsp;&emsp;这类工作主要是基于 Segment Level Recurrence 来减少内存占用，以使模型可处理更长的输入序列，是一种分治思想。

#### 1.1 Transformer-XL
&emsp;&emsp;原始的 Transformer 模型有着限定长度的注意力范围，每个 token 只可能关注其自身字段（segment）内的其它 token 信息且无法在不同的字段之间传递上下文信息。  
&emsp;&emsp;Transformer-XL 试图通过重用不同字段之间的隐态表征的方法来解决输入长度限制。此外，该模型还使用了一种新的位置编码。  
&emsp;&emsp;Segment Level Recurrence 并不是显式地将 dense 自注意力矩阵优化为 sparse，而是像RNN一样使用 recurrent 机制来连接不同的字段。它将来自前一个字段的表征缓存下来，然后作为当前字段的扩展上下文来使用。这一方法将最长文本长度扩大了 N 倍。其中 N 为网络的深度。

#### 1.2 Compressive Transformers
&emsp;&emsp;Compressive Transformers 是 Transformer-XL 的进一步改进。二者之间最大的区别是后者只保留前一个字段的激活（activation），而前者通过使用二级缓存机制，将过去所有字段的激活都保存了起来。因此，Compressive Transformers 可以处理比 Transformer-XL 更长的文本长度。

### 2. Sparse Attention
&emsp;&emsp;稀疏注意力不是在完整的注意力矩阵中计算所有可能的分数对，而是从输入序列中计算有限的分数对以构建一个稀疏矩阵。在选择有限对进行评分方面有几种稀疏模式。常见的包括：  
- sliding window attention (band attention)
- dilated sliding window attention
- global attention
- random attention
- block local attention

![稀疏注意力模式](/img/NLP/sparse_attention_patterns.png) 

#### 2.1 Longformer
&emsp;&emsp;[Longformer](https://arxiv.org/abs/2004.05150) 使用了多种稀疏注意力模式：它在与序列程度成线性关系的同时，将局部（sliding window attention）与全局（global attention）信息结合了起来。在每个注意力层中，复杂度从${\rm O}(N^2)$减少到${\rm O}(N\times W)$。其中$N$为输入长度而$W$为窗口大小。  
![Longformer](/img/NLP/Longformer.png) 

#### 2.2 Big Bird
&emsp;&emsp;与Longformer 类似，[Big Bird](https://arxiv.org/abs/2007.14062)也结合了 random attention， sliding window attention 以及 global attention 来构建自己的稀疏性。  
![Longformer](/img/NLP/BigBird.png) 
&emsp;&emsp;上图中的稀疏注意力模式多是基于位置的，这意味着选择从输入序列中选择有限对 tokens/pixels 对。另一种生成稀疏注意力矩阵的策略是基于内容或输入 tokens 的方法。Routing Transformer 和 Performer 是典型的基于内容的稀疏注意力矩阵，基本概念是聚类/分桶。

#### 2.3 Routing Transformer
&emsp;&emsp;[Routing Transformer](https://arxiv.org/abs/2003.05997)的想法是学习动态稀疏注意力模式，避免分配计算和内存来关注与感兴趣的 query 无关的内容。它利用 k-means 聚类来聚类 query 和 keys，因此每个 query 仅关注属于同一个 cluster 的 keys 来创建其稀疏性。

#### 2.4 Reformer
&emsp;&emsp;[Reformer](https://arxiv.org/abs/2001.04451)引入了多种全新的技术，比如 multi-round LSH attention 以及 reservable transformer。与 Routing Transformer 使用 k-means 聚类来减少注意力范围并创建稀疏性的方法类似，Reformer 使用一个基于哈希的相似性度量（multi-round Locality Sensitive Hashing）来高效准确地实现 token 分桶并将它们切块成同样大小来方便并行计算。Reformer 声称可以在单处理器上，仅使用16 GB内存就处理长达一百万词的上下文窗口。总的来说，Reformer 使用的用来减少计算复杂度与内存空间的两个关键技术：  
- Locality Sensitive Hashing（LSH）用来减少关注长序列的复杂性，将复杂度从${\rm O}(N^2)$减少到${\rm O}(N\log N)$
- Reversible Residual Layers 被用来更高效地利用内存，使得内存不再与层数成线性关系

![Longformer](/img/NLP/Reformer.png) 
&emsp;&emsp;理论上，LSH可以帮助减少复杂度为${\rm O}(N\log N)$，但实际中，Reformer只有对输入长度>2048的情况才会显现出收益。此外，multi-round LSH attention 也添加了额外的操作，从而造成了性能下降。  

#### 2.5 Sparse Attention 局限性
&emsp;&emsp;稀疏注意力也有其局限性：  
- 要求高效的稀疏矩阵乘运算，这在许多加速器中是不支持的
- 一切运算是无法稀疏化的，比如 Softmax
- 没有严格的理论证明稀疏注意力有足够强的表达能力
- 为了补偿稀疏表征带来的性能损失，往往需要堆叠更多的注意力层，这带来了额外的性能损失  

### 3. Approximation
&emsp;&emsp;低秩近似和核近似是另一种流行的将自注意力矩阵化简为线性复杂度的方法。总的来说，这类方法关注近似或者化简 QKV attention 中的矩阵乘。[Linformer](https://arxiv.org/abs/2006.04768) 和 [Perceiver](https://arxiv.org/abs/2103.03206) 是两个使用低秩近似技巧来做投影的例子，它们声称可以将注意力矩阵优化为线性时间复杂度。  
#### 3.1 Linformer
&emsp;&emsp;Linformer 通过向 K 和 V 矩阵添加一个投影层，将原始的全注意力点乘运算分解为更小的注意力。具体来说，它将 $n\times d$ 维的 keys 和 values 投影为$k\times d$维，即将输入序列的长度投影为一个更短的值，同时保持 key 和 value embedding的维度不变。  

![Linformer](/img/NLP/Linformer.png)   

&emsp;&emsp;当 $k$ 比 $n$ 小很多的时候，Linformer可以显著减少内存占用。此外，Linformer 还引入了其它提高效率和性能的技术：  
- 在投影之间做参数共享，相同的投影可以在不同的 head 和 layer 之间共享（如一个12层，12头堆叠的 Transformer 模型，head-wise sharing，key-value sharing 以及 layer-wise sharing 将会分别引入24，12及1个不同的线性投影矩阵）
- 投影维度 $k$ 对不同的 head 数及 layer 数可调整
- 可以使用如 max/mean pooling 或者卷积等其它投影方法  

#### 3.2 Performer
&emsp;&emsp;[Performer](https://arxiv.org/abs/2009.14794) 在不显式地计算$N\times N$自注意力矩阵的情况下来近似注意力。同类方法还有 [Linear Transformer](https://arxiv.org/abs/2006.16236)，[RFA](https://arxiv.org/abs/2103.02143)（Random Feature Attention），[Nyströmformer](https://arxiv.org/ab1s/2102.03902) 使用近似方法来简化 QKV 注意力矩阵计算。  
&emsp;&emsp;相比于原始注意力矩阵，Performer 首先首先逼近较低秩的随机 Q 和 K 矩阵，然后使用矩阵关联属性以不同的顺序计算最终的注意力。即相比于先做 QK 乘结果再与 V 乘，先做 KV 乘结果再与 Q 乘。

![Linformer](/img/NLP/Performer.jpg)  

&emsp;&emsp;与 Linformer 和之前的稀疏注意力技巧不同，Performer 设计的巧妙之处在于它不显式地计算和存储注意力矩阵 A，因此避免了$N^2$的开销。

### 4. Model Selection
&emsp;&emsp;[下图](https://arxiv.org/abs/2009.06732)包括了当下大多数相关的优化工作。  

![Linformer](/img/NLP/TransformerModels.jpg)   

&emsp;&emsp;各个模型之间的关系图如下：
![Linformer](/img/NLP/Taxonomy_of_Transformer_models.jpg)

&emsp;&emsp;那么，如何选择合适的模型呢？来自 Google 的一项[工作](https://arxiv.org/abs/2011.04006)关注于评估具有不同模式的较长输入序列的模型性能，他们用6个不同的任务从6个不同的角度来评估这些模型。  

![Linformer](/img/NLP/performance_comparision.png) 

&emsp;&emsp;总的来说，这些模型在不同的任务上有不同的性能表现。一般基于 **low-rank** 和 **kernel approximation** 的方法（如 Performer，Linformer，Linear Transformer）在推理速度，任务表现及内存占用方面表现出更好的平衡性。  

### 5. Inference Acceleration
&emsp;&emsp;除了模型层面的优化，还有很多工作关注在架构及应用层面的优化。推理加速方面有像 NVIDIA的 [Faster Transformer](https://github.com/NVIDIA/FasterTransformer) 和腾讯的 [Turbo Transformer](https://github.com/Tencent/TurboTransformers) 等致力于提升推理性能的工作。  

#### 5.1 Faster Transformer
&emsp;&emsp;Faster Transformer 不修改模型架构而是在计算加速层面优化 Transformer 的 encoder 和 decoder 模块。做的事情包含：  
- 尽可能多地融合除 GEMM 以外的操作
- 支持 FP16 和 INT8
- 移除 encoder 输入中无用的 padding 来减少计算开销（借鉴自字节跳动的工作 [Effective Transformer](https://github.com/bytedance/effective_transformer)）  

&emsp;&emsp;用 Transformer的 encoder 阶段（即 BERT）做 benchmarking 的结果显示：Faster Transformer 在小的 batch size 及序列长度时可以带来3倍的加速；在大 batch size 及序列长度时，利用 INT8-v2 量化可以带来5倍的加速。
![Linformer](/img/NLP/Faster_Transformer_TF_Encoder.png)

&emsp;&emsp;对于大 batch size 和序列长度，EFF-FT （Effective FasterTransformer） 和 FT-INT8-v2 都带来了 2 倍的加速。同时使用 EFF-FT 和 INT8-v2 相比 FasterTransformer FP16 可以带来大约 3.5 倍的加速。  
![Linformer](/img/NLP/Faster_Transformer_benchmark.png)

&emsp;&emsp;Decoder部分，与 TensorFlow 相比，FT-Decoder 算子提供了1.5~3倍的加速，FT-Decoding 算子提供了4~18倍的加速。
![Linformer](/img/NLP/Faster_Transformer_Decoder.png)

#### 5.2 Turbo Transformer
&emsp;&emsp;[Turbo Transformer](https://arxiv.org/pdf/2010.05680.pdf) 是另一个专注于提高 Transformer 推理性能的项目，它由 computation runtime 及 serving framework 组成。加速适用于 CPU 和 GPU，最重要的是，它可以无需预处理便可处理变长的输入序列。一些要点如下：  
- 与 FasterTransformer 类似，它融合了除 GEMM 之外的操作以减少计算量
- smart batching，对于一个 batch 内不同长度的序列，它也最小化了 zero-padding 开销
- 对 LayerNorm 和 Softmax 进行批处理，使它们更适合并行计算
- 引入了模型感知分配器，以确保在可变长度请求服务期间内存占用较小  

&emsp;&emsp;Turbo Transformer 的实现与 Pytorch、Tensorflow 和 Faster Transformers 相比，Turbo Transformer 总的来说实现了更高的 QPS/throughput。  
&emsp;&emsp;就固定长度的输入来说，Turbo Transformer 的 runtime 大约比 XLA 和 ONNXruntime 快10%，但比 Faster Transformers 和 TensorRT 要慢越10%。因为 TensorRT 需要offline tuning 过程，在此期间它可以为 GEMM kernel 选择最优参数且可能识别出对于非 GEMM kernel 来说最优的 CUDA 线程块大小。