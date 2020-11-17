---
layout: post
title: "模型压缩"
subtitle: "Model Compression"
author: "Roger"
header-img: "img/ModelCompression/model_compression.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Model Compression
---

## 模型压缩简介  
> 总结自[李宏毅](http://speech.ee.ntu.edu.tw/~tlkagk/index.html)老师的视频教程：Network Compression  

&emsp;&emsp;为了将神经网络模型部署到手机、智能手表等计算资源有限的设备上，动辄几个GB大小参数的网络自然是不现实的，需要压缩模型到合适的规模。模型压缩方法主要有如下几类：
- 网络剪枝（Network Pruning）
- 知识蒸馏（Knowledge Distillation）
- 参数量化（Parameter Quantization）
- 网络结构设计（Architecture Design）
- 动态计算（Dynamic Computation）
- 硬件设计

## 1. 网络剪枝  
&emsp;&emsp;神经网络通常有很多参数是冗余的（即某些参数很小，或某些neuron在大多数时间里输出都为0或很小）。因此，可以对这些参数进行剪枝以达到减小网络参数的目的。  
**Q**：为什么不直接训练一个小的网络呢？  
**A**：这是因为大的网络往往更容易训练（网络参数越多，遇到local minimum的概率越小，大多数情况是saddle point，且local minimum处的loss很接近于global minimum），可以得到更好的结果。而直接训练一个小的网络结果不如训练大网络的效果好。  
&emsp;&emsp;网络剪枝的流程是：先评估哪些参数不重要可以移除→移除后用训练数据进行Fine-tuning以恢复损失的精度→重复上述过程直至满意的结果。weight pruning会造成不同的neuron的输入/输出不一致，这样的话无法利用GPU加速。为了规范数据以利用GPU加速，可以使用neuron pruning，即直接删除对应的neuron（等效于缩小hidden layer的规模）。  
## 2. 知识蒸馏
&emsp;&emsp;该方法是先训练一个大的神经网络来学习任务，在此基础上再训练一个小的网络来“模仿”该大网络的行为。即给定相同的输入，小网络要模仿大网络的输出。这与直接使用数据标签进行训练的区别是：以多分类任务为例，数据标签只能提供单一的信息（是否是该类），无法提供更详细的类与类之间的关系。而大网络不是只输出单个结果，而是输出一个分布，有助于小网络从中学到更丰富的信息。  
&emsp;&emsp;一个有用的trick：Temperature：  
![Temperature](/img/ModelCompression/Temperature.jpg)  
## 3. 参数量化  
- 使用更少的bit来表示一个值，如float32变为float16
- 权值聚类  
  比如将参数聚为4类，那么可用2 bit来表示四种不同的权值，保留每种权值与对应数值的组合（即保存一个映射表，其中对应的数值可以为该类参数的均值）  
- 在权值聚类基础上更进一步，使用更少的bit来表示聚类簇的频率，如哈夫曼编码
- Binary weights，网络参数值只可能是1/-1，先随机初始化实值网络参数，以及多组不同的由±1权值组成的网络参数组合。计算梯度时，不使用实值参数来计算，而是用和当前实值参数最接近的Binary network的参数来进行计算梯度，用得到的梯度来更新实值网络。  

## 4. 网络结构设计
&emsp;&emsp;对于全连接层网络，假设原网络是由两层Dense层连接组成，两层neuron数分别为$m$、$n$，此时在两层中间再插入一层neuron数为$k$（$k \lt min(m, n)$）的Dense层。那么之前两层之间的参数大小为$m\cdot n$，插入后参数大小变为$m\cdot k+n\cdot k=k\cdot (m+n)$。k选择较小可以使整体参数变小，但同时网络的表达能力也变弱。  
&emsp;&emsp;对于卷积网络，用Depthwise Seperable（Depthwise Convolution + Pointwise Convolution）来代替普通的卷积网络。  
![VanillaConv](/img/ModelCompression/vanillaConv.jpg "Vanilla CNN")   
![DepthwiseConv](/img/ModelCompression/depthwiseSeperableConv.jpg "Depthwise Seperable CNN") 

## 5. 动态计算
&emsp;&emsp;在网络电量低或者计算资源不足时，不使用整个网络得出预测结果，而是使用部分网络给出预测结果。比如对于多个Dense层堆叠的网络，在每个（或每几个）层上都加上结果的输出，这样在有需要时，可以只用网络的低层输出的结果作为预测结果，从而减少计算量。  
&emsp;&emsp;这样做有一个缺点：由于低层网络之前只需要学习局部/底层特征，加上结果输出后该部分网络“被迫”学习更加全局的特征，从而破坏了整体的网络布局。研究解决该问题的论文有：Multi-Scale Dense Networks。