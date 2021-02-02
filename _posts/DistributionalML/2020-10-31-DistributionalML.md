---
layout: post
title: "分布式机器学习"
subtitle: "Distributional Machine Learning"
author: "Roger"
header-img: "img/distributed_ML.png"
header-mask: 0.4
mathjax: true
tags:
  - Distributed System
---

## 为什么要并行
&emsp;&emsp;现在的机器学习训练数据集越来越大，模型参数规模也越来越大（如BERT），训练时间会变得很长，更不用说模型调参了。而单个主机可扩展的CPU/GPU数量有限，想要做大规模训练就需要用到大量节点来同时计算。  
**重要概念**  
- 通信方式：共享内存 V.S. 消息传递
- 节点结构：客户端-服务器端 V.S. 点对点
- 同步方式：批量同步 V.S. 异步
- 并行方式：数据并行 V.S. 模型并行

## 线性回归简介
&emsp;&emsp;对于线性回归模型，我们要求的是对于给定输入$x\in\mathbb{R}^d$，预测$f(x)=x^Tw$，其中$w$为模型参数。模型参数可以通过最小化损失函数$L(w)=\sum_{i=1}^{n}\frac{1}{2}(x_i^Tw-y_i)^2$得到。计算该损失函数的梯度为$g(w)=\frac{\partial L(w)}{\partial w}=\sum_{i=1}^{n}g_i(w)=\sum_{i=1}^{n}(x_i^Tw-y_i)x_i$，求得的结果$g(w)$为一个维度和$w$相同的向量，求得梯度后可以使用梯度下降方法来调整参数。可以看到$g_i$是与独立的样本相关联的，这使得分布式计算可以相互独立进行。线性回归梯度的计算是最耗时的，这一部分的时间复杂度为$O(m\ast n)$，其中$m$和$n$分别为样本大小和参数维度。通过将样本及参数$w$均匀地分配到不同的处理器上，分别计算梯度$g$后再累加起来便可得到最终梯度。  
## 处理器间通信
1. 共享内存的方法比较容易，但无法大规模并行，因为需要在同一主机上运行，无法扩展到大规模集群。
2. 消息传递方式，多个节点，每个节点都有若干处理器，各个节点之间可通过TCP/IP等协议进行通信以传递必要的消息。这种方法有两种架构方式：  
   - Client-Server架构，选定一个节点作为server节点，其余节点为worker节点，每个worker节点负责实际的运算，server节点负责协调任务、汇总各节点返回结果
   - Peer-to-Peer架构，所有的节点具有相同的地位，每个节点可以和其邻居节点通信  

## 实现方法（编程模型）
### 1. MapReduce（同步算法）
&emsp;&emsp;MapReduce最早是由[Google提出](https://research.google.com/archive/mapreduce-osdi04.pdf)，有很多开源实现（如Hadoop，Spark）。MapReduce是**Client-Server**架构，使用消息传递方式通信，以同步（即每一轮需worker全部完成任务后再进行下一轮运算）的方式并行。数据存在于worker上由各worker进行计算（Map），由server负责调度并汇总所有worker的计算结果（Reduce）。  
&emsp;&emsp;以线性回归的数据并行应用为例，server在对worker广播模型参数$w$，各worker利用本地样本以及得到的模型参数计算本地梯度并求和得到本地梯度向量（Map操作）。然后，server与worker通信收集各worker的梯度向量并相加获得最终梯度（Reduce操作），server获得最终梯度后对模型参数进行梯度下降，然后再将新的模型参数分发到各worker上，进行下一轮迭代。  
#### 使用MapReduce实现并行梯度下降
&emsp;&emsp;算法流程如下：
- 广播参数：server将最新参数广播给所有worker
- Map：worker在本地计算
  - 将$(x_i,y_i,w_t)$映射为$g_i=(x_i^Tw-y_i)x_i$
  - n个样本得到n个向量：$g_1,g_2,g_3,\cdots,g_n$
- Reduce：每个worker通过求和得到本地梯度向量$g=\sum_{i=1}^{n}g_i$，然后server将m个worker的梯度结果相加得到最终的梯度向量$g$
- server更新参数：$w_{t+1}=w_t-\alpha\cdot g$  

&emsp;&emsp;若有m个worker，理想情况下算法的运行时间会减少到原来的$\frac{1}{m}$，但使用更多的worker与减少的训练时间不是线性关系的，这是由节点之间通信开销以及同步开销造成的。实际使用中需要在增加核数带来的加速与额外开销之间进行权衡。  
#### 通信开销
&emsp;&emsp;通信开销需要考虑以下几方面：  
- **通信复杂度**随着模型参数的增加、worker节点数的增加而增加
- **网络延迟**节点之间收发数据包在网络中传递所产生的延迟

&emsp;&emsp;常用的估计通信时间公式：$\frac{\mathrm{complexity}}{\mathrm{bandwidth}}+\mathrm{latency}$  
#### 同步开销
&emsp;&emsp;由于是同步并行的，所以每轮迭代计算时间是由最慢的worker决定的。且如果一个节点故障重启，则会造成整体耗时的增加，而且节点越多，故障的概率也就越大，同步时间开销也就越大。  
### 2. Parameter Server（异步算法）
&emsp;&emsp;[2011年](https://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent.pdf)后开始流行。同步算法理论上更快，但由于同步开销等原因，使得异步算法实际上更快。异步算法的Parameter server架构也是**Client-Server**架构，其中一个（或几个）节点作为server，用于调度协调其它节点，其它则为worker节点。worker计算梯度并将梯度发给server，server完成梯度并将更新后的参数发给worker。  
&emsp;&emsp;Parameter server和MapReduce很像：架构均为client-Server，通信方式均为message-passing，二者的主要区别在于通信的同步/异步。同步算法在每一轮迭代中需要等待所有worker完成运算后才可以进行通信，这使得每一轮迭代时间由最慢的worker来决定，大量时间浪费在等待上。而异步算法不需要等待所有worker完成计算，每个worker完成运算后立即与server通信并开始下一轮计算，无需等待其它worker，效率很高。异步梯度下降流程如下（对于数据并行引用场景）：  
1. 将数据平均划分到所有m个worker中，每个节点拥有部分数据（inputs & targets）
2. worker端和server端以不同的方式进行计算：

|第i个worker|server|
|:--:|:--:|
|1. 从server中获取最新的模型参数$w$|1. 接收worker发来的梯度$\tilde{g}_i$|
|2. 使用本地数据及参数$w$计算梯度$\tilde{g}_i$|2. 更新参数：$w\leftarrow w-\alpha\cdot\tilde{g}_i$|
|3. 将梯度$\tilde{g}_i$发送给server||  

#### 异步算法优缺点
- 理论上，同步算法收敛更快，异步算法收敛速度更慢
- 实际上，异步算法比同步算法更快
- 异步算法使用的限制：不能有相比其它worker慢很多的worker，如联邦学习场景（__原因__：等这个特别慢的worker计算得到梯度时，其它的worker已经通过多次更新使得模型参数得到改进，而这个慢的worker仍使用的是旧的参数，得到的也是旧的、相对不准确的梯度，如果用于server更新的话可能反而有害）  

### 3. Decentralized Network
&emsp;&emsp;Mapreduce和Parameter server都是client-server架构，而decentralized network是peer-to-peer（P2P）架构，每一个节点都是worker。Decentralized Network通信方式也是message passing，每个节点和其邻区节点通信。去中心化算法实现的梯度下降被[证明](https://arxiv.org/abs/1705.09056)是可以收敛的。算法的收敛速度取决于节点之间的连接方式，在节点拓扑为全连接时收敛很快，若图不是强连接的则算法根本不会收敛。  
&emsp;&emsp;Decentralized network算法，对于数据并行（Data parallelism）的应用方式，数据被划分到节点上，每个节点都有本地数据及当前参数$w$的一个副本，不过这些参数副本都略有差别，在算法的最终会收敛到相同结果。每个节点重复进行如下4步计算：  
1. 使用本地数据及本地当前参数$\tilde{w}_i$计算梯度$\tilde{g}_i$
2. 从邻居节点中拉取参数$\lbrace\tilde{w}_k\rbrace$
3. 计算参数$\tilde{w}_i$以及$\lbrace\tilde{w}_k\rbrace$的加权平均并将结果作为新的$\tilde{w}_i$
4. 进行梯度更新：$\tilde{w}_i\leftarrow\tilde{w}_i-\alpha\cdot\tilde{g}_i$  

### 4. 总结

| |MapReduce|Parameter Server|Decentralized Network|
|:--:|:--:|:--:|:--:|
|通信方式|message passing|message passing|message passing|
|节点架构|client-server|client-server|peer-to-peer|
|同步方式|同步|异步|同步/异步|