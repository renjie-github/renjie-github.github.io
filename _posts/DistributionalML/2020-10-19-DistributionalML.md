---
layout: post
title: "分布式机器学习"
subtitle: "Distributional Machine Learning"
author: "Roger"
header-img: "img/post-create-blog.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Blog
---

## 线性回归简介
&emsp;&emsp;对于线性回归模型，我们要求的是对于给定输入$x\in\mathbb{R}^d$，预测$f(x)=x^Tw$，其中$w$为模型参数。模型参数可以通过最小化损失函数$L(w)=\sum_{i=1}^{n}\frac{1}{2}(x_i^Tw-y_i)^2$得到。计算该损失函数的梯度为$g(w)=\frac{\partial L(w)}{\partial w}=\sum_{i=1}^{n}g_i(w)=\sum_{i=1}^{n}(x_i^Tw-y_i)x_i$，求得的结果$g(w)$为一个维度和$w$相同的向量，求得梯度后可以使用梯度下降方法来调整参数。可以看到$g_i$是与独立的样本相关联的，这使得分布式计算可以相互独立进行。线性回归梯度的计算是最耗时的，这一部分的时间复杂度为$O(m\ast n)$，其中$m$和$n$分别为样本大小和参数维度。通过将样本及参数$w$均匀地分配到不同的处理器上，分别计算梯度$g$后再累加起来便可得到最终梯度。  
## 处理器间通信
1. 共享内存的方法比较容易，但无法大规模并行，因为需要在同一主机上运行，无法扩展到大规模集群。
2. 消息传递方式，多个节点，每个节点都有若干处理器，各个节点之间可通过TCP/IP等协议进行通信以传递必要的消息。这种方法有两种架构方式：  
   - Client-Server架构，选定一个节点作为server节点，其余节点为worker节点，每个worker节点负责实际的运算，server节点负责协调任务、汇总各节点返回结果
   - Peer-to-Peer架构，所有的节点具有相同的地位，每个节点可以和其邻居节点通信  

## 实现方法（编程模型）
### MapReduce（同步算法）
&emsp;&emsp;MapReduce是Client-Server架构，使用消息传递方式通信，以同步（即每一轮需worker全部完成任务后再进行下一轮运算）的方式并行。数据存在于worker上由各worker进行计算（Map），由server负责调度并汇总所有worker的计算结果（Reduce）。  
&emsp;&emsp;线性回归并行中，server在对worker广播模型参数$w$，各worker利用本地样本（数据并行）以及得到的模型参数计算本地梯度并求和得到本地梯度向量。然后，server与worker通信收集各worker的梯度向量并相加获得最终梯度，server获得最终梯度后对模型参数进行梯度下降，然后再将新的模型参数分发到各worker上，进行下一轮迭代。  
&emsp;&emsp;使用更多的worker与减少的训练时间不是线性关系的，这是由节点之间通信开销以及同步开销造成的。实际使用中需要在增加核数带来的加速与额外开销之间进行权衡。  
#### 通信开销
&emsp;&emsp;通信开销需要考虑以下几方面：  
- **通信复杂度**随着模型参数的增加、worker节点数的增加而增加
- **网络延迟**节点之间收发数据包在网络中传递所产生的延迟

&emsp;&emsp;常用的估计通信时间公式：$\frac{\mathrm{complexity}}{\mathrm{bandwidth}}+\mathrm{latency}$  
#### 同步开销
&emsp;&emsp;由于是同步并行的，所以每轮迭代计算时间是由最慢的worker决定的。且如果一个节点故障重启，则会造成整体耗时的增加，而且节点越多，故障的概率也就越大，同步时间开销也就越大。  
### Client-Server（异步算法）
&emsp;&emsp;同步算法理论上更快，但由于同步开销等原因，使得异步算法实际上更快。

### Decentralized Network
