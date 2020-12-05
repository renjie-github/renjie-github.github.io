---
layout: post
title: "少样本学习"
subtitle: "Few-shot Learning"
author: "Roger"
header-img: "img/distributed_ML.png"
header-mask: 0.4
mathjax: true
tags:
  - Few-Shot Learning
---

## Few-shot Learning简介
&emsp;&emsp;Few-shot Learning属于Meta Learning，Meta learning的目的是让模型学会学习。Few-shot Learning用于从很少的样本（**不要求在训练数据集中出现过**）中识别事物之间的异同。比如给定两张图片，模型可以区分二者是否是同一事物；或者给定一张图片（Query）和一组待标签的图片（Support Set，对应多个事物类别，每个类别下有一张或多张图片，用于在预测时提供额外信息），模型可以得出Query属于Support Set中的哪一类别。  
&emsp;&emsp;Few-shot Learning中Support set的类别数为$k$，有$k$个类别的Support set叫做$k$-way Support set。每个类别中样本个数记作$n$，每个类别有$n$个样本的Support set则称作$n$-shot。Support set中的类别数越多，每个类别的样本数越少，任务越难。  
## Few-shot Learning方法介绍
&emsp;&emsp;该方法的思想是**学习数据之间的相似性**。首先在一个很大的数据集上学习一个相似性函数，训练完成后，在预测时通过比较Query和Support set中每个样本之间的相似性来找到具有最高相似性分数的样本。
### Siamese Network
&emsp;&emsp;有两种训练Siamese Network的方法：
1. Pairwise Similarity Score  
  &emsp;&emsp;一种是每次取两个样本，计算二者之间的相似度。样本从数据集中构建，每次随机从同一类别中选择两个样本，标签为1；每次从不同的两个类别中随机各取一个样本，标签为0。以图片为例，训练时，两张图片经过**同一个**CNN提取特征$h_1$，$h_2$，在计算二者的绝对误差$z=\|h_1-h_2\|$，再将$z$经过Dense层，以Sigmoid激活函数输出，用标签计算损失函数。  

2. Triplet Loss
  &emsp;&emsp;每次从训练集中选择三个样本，其中两个属于同一类（anchor样本 + 正样本），另一个样本与前两个样本属于不同类（负样本）。以图片为例，训练时，三张图片经过**同一个**CNN提取特征，然后分别计算anchor样本与正样本的特征向量之间距离的2范数的平方$d^+$，以及anchor样本与负样本的特征向量之间距离的2范数的平方$d^-$。如果训练有效，应使得$d^+$尽量小，$d^-$尽量大。定义一个超参数margin：$\alpha$。如果$d^-\ge d^++\alpha$，则认为样本分类正确，loss为0，否则loss为$d^++\alpha-d^-$。综上，定义损失函数（triplet loss）为：$max\lbrace0,d^++\alpha-d^-\rbrace$，利用梯度下降来更新网络参数。

3. Pretraining & Fine Tuning  
  &emsp;&emsp;以Siamese Network或其他监督学习的方法预训练一个神经网络，这样，给定一个样本便能得到其对应的embedding特征向量。使用时，对于support set中每一类中的样本，分别计算得到特征向量，然后对该类中样本的特征向量取平均得到该类的特征向量均值。最后，用query的特征向量分别与各类的特征向量均值进行对比（将各类的特征向量均值拼成一个矩阵，再与query的特征向量做矩阵乘），对比方式可以是余弦相似度等指标，经过softmax输出概率。  
  &emsp;&emsp;在预训练的基础上，使用support set中的样本，通过最小化预测结果与label之间的交叉熵来进行fine-tuning，以调节参数矩阵$W$和偏置$b$。这里参数矩阵$W$初始化为预训练得到的各类特征向量均值拼接成的矩阵，偏置$b$初始化为全0向量。  
  &emsp;&emsp;因为support set中的样本量很少，所以需要加正则化来防止过拟合。这里用熵正则（即所有query样本输出概率entropy的均值）比较好，entropy loss部分越小，说明模型对于自己的判断越“确信”。  
  &emsp;&emsp;模型的最终输出为一个Softmax分类器：  
  $$
  p=\text{Softmax} \left(Wq+b\right)=\text{Softmax} \left(\left[\begin{array}{c}
  w_{1}^{T}q+b_1\\
  w_{2}^{T}q+b_2\\
  w_{3}^{T}q+b_3\\
  \end{array} \right]\right) \tag{1}
  $$  
  &emsp;&emsp;将上式中点积替换为余弦相似度（等效于对内积的结果做归一化）有助于提高分类效果，则上式变为：  
  $$
  p=\text{Softmax}\left(\left[
    \begin{array}{c}
    \text{sim}(w_1,q)+b_1\\
    \text{sim}(w_2,q)+b_2\\
    \text{sim}(w_3,q)+b_3\\
    \end{array}
    \right]\right) \tag{2}
  $$  
  &emsp;&emsp;其中$\text{sim}$代表余弦相似度：  
  $$
  \text{sim}(w,q)=\frac{w^Tq}{\lVert w\rVert_2\cdot\lVert q\rVert_2} \tag{3}
  $$






