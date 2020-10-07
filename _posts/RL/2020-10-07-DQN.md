---
layout: post
title: "强化学习之——算法分类"
subtitle: "Algorithm Classification"
author: "Roger"
header-img: "img/RL.jpg"
header-mask: 0.4
mathjax: true
tags:
  - RL
  - DL
---

## 强化学习算法分类
### **Model-free vs Model-based**  
&emsp;&emsp;在Model-free与Model-based中，**“model”所指的都是环境的模型**。对应的Model-based是指对环境建立一个模型，该模型可以根据给定的observation，给出相应的immediate reward以及next observation。我们所遇到的大多数算法都属于model-free，此时我们的agent执行动作并用获得observation、reward训练，以及最终策略的运行都是在一个环境中进行的。
### **Value-based vs Policy-based**  
&emsp;&emsp;Value-based方法是基于Markov过程，对每个状态定义一个V（或每个（状态，动作）定义一个Q），通过动态规划的方法来逐渐逼近真实value。其对应的策略则是eager策略，即总是选择能获得最大收益（即V或Q最大）的动作。代表算法为DQN系列。  
&emsp;&emsp;从上面描述看出**value-based方法是通过计算所有可能动作的value来间接地优化策略，policy-based采用的是直接近似policy的方法**，策略通常被表示为对所有可能动作的概率分布。**Policy-based方法通常样本效率低，这意味着它们要求与环境有更多的交互。**Value-based方法可以从大的replay buffer中受益。Policy-based方法虽然样本效率低，但该方法优化目标更直接，且每轮训练期间只需处理一批样本的状态以得到关于动作的概率分布，其总的计算效率比value-based方法更好。  
### **On-policy vs Off-policy**  
&emsp;&emsp;**Off-policy方法不依赖于“新鲜”数据，这意味着可以从很旧的数据中学到策略**，因此可以用很大的replay buffer，样本效率高。这类算法的代表有DQN系列算法、DDPG等。**On-policy方法很大程度上依赖于根据我们正在更新的当前策略采样的训练数据，这是因为on-policy方法试图直接优化当前策略**。二者的选择需要视具体应用场景来定，如果样本获取成本低，使用on-policy方法更好，否则应考虑off-policy方法。