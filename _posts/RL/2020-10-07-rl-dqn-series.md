---
layout: post
title: "强化学习之——DQN系列算法"
subtitle: "DQN Series"
author: "Roger"
header-img: "img/RL.jpg"
header-mask: 0.4
mathjax: true
tags:
  - RL
  - DL
---

### **DQN** 
&emsp;&emsp;上文中提高的Q-learning方法可以通过动态规划的方法，对Q值表进行迭代更新来实现。但是该方法有个问题：当状态空间数很大时，需要维护一个很大的Q值表，对于连续状态空间甚至无法给出这样的Q值表。为解决这类问题，可以采用神经网络来学习关于Q值的一个“函数”，以此来代替Q值表的存储。DQN便是这一方法的典型代表。  
&emsp;&emsp;该算法在2013年被DeepMind团队提出，15年发表于Nature的文章[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)加入了target network以提高算法稳定性。  
&emsp;&emsp;该算法