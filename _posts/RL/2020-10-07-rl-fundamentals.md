---
layout: post
title: "强化学习之——基础介绍"
subtitle: "fundamentals"
author: "Roger"
header-img: "img/RL.jpg"
header-mask: 0.4
mathjax: true
tags:
  - RL
  - DL
---

### **基础介绍**    
- Markov决策过程  
&emsp;&emsp;维基百科中[Markov决策过程](https://en.wikipedia.org/wiki/Markov_decision_process)定义为一个离散时间随机控制过程，它提供了一种数学框架，用于在回报部分是随机产生，部分是由决策者控制的情况下，对决策过程进行建模。Markov决策过程是Markov链的扩展，允许进行动作选择并获得相应的奖励。  
&emsp;&emsp;一个Markov决策过程由一个4元元组$(S, A, P_a, R_a)$构成：  
  - $S$是状态空间集合
  - $A$是动作空间集合
  - $P_a(s, s')=P_r(s_{t+1}=s'\|s_t=s, a_t=a)$为在$t$时刻，状态$s$下采取动作$a$后到达状态$s'$的概率
  - $R_a$是从状态$s$下采取动作$a$到达状态$s'$后获得的即时奖励（或期望的即时奖励）

  &emsp;&emsp;Markov决策过程的目标是找到一个“好的策略”：即函数$\pi$，该函数执行在状态$s$下采取的动作$\pi(s)$。Markov决策过程与策略结合后，对于每个状态下的动作也就确定了，这使得它表现地类似Markov链的形式，即$P_a(s, s')=P_r(s_{t+1}=s'\|s_t=s, a_t=a)$被简化为了$P_a(s,s')=P_r(s_{t+1}=s'\|s_t=s)$。  
&emsp;&emsp;Markov决策过程的优化目标是最大化某种随机奖励的累积函数，对于未来无限长的过程，目标函数是期望的折现奖励（expected discount reward）:  
$R=E[\sum_{t=0}^{\infty}{\gamma^tR_{a_t}(s_t, s_{t+1})}],\ where\ a_t=\pi(s_t)\ and \ 0\leq\gamma\leq1\tag{1.1}$  
&emsp;&emsp;多数情况下无法显式地获得转移概率$P_a(s, s')$，此时便需要一个仿真环境，通过采样转移分布来隐式地对MDP建模。隐式MDP建模的一种常见形式是一个**情景式（episodic）仿真器**，其可以从某个初始状态出发，没个时间步接收一个动作输入后产生后续状态及奖励。以该形式产生的states, actions, rewards轨迹（trajectories）叫做episodes。  
&emsp;&emsp;另一种仿真器形式是**生成式（generative）仿真器**，给定任意状态及动作，单步仿真器可以产生下一状态及奖励，即$s',r\leftarrow G(s,a)$。与情景式仿真器相比，生成式仿真器的优点是它可以从任意状态产生数据，而不是仅从轨迹中遇到的状态产生。  
&emsp;&emsp;显式模型通过从分布中采样来简单地产生生成式模型，而重复应用生成模型则可以生成情景式模拟器。相反，只能通过回归的方式来学习近似模型。**对于特定的MDP，可获得的模型类型在确定合适的解决方案算法中起着重要作用**。例如，动态规划算法要求有显式的模型（知道转移概率），而蒙特卡洛树搜索（如Alpha zero）则要求一个生成式模型（或囊括了所有状态的情景式模型），大多数强化学习算法只要求情景式模型。  
- 部分可观测MDP  
&emsp;&emsp;Markov决策过程假设采取动作时状态$s$是已知的，否则无法计算$\pi(s)$。当该假设不成立时，对应的问题叫做部分可观测的Markov决策过程（partially observable Markov decision process, POMDP）。  
- Q-learning  
&emsp;&emsp;转移概率或奖励未知时的问题叫做强化学习问题。定义一个值函数有助于解决对于这类问题，该函数对应不断按照最优策略（或根据当前有的策略）采取一系列动作：  
$Q(s,a)=\sum_{s'}{P_a(s,s')(R_a(s,s')+\gamma V(s'))}\tag{1.2}$  
&emsp;&emsp;上式中转移概率$P_a(s, s')$是未知的，学习期间的经验是基于$(s,a,s')$对，因此，可以构造一个Q值表并使用经验来对其进行更新，这便是Q-learning。  
强化学习可以在不显式给定转移概率的情况下解决MDP问题。值与策略迭代需要用到转移概率的值，在强化学习中，转移概率是通过仿真器来获取，该仿真器通常从一个均匀随机初始状态开始运行多次。强化学习可以与函数近似方法（如神经网络）结合来处理状态树很大的问题。