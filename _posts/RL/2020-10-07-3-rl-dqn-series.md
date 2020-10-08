---
layout: post
title: "强化学习之——DQN"
subtitle: "DQN"
author: "Roger"
header-img: "img/RL.jpg"
header-mask: 0.4
mathjax: true
tags:
  - RL
  - DL
---

### **DQN算法** 
#### **Q-learning**
&emsp;&emsp;Q-learning方法可以通过动态规划的方法，对Q值表进行迭代更新来实现。其具体步骤如下：
1. 初始化一个状态-动作值映射的空表（全0）；  
2. 通过与环境交互获得$(s,a,r,s')$，在这一步需要决定要采取何种动作，如何选择动作没有唯一的正确方法（exploration vs expolitation）  
3. 使用Bellman方程近似更新$Q(s,a)$：  
   $Q(s,a)\leftarrow r+\gamma max_{a'\in A}Q(s',a')\tag{1}$  
4. 检查收敛条件，若不满足，从步骤2.继续这一过程。  

&emsp;&emsp;由于从环境中采样，在现有的value上直接赋以新的值通常会使得训练变得不稳定。实际中是使用混合（“blending”）技巧来近似更新$Q(s,a)$，具体是：使用值在0~1之间的学习率$\alpha$来平衡新、旧Q值，以使得Q值更加平滑地收敛（即使环境有噪声）：  
$Q(s,a)\leftarrow (1-\alpha)Q(s,a)+\alpha(r+\gamma max_{a'\in A}Q(s',a'))\tag{2}$  
&emsp;&emsp;上式中的Bellman更新可以重写为：  
$Q(s,a)\leftarrow Q(s,a)+\alpha\delta_k(s,r,s')\tag{3}$  
&emsp;&emsp;其中的$\delta_k(s,r,s')=r+\gamma max{a'\in A}Q(s',a')-Q(s,a)$就是所谓的TD error。  
#### **Deep Q Network**
&emsp;&emsp;但是值迭代法有个问题：当状态空间数很大时，需要维护一个很大的Q值表，对于连续状态空间甚至无法给出这样的Q值表。为解决这类问题，可以采用神经网络来学习关于Q值的一个“函数”，以此来代替Q值表的存储。DQN便是这一方法的典型代表。  
&emsp;&emsp;该算法在2013年被DeepMind团队提出，15年发表于Nature的文章[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)加入了target network以提高算法稳定性。  
&emsp;&emsp;该算法使用神经网络来近似Q指标，这样做可以将存储Q表替换为存储能将对应的状态映射为Q值的函数（网络参数）。此外，这样做还使得网络可以处理连续的状态空间而无需对其进行离散化。  
&emsp;&emsp;DQN算法将值迭代过程转化为类似于监督学习的方式，通过梯度下降的方法进行训练学习。在训练的每次迭代过程中调节网络参数以减少Bellman方程中的均方误差。其中目标值$r+\gamma max_{a'}Q^\*(s',a')$被近似替换为$r+\gamma max_{a'}Q(s',a';\theta_i^-)$，$\theta_i^-$是前轮迭代的参数。最终损失函数为：  
$L_i = \mathbb E[((r+\gamma max_{a'}Q(s',a';\theta_i^-))-Q(s,a;\theta_i))^2]\tag{4}$  
&emsp;&emsp;DQN是**model-free**的：因为它通过直接使用从仿真器得到的样本来解决强化学习任务，而不需要显式地估计奖励与转移的动态特性$P(r,s'\|s,a)$。DQN也是**off-policy**的：它在训练过程中学习的是贪心策略$a=argmax_{a'}Q(s,a';\theta)$，而其动作分布却保证了对状态空间一定程度的探索（**$\varepsilon-greedy$**），即以概率$\varepsilon$随机选择动作，以概率$1-\varepsilon$遵循贪心策略。  
&emsp;&emsp;由于DQN是off-policy的，所以可以从过去的旧的经验中学习，即可以采用**经验回放（experience replay）**来存储过往经验，然后从经验回放池中每次均匀随机采样一个minibatch数据来对网络进行训练。这样做的好处：每个样本都可以被多次使用，从而提高了样本效率；从连续的样本中学习是低效的，因为连续的样本之间有较强的相关性，这与梯度下降中要求的数据独立同分布条件矛盾，从经验池中随机采样可以减少这一相关性的影响。此外，从由相关性的样本中学习也会给策略带来bias，使得参数陷入局部最小值，导致错误的策略甚至算法不收敛。  
&emsp;&emsp;为了提高使用神经网络后的稳定性，DQN使用了一个独立的目标网络来产生梯度下降过程中的目标值。训练过程中，每隔C次更新将网络$Q$的参数复制给目标网络$\hat{Q}$，并使用目标网络来产生用于接下来C次更新所需的目标值。对于online Q-learning算法，一次增加$Q(s_t,a_t)$的更新也会增加所有动作$a$的$Q(s_{t+1},a)$，从而也增加了目标值，导致策略的震荡甚至发散。使用旧参数充当目标网络在更新Q与该次更新影响目标值的时间之间增加了时延，从而减少了震荡或发散的可能性。  
&emsp;&emsp;作者还发现将误差项$r+\gamma max_{a'}Q(s',a';\theta_i^-)-Q(s,a;\theta_i)$裁剪到(-1, 1)的范围内可提高算法稳定性。因为绝对值损失函数$\|x\|$对于有负值的$x$导数为-1，正值$x$导数为1。对误差项进行裁剪等效于：对超过(-1, 1)范围的误差使用绝对值损失函数。**[P.S.]这里类似于深度学习损失函数中huber loss的作用，减少了异常值的影响，避免了过大的参数更新**。  
&emsp;&emsp;完整的DQN算法如下：  
<img src="/img/RL/DQN_algo.jpg" width=600 height=700 div align=center />  

