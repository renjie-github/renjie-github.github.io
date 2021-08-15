---
layout: post
title: "置信域策略优化算法"
subtitle: "TRPO & PPO"
author: "Roger"
header-img: "img/RL/TRPO.jpg"
header-mask: 0.4
mathjax: true
tags:
  - RL
---

# 置信域策略优化算法
## 算法概述
置信域策略优化算法（[TRPO(2015)](https://arxiv.org/abs/1502.05477)，Trust Region Policy Optimization）是强化学习中的一种策略梯度（Policy Gradient）算法，其通过限制KL散度（或策略改变范围）来避免每次迭代中，策略参数过大的变化。PPO算法是在TRPO基础上的改进，在实际使用中实现更简单，计算量更小。这类算法在每一次迭代时都需要进行参数更新，因此计算量较大，但从实际应用中的表现来看，这类算法的表现更加稳定，更容易收敛。与策略梯度算法相比，置信域优化算法更加稳定，对超参更加鲁棒，且sample efficiency更高。  
## 什么是策略梯度
策略梯度（Policy Gradient）之前已经讲过，这里再提一下。策略梯度法是强化学习中的一种优化策略：该方法使用随机梯度下降（实际上是做随机梯度上升），根据期望回报（长期累积奖励）来优化参数化的策略，与之相对应的是值优化方法（Value-based）。  
策略网络表示为$\pi(a|s;\theta)$，状态价值函数为动作价值函数关于动作$A\sim\pi$的期望：    
$$V_\pi(s)=\mathbb{E}_{A\sim\pi}\left[Q_\pi(s, A)\right] = \sum_a\pi(a|s;\theta)\cdot Q_\pi(s, a) \tag{1} \label{eq1}$$  
目标函数:  
$$J(\theta)=\mathbb{E}_S[V_\pi(S)] \tag{2}$$
## 什么是置信域  
在数值优化邻域，置信域（Trust Region）是使用模型函数（通常为二次函数）近似的目标函数域的一个子集。如果在置信域内找到了目标函数的适合模型，则扩展该域；反之，则收缩该域。  
评估域“拟合”效果的评估标准：比较通过模型近似得到的期望改善与在目标函数中观察到的实际改善的**比值**。通过给该比值简单设置一个阈值来作为扩展或收缩域的标准：只有当一个域提供了合理的近似时，该模型函数才会被信任（trusted）。  
置信域与线搜索（line-search）方法相对，置信域方法首先选择一个步长（step size，即置信域的规模）以及一个方向（step direction），而线搜索方法首先选择一个方向，然后再选择一个步长。
## 置信域优化算法
对于策略梯度算法，模型的策略参数需要根据目标函数$J$来进行更新。令当前的策略参数为$\theta_{old}$，定义当前策略的邻域为：  
$$
\mathcal{N}(\theta_{old})=\left\{\theta|\left\|\theta-\theta_{old}\right\|_2\le\Delta\right\} \tag{3}
$$  
可将上述邻域想象为以参数$\theta_{old}$为中心，半径为$\Delta$的超球面。如果**在上述邻域中**存在函数$L\left(\theta|\theta_{old}\right)$可以近似目标函数$J(\theta)$，则称该邻域为“置信域”。在置信域中，若$L$可以近似$J$，则若优化参数$\theta$能优化$L$，则同时也可以优化$J$，这便是置信域优化算法的基本思想。  
置信域优化算法就是不断重复下述两步：
- 近似阶段
  给定当前参数$\theta_{old}$，在其邻域（置信域）构建目标函数$J$的近似函数$L(\theta|\theta_{old})$
- 最大化阶段
  在**置信域**中，找到能够最大化模型函数$L$的参数$\theta_{new}$:  
  $$
  \theta_{new}\leftarrow\rm{argmax}_{\theta\in\mathcal{N}(\theta_{old})}L(\theta|\theta_{old}) \tag{4}
  $$
# TRPO算法
## 算法推导
对状态价值函数公式$\eqref{eq1}$进行改写：  
$$
\begin{align}
V_\pi(s) &= \sum_a\pi(a|S;\theta)\cdot Q_\pi(S, a) \\
&= \sum_a\pi(a|S;\theta_{old})\cdot \frac{\pi(a|S;\theta)}{\pi(a|S;\theta_{old})} \cdot Q_\pi(S, a) \\
&= \mathbb{E}_{A\sim\pi(\cdot|S;\theta_{old})}[\frac{\pi(A|S;\theta)}{\pi(A|S;\theta_{old})} \cdot Q_\pi(S, A)]
\end{align} \tag{5}
$$  
于是目标函数可以写为：  
$$
\begin{align}
J(\theta) &= \mathbb{E}[V_\pi(S)] \\
&= \mathbb{E}_S\left[\mathbb{E}_A[\frac{\pi(A|S;\theta)}{\pi(A|S;\theta_{old})} \cdot Q_\pi(S, A)]\right] \\
&= \mathbb{E}_{S, A}[\frac{\pi(A|S;\theta)}{\pi(A|S;\theta_{old})} \cdot Q_\pi(S, A)]
\end{align} \tag{6}
$$  
其中状态$S$是从环境中随机sample的一个状态，而动作$A$则是由策略网络在当前状态下随机抽样得到的输出，即$\pi(A|S;\theta_{old})$。实际就是从一个随机状态出发，策略网络持续与环境交互得到轨迹（trajectory）的过程，一条轨迹记为$s_1, a_1, r_1, s_2, a_2, r2, \cdots, s_n, a_n, r_n$。用实际观测到的(s, a, r)样本来估计实际的期望值的过程就是Monte Carlo近似。可以用一条轨迹中所有的平均来更加准确地估计期望：  
$$
L(\theta|\theta_{old}) = \frac{1}{n}\sum_{i=1}^{n}\frac{\pi(a_i|s_i;\theta)}{\pi(a_i|s_i;\theta_{old})} \cdot Q_\pi(a_i, s_i) \tag{7}
$$  
上式中，可以用每一步的discounted return来近似估计该步的$Q_\pi(a_i, s_i)$，即：  
$$
Q_\pi(s_i, a_i) \approx u_i=r_i + \gamma\cdot r_{i + 1} + \gamma^2\cdot r_{i + 1} + \cdots + \gamma^{n-i} \cdot r_n \tag{8}
$$  
则目标函数$L$可以进一步近似为：  
$$
\tilde{L}(\theta|\theta) = \frac{1}{n}\sum_{i=1}^{n}\frac{\pi(a_i|s_i;\theta)}{\pi(a_i|s_i;\theta_{old})} \cdot u_i \tag{9}
$$  
至此，完成了算法的第一步：近似。  
接下来推导第二步：最大化目标函数。这一步需要最大化在置信域中，使用**置信域优化算法**进行（对比策略梯度算法使用的是随机梯度上升），以保证策略不会由于一个太差的近似而发散。如何保证两步策略的变化足够小，优化在置信域中进行呢？  
有两种方法：一种是通过$\theta$和$\theta_{old}$之间的二范数小于某阈值；另一种是限制新的策略函数$\pi(\cdot|s_i;\theta)$和旧的策略函数$\pi(\cdot|s_i;\theta_{old})$之间的KL散度小于某阈值。  
以方法一为例（论文中用了方法二），最终的参数更新公式为：  
$$
\theta_{new} \leftarrow \rm{argmax}_\theta\,\tilde{L}(\theta|\theta_{old});\quad s.t. \left\|\theta-\theta_{old}\right\| \le \Delta \tag{10}
$$  
对于求解带限制条件的优化问题，可使用Gradient projection method。置信域优化算法  
## PPO
[PPO](https://arxiv.org/abs/1707.06347)算法是对TRPO优化效率上一个改进，其通过修改TRPO算法，使其可以使用SGD算法来做置信域更新，并且用clipping的方法方法来限制策略的过大更新，保证优化在置信域中进行。PPO算法在实际表现中大多情况下也优于TRPO算法。