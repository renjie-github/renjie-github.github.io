---
layout: post
title: "强化学习之——策略梯度"
subtitle: "Policy Gradient"
author: "Roger"
header-img: "img/RL.jpg"
header-mask: 0.4
mathjax: true
tags:
  - RL
  - DL
---

## 1. 策略网络
&emsp;&emsp;相比于DQN等基于值函数估计以及贪心策略的算法，直接对策略进行优化是一种更为直接的优化方式。策略函数$\pi(a|s)$是在状态$s$下，关于动作的**概率密度函数**（Probability Density Function）。Agent从该概率分布中随机抽样要执行的动作$a$。该策略函数可以是一个表的形式，但如果要解决问题有很大的状态空间，那么存储该表便是一个很大的问题，因此可以使用函数（线性函数/核函数/神经网络等）来近似这一策略函数。使用神经网络近似的策略函数叫做策略网络（Policy Network）：$\pi(a|s;\theta)$，其中$\theta$为可学习的网络参数。  

## 2. 状态价值函数近似
- **折现回报**  
&emsp;&emsp;折现回报$U_T$（Discounted return）定义为**从T时刻起，未来奖励的加权和**：  
$$U_T=\sum_{t=0}^{\infty}\gamma^t R_{T+t}\tag{1}$$  
&emsp;&emsp;上式中奖励$R$、动作$A$以及状态$S$均具有随机性，奖励$R$的随机性是由前一时刻状态$s$与动作$a$造成的，动作$A$的随机性是有策略函数$\pi$，状态$S$的随机性来自状态转移函数$P$。由于$U_T$为未来所有奖励的加权和，所以$U_T$的随机性来自于未来所有动作$A$及状态$S$。  
- **动作价值函数**  
&emsp;&emsp;动作价值函数$Q_\pi(s_t|a_t)$是**折现回报$U_T$的条件期望**，用于评价在状态$s_t$下选择动作$a_t$的好坏程度。通过求期望，动作价值函数公式中将不包含$t$时刻后的动作及状态，仅依赖于当前时刻的动作$a_t$、状态$s_t$以及策略函数$\pi$：  
$$Q_\pi(s_t,a_t)=\mathbb{E}[U_t|S_t=s_t,A_t=a_t]\tag{2}$$  
- **状态价值函数**  
&emsp;&emsp;状态价值函数$V_\pi(s_t)$是**动作价值函数的期望**，通过求期望来消除动作随机变量$A\sim\pi(\cdot|s_t)$，最终状态价值函数$V_\pi(s_t)$仅与策略函数$\pi$以及状态$s_t$有关：  
$$V_\pi(s_t)=\mathbb{E}_{A}[Q_\pi(s_t,A)]\tag{3}$$  
&emsp;&emsp;给定策略函数$\pi$，$V_\pi(s_t)$代表当前状态的好坏；给定状态$s_t$，$V_\pi(s_t)$可衡量策略$\pi$的好坏（值越大，代表越好）。如果动作是离散的，那么可以重写公式消除随机变量$A$（对于连续动作则用积分）：  
$$V_\pi(s_t)=\mathbb{E}_{A}[Q_\pi(s_t,A)]=\sum_{a}\pi(a|s_t)\cdot Q_\pi(s_t,a)\tag{4}\label{eq4}$$  
&emsp;&emsp;用神经网络近似策略函数后，状态价值函数中$\pi(a|s_t)$变为$\pi(a|s_t;\theta)$：  
$$V(s_t;\theta)=\sum_{a}\pi(a|s_t;\theta)\cdot Q_\pi(s_t,a)\tag{5}\label{eq5}$$  

## 3. 策略梯度的近似计算  
&emsp;&emsp;有了状态价值函数$\eqref{eq5}$，我们可以通过调节参数$\theta$来使状态价值函数值变大，即最大化目标函数：$$J(\theta)=\mathbb{E}_S[V(S;\theta)]$$。具体过程是**观察到一个状态$s$后，使用策略梯度$\frac{\partial V(s;\theta)}{\partial\theta}$来更新参数$\theta$**：$\theta\leftarrow\theta+\beta\cdot\frac{\partial V(s;\theta)}{\partial\theta}$。这里类似于随机（状态$s$随机）梯度上升，真正的梯度是$J(\theta)$关于$\theta$的导数。  
&emsp;&emsp;策略梯度公式如下，需要注意的是$Q_\pi(s,a)$也与$\pi$有关，不过求导时是否考虑该项不影响最终结果的形式：  
$$
\begin{align}
\frac{\partial V(s;\theta)}{\partial\theta}&=\sum_{a}\frac{\partial\pi(a|s;\theta)}{\partial\theta}\cdot Q_\pi(s,a)\tag {6} \label {eq6}\\
&=\sum_{a}\pi(a|s;\theta)\cdot\frac{\partial \rm{ln}\pi(a|s;\theta)}{\partial\theta}\cdot Q_\pi(s,a)\\
&=\mathbb{E}_{A\sim\pi(\cdot|s;\theta)}[\frac{\partial \rm{ln}\pi(a|s;\theta)}{\partial\theta}\cdot Q_\pi(s,a)] \tag {7} \label {eq7}\\
&=\mathbb{E}_{A\sim\pi(\cdot|s;\theta)}[g(A)]
\end{align}
$$  
&emsp;&emsp;对于离散的动作空间，可以用公式$\eqref{eq6}$形式（$\eqref{eq7}$形式也适用）。若动作空间非常大或是连续的，则可以使用公式$\eqref{eq7}$形式。由于$A$是连续的，需要在所有的可能的策略函数空间上求定积分，而该策略函数空间又是非常复杂，无法计算的，解决方法是对该期望用蒙特卡洛近似：  
1. 从概率密度函数$\pi(\cdot\|s;\theta)$中随机抽样一个动作$\hat{a}$
2. 计算$g(\hat{a},\theta)=\frac{\partial \rm{ln}\pi(\hat{a}\|s;\theta)}{\partial\theta}\cdot Q_\pi(s,\hat{a})$
3. 显然有$\mathbb{E}_A[g(A,\theta)]=\frac{\partial V(s;\theta)}{\partial\theta}$。由于$g(\hat{a},\theta)$是由概率密度函数$\pi(\cdot\|s;\theta)$采样得到，所以$g(\hat{a},\theta)$为$\frac{\partial V(s;\theta)}{\partial\theta}$的无偏估计，可以用它来近似策略梯度（蒙特卡洛近似）。  

## 4. 动作价值函数的估计
&emsp;&emsp;策略梯度算法需要估计动作价值函数$Q_\pi(s_t,a_t)$，有两种估计方法：  
1. REINFORCE方法（需要完成一整个的episode后才可以更新）
  - 用策略网络$\pi$来控制agent执行动作，记录下完整的轨迹：$(s_1,a_1,r_1),(s_2,a_2,r_2,s_3),\cdots,(s_T,a_T,r_T)$
  - 计算该轨迹的折后回报（discounted return）$u_t=\sum_{k=t}^{T}\gamma^{k-t}r_k$
  - 用$u_t$近似$q_t$（蒙特卡洛近似，因为根据定义，$Q_\pi(s_t,a_t)=\mathbb{E}[U_t]$）  
2. 函数近似法
  - 用神经网络来近似$Q_\pi$，即Actor-Critic算法

## 5. 策略梯度算法
&emsp;&emsp;算法流程如下：  
1. 观察环境得到状态$s_t$
2. 根据策略函数$\pi(\cdot\|s_t;\theta_t)$随机采样动作$a_t$
3. 估计动作价值函数的值$q_t\approx Q_\pi(s_t,a_t)$（用REINFORCE或Critic）
4. 对策略网络求关于参数$\theta$的导数：$d_{\theta,t}=\frac{\partial \rm{ln}\pi(a_t\|s_t,\theta)}{\partial\theta}\|_{\theta=\theta_t}$
5. 计算（近似的）策略梯度：$g(a_t,\theta_t)=q_t\cdot d_{\theta,t}$（蒙特卡洛近似）。【注】由于$a_t$是随机抽样得到的，所以$g(a_t)$是随机梯度。
6. 更新策略网络参数：$\theta_{t+1}=\theta_t+\beta\cdot g(a_t,\theta_t)$  

## 6. 添加Baseline
&emsp;&emsp;向策略梯度中添加baseline可以起到降低方差，加快收敛的作用。baseline可以是**任意与动作$A$无关的值**，但选取合适的baseline可以获得更好的收敛。添加baseline不影响策略梯度的值，证明如下：  
$$
\begin{align}
\mathbb{E}_{A\sim\pi}[b\cdot\frac{\partial \rm{ln}\pi(a_t\|s_t,\theta)}{\partial\theta}]&=b\cdot\mathbb{E}_{A\sim\pi}[\frac{\partial \rm{ln}\pi(a_t\|s_t,\theta)}{\partial\theta}]\\
&=b\cdot\sum_{a}\pi(a\|s;\theta)\cdot[\frac{1}{\pi(a|s;\theta)}\cdot\frac{\partial\pi(a_t\|s_t,\theta)}{\partial\theta}]\\
&=b\cdot\sum_{a}\frac{\partial\pi(a_t\|s_t,\theta)}{\partial\theta}\\
&=b\cdot\frac{\partial\sum_{a}\pi(a_t\|s_t,\theta)}{\partial\theta}\\
&=b\cdot\frac{\partial 1}{\partial\theta}\\
&=b\cdot0\\
&=0 \tag {8} \label {eq8}
\end{align}
$$  
&emsp;&emsp;根据上述证明，向策略梯度公式中添加baseline不会影响结果，即下式与$\eqref{eq7}$等价：  
$$\frac{\partial V(s;\theta)}{\partial\theta}=\mathbb{E}_{A\sim\pi(\cdot|s;\theta)}[\frac{\partial \rm{ln}\pi(a|s;\theta)}{\partial\theta}\cdot(Q_\pi(s,a)-b)] \tag {9} \label {eq9}$$  
&emsp;&emsp;根据$\eqref{eq8}$的证明，baseline的添加不影响策略梯度，但会影响蒙特卡洛近似$g(a_t)$的效果，从而起到降低（近似）方差、加速收敛的作用。策略梯度的蒙特卡洛近似$g(a_t)$变为了：  
$$g(a_t)=\frac{\partial \rm{ln}\pi(a|s;\theta)}{\partial\theta}\cdot(Q_\pi(s,a)-b) \tag{10} \label{10}$$  
### 常见的baseline
- baseline取0（无baseline），即标准的策略梯度算法
- baseline取$b=V_\pi(s_t)$，因为状态$s_t$是在动作$A_t$之前观察到的，所以不依赖于$A_t$（满足条件）。当baseline选取接近$Q_\pi(s_t,A_t)$时，对期望做近似的方差会减小，从而加速算法收敛。而$V_\pi(s_t)$就很接近该值，见$\eqref{eq4}$：$$V_\pi(s_t)=\mathbb{E}_{A_t}[Q_{\pi}(s_t,A)]$$。

## 总结
1. $$U_t=R_t+\gamma\cdot R_{t+1}+\gamma^2\cdot R_{t+2}+\gamma^3\cdot R_{t+3}+\cdots$$  
2. $$Q_\pi(s_t,a_t)=\mathbb{E}[U_t\|s_t,a_t]$$  
3. $$V_\pi(s_t)=\mathbb{E}_A[Q_\pi(s_t,A)|s_t]$$  
4. 用随机梯度$g(a_t)$近似策略梯度（**蒙特卡洛近似**，使用一个样本来近似期望）
5. 用$u_t$近似$Q_\pi(s_t,a_t)$（**蒙特卡洛近似**，使用一个轨迹来近似动作价值函数）:
  - 观察轨迹：$(s_1,a_1,r_1),(s_2,a_2,r_2,s_3),\cdots,(s_T,a_T,r_T)$
  - 对轨迹计算折现回报：$u_t=\sum_{k=t}^{T}\gamma^{k-t}r_k$
  - $u_t$为$Q_\pi(s_t,a_t)$的无偏估计
6. 用值函数网络$v(s;w)$**近似**$V(s;\theta)$
7. $$\frac{\partial V(s_t)}{\partial\theta}=\frac{\partial \rm{ln}\pi(a_t|s_t;\theta)}{\partial\theta}\cdot(u_t-v(s_t,w))$$
8. 用策略梯度更新策略网络：$\theta\leftarrow\theta-\beta\cdot\delta_t\cdot\frac{\partial \rm{ln}\pi(a_t\|s_t;\theta)}{\partial\theta}$
9. 记$-\delta_t=(u_t-v(s_t,w))$，用$\frac{1}{2}\delta_t^2$作为损失函数，用梯度下降$w\leftarrow w-\alpha\cdot\delta_t\cdot\frac{\partial v(s_t,w)}{\partial w}$更新值函数网络