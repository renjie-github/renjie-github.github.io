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
  
#### Markov决策过程  
&emsp;&emsp;维基百科中[Markov决策过程](https://en.wikipedia.org/wiki/Markov_decision_process)定义为一个离散时间随机控制过程，它提供了一种数学框架，用于在回报部分是随机产生，部分是由决策者控制的情况下，对决策过程进行建模。Markov决策过程是Markov链的扩展，允许进行动作选择并获得相应的奖励。  
&emsp;&emsp;一个Markov决策过程由一个4元元组$(S, A, P_a, R_a)$构成：  
  - $S$是状态空间集合
  - $A$是动作空间集合
  - $P_a(s, s')=P_r(s_{t+1}=s'\|s_t=s, a_t=a)$为在$t$时刻，状态$s$下采取动作$a$后到达状态$s'$的概率
  - $R_a$是从状态$s$下采取动作$a$到达状态$s'$后获得的即时奖励（或期望的即时奖励）

  &emsp;&emsp;Markov决策过程的目标是找到一个“好的策略”：即函数$\pi$，该函数执行在状态$s$下采取的动作$\pi(s)$。Markov决策过程与策略结合后，对于每个状态下的动作也就确定了，这使得它表现地类似Markov链的形式，即$P_a(s, s')=P_r(s_{t+1}=s'\|s_t=s, a_t=a)$被简化为了$P_a(s,s')=P_r(s_{t+1}=s'\|s_t=s)$。  
&emsp;&emsp;Markov决策过程的优化目标是最大化某种随机奖励的累积函数，对于未来无限长的过程，目标函数是期望的折现奖励（expected discount reward）:  
$R=E[\sum_{t=0}^{\infty}{\gamma^tR_{a_t}(s_t, s_{t+1})}],\ where\ a_t=\pi(s_t)\ and \ 0\leq\gamma\leq1\tag{1}$  
&emsp;&emsp;多数情况下无法显式地获得转移概率$P_a(s, s')$，此时便需要一个仿真环境，通过采样转移分布来隐式地对MDP建模。隐式MDP建模的一种常见形式是一个**情景式（episodic）仿真器**，其可以从某个初始状态出发，没个时间步接收一个动作输入后产生后续状态及奖励。以该形式产生的states, actions, rewards轨迹（trajectories）叫做episodes。  
&emsp;&emsp;另一种仿真器形式是**生成式（generative）仿真器**，给定任意状态及动作，单步仿真器可以产生下一状态及奖励，即$s',r\leftarrow G(s,a)$。与情景式仿真器相比，生成式仿真器的优点是它可以从任意状态产生数据，而不是仅从轨迹中遇到的状态产生。  
&emsp;&emsp;显式模型通过从分布中采样来简单地产生生成式模型，而重复应用生成模型则可以生成情景式模拟器。相反，只能通过回归的方式来学习近似模型。**对于特定的MDP，可获得的模型类型在确定合适的解决方案算法中起着重要作用**。例如，动态规划算法要求有显式的模型（知道转移概率），而蒙特卡洛树搜索（如Alpha zero）则要求一个生成式模型（或囊括了所有状态的情景式模型），大多数强化学习算法只要求情景式模型。  
#### 部分可观测MDP  
&emsp;&emsp;Markov决策过程假设采取动作时状态$s$是已知的，否则无法计算$\pi(s)$。当该假设不成立时，对应的问题叫做部分可观测的Markov决策过程（partially observable Markov decision process, POMDP）。  
#### Q-learning  
&emsp;&emsp;转移概率或奖励未知时的问题叫做强化学习问题。定义一个值函数有助于解决对于这类问题，（状态）值函数$V_{\pi}(s)$定义为从状态$s$开始，遵循策略$\pi$的期望回报值。因此，基本上值函数估计给定的状态有“多好”。  
&emsp;&emsp;**值函数方法试图通过为某个策略（通常是当前策略（on-policy）或者是最优策略（off-policy）），维护一套期望回报的估计值来寻找最大化回报的策略**。值函数方法依赖于MDP理论，其中最优性被定义为在某种意义上比前一个更强：如果某个策略能从**任意初始状态**中获得最好的期望回报，则称该策略是最优的。为了定义最优性，策略$\pi$的值被定义为：$V^\pi(s)=E[R\|s,\pi]$，**这里$R$代表从初始状态$s$开始遵循策略$\pi$得到的回报**。定义$V^\*(s)$为$V^\pi(s)$可能的最大值：$V_{\pi}^*(s)=max_{\pi}V^\pi(s)$，其中$\pi$是可变的。在任意状态都能够实现这一最优值的的策略即最优策略。尽管状态值已经足够定义最优性，但定义动作值更有用。给定一个状态$s$，动作$a$以及策略$\pi$，在策略$\pi$下$(s,a)$对的动作值定义为：$Q^{\pi}(s,a)=E[R\|s,a,\pi]$，**这里$R$定义为在状态$s$下首先采取动作$a$，并在后来过程中遵循策略$\pi$得到的随机回报**。根据MDP理论，如果$\pi^\*$是一个最优策略，我们在每个状态$s$下，通过从$Q^{\pi^\*}(s,\cdot)$中选择最优动作执行。对应该最优策略$Q^{\pi^\*}$的动作值函数叫做最优动作值函数，记作$Q^\*$。总的来说，只需知道最优动作值函数，我们便知道如何进行动作是最优的。   
&emsp;&emsp;动作值函数对应不断按照最优策略（或根据当前有的策略）采取一系列动作：  
$Q(s,a)=\sum_{s'}{P_a(s,s')(R_a(s,s')+\gamma V(s'))}\tag{2}$  
&emsp;&emsp;上式中转移概率$P_a(s, s')$是未知的，学习期间的经验是基于$(s,a,s')$对。**有两种方法来计算最优动作值函数：值迭代与策略迭代**，两中算法都是计算一系列函数$Q_k(k=0,1,2,...)$，最终收敛到$Q^\*$。计算这些函数涉及到在整个状态空间上计算期望，这对于大状态空间问题是不切实际的。在强化学习中，期望是通过在样本上做平均来近似的，并使用函数近似技巧（如神经网络）来应用大的状态-动作空间。      
&emsp;&emsp;强化学习可以在不显式给定转移概率的情况下解决MDP问题。值与策略迭代需要用到转移概率的值，在强化学习中，转移概率是通过仿真器来获取，该仿真器通常从一个均匀随机初始状态开始运行多次。强化学习可以与函数近似方法（如神经网络）结合来处理状态树很大的问题。 
#### 蒙特卡洛法  
&emsp;&emsp;蒙特卡洛法可用在模拟策略迭代的算法中。**策略迭代由两部分组成：策略评估与策略改进。蒙特卡洛法是用在策略评估步骤中**。在这一步中，给定一个静态的，确定性的策略$\pi$，目标是为所有的状态-动作对$(s,a)$计算函数值$Q^\pi(s,a)$（或好的近似值）。假设MDP是有限的且有足够的物理资源来处理动作值，并且问题是情景式的，每个episode之后会从某个随机初始状态开始一个新的episode。然后，对一个给定状态-动作对$(s,a)$的估计值可通过对从$(s,a)$开始到结束的样本回报进行平均获得。给定足够的时间，上述过程可以构建动作值函数$Q^\pi$的一个准确估计。  
&emsp;&emsp;在策略改进步骤，下一个策略是通过关于Q值计算贪心策略获得的：给定状态$s$，新的策略返回一个最大化$Q(s,\cdot)$的动作。在实际中，使用惰性评估（lazy evaluation）可以推迟最大化动作的计算，直到需要用到的时候再计算。  
&emsp;&emsp;蒙特卡洛法的缺点：
1. 在评估次优策略上花费太多时间
2. 使用了长的trajectory中的样本，却仅用来估计该trajectory起始点的单状态-动作对，样本使用效率低
3. 当沿着trajectory的回报有高方差时，收敛很慢
4. 仅对于情景式问题（episodic problem）有效
5. 仅对于小的，有限MDP有效  
 
#### 时间差分法  
&emsp;&emsp;针对蒙特卡洛法的五个缺点：  
- 问题1是通过**允许在值固定之前（在某个或所有状态）改变策略**来纠正的，但这一操作也可能使得算法难以收敛。当前大多数算法都这样做，从而产生了一类算法：广义策略迭代算法（generalized policy iteration algorithm）。许多actor-critic方法都属于这一类。  
- 问题2可通过允许trajectories对任意状态-动作对（的值估计）产生贡献来纠正。这样做某种程度上也有助于解决问题3，不过当回报有高方差时，一个更好的解决方法是时间差分法（temporal difference method）。该方法是基于递归Bellman方程。TD方法中的计算可以是增量的（在每次状态转移后，记忆被改变然后丢弃该状态转移），也可以是批处理（当对状态转移进行批处理并且估计值仅利用该批数据进行一次计算）。批处理方法，如最小二乘时间差分法，或许能更好地使用样本中的信息，而增量式方法对于批处理方法的计算时间/空间复杂度太高时是唯一的选择。基于时间差分的方法也克服了问题4。
- 使用函数近似法来解决问题5。
  
&emsp;&emsp;值迭代也可以用作起点，从而产生了Q-learning算法及其许多变体。  
&emsp;&emsp;使用动作值的问题是它可能需要对候选动作给出准确的估计，而这一要求对于回报有噪声时难以满足（即使TD方法某种程度上能减少这一影响）。使用兼容函数近似法会损害一般性及效率。使用TD的另外一个问题来自于对递归Bellman方程的依赖。**大多数TD方法有$\lambda$参数$(0\leq\lambda\leq1)$，该参数使得可以在不依赖Bellman方程的蒙特卡洛法与完全依赖于Bellman方程的TD方法之间进行差值**。  
#### 直接策略搜索  
&emsp;&emsp;另一种方法是直接在策略空间搜索，此时问题变为随机优化问题。该方法的两个实现是gradient-based方法以及gradient-free方法。  
&emsp;&emsp;Gradient-based方法（policy-gradient法）将有限维的参数空间映射到策略空间：给定参数向量$\theta$，令$\pi_\theta$代表与$\theta$有关的策略，定义性能函数：$\rho(\theta)=\rho^{\pi_\theta}$。在一些不严格的条件要求下，该函数将对于参数$\theta$可导。一旦已知了梯度$\rho$，则可以使用梯度上升来进行训练。无法获得梯度的解析表达式，只能获得其有噪声的估计值。该估计值可以以多种方式构建，比如强化方法（REINFORCE method）。许多策略搜索方法有陷入局部最优的问题（因为它们基于局部搜索）。  
&emsp;&emsp;一大类gradient-free方法避免依赖于梯度信息，这些方法包括模拟退火算法，交叉熵搜索或进化算法。许多gradient-free方法可以在理论上实现全局最优。  
&emsp;&emsp;策略搜索算法可能在有噪声的数据上收敛很慢。比如在trajectory很长且回报的方差很大的情景式问题中时。此时使用时间差分的值函数的方法可能会有帮助。二者的结合有actor-critic方法。