---
layout: post
title: "强化学习之——DQN的变种"
subtitle: "DQN variants"
author: "Roger"
header-img: "img/RL.jpg"
header-mask: 0.4
mathjax: true
tags:
  - RL
  - DL
---

### **DQN的变种** 
#### **Double DQN**
&emsp;&emsp;Double DQN是于2015年发表的文章["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/abs/1509.06461)中提出的，目的是减少DQN中常见的“高估”（overestimation）问题。  
&emsp;&emsp;DQN中目标Q值的计算公式为：  
$Q_{target}(s,a)\leftarrow r+\gamma max_{a'\in A}Q(s',a')\tag{1}\label{eq1}$  
&emsp;&emsp;Bellman更新公式$\eqref{eq1}$中max算子使用同一个Q值来同时做动作选择及动作评估，这使得它更容易选择高估的值，从而导致过于乐观的值估计。Double DQN的核心思想：将动作选择与评估解耦可以减小上述效应的影响。  
&emsp;&emsp;作者证明了任何原因造成的估计误差都会引起上偏（up bias）。[Thrun and Schwartz (1993)](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)证明：如果动作值包含均匀分布在$[-\epsilon, \epsilon]$内的随机误差，则每个目标值被高估的误差上界是$\gamma \epsilon \frac{m-1}{m+1}$。Double DQN作者给出了相应的误差下确界，并证明了在造成经典DQN高估的同样条件下，Double DQN的误差下确界是0：  
<img src="/img/RL/DQN-variant-theorem.jpg" width=500 height=500 div align=center />  
&emsp;&emsp;与经典DQN的区别是：使用online network来评估贪心策略（进行动作选择），使用目标网络来估计值（进行值估计）。所以，Double DQN的目标Q值计算公式变为：  
$Q_{target}(s,a)\leftarrow r+\gamma Q(s', argmax_{a}(s',\theta_t),\theta_{t}^{-})\tag{2}$  
&emsp;&emsp;上式中$\theta_{t}^{-}$是target网络的参数，$\theta_{t}$是当前online网络的参数。对应的[TensorFlow代码实现](https://github.com/renjie-github/RLToolKit/blob/main/DoubleDQN.ipynb)为
```python
import tensorflow as tf

def train_on_batch(self, states, actions, rewards, next_states, dones):
    actions = tf.cast(actions, dtype=tf.uint8)
    # using target network to calculate Q values
    next_Q_values = self.model_target(next_states)
    # using online network to select eager action 
    next_mask = tf.one_hot(np.argmax(self.model(next_states), axis=1), depth=2, dtype=tf.float32)  
    target_Q_values = (rewards + (1 - dones) * self.gamma * 
                        tf.reduce_sum(next_Q_values * next_mask, axis=1, keepdims=True))

    mask = tf.squeeze(tf.one_hot(actions, depth=2, dtype=tf.float32))
    with tf.GradientTape() as tape:
        Q_values_ = self.model(states)
        Q_values = tf.reduce_sum(Q_values_ * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(tf.keras.losses.MSE(target_Q_values, Q_values))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
```
#### **Dueling DQN**

#### **Prioritized Double DQN**

#### **Noisy DQN**