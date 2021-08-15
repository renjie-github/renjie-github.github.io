---
layout: post
title: "模型训练性能调优笔记"
subtitle: ""
author: "Roger"
header-img: "img/post-create-blog.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Model Training
---

# Table of Content
- [Table of Content](#table-of-content)
- [PyTorch](#pytorch)
  - [开启异步数据加载及数据增强](#开启异步数据加载及数据增强)
- [TensorFlow](#tensorflow)
- [References](#references)
# PyTorch
## 开启异步数据加载及数据增强  
PyTorch的DataLoader支持异步数据加载/增强，默认设置是{"num_workers: 0, "pin_memory": False}，设置num_workers > 0可以开启异步数据处理，使用设置pin_memory=True会带来更好的性能，但也会增加显存的占用。
<p align="center">
  <img src=/img/ModelTraining/data_loader.png alt="async data loading" width="400", height="300" />
</p>
<!-- ![VanillaConv](/img/ModelTraining/vanillaConv.jpg "Vanilla CNN")  -->
# TensorFlow


# References
[1] ECCV 2020 Tutorial on Accelerating Computer Vision with Mixed Precision