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
  - [开启cuDNN AUTOTUNER](#开启cudnn-autotuner)
  - [增大batch size](#增大batch-size)
  - [拿掉后面紧跟Batch Norm的卷积层的bias](#拿掉后面紧跟batch-norm的卷积层的bias)
  - [用parameter.grad = None代替model.zero_grad()](#用parametergrad--none代替modelzero_grad)
  - [在最终的训练中关闭debug API](#在最终的训练中关闭debug-api)
  - [用DistributedDataParallel代替DataParallel](#用distributeddataparallel代替dataparallel)
  - [在多个GPU之间做负载均衡](#在多个gpu之间做负载均衡)
  - [使用APEX中的fused building blocks](#使用apex中的fused-building-blocks)
  - [用checkpointing重计算中间结果](#用checkpointing重计算中间结果)
  - [使用PyTroch JIT](#使用pytroch-jit)
- [TensorFlow](#tensorflow)
- [References](#references)
# PyTorch
## 开启异步数据加载及数据增强  
PyTorch的DataLoader支持异步数据加载/增强，默认设置是{"num_workers: 0, "pin_memory": False}，设置num_workers > 0可以开启异步数据处理，使用设置pin_memory=True会带来更好的性能，但也会增加显存的占用。

![dataloader](/img/ModelTraining/data_loader.png "Vanilla CNN") 

## 开启cuDNN AUTOTUNER
cuDNN支持多种算法来计算卷积，autotuner会运行一个较短的benchmark来选择（对于当前平台）具有最佳性能的算法。  
具体用法是设置：
```python
torch.backends.cudnn.benchmark = True
```  
![autotuner](/img/ModelTraining/cudnn_autotuner.png "autotuner")

## 增大batch size
增大batch size以最大化利用GPU显存，使用AMP（Automatic Mixed Precision）可以进一步减少显存要求从而进一步增大batch size。  
增加batch size的同时，还需要：  
- 调节学习率，增加学习率warmup，学习率decay，调节weight decay  
- 或者切换到转为大batch训练而设置的优化器：  
  - LARS
  - LAMB
  - NVLAMB
  - NovoGrad  

## 拿掉后面紧跟Batch Norm的卷积层的bias
因为BN操作会移除bias，前面卷积层加了bias也只会增加无用开销。  

## 用parameter.grad = None代替model.zero_grad()  
model.zero_grad()用于下一个迭代前清空梯度，避免对下一个迭代造成干扰。该操作本质上会在关于每个参数的循环中用memset置零。该循环会为每个参数调用单独的CUDA kernel，这会导致效率的下降。此外，在反向传播阶段，PyTorch框架会使用“+=”算子来更新梯度，该算子先读后写，最后将值存到显存中。但实际上读的操作是不必要的，因为显存会被预置为0，所以不需要先读（而且要置为零，无需读）。  
代替的方法是：  
```python
for param in model.parameters():
    param.grad = None
```  
这一方法不会为每个参数都执行memset，显存被PyTorch的allocator以一种更高效的方式置零。此外，反向传播阶段，PyTorch会用“=”算子（写）来更新梯度，避免了读的操作。

## 在最终的训练中关闭debug API 
为了支持debugging会引入额外的开销，debugging API如下：  
- anomaly detection:
  - torch.autograd.detect_anomaly
  - torch.autograd.set_detect_anomaly(True)
- autograd profiler:
  - torch.autograd.profiler.profile
- automatic NVTX ranges:
  - torch.autograd.profiler.emit_nvtx
- autograd gradcheck
  - torch.autograd.gradcheck
  - torch.autogra.gradgradcheck

## 用DistributedDataParallel代替DataParallel
DataParallel是用一个CPU core来驱动多个GPU（python GIL），最多只能用于单个节点。而DistributedDataParallel一个CPU core/python进程对应一张GPU卡，这样GPU就不用等待CPU。该方法适用于单/多节点（API相同），有着高效的实现方式：  
- 梯度all-reduce的自动分箱（automatic bucketing）
- all-reduce（memory-bound）操作与反向传播（compute-bound）操作（时间上）重叠  

这也带来了一点缺点：要求一点多进程编程基础，要注意对共享资源的保护，比如一个时间只能有一个worker写checkpoint文件。需要对每个worker写单独的锁文件。  

## 在多个GPU之间做负载均衡
训练过程涉及forward，backward，all-reduce，optimizer四个部分，多卡中的任一一张卡在某个阶段缓慢/阻塞都会影响整个计算效率（all-reduce结束后才可以优化），此时别的显卡处于idle状态。为了减少这种情况的发生，每个GPU卡处理的数据量应尽量接近。  
有很多方法可以实现这一负载均衡，比如将数据按照相似的长度分箱/排序，取决于具体的问题。  

## 使用APEX中的fused building blocks
NVIDIA的APEX（A PyTorch Extension）提供了许多优化过的，可重用的构建模块。其包含的组件有：  
- distributed training
  - apex.parallel.SyncBatchNorm，是针对distributed data parallel设计的batch normalization
- fused optimizers
  - apex.optimizers.FusedAdam
  - apex.optimizers.FusedLAMB
  - apex.optimizers.FusedNovoGrad
  - apex.optimizers.FusedSGD
- apex.normalization.FusedLayerNorm

## 用checkpointing重计算中间结果  
常规的训练会在forward阶段保存所有输出的结果（但也会占用更多的显存），backward阶段则不用再次计算，这会限制最大可达到的batch size数。  
通过使用activation checkpointing，在forward阶段只保留某些运算的输出（占用更少的显存），在backward阶段，剩余的中间值被重新计算（额外的计算开销），这使得可以使用更大的batch size，从而获得更大的GPU及TensorCores的利用率。  
相关的PyTorch的API是torch.utils.checkpoint。选择重新计算哪些运算将变得很关键，应该选择重新计算开销较小，结果占用大量显存的运算进行checkpointing，比如：  
- activation (ReLU, Sigmoid, ...)
- up/down sampling
- maxtrix-vector ops with small accumulation depth


## 使用PyTroch JIT
PyTroch JIT可以将逐点运算融合（fuse）到单个CUDA kernel上。Unfused的逐点运算属于memory-bound，对于每个unfused op，PyTorch需要：  
- 启动一个独立的CUDA kernel
- 从global memory中加载数据
- 执行运算
- 将计算结果保存回global memory  
  
使用方式及性能对比如下：
![JIT](/img/ModelTraining/JIT.png "autotuner")


# TensorFlow


# References
[1] ECCV 2020 Tutorial on Accelerating Computer Vision with Mixed Precision