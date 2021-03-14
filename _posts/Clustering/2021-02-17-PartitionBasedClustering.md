---
layout: post
title: "层次聚类"
subtitle: "Hierarchical Clustering Methods"
author: "Roger"
header-img: "img/hierarchical_clustering.png"
header-mask: 0.4
mathjax: true
tags:
  - Clustering
---

## 层次聚类方法基本概念
&emsp;&emsp;层次聚类的特点有：
- 层次地进行聚类，聚类结果构成一棵树（dendrogram）
- 无需指定簇个数K
- 聚类结果是更确定性的（deterministic，相比于K-Means需要多次初始化）
- 无需通过不断迭代来细化聚类结果  

&emsp;&emsp;层次聚类主要分为两类：
- Agglomerative（自底向上）：一开始将每个样本看做一个cluster，然后自底向上通过不断合并cluster来层次地构建聚类树
- Divisive（自顶向下）：一开始将所有样本整体看做一个cluster，然后自顶向下通过不断分裂cluster来层次地构建聚类树

## Agglomerative Clustering
&emsp;&emsp;层次聚类中agglomerative clustering的代表是AGNES（AGglomerative NESting），由Kaufmann和Rousseeuw于1990年提出。该算法使用single-link法以及dissimilarity matrix，通过不断的合并有着least dissimilarity的cluster，最终将所有的样本合并为单个cluster。  
&emsp;&emsp;各Agglomerative clustering算法的主要区别在于cluster之间相似性的度量方法：
- Single link（nearest neighbor）：两个cluster之间的相似度用两个cluster之间最相似（最近）的样本之间的相似度。该度量更强调cluster间靠近的区域而忽略cluster的整体结构，因此可以用于聚类non-elliptical的cluster。同时带来的问题就是对噪声和异常值更加敏感
- Complete link（diameter）：两个cluster之间的相似度用两个cluster之间最不相似（最远）的样本之间的相似度。合并cluster是按照合并后具有最小diameter的原则进行。这一形为不是局部性的，可以获得更加“紧凑”的cluster。此外，该方法也对异常值敏感
- Average link（group average）：两个cluster中的所有样本的平均距离，假如cluster$C_a$样本数为$N_a$，cluster$C_b$样本数为$N_b$，那么需要计算$N_a\times N_b$对样本距离的平均。随之而来的是计算代价高
- Centroid link（centroid similarity）：用两个cluster之间的centroids距离计算
- Grouped Averaged Agglomerative Clustering（GAAC）：假设cluster$C_a$和cluster$C_b$合并为cluster$C_{a\cup b}$.则新的centroid为:
  $$
  C_{a\cup b}=\frac{N_a\boldsymbol{c}_a+N_b\boldsymbol{c}_b}{N_a+N_b}
  $$
  ，其中$\boldsymbol{c}_a$和$\boldsymbol{c}_b$分别为两个cluster的centroid。GAAC的相似性度量是它们之间距离的平均
- Ward's Criterion：使用合并之后SSE（Sum of Squares Error），其中SSE的计算是对每个cluster内所有样本计算的。当两个cluster合并后，SSE会增加，Ward's criterion方法使用的便是选择合并后SSE增加量最小的两个cluster进行合并：  
  $$
  W(C_{a\cup b}, \boldsymbol{c}_{a\cup b})-W(C, \boldsymbol{c})=\frac{N_aN_b}{N_a+N_b}d(\boldsymbol{c}_a, \boldsymbol{c}_b)
  $$

## Divisive Clustering
&emsp;&emsp;层次聚类中divisive clustering的代表是DIANA（Divisive Analysis），由Kaufmann和Rousseeuw于1990年提出。该方法是AGNES的反向操作，从一整个cluster开始，不断地划分出更多的cluster。该方法相比于agglomerative clustering更加高效，因为分裂时没有太多的选择，只是决定如何拆分并以递归地方式完成。  
&emsp;&emsp;该方法决定划分哪个cluster的原则是：选择具有最大SSE的cluster进行划分。划分方式：可以是使Ward's critetion减少最多的划分方式。对于categorical数据，可以使用Gini-index。该方法使用一个阈值来决定终止条件，不产生太小的cluster（因为这些cluster包含的主要是噪声）。

## 层次聚类的缺陷
- 执行是单向的，一旦划分/合并cluster就无法再撤回
- 时间复杂度至少是$O(n^2)$，其中$n$是样本数

## 更多的层次聚类算法
### 1. BIRCH
&emsp;&emsp;BIRCH（Balanced Iterative Reducing and Clustering Using Hierarchies, 1996）增量式地构建CF tree（clustering feature tree），CF-tree是一种用于**多阶段聚类**的层次化数据结构：
1. 扫描一遍数据以构建一个初始的CF tree（一种数据的多级压缩，用于保留数据潜在的聚类结构）
2. 使用任意一种聚类算法对CF tree的叶节点进行聚类  
   
&emsp;&emsp;该方法的核心思想是**多级聚类（Multi-level clustering）**，包括low-level的micro-clustering：减少复杂度并增加可扩展性，以及high-level的macro-clusteing：为更高层次的聚类留出足够的灵活性。BIRCH的时间复杂度是线性的，它通过扫描一遍数据来找到一个比较好的聚类，然后通过额外的几次扫描来改善聚类质量。  
&emsp;&emsp;BIRCH的聚类特征（clustering feature）是对于给定sub-cluster的统计特征的概况：sub-cluster的0阶、1阶、2阶矩（moment），通过保存关键的量测数据以更高效地利用存储。特征包括：（N，LS，SS），分别对应数据点数$N$、$N$个点的线性求和$\sum_{i=1}^{N}X_i$、$N$个点的平方和$\sum_{i=1}^{N}X_i^2$。  
&emsp;&emsp;Cluster的度量有：
- Centroid：cluster中所有样本的平均向量：$$\boldsymbol{x}_0=\frac{\sum_{i}^{N}\boldsymbol{x}_i}{N}$$
- Radius：cluster中所有样本到该cluster的centroid的均方根：$R=\sqrt{\frac{\sum_{i}^{N}(\boldsymbol{x}_i-\boldsymbol{x}_0)^2}{N}}$
- Diameter：cluster中所有样本对距离的均方根：$D=\sqrt{\frac{\sum_{i}^{N}\sum_{j}^{N}(\boldsymbol{x}_i-\boldsymbol{x}_j)^2}{N(N-1)}}$

&emsp;&emsp;CF tree很像B+ tree，可以增量地插入新的数据点。这意味着当一个新的数据点到来时，可以从root开始找到最近的叶子项（leaf entry），将数据点加入到该leaf entry并更新CF tree。如果entry diameter > max_diameter，那么将该leaf entry（可能包括其父节点）按照B+ tree算法分裂。CF tree有两个参数：（1）Branching factor：最大子节点数目（2）存储在叶子节点的Maximum diameter of sub-clusters。总的来说，CF tree是一种存储聚类特征的height-balanced tree，它的非叶节点存储它孩子们的CF的和.  
&emsp;&emsp;BIRCH算法特点是：
- 对数据点的插入顺序敏感
- 由于采用固定大小的叶子节点，cluster可能不够自然
- 由于所使用的radius和diameter度量，最终的cluster倾向于变为球形（spherical）

### 2. CURE（1998）
&emsp;&emsp;用一组分散的有代表性的点来代表一个cluster
### 3. CHAMELEON（1999）
&emsp;&emsp;在数据的K近邻图上使用图划分方法（graph partitioning methods）