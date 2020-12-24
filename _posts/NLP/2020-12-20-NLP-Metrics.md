---
layout: post
title: "NLP任务评价指标"
subtitle: "NLP evaluation metrics"
author: "Roger"
header-img: "img/NLP/NLP.jpg"
header-mask: 0.4
mathjax: true
tags:
  - NLP
---

## 1. BLEU  
&emsp;&emsp;BLEU (bilingual evaluation understudy)用于评估**从一种语言翻译成另一种语言的文本的质量**。这里“质量”的好坏被定义为与人类翻译结果的一致性高低。BLEU分数的计算是对于独立的翻译片段（一般是句子）而言的，通过与高质量的翻译“参照”进行比较得出。对于整个语料的得分则是所有翻译片段得分的平均。**该度量方式不考虑可理解性及语法的正确性**。BLEU的值介于0和1之间，越接近于1代表翻译结果越接近于“参照”。如果值为1，代表翻译“参照”中有一个与翻译结果相同（这也意味着，更多的“参照”将带来更高的分数）。  
&emsp;&emsp;BLEU使用一种修正的precision形式来比较翻译候选与多个翻译参考之间的异同，其定义是：**翻译句子中的短语，出现在参考句子中的比例**。首先根据n-gram划分一个短语包含单词的数量，如1-gram（unigram）就是将语料划分成1个单词的短语。然后统计这些短语出现在参考译文中的个数，最后除以划分总数，得到BLEU-1分数。Unigram的准确率可以用于衡量单词翻译的准确性，更高阶的n-gram 的准确率可以用来衡量句子的流畅性。  
&emsp;&emsp;BLEU的一个缺点是**该指标倾向于选择更短的翻译**（从而产生更高的分数），比如只有一个单词，该单词在参考译文中，那么分数为1。候选翻译太短往往意味着有些词漏翻译了，即召回率（候选翻译中n-gram出现在参考翻译中的比例）低。为了产生整个语料的分数，使用几何平均数乘以简洁度惩罚（brevity penalty）来组合段的修改后的精确度分数，以防止非常短的候选者获得太高的分数。记“参照”句子的长度为$r$，翻译的语料长度为$c$。当$c\le r$时，施加简洁度惩罚：$e^{1-r/c}$（对于多个参考句子的情况，$r$代表最接近翻译句子长度的所有参考句子的长度和；也有用最短参照句子长度的）。  
&emsp;&emsp;BLEU的另外一个缺点是虽然原则上能够评价任何语言的翻译，但**BLEU目前的形式不能处理没有字界（word boundary，如汉语）的语言**。  
&emsp;&emsp;NLTK中可以计算独立的N-Gram得分以及各N-Gram的加权几何平均（通常会统计BLEU1到BLEU4的加权累计值作为评估文本生成系统效果的参照）。
## 2. METEOR  
&emsp;&emsp;METEOR（**M**etric for **E**valuation of **T**ranslation with **E**xplicit **OR**dering）是用来评估机器翻译效果的度量指标。**该指标基于precision和recall的调和平均，且recall的权重高于precision**。该指标被设计来修正BLEU的某些缺陷，并且在句子或语段层面上与人类判断产生良好的相关性。该指标的**主要思想是有时翻译模型是正确的，但它优势和参考翻译不一样（比如适用了同义词）。所以可以使用WordNet等知识源来扩展同义词集。此外，还考虑了单词的形式（有着相同词干的单词背认为是部分匹配的，也应该给予奖励，例如将likes翻译成like比翻译成其它的词要好）**。METOR与BLEU度量的不同之处在于，BLEU在语料库层面寻求相关性。  
&emsp;&emsp;与BLEU一样，METEOR评估的基本单位是句子，算法首先在两个句子(候选翻译和参考翻译)之间创建一个“对齐”（alignment）。其中，“对齐”是unigram之间的一个映射，即一个字符串中的一个unigram与另一个字符串中的unigram之间的对应关系，通过选择映射关系来产生“对齐”。映射规则如下：候选翻译中的每个unigram必须被映射到参考翻译中的0个或1个unigram。如果有多个映射方式备选，那么选择映射中unigram交叉最少的（即unigrams之间的映射连线交叉最少）。该过程分阶段进行，在每个阶段只将之前阶段没有匹配的unigrams添加到“对齐”中去。一旦计算得到最终的“对齐”，METEOR分数采用如下方式计算：  
- Unigram precision P：$P=\frac{m}{w_t}$，其中$m$是候选翻译中，存在于参考翻译中的unigram数。$w_t$是候选翻译中的unigram数
- Unigram recall R：$R=\frac{m}{w_r}$，其中$w_r$是参考翻译中的unigram数
- 使用调和平均来组合precision和recall：$F_{mean}=\frac{10PR}{R+9P}$，即recall的权重是precision的9倍  

&emsp;&emsp;上述所引入的度量方法只考虑到单个单词的一致性，而没有考虑到参考翻译和候选翻译中出现的更大的片段（即句子的流畅性）。为解决此问题，需要使用更长的n-gram匹配来为该“对齐”计算一个惩罚$p$。**参考翻译和候选翻译中不相邻的映射越多，惩罚越高**。为计算该惩罚，将unigram被分组到元素尽可能少的“块”（chunk，候选翻译和参考翻译能够对齐的、空间排列上连续的单词形成一个chunk）中，其中“块”被定义为候选翻译和参考翻译中相邻的一组unigram。候选翻译和参考翻译中的相邻映射越长，“块”越少（候选和参考完全相同，则只有一个“块”）。惩罚$p$的计算方法为：  
$$p=0.5(\frac{c}{u_m})^3 \tag{2.1}$$  
&emsp;&emsp;其中$c$是“块”的数量，$u_m$是被映射的unigram数。最终的METEOR分数M的计算公式如下。加入惩罚的作用为：如果没有二元或更长的匹配，则将$F_{mean}$最多减少50%。  
$$M=F_{mean}(1-p) \tag{2.2}$$  
&emsp;&emsp;为了在整个语料上计算该分数，分别计算$P$，$R$以及惩罚$p$，然后使用相同的公式来组合。该算法对于多个参考翻译的情况也适用，此时算法将候选翻译与所有参考翻译进行比较并且选择得分最高的那个。  
> 示例  
> Hypothesis	on	the	mat	sat	the	cat  
> Score: 0.5000 = Fmean: 1.0000 × (1 − Penalty: 0.5000)  
> Fmean: 1.0000 = 10 × Precision: 1.0000 × Recall: 1.0000 / （Recall: 1.0000 + 9 × Precision: 1.0000）  
> Penalty: 0.5000 = 0.5 × (Fragmentation: 1.0000 ^3)  
> Fragmentation: 1.0000 = Chunks: 6.0000 / Matches: 6.0000

> Reference	the	cat		sat	on	the	mat  
> Hypothesis	the	cat	was	sat	on	the	mat  
> Score: 0.9654 = Fmean: 0.9836 × (1 − Penalty: 0.0185)  
> Fmean: 0.9836 = 10 × Precision: 0.8571 × Recall: 1.0000 / （Recall: 1.0000 + 9 × Precision: 0.8571）  
> Penalty: 0.0185 = 0.5 × (Fragmentation: 0.3333 ^3)  
> Fragmentation: 0.3333 = Chunks: 2.0000 / Matches: 6.0000

> Reference	the	cat	sat	on	the	mat  
> Hypothesis	the	cat	sat	on	the	mat  
> Score: 0.9977 = Fmean: 1.0000 × (1 − Penalty: 0.0023)  
> Fmean: 1.0000 = 10 × Precision: 1.0000 × Recall: 1.0000 / （Recall: 1.0000 + 9 × Precision: 1.0000）  
> Penalty: 0.0023 = 0.5 × (Fragmentation: 0.1667 ^3)     
> Fragmentation: 0.1667 = Chunks: 1.0000 / Matches: 6.0000

## 3. ROUGE
&emsp;&emsp;ROUGE（Recall-Oriented Understudy for Gisting Evaluation）是一组度量，用于比较自动生成的摘要或翻译与人类生成的参考摘要或翻译之间的相似性。**ROUGE与BELU的区别是ROUGE只考虑召回率**，即不关心翻译结果是否流畅，只关注翻译是否准确：候选翻译中包含多少参考译文中的n-gram。共包含如下5中评估度量：  
- ROUGE-N：系统给出摘要与参考摘要之间N-grams的重叠水平
  - ROUGE-1指的是系统给出摘要和参考摘要之间unigram的重叠：$\frac{共现unigram个数}{参考摘要unigram个数}$
  - ROUGE-2指的是系统给出摘要和参考摘要之间bigram的重叠
- ROUGE-L：基于最长公共子序列(Longest Common Subsequence，LCS)的统计指标。最长公共子序列问题自然地考虑了句子层面的结构相似性，自动识别序列n-gram中最长的共现词。计算公式如下，其中$m$、$n$分别为参考摘要、候选摘要的长度（词个数）：  
  $$
  R_{lcs}=\frac{LCS(candidate, reference)}{m} \\
  P_{lcs}=\frac{LCS(candidate, reference)}{n} \\
  F_{lcs}=\frac{(1+\beta^2)R_{lcs}P_{lcs}}{R_{lcs}+\beta^2P_{lcs}} \tag{3.1}
  $$
- ROUGE-W：加权的（weighted）基于最长公共子序列(Longest Common Subsequence，LCS)的统计指标，支持连续的LCSes
- ROUGE-S：基于Skip-bigram的共现词统计指标。其中Skip-bigram是按句子顺序排列的任意一对单词
- ROUGE-SU：Skip-bigram加上Unigram的共现词统计指标  

## 4. CIDEr
&emsp;&emsp;CIDEr（Consensus-based Image Description Evaluation）是Vedantam于2014年提出的**针对图片摘要的度量标准**。该标准是基于共识（consensus-based）的评价标准，基本原理是通过度量待评测语句与其它大部分人工描述句之间的相似性来评价其与人的相似性（human-like），相比于前面指标，和人类共识具有更高的相关性。CIDEr也是在句子层面上的，它将每个句子看做一个文档，然后计算TF-IDF向量的余弦相似性，整个文档的CIDEr分数是所有句子分数的平均。该指标的优势是不同的n-grams有不同的权重（因为不同的TF-IDF），这是因为越常见的n-grams包含的信息越少。  
&emsp;&emsp;首先将n-grams在参考句中出现的频率编码进来，n-grams在数据集所有图片中经常出现的图片的权重应该减少，因为其包含的信息量更少，每个n-gram的权重通过TF-IDF来计算。将句子用n-grams表示成向量形式。每个参考句与候选翻译之间通过计算TF-IDF向量的余弦距离来度量其相似性。假设$c_i$为候选翻译，参考句子的集合为$S_i={s_{i1},s_{i2},\cdots,s_{im}}$，那么有：  
$$CIDEr_n(c_i,S_i)=\frac{1}{m}\sum_j\frac{g^n(c_i)\cdot g^n(s_{ij})}{\|g^n(c_i)\|\cdot\|g^n(s_{ij})\|} \tag{4.1}$$  
&emsp;&emsp;上式中$g^n(c_i)$和$g^n(s_{ij})$为句子中所有n-grams的TF-IDF分数组成的向量。与BLEU类似，当使用多种长度的n-grams时，CIDEr取其均值：  
$$CIDEr(c_i,S_i)=\frac{1}{N}\sum_{n=1}CIDEr_n(c_i,S_i) \tag{4.2}$$  
&emsp;&emsp;令$w_k$表示第k组可能的n-grams，$h_k(c_i)$表示$w_k$在候选翻译$c_i$中出现的次数，$h_k(s_{ij})$表示$w_k$在候选翻译$s_{ij}$中出现的次数。对于$w_k$，计算权重TF-IDF向量$g_k(s_{ij})$：  
$$g_k(s_{ij})=\frac{h_k(s_{ij})}{\sum_{w_l\in\Omega} h_l(s_{ij})}\log(\frac{|I|}{\sum_{I_{p\in I}}\min(1,\sum_qh_k(s_{pq}))}) \tag{4.3}$$  
&emsp;&emsp;上式中$\Omega$是n-grams的字表，$I$是所有图片集合。第一项是TF项，度量了每个n-gram的词频，第二项为IDF项，度量了每个n-gram的稀缺性。二者是相互制约关系。






