---
layout: post
title: "博文模板"
subtitle: "Blog template"
author: "Roger"
header-img: "img/post-create-blog.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Blog
---

# 标题1
## 标题2
### 标题3
#### 标题4
##### 标题5


> **加粗文字块文字块**

1. 序号
   1. 子序号
      1. 子序号
2. 序号




* 测试字段1
* 测试字段2

<https://www.baidu.com>

[百度url](https://www.baidu.com)

### 公式
- 行内公式

  文字$\lambda_{0}$文字

- 行间公式

\begin{equation}
\begin{aligned}
V(s_t;\theta)=\sum_{a}\pi(a|s_t;\theta)\cdot Q_\pi(s_t,a)
\end{aligned}
\end{equation}

  $E=mc^{2}$

  $x+y=z\tag{1.1}$

  结束文字

- 矩阵  
  $$
  \left(\begin{array}{cc} 
  0.8944272 & 0.4472136\\
  -0.4472136 & -0.8944272
  \end{array}\right)
  \left(\begin{array}{cc} 
  10 & 0\\ 
  0 & 5
  \end{array}\right)
  $$ 

### 表格

|列名1|列名2|列名3|
|a|b|c|
|d|e|f|

### 图片
* 插入网络图片

![网络图片](https://bkimg.cdn.bcebos.com/pic/3c6d55fbb2fb4316b81c19dd2ca4462309f7d312?x-bce-process=image/resize,m_lfit,w_268,limit_1/format,f_jpg)

* 插入本地图片
![本地图片](/img/404-bg.jpg)

* 插入base64编码
  
  ![编码图片][引用内容]
  然后文末添加  
  [引用内容]

### 代码
```javascript
$(document).ready(function () {
    alert('RUNOOB');
});
```

```python
print("test case of python")
```

### 段落换行

文字（末尾加两个空格后回车）内容  
文字内容文字内容文字内容文字内容

文字内容（中间空一行）文字内容

文字内容文字内容文字内容文字内容

### HTML
<kbd>F5</kbd> <kbd>Alt</kbd>

&emsp;&emsp;