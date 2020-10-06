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

  貌似双$符不起作用$$\t_{0}$$，仍以行内形式实现

  $E=mc^{2}$

  $$x+y=z\tag{1.1}$$

  \[x + y = z\]    
  \(x + y = z\)
  
  结束文字

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

$$ (b_{n - 1}b_{n - 2}\dots b_1 b_0)b = \sum{i = 0}^{n - 1} b_i \times b^i。 $$