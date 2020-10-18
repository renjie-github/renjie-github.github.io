---
layout: post
title: "Vim常用命令"
subtitle: "Vim Common Command"
author: "Roger"
header-img: "img/Linux/vim.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Linux
---

### 简单介绍
&emsp;&emsp;vim是Unix系统都会默认内置的一款轻量级编辑器。虽然目前有很多诸如VSCode、Sublime等图形界面文本编辑器，但对于服务器等需要远程操作且没有安装图形界面的场景，vim几乎是唯一的选择。此外，vim也支持很多定制的设置，熟练的程序员通过大量的快捷键可以极大地提高编程效率。  
&emsp;&emsp;这里只介绍一些简单的入门命令，使初学者可以完成快速上手基本的vim操作。**【注】vim默认进入的是命令模式，按键<kbd>i</kbd>可进入文本编辑模式，按<kbd>ESC</kbd>键能从文本编辑模式退回命令模式。**
### 显式相关
- 显示行号  
&emsp;&emsp;命令模式下输入:set number (或:set nu)  
```shell
:set number
```
- 取消行号显式  
&emsp;&emsp;命令模式下输入:set nonumber (或:set nonu)  
```shell
:set nonumber
```
- 转到第n行  
&emsp;&emsp;命令模式下输入:n (n为行号)  
```shell
:n
```
- 转到首/尾行
&emsp;&emsp;命令模式下输入:n (n为行号)  
```shell
:gg # to head line
# or use :G to locate to end line
```

### 插入操作（aio）
在命令模式下
- a&emsp;&emsp;在光标所在字符后插入内容
- A&emsp;&emsp;在光标所在行尾插入内容（跳到行尾进入文本编辑模式，下文类似）
- i&emsp;&emsp;在光标所在字符前插入内容
- I&emsp;&emsp;在光标所在行首插入内容
- o&emsp;&emsp;在光标所在行上插入内容
- O&emsp;&emsp;在光标所在行下插入内容
- $&emsp;&emsp;光标移至行尾
- 0&emsp;&emsp;光标移至行首

### 删除操作
在命令模式下
- x&emsp;&emsp;删除光标处字符
- nx&emsp;&emsp;删除光标位置后n个字符
- dd&emsp;&emsp;删除光标所在行
- ndd&emsp;&emsp;删除光标所在算起n行
- dG&emsp;&emsp;删除光标所在至文档末尾所有内容
- :n1,n2d&emsp;&emsp;删除n1-n2行内容

### 剪切-复制-粘贴
在命令模式下
- yy&emsp;&emsp;复制当前行
- nyy&emsp;&emsp;复制当前行算起下n行
- dd&emsp;&emsp;剪切当前行
- ndd&emsp;&emsp;剪切当前行算起下n行
- p&emsp;&emsp;粘贴在当前光标所在行上
- P&emsp;&emsp;粘贴在当前光标所在行下

### 替换操作
在命令模式下
- r&emsp;&emsp;替换光标处字符
- R&emsp;&emsp;从光标所在处开始不断替换字符，直到按<kbd>ESC</kbd>才会退出

### 搜索及搜索并替换
在命令模式下  
- /字符&emsp;&emsp;搜索"字符"
- n&emsp;&emsp;跳到下一个匹配位置，先搜索后再执行
- :%s/旧字符/新字符/g&emsp;&emsp;全文替换旧字符为新字符（g表示无需确认）
- :n1,n2s/旧字符/新字符/g&emsp;&emsp;在n1行到n2行范围内替换旧字符为新字符

### 取消上一步操作（undo）
在命令模式下  
- u&emsp;&emsp;取消上一步操作

### 保存退出
在命令模式下  
- :w&emsp;&emsp;保存修改
- :w filename&emsp;&emsp;另存为名为filename的文件
- :wq&emsp;&emsp;保存修改后退出(等效为ZZ)
- :q!&emsp;&emsp;不保存修改，退出
- :wq!&emsp;&emsp;保存修改并退出(强制)

### 导入文件内容到当前
在命令模式下  
- :r /路径/文件名（或!命令）&emsp;&emsp;将文件内容（或命令执行结果）导入当前光标所在处
- :!命令&emsp;&emsp;不退出vim情况下执行linux系统命令

### 自定义快捷键
在命令模式下  
- 快捷键定义  
  :map 自定义快捷键 对应操作&emsp;&emsp;如为了用<kbd>Ctril</kbd>+<kbd>M</kbd>给行首注释，连续按<kbd>Ctrl</kbd>+<kbd>V</kbd>+<kbd>M</kbd>，后面再跟I#<ESC>（I为跳到行首并转为插入模式，加入#，最后按<kbd>ESC</kbd>退出）  

  取消注释&emsp;&emsp;:map <kbd>Ctrl</kbd>+<kbd>V</kbd>+<kbd>O</kbd> 0x（跳到行首并删除）  

  插入固定内容&emsp;&emsp;:map <kbd>Ctrl</kbd>+<kbd>V</kbd>+<kbd>H</kbd> iroger@mail.com（i转插入模式，内容为roger@mail.com邮箱）  
- 连续注释多行  
  :n1,n2s/^/#/g&emsp;&emsp;n1-n2行行首（^）插入#号  
  :n1,n2s/^#//g&emsp;&emsp;n1-n2行删除行首#号  
  :n1,n2s/^//g&emsp;&emsp;n1-n2行行首（^）删除行首内容（比如#）  
  :n1,n2s/^/\/\//g&emsp;&emsp;n1-n2行行首（^）插入//（需加转移符\）  
- 默认替换  
  :ab myname myNameisRoger&emsp;&emsp;输入myname自动会替换为myNameisRoger（即a替换为b）  
- 保存自定义快捷键  
  需要将其写入用户home（管理员为root）目录下的配置文件.vimrc中