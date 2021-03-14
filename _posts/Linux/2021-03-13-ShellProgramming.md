---
layout: post
title: "Shell编程"
subtitle: "common commands"
author: "Roger"
header-img: "img/Linux/shell.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Linux
---

### 基本介绍
&emsp;&emsp;本文用于记录常见的shell编程命令，便于使用时查询。
### 常用命令
#### 1. read命令：输入
read命令：read [Options] [name...]
>
  -s：用于输入密码等场景时不打印到屏幕   
  -p: 打印输入提示字符串  
  -r: 禁用反斜杠转义字符“\”  
  -t: 设置最长等待时间（秒）  
  -a: 将输入分配给一个数组（array），而不是一个变量名。例如read -r -a MY_VAR <<< "Linux is awesome"这个命令会将"Linux is awesome"拆分成三项后赋值给MY_VAR  

```shell
read -t 30 -p "input your value:" my_var
# 打印输入提示字符串，最多等待30秒，将输入的结果赋给变量my_var
```

#### 2. nmap命令：扫描当前计算机
nmap -sT 192.168.0.1 扫描目标服务器TCP端口


#### 3.grep命令：对行进行正则匹配
grep [Option] 文件名 （按行提取文件内容）  
grep -v string1: 只匹配**不包含**"string1"的结果  

#### 4. cut命令：按列提取文件内容
cut [Option] 文件名
>
  -f: 指定选取第几列  
  -d: 指定分隔符  

cut只能识别固定的分隔符，对于长度不一的（如空格）分隔符就不行了，这是需要使用awk

#### 5. awk命令：按列提取文件内容
awk '条件1{动作1}{条件2}{动作2}...' 文件名  
- 条件
x >10, x>=10, x<=10等  
- 动作
格式化输出/流程控制语句 

```shell
awk 'BEGIN {FS=":"} {printf $1 "\t" $3 "\n"}' /etc/passwd
# BEGIN作用是在程序一开始需要执行后续动作（这里是定义分隔符为冒号）。
# 第二个动作不需要条件，可以直接执行。如果用print代替printf可以省掉最后的"\n"
```

#### 5. sed命令：文本处理
sed命令：sed [Option] '[动作]' 文件名  
>
  -n：加上这项后，只会把经过sed命令处理的行输出到屏幕（否则全部输出）  
  -e：允许对输入数据应用多条sed命令编辑  
  -i：用sed的修改结果直接修改读取数据的文件，而不是由屏幕输出  
    动作：
  a：追加（当前行后追加一行或多行，追加多行时，除最后一行外，每行末尾需要用“\”代表数据未完结）  
  c：行替换（用c后的字符串替换原数据行，替换多行时，除最后一行外，每行末尾需要用“\”代表数据未完结）  
  i：插入（当前行前面插入一行或多行，插入多行时，除最后一行外，每行末尾需要用“\”代表数据未完结）  
  d：删除指定行  
  p：打印指定行  
  s：字符串替换，格式为“行范围s/旧字符串/新字符串/g”（与vim的替换相似）

sed是轻量级流编辑器。主要用来将数据进行选取、替换、删除、新增的命令。（对应vim只能修改文件，不能修改管道命令的结果，而sed都可以）
```shell    
sed -n '2p' test.txt # 只输出文件第二行（不然还会输出整个文件内容）  
sed '2,4d' test.txt # 删除第二行到第四行的数据，但不修改文件本身  
sed '2a hello' test.txt # 在第二行后追加hello  
sed '2i hello \   
world' test.txt # 在第二行前插入两行数据  
sed '2c stringstringstring' test.txt # 用stringstringstring替换第二行  
sed -i '3s/旧字符/新字符/g' test.txt # 把第三行中的旧字符替换成新字符，同时用结果覆盖文件（不输出）  
sed -e 's/string1//g; s/string2//g' test.txt # 同时把string1和string2替换为空  
```

#### 6. sort命令：排序
sort命令：sort [Option] 文件名
>
  f：忽略大小写  
  -n：以数值型进行排序，默认使用字符串型排序  
  -r：逆序  
  -t：指定分隔符，默认为制表符  
  -k n[,m]：按照指定的字段范围排序。从第n字段开始，m字段结束（默认到行尾）  

```shell
sort -t ":" -k 3,3 /etc/passwd
# 指定分割符为：，用第三字段开头，第三字段结尾排序（只用第三字段排序）
```

#### 7. wc命令：文本统计
wc命令：wc [Option] 文件名
>
  -l：只统计行数  
  -w：只统计单词数  
  -m：只统计字符数

#### 8. 其它 
- bash脚本的开头必须写：#!/bin/bash  
- 命令output &>/dev/null 作用相当于将命令输出丢弃到回收站  
- 只有用双()括起来，才可以进行数学运算  
- date=$(date +%y%m%d) 只显示时间的数字形式并将表达式的结果赋值给date变量  

### 条件判断
#### 1. 按照文件是否存在判断
>
  -d：判断该文件是否为目录文件  
  -e：判断该文件是否存在  
  -f：判断该文件是否为普通文件  

```shell
test -e test.txt 或 [ -e test.txt ]
# 之后可以用echo $?判断是否存在（为0则存在）或者用
[ -e test.txt ] && echo "yes" || echo "no"
```

#### 2. 按照文件权限判断
>
  -r：判断文件是否有读权限  
  -w：判断文件是否有写权限  
  -x：判断文件是否有执行权限  

```shell
[ -w test.txt ] && echo yes || echo no
```
#### 3. 按照文件新旧判断
>
  文件1 -nt 文件2：文件1是否比文件2新  
  文件1 -ot 文件2：文件1是否比文件2旧  
  文件1 -nt 文件2：文件1与文件2是否为同一个文件（判断是否为硬链接）  

#### 4. 整数之间的比较
>
  -eq  
  -ne  
  -gt  
  -lt  
  -ge  
  -le  

#### 5. 字符串的判断
>
  -z：字符串是否为空  
  -n：字符串是否非空  
  string1==string2：字符串是否相等  
  string1!=string2：字符串是否不等（!与后面的条件之间也有空格）  

```shell
name=test
[ -z "$name" ] && echo yes || echo no
```

#### 6. 逻辑连接多重判断
>
  判断1 -a 判断2：逻辑与  
  判断1 -o 判断2：逻辑或  
  ！判断：逻辑非（使原始判断式取反）  

```shell
aa=24
[ -n "$aa" -a "$aa" -gt 23] && echo "yes" || echo "no"
```

### 流程控制-if语句
注意方括号两侧空格不能省略  
```shell
if [ 条件判断式 ]; then
    程序
fi
或者：
if [ 条件判断式 ]
    then
        程序1
fi

if [ 条件判断式 ]
    then
        程序1
    else
        程序2
fi


if [ 条件判断式 ]
    then
        程序1
elif [条件判断式2]
    then
        程序 2
else
    程序3
fi
```

### 流程控制-case语句
```shell
case $变量名 in
    "值1")
	如果变量值等于值1，执行程序1
	;;
    "值2")
	如果变量值等于值2，执行程序2
	;;
    *)
	如果变量值与上述值都不匹配，执行该程度
	;;
esac
```

### 流程控制-for循环
```shell
# 形式1：
for 变量 in 值1 值2 值3
    do
        程序
    done
```

以批量解压缩脚本为例：
```shell
#!/bin/bash 
cd /lamp
ls *.tar.gz > ls.log
for i in $(cat ls.log)
    do
        tar -zxf $i &>/dev/null
    done
rm -rf /lamp/ls.log
```

```shell
# 形式2：
for ((初始值; 循环控制条件;变量变化))
    do
        程序
    done
```

以从1加到100为例：  
```shell
#!/bin/bash
s=0
for ((i=1;i<=100;i=i+1))
    do
        s=$(($s+$i))
    done
echo "Sum is: $s"
```

### 流程控制-while循环
一般用于不确定循环几次的场景，for循环一般用于固定循环。  
```shell
while [ 条件判断式 ]
    do
        程序
    done
```

以从1加到100为例：  
```shell
#!/bin/bash
i=1
s=0
while [ $i -le 100 ]
    do
        s=$(($s + $i))
        i=$(($i + 1))
    done
echo "Sum is: $s"
```

### 流程控制-until循环
只要条件不成立，执行循环，直到条件成立。与while循环相反。  
```shell
until [ 条件判断式 ]
    do
        程序
    done
```






