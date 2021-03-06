---
layout: post
title: "Linux之：系统进程及资源管理，计划任务"
subtitle: "common commands"
author: "Roger"
header-img: "img/Linux/process_management.png"
header-mask: 0.4
mathjax: true
tags:
  - Linux
---

### 基本介绍
&emsp;&emsp;本文用于Linux系统进程查询、管理。
### 系统资源查看
#### 1. ps命令：查看系统所有进程
ps aux：使用BSD（Unix）系统格式，a代表查看所有前台进程，x代表查看所有后台进程，u代表显示产生进程的用户  
ps -le：使用Linux系统格式，-l代表显示详细信息，-e代表显示所有进程  
ps aux显示信息从左到右：   

|用户|PID|CPU消耗百分比|内存消耗百分比|虚拟内存消耗(KB)|实际物理内存消耗(KB)|连接终端|进程状态|启动时间|累积CPU时间|启动命令|
|USER|PID|%CPU|%MEM|VSZ|RSS|TTY|STAT|START|TIME|COMMAND|

>
TTY用于判断是在哪个终端运行的，tty1-tty7代表本地控制台终端，tty1-tty6是本地字符界面终端，tty7是图形终端。pts/0-255代表虚拟终端  
STAT：S代表休眠、T代表停止、R代表运行、s代表包含子进程、+代表位于后台  

#### 2. top命令：查看系统健康状态
```shell
top [Option]
```

>
  -d 秒数：每隔几秒更新（默认3秒）

top命令交互模式中可以执行的命令：
>
  ?或h：显示帮助  
  P：以CPU使用频率排序（默认）  
  M：以内存使用率排序  
  N：以PID排序  
  q：退出top  

top命令显示信息：
- 第一行：系统当前运行的时间，当前系统登录了几个用户，系统在前1分钟、5分钟、15分钟的平均负载（一般认为<核数时负载较小），用uptime命令也可以显式该行
- 第二行：进程信息，总共有多少进程，运行、休眠、停止、僵尸进程（正在结束还没完全停止的进程，或结束时卡死的进程）
- 第三行：CPU信息，用户模式占用的CPU百分比，系统模式占用的CPU百分比，改变过优先级的用户进程占用的百分比，等待输入/出的进程占用的百分比，硬中断请求服务占用的CPU百分比，软中断请求服务占用的CPU百分比，虚拟时间百分比（即有虚拟机时，虚拟CPU等待实际CPU的时间百分比）
- 第四行：物理内存信息，物理内存总量、空闲物理内存数、已经使用的物理内存数、作为缓冲的内存数
- 第五行：交换分区信息，分区总量、空闲分区量、已经使用总量、作为缓存的交换分区大小

#### 3. vmstat命令：监控系统资源
```shell
vmstat [刷新延时 刷新次数]

vmstat 1 3
```

#### 4. free命令：查看内存使用状态
```shell
free [Option]
```

>
  -b：字节为单位  
  -k：KB  
  -m：MB  
  -g：GB  

#### 5. pstree：查看进程树
```shell
pstree [Option]
```

>
  -p：详细显示进程的PID  
  -u：显示进程的所属用户  

#### 6. kill命令：结束进程
kill -l：查看可用的进程信号  
```shell
kill [Option] PID
```
>
  -1：SIGHUP，重启进程  
  -9：SIGKILL，强制结束进程  
  -15：SIGTERM，正常结束进程  

```shell
killall [Option] 进程名 # 注意，不是PID。该命令用于杀死所有相关进程
```
```shell
pkill [Option] [信号] 进程名
  -t：按照终端号踢出用户  
先用w查看登录本机已经登录的用户，然后pkill -9 -t tty1（强制结束本地终端登录tty1）
```

### 系统工作管理
#### 1. 放入后台执行
- 方法1：执行命令 & （放入后台后还在运行）
- 方法2：执行命令的过程中，CTRL + Z （放入后台后就暂停了）  

查看后台进程：jobs -l  
恢复后台进程：fg %工作号 （注：工作号 != 进程号）  
恢复到后台：bg %工作号 （程序不能和前台有交互）  
>
【P.S.】缓存（cache）是用来加速从硬盘中“读取”数据的（避免重复读取），缓冲（buffer）是用来加速向硬盘“写入”数据的（批量写入，避免小数据频繁写入操作）

#### 2. uname：查看系统与内核相关信息
```shell
uname [Option]
```
>
  -a：查看系统所有相关信息  
  -r：查看内核版本  
  -s：查看内核名称  

#### 3. file命令：查看文件类型
```shell
file /bin/ls # 用file命令查询shell的外部命令会顺带显示系统位数，可用来查看系统位数
```
#### 4. lsof命令：列出进程打开或使用的文件信息
```shell
lsof [Option]
```
>
  -c 字符串：只列出以字符串开头的进程打开的文件  
  -u 用户名：只列出某个用户的进程打开的文件  
  -p pid：只列出某个PID进程打开的文件  

#### 5. 其它命令
- dmseg命令：查看开机时内核检测信息
- /proc/cpuinfo：包含CPU信息
- lsb_release -a：查看当前系统的发行版本

### 系统定时任务
crond：系统定时任务
> 
service crond restart 重启服务  
chkconfig crond on 检查启动

```shell
crontab [Option]
```
>
  -e：编辑crontab定时任务  
  -l：查询crontab任务  
  -r：删除当前用户所有的crontab任务  

crontab -e会打开vim编辑器，输入：
```vim
* * * * * 执行的任务
```

上述5个*分别为：
>
一小时当中的第几分钟（0-59），一天当中的第几小时（0-23），一月当中的第几天（1-31），一年当中的第几月（1-12），一周当中的星期几（0-7,0与7均为周日）

特殊符号：
>
  *：代表任何时间  
  ,：代表不连续的时间  
  -：代办连续的时间范围  
  */n：代表每隔多久执行一次  

45 22 * * * 命令：在22点45分执行命令
0 17 * * 1 命令：每周一的17点0分执行命令
0 5 1, 15 * * 命令：每月1号和15号的凌晨5点0分执行命令
40 4 * * 1-5 命令：每周一到周五的凌晨4点40分执行命令
*/10 4 * * * 命令：每天的凌晨4点，每隔10分钟执行一次命令
0 0 1,15 * 1 命令：每月1号和15号，每周一的0点0分都会执行命令 【星期几和几号最好不要同时出现】
5 5 * * 2 /sbin/shutdown -r now 每周二凌晨05:05重启
0 5 1,10,15 * * /root/sh/autoback.sh 每个月的1,10,15凌晨五点执行脚本

【注】定时任务中%有特殊函数，要当%用时前面应加转义符\




