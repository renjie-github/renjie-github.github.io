---
layout: post
title: "Git常用命令"
subtitle: "Git common commands"
author: "Roger"
header-img: "img/post-create-blog.jpg"
header-mask: 0.4
mathjax: true
tags:
  - Blog
---

# Git常用命令
## Git基本配置
配置用于提交的作者名、邮箱  
```shell
git config --global user.name "Roger"
git config --global user.email "Roger@***.com"
```  
## 创建一个本地Git仓库
```shell
git init
```  
## 克隆一个仓库
```shell
git clone /path/to/repo
git clone username@host:/path_to/repo
```
## 添加文件到staging暂存区（add）
```shell
git add -A # 添加项目所有变化到staging
或 
git add . # 添加当前目录所有变化，不包括.gitignore内的文件
或
git add * # 会忽略.gitignore把所有文件都加入
或
git add <filename> # 添加特定文件
```  
## 将文件写入本地仓库（commit）
```shell
git commit -m "commit message"

git commit -a # 包含了git add步骤，但只适用于修改及删除，新文件还是需要git add
```  
## 将代码发送到远程仓库
```shell
git push origin <master> # 提交到master分支
```  
## 查看状态
```shell
git status
```  
## 连接到远程仓库
```shell
# 如果没有将本地仓库连接到远程server，需要添加该server来允许push, origin为远程地址的别名
git remote add <origin> <server url> 

# 列出所有当前依据配置过的远程仓库
git remote -v

# 显式某个远程仓库的信息
git remote show <remote>

# 删除远程仓库
git remote rm <name>
git remote rename <old_name> <new_name>
```  
## 分支相关
```shell
# 创建一个分支并切换到该分支
git checkout -b <branch name>

# 切换到一个分支
git checkout <branch name>

# 列出所有分支, *标记当前分支
git branch

# 删除本地分支
git branch -d <branch name>

# 将制定分支push到远程仓库, origin为远程主机别名
git push origin <local branch name>:<remote branch name>

# 将所有分支push到远程仓库
git push --all origin

# 在远程仓库删除一个分支
git push origin :<branch name>
或
git push origin --delete <branch name>
```  
## 从远程仓库更新
```shell
# 从远程仓库拉取并merge改变到本地工作目录，功能上等于git fetch + git merge，但隐藏了一些细节
git pull [--update]

# 从远程获取代码内容，fetch 后可以git diff检查
git fetch [alias]/[branch]

# merge前查看代码变动
git diff # 显示所有未stage的文件差异
git diff stat # 显示摘要而非所有
git diff -staged # 比较staging中文件和最新版本文件的差异
git diff <source branch> <target branch> # 比较不同分支差异

# 将fetch获取到的内容merge到活跃分支
git merge [alias]/[branch]
```  
## 重要阶段打标签
```shell
git tag [-a] v1.0 # -a命令会记录是谁在何时打的标签，会打开编辑器让自己写一些注解
git tag -a <tag name，如1.0.0> <commit ID>

# 打标签时添加信息
git tag -a <tag name> -m "my tag info" <commit ID>

# 显示所有tag信息
git tag

# 将所有tags push到远程仓库
git push --tags origin
```  
## 从缓冲区中移除文件，但保留文件内容
```shell
# 从stage中移除文件但本地所做的改变还在
git reset [file]

# 撤销指定commit后的所有commits并保留本地的文件改变内容
git reset [commit]

# 丢弃指定commit后所有的历史并回退到特定commit状态
git reset --hard [commit]
```  
## 搜索内容
```shell
git grep "foo()"
```  
