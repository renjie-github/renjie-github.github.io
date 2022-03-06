---
layout: post
title: "REST API"
subtitle: "concepts"
author: "Roger"
header-img: "img/Linux/restful_api.png"
header-mask: 0.4
mathjax: true
tags:
  - Blog
---

# Restful API
## 基本介绍
&emsp;&emsp;REST API（也叫作RESTful API）是一个遵守REST架构风格并且允许RESTful Web服务之间交互的应用程序接口（API）。REST全称是REpresentational State Transfer，当一个client通过一个RESTful API发送请求，它事实上transfer a representation of the state of the resource to the requester or endpoint。这一information或representation，通过HTTP以如下几种格式之一传输：JSON（Javascript Object Notation），HTML，XLT，Python，PHP，plain text，其中JSON是最常见的格式。  
&emsp;&emsp;Header以及参数在RESTful API HTTP请求的HTTP方法中也很重要，它们包含了重要的识别信息，比如request metadata，authorization，unifrom resource identifier（URI），caching，cookies等。有request header以及response header两种，每种都有其自己的HTTP连接信息和状态码。  
&emsp;&emsp;一个RESTful API需要遵循以下限制：
- 一个由clients，servers，resources以及通过HTTP管理的requests组成的client-server架构
- 无状态（stateless）的client-server通信，意味着在get requests之间没有client信息的存储，每个request是独立的
- 在组件之间有一个通用的接口，从而信息以一个标准的形式传输。这要求：
  - 请求的资源是可识别的，并且与发送给client的表示分离
  - 资源可以被client通过其接收到的representation处理，因为representation包含了足够的信息
  - 返回给client的自我描述的消息包含了足够的信息来告诉client如何处理该信息
  - hypertext/hypermedia是可获取的，这意味着在client访问资源后，client应该使用hyperlink来找到所有当前可采取的操作
- 是一个分层的系统，组织了涉及到的不同类型的server（负责安全，负载均衡等），涉及到将请求信息检索到层级结构中，对于client是不可见的
- Code-on-demand（可选）：即在被client请求时，从server发送可执行代码到client，以扩展client功能的能力

## HTTP methods
- GET: 获取数据
- POST：写“新”数据，期望其每执行一次都会创建一条新的record
- PUT：“更新/替换 ”数据，一般期望其执行很多次都不会有side-effects
- DELETE：删除数据

&emsp;&emsp;上面四种方法对应数据库操作中的CRUD（Create，Read，Update，Delete）

## REST API Example
### 创建一个Python API
- 设置Python环境
  - 创建一个虚拟环境 `python3 -m venv .venv`
  - 激活虚拟环境 `source .venv/bin/activate`
  - 安装flask `pip3 install flask`
  - 安装数据库 `pip3 install flask-sqlalchemy`
  - 导出dependency `pip3 freeze > requirements.txt`
- 创建app文件
  - `touch application.py`
  - 编写代码

  ```Python
  from flask import Flask
  app = Flask(__name__)

  @app.route('/')
  def index():
    return 'Hello!'
  ```  
  - 测试app运行
    - 设置环境变量`export FLASK_APP=application.py`以及`FLASK_ENV=development`
    - 启动app `flask run`
  - 添加逻辑
   
  ```Python
  from flask import Flask
  app = Flask(__name__)

  @app.route('/')
  def index():
    return 'Hello!'

  @app.route('/drinks')
  def get_drinks():
    return {"drinks": "drink data"}
  ```   

&emsp;&emsp;详细[代码](https://github.com/renjie-github/ToyProjects/tree/main/rest_api/flask/api)