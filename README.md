<p align = "center"> <img width = "350％"  src = "logo/logo.jpeg"/> </p>


# 机器学习实战笔记(附:[源代码](https://github.com/shiyipaisizuo/machine_learning_in_action/tree/master/src)) 基于 *[GNU3.0](https://github.com/shiyipaisizuo/machine_learning_in_action/blob/master/LICENSE)* 协议

## 第一部分 分类

### 第一章 机器学习基础([代码](https://github.com/shiyipaisizuo/machine_learning_in_action/tree/master/src/ch01))

- **熟悉[Python](https://www.python.org/)即可。**
- **开发机器学习应用程序步骤**

    - <p>1.收集数据。</p>
    - <p>2.准备输入数据。</p>
    - <p>3.分析输入数据。</p>
    - <p>4.训练算法。</p>
    - <p>5.测试算法。</p>
    - <p>6.使用算法。</p>
   
- **掌握[numpy](http://www.numpy.org/)函数库基础**

    `>> from numpy import *`

### 第二章 K-近邻算法([代码](https://github.com/shiyipaisizuo/machine_learning_in_action/tree/master/src/ch02))

- **K-近邻算法优缺点**

    - 优点:精度高,对异常值步敏感，无数据输入假定。
    - 缺点:计算复杂度高，空间复杂度高。
    - 范围:数值型和标称型。
    
    
- **测试分类器**

    **`错误率是常用的评估方法，完美评估器为0，最差的评估器为1.0`**

#### 例子:使用k-近邻算法改进约会网站的配对效果

- 准备数据:从文本数据中解析出数据，用`numpy`转化文本为矩阵，同时进行归一化数值操作(将对数据有影响的数值归纳为`0～1`之间)。
- 分析数据:使用`matplotlib`实现数据可视化。
- 测试数据:错误评估 **`训练数据/测试数据 = 90%/10%`**。
- 使用算法:基于用户的输入，自动匹配。

#### 例子:手写识别系统

- 准备数据:将图像分为`32*32`的二进制图像转化为`1*1024`的数组，每次读取`32`行,存入数组，并且返回数组。
- 分析数据:确保数据准确无误。
- 测试数据:随机选取数据测试。
- 使用数据:将评估错误率，选择`最低评估错误率`来作为首选算法。

#### 小节

**K-近邻算法是最简单的分类算法，如果数据量太大，会变得非常耗时。**

### 第三章 决策树([代码](https://github.com/shiyipaisizuo/machine_learning_in_action/tree/master/src/ch03))

- **决策树算法优缺点**

    - 优点:计算复杂度不高，输出结果易于理解，对中间值不明干，可以处理不想管特征数据。
    - 缺点:可能会产生过度匹配。
    - 范围:数值型和标称型。
    
