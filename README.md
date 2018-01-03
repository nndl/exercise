# 《神经网络与深度学习》课程练习

复旦大学 计算机学院 COMP630068
复旦大学 大数据学院 DATA130011 

Neural Network and Deep Learning

### 环境设定
本次作业需要首先安装 anaconda3 下载地址 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ ， 安装的tensorflow 版本大于0.12即可，目前的版本都远大于0.12。 windows 用户，使用tensorflow 只支持python3.5 ，对应的anaconda3-4.2.0。mac 或者linux系统没有这个问题。

## Exercise 

### exercise 1 - warmup
本作业主要是让大家熟悉numpy.numpy 是一个很实用的数据科学计算的工具，它集成了很多矩阵操作和数学函数，对我们的学习和研究都很有帮助。本作业的内容是按照题目的文字要求，填写对应的语句，然后执行自己填写的语句。最后需要保存自己的ipynb 文件，并放到自己的压缩包里面。

### exercise 2 - simple neural network
本作业主要是熟悉写简单的神经网络的方法。尝试用numpy 和tensorflow的两种方法实现这个神经网络。

1. 第一个文件是numpy实现的全连接神经网络，缺少训练函数部分，主要内容就是更新权重,请同学们补充完整。

2. 另一个文件是用tensorflow实现的全连接神经网络，交叉熵损失函数的节点和训练函数的节点未写出，请同学们补充完整。

3. 数据集 NMIST数据集

   MNIST数据集包括60000张训练图片和10000张测试图片。图片样本的数量已经足够训练一个很复杂的模型（例如 CNN的深层神经网络）。它经常被用来作为一个新 的模式识别模型的测试用例。而且它也是一个方便学生和研究者们执行用例的数据集。除此之外，MNIST数据集是一个相对较小的数据集，可以在你的笔记本CPUs上面直接执行。

### exercise 3 - convolutional neural network

### exercise 4 - recurrent neural network

### exercise 5 - restricted Boltzmann machine



## Seminar Project

### project 1 - deep reinforcement learning



## 提交要求

作业需要提交到elearning 上面，提交作业的时候，请提交一个“14300000001.zip”压缩文件即可。
该作业需要同学们独立完成，如果发现两个人提交的完全一样，本次作业将记0分处理。

每次练习请在截止日期之前提交。过期提交请在压缩文件名的学号前面注明 “迟交”，（例如 “迟交14300000001.zip”）迟交将会扣除一定分数，迟交的最后截止日期是为截止日期之后的2日内，此后提交的将会不得分。（如果有特殊情况，确实提交不了，请预先发邮件给TA）

### 报告

 你需要提交你的源代码（请提交 ipynb 格式的文件）和你的报告(格式为pdf文件)。报告包括 至少包括 “运行结果的截图” + 讨论分析。

如果尝试了新的办法，提高了预测精度，可以加分。	