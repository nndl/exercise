# 线性回归



## 问题描述：

有一个函数![image](http://latex.codecogs.com/gif.latex?f%3A%20%5Cmathbb%7BR%7D%5Crightarrow%20%5Cmathbb%7BR%7D) ，使得。现 ![image](http://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29)在不知道函数![image](http://latex.codecogs.com/gif.latex?f%28%5Ccdot%20%29) 的具体形式，给定满足函数关系的一组训练样本![image](http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cleft%20%28%20x_%7B1%7D%2Cy_%7B1%7D%20%5Cright%20%29%2C...%2C%5Cleft%20%28%20x_%7BN%7D%2Cy_%7BN%7D%20%5Cright%20%29%20%5Cright%20%5C%7D%2CN%3D300)，请使用线性回归模型拟合出函数![image](http://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29)。(可尝试

一种或几种不同的基函数，如多项式、高斯或sigmoid)






## 数据集: 

 	根据某种函数关系生成的train 和test 数据。





## 题目要求： 

- [ ] 使用训练集train.txt 进行训练，使用测试集test.txt 进行评估（标准差），训练模型时请不要使用测试集。
- [ ] 请使用线性回归模型解决此问题，不建议使用神经网络等其他方法。
- [ ] 请使用代码模板linear_reg.py，补全其中缺失的代码。尽量不要改动主要接口，可自由添加所需的函数。
- [ ]  推荐使用python 及numpy 编写代码，教程可参考cs231n-numpy-tutorial。
