#<center> Linear Regression </center> #

### 介绍

**回归**是一种估计变量之间关系的机制，而线性关系是变量之间最为简单的一种关系，因此线性回归即是估计变量之间存在的线性关系的模型。

### Simple linear regression

简单线性回归拟合的线为 y=wx+b​，x 称为自变量（**independent variable or predictor variable**），y​称为因变量（**dependent  variable or response variable**）。

- w为直线的斜率（**slope**），w趋近于0时候表示两者并无线性关系，绝对值越大表明有很强的线性关系。
- b为直线的截距（**intercept**）

线性回归的损失函数为：
$$
\begin{array}{l}{\mathrm{L}(w, b)} \\ {=\sum_{n}\left(\hat{y}^{n}-\left(b+w \cdot x^{n}\right)\right)^{2}}\end{array}
$$
故目标函数与对应的参数为：
$$
\begin{aligned} f^{*}=& \arg \min _{f} L(f) \\ w^{*}, b^{*} &=\arg \min _{w, b} L(w, b) \\ &=\arg \min _{w, b} \sum_{n}\left(\hat{y}^{n}-\left(b+w \cdot x^{n}\right)\right)^{2} \end{aligned}
$$
接着使用最小二乘法或者梯度下降的方法来进行求解即可。

在进行建模之前，应该先根据样本来看变量之间是否线性相关，可以根据画出散点图或者计算相关系数。相关系数的计算公式如下：
$$
r=\frac{1}{n-1} \sum_{i=1}^{n}\left(\frac{x_{i}-\overline{x}}{s_{x}}\right)\left(\frac{y_{i}-\overline{y}}{s_{y}}\right)
$$
其中$S_x$和$S_y$表示样本的标准差，$\bar{x}$和$\bar{y}$表示均值。相关系数在-1到1之间，而相关系数的平方始终为正，称为**coefficientof  determination**。

**注1：R-squared：equal  to  the  proportion  of  the  totalvariability that’s explained by a linear model.**

**注2:相关（correlation）不等于导致（causation）**，如溺水和冰淇淋的销量是correlation，因为都是在天气炎热的情况下发生的，但是不能说是溺水导致冰淇淋销量增高。

### Multiple Linear Regression

多元线性回归是对于每个数据点不是标量而是一个向量，假设有n个数据，每个数据有p个特征，因此目标函数为
$$
y = w_1x_1+w_2x_2+\cdots+w_px_p
$$
使用矩阵的方式来表示目标函数则有：
$$
y = X\beta+b
$$
其中$\beta$是一个p维的系数向量，b是表示含有n个元素的矩阵，同简单线性回归中b的含义相同。因此损失函数为：
$$
\min _{\beta} \sum_{i=1}^{n}\left(y_{i}-X_{i} \beta\right)^{2}
$$
使用最小二乘法和线代的基本知识可以计算出结果为：
$$
\hat{\beta}=\left(X^{T} X\right)^{-1} X^{T} y
$$

### Model Evaluation





















### 参考

[1.MIT linear regression](http://www.mit.edu/~6.s085/notes/lecture3.pdf)

[2.李宏毅ppt](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/Regression.pdf)

