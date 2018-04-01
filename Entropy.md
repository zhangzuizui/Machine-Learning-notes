# Entropy
熵是表示随机变量不确定性的度量，熵越大，随机变量的不确定性越大:
$$ H(p) = -\sum_{i=1}^n p_i\log p_i $$
若随机变量$p$只有两个值，那么绘制成函数图像的话是一个典型的凹函数。
**（有空插个图）**
## 条件熵
条件熵的定义为，$X$给定条件下$Y$的条件概率分布的熵，对$X$的数学期望，即
$$H(Y|X) = \sum_{i=1}^np_iH(Y|X=x_i)$$
其也可以表示为，$X$和$Y$的联合分布的熵减去$X$的部分，即
$$H(Y|X) = H(X,Y) - H(X)$$
推导过程大致是
$$
    \begin{aligned}
    H(X, Y) - H(X)&=-\sum_{x,y}P(x,y)\log P(x,y) + \sum_{x}P(x)\log P(x)\\
    &=\sum_x(P(x)\log P(x)-\sum_yP(x,y)\log (P(y|x)P(x)))\\
    &=\sum_x(P(x)\log P(x)-\sum_yP(x,y)\log P(y|x)-\sum_yP(x,y)\log P(·x))\\
    &=-\sum_{x,y}P(x,y)\log P(y|x)
    \end{aligned}
$$

## 信息增益
在已知$A$的条件下，信息增益为：
$$g(D,A) = H(D) - H(D|A)$$

## 信息增益比
特征$A$的信息增益与训练集$D$关于特征值$A$的熵的比，
$$g_R(D,A) = \frac{g(D,A)}{H_A(D)}$$
其中，
$$H_A(D) = -\sum_{i=1}^n\frac{|D_i|}{|D|}\log \frac{|D_i|}{|D|}$$
即特征值$A$一共有$n$个取值，通过这$n$个取值，把数据集$D$分成了$n$个部分。

## 交叉熵
$$H(p, q) = -\sum_{x}p(x)\log q(x)$$
其中$p$是数据的真实分布情况，$q$是非自然（预测）的分布。 

# 熵值法
评价数据离散程度的方法，熵越小，离散程度越大，指标对综合评价的影响越大（理解：当所有样本的概率都相等时，熵最大，此时该指标对结果无任何影响甚至可以剔除）。

$$E_i = -\frac{1}{\log n}\sum_{i=0}^ny_i\log y_i$$
其中$y_i$是第$i$个样本的特征$y$占所有样本的特征$y$的比重。

$$w_i=\frac{1-E_i}{\sum_{j=0}^m(1-E_j)}$$
归一化一下即可得到第$i$个特征的权重。

# 最大熵模型
给我的直观感觉是最大熵模型实际上是在最大化后验概率的条件熵，即
$$H(p)=-\sum_{x,y}P(x,y)\log P(y|x)$$
利用经验概率对其进行转换，其中
$E_p(f)=\sum_{x,y} \hat P(x)P(y|x)f(x,y)$
$E_{\hat p}(f)=\sum_{x,y}\hat P(x,y)f(x,y)$
其中$f$是一个二值函数，满足某条件就得1，否则为0，是$x$和$y$的特征函数
$$
    \begin{aligned}
        &\max H(p)=-\sum_{x,y}\hat P(x)P(y|x)\log P(y|x) \\
        &\begin{aligned}
            s.t. \quad &\sum_{i=1}^nE_{\hat P}(fi)-E_P(f_i)=0, \quad i=1,2,...,n \\
            &\sum_{y}P(y|x)=1
        \end{aligned}
    \end{aligned}
$$
然后用拉格朗日乘子法将其转化为对偶问题进行求解。

而经验概率分布$P(X,Y)$的最大似然函数为
$$L(p)=\prod _{x,y}\hat P(x,y)^{\hat P(x,y)}=\prod _{x,y}(\hat P(x)P(y|x))^{\hat P(x,y)}=\prod _{x,y}P(y|x)^{\hat P(x,y)}$$
然后进行求解。实质上与最大熵模型的对偶函数是等价的。

