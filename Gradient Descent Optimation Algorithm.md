# 市面上各种乱七八糟的梯度下降类最优化方法
首先最基本的梯度下降，BGD和SGD就不提了。全文看这篇[博客](http://ruder.io/optimizing-gradient-descent/)
## Momentum
Momentum是一种加速SGD的方法，其思想在于当函数图像比较“扁”的时候，梯度下降最快的方向并没有直接指向最优点，导致迭代过程中波动很大，梯度下降缓慢。Momentum意指使用类似于球从山上滚落的理念来进行SGD，就是在往下滚的过程中会积累动量（类似于一个惯性），这样就能对SGD进行加速，公式为：
$$v_t=\gamma v_{t-1}+\eta\frac{\partial J(\theta)}{\partial\theta}$$
$$\theta=\theta-v_t$$
相当于通过乘一个系数$\gamma$的方式，将之前的梯度全部都累计起来了

## Nesterov accelerated gradient
NAG旨在考虑到这样一种情况，在梯度大的时候减小参数的变化量，在梯度小时增大参数的变化量，就是在函数图像陡的地方使参变化得慢点，因为此时本来就变得很快，以防止变大幅度太大直接跨越了极值点的情况，然后在函数图像比较平稳的地方使参数变化大点，以加快迭代速度，应该也是针对SGD的优化方法。公式为
$$v_t=\gamma v_{t-1}+\eta\frac{\partial J(\theta-\gamma v_{t-1})}{\partial\theta}$$
$$\theta = \theta-v_t$$

## Adagrad
Adagrad的思想是让不同的参数拥有不同的学习率，学习率大小根据参数出现的频率而定，出现频率小的参数学习率大，频率大的参数学习率小，因此它很适合于在稀疏数据上使用，并且大幅度的提升了SGD的鲁棒性，这也是在GloVe中使用Adagrad进行词向量训练的原因。
$$g_{t,i}=\frac{\partial J(\theta_{t,i})}{\partial\theta_{t,i}}$$
然后Adagrad对学习率进行了修正，其参数修改的方程如下
$$\theta_{t+1,i}=\theta_{t,i}-\frac{\eta}{\sqrt{G_{t,ii}+\epsilon}}\cdot g_{t,i}$$
其中$G_t$是一个对角阵，$ii$是它对角线上的元素，其值等于从$0$时刻到$t$时刻累计的$\theta_i$的平方和(显然这样学习率会越来越小)。
AdaGrad的优势是自适应学习率，同时也是其劣势，因为学习率会一直缩减直到0，因此有了Adadelta来解决这个问题

## Adadelta
Adadelta没有低效的累积梯度的平方，而是递归的定义了过去所有平方梯度的衰减均值，其定义如下。
$$E[g^2]_t=\gamma E[g^2]_{t-1}+(1-\gamma)g_t^2$$
然后把参数更新的方程以向量的形式表示
$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}g_{t}$$
容易知道，Adadelta不需要默认学习率，调参都省了

## RMSprop
Geoff Hinton在Coursera中提出的方法，其实就是Adadelta，然后Hinton建议默认$\eta = 0.001$，$\gamma = 0.9$

## Adam
Adam=适应性动量估计法，其实就是一个前面提到的adaxx算法与动量估计法结合的版本。
它同时储存了Adadelta和Momentum中的关键项，即
$$
    \begin{aligned}
        m_t&=\beta_1 m_{t-1}+(1-\beta_1)g_t \\
        v_t&=\beta_2 v_{t-1}+(1-\beta_2)g_2^2
    \end{aligned}
$$
然后因为Adam的作者观察到当$m_t$和$v_t$一开始被初始化为0向量时，他们会偏向于0（biased towards zero），尤其是在最初几次迭代和衰减率很小（$\beta_1$和$\beta_2$趋近于1）时。因此使用了方法来绣着这个偏差
$$
    \begin{aligned}
        \hat{m_t}&=\frac{m_t}{1-\beta_1^t} \\
        \hat{v_t}&=\frac{v_t}{1-\beta_2^t}
    \end{aligned}
$$
最后的参数修正方程（向量形式）为
$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t}$$
作者设置的默认参数为
$$
    \begin{aligned}
    \beta_1&=0.9 \\
    \beta_2&=0.999 \\
    \epsilon&=10^{-8}
    \end{aligned}    
$$

## 其他最优化方法
AdaMax, Nadam, AMSGrad
在作者给出的可视化示例中可以发现，Adadelta的效果已经特别好了，对于鞍点和函数十分扁平的地方都能处理的较好。