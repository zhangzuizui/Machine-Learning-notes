# OLS, L1 and L2
这篇note主要讲OLS，L1和L2正则是怎么推出来的，为什么work。
顺便的
[扩展阅读](https://cosx.org/2013/01/story-of-normal-distribution-1/)
[扩展阅读2](https://www.jianshu.com/p/a47c46153326)

## 高斯分布和拉普拉斯分布
高斯分布就是正态分布，其概率密度函数如下
$$f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
顺便的，二元高斯分布的概率密度如下
$$f(x,y)=\frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}}\exp(-\frac{1}{2(1-\rho^2)}(\frac{(x-\mu_X)^2}{\sigma_x^2}+\frac{(y-\mu_Y)^2}{\sigma_y^2}-\frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_x\sigma_y}))$$
顺便，$\rho$是皮尔逊相关系数
$$\rho=\frac{cov(X,Y)}{\sigma_x\sigma_y}$$
拉普拉斯分布跟高斯分布比较像，其概率密度函数如下
$$f(x)=\frac{1}{2b}\exp(-\frac{|x-\mu|}{b})$$
其函数图像与高斯分布不一样的地方是，在极大值点是一个“尖”，极值点不存在梯度。

##OLS的推导
假设有函数
$$f(x)=xw^T+\varepsilon$$
这里$\varepsilon$指的是误差
设$\varepsilon$服从高斯分布$N(0, \sigma^2)$，即$(y_i-x_iw^T)$服从分布$N(0, \sigma^2)$，那么就有$y_i$服从分布$N(x_iw^T,\sigma^2)$
现在求该函数的最大似然估计
$$
    \begin{aligned}
        \arg\underset{w}\max L(w) &= \prod_{i=1}^n\frac{1}{\sigma\sqrt{2\pi}}\exp(-\frac{(y_i-x_iw^T)^2}{2\sigma^2}) \\
        \arg\underset{w}\max\ln L(w)&=-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-x_iw^T)^2-n\ln\sigma\sqrt{2\pi} \\
        \arg\underset{w}\min\log L(w)&=\sum_{i=1}^n(y_i-x_iw^T)^2
    \end{aligned}
$$
也就是说，对于自然数据集来说，一般都假设其服从高斯分布，那么在这种情况下求平方误差实际上等价于求最大似然估计。

##L1，L2推导
L1正则和L2正则的推导过程实质上就是在对上面OLS推导的基础上，从求函数的最大似然估计变成了求最大后验概率估计。
假设$w$服从拉普拉斯分布
$$
    \begin{aligned}
        \arg\underset{w}\max L(w) &= \prod_{i=1}^n\frac{1}{\sigma\sqrt{2\pi}}\exp(-\frac{(y_i-x_iw^T)^2}{2\sigma^2})\prod_{j=1}^m\frac{1}{2b}\exp(-\frac{|w_j|}{b})\\
        \arg\underset{w}\max\ln L(w)&=-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-x_iw^T)^2-n\ln\sigma\sqrt{2\pi} -\sum_{j=1}^m\frac{|w_j|}{b}-m\ln 2b\\
        \arg\underset{w}\min\log L(w)&=\sum_{i=1}^n(y_i-x_iw^T)^2+\frac{1}{b}\sum_{j=1}^m|w_j|
    \end{aligned}
$$

同理，在$w$服从高斯先验时可以得到
$$\arg\underset{w}\min\log L(w)=\sum_{i=1}^n(y_i-x_iw^T)^2+\frac{1}{2\sigma_w^2}\sum_{j=1}^mw_j^2$$

##L1,L2的优势
L1：得到$w$的稀疏矩阵；
L2：对模型增加一个结构风险，防止过拟合。

**为什么L1能够使$w$变稀疏**
首先有个很重要的点是，在什么情况下某点为损失函数$L$的极值点：
$$\frac{\partial L}{\partial w}|_{w=k}=0$$
或者
$$sign(\frac{\partial L}{\partial w}|_{w=k^-})!=sign(\frac{\partial L}{\partial w}|_{w=k^+})$$
假设在引入正则项以前
$$\frac{\partial L}{\partial w}|_{w=0}=m$$
即$w=0$的点不是其极值点，在引入L1正则后，有
$$\frac{\partial L+\lambda|w|}{\partial w}|_{w=0^-}=m-\lambda$$
$$\frac{\partial L+\lambda|w|}{\partial w}|_{w=0^+}=m+\lambda$$
如果有
$$m-\lambda<0 \quad \&\& \quad m+\lambda>0$$
即
$$|m|<\lambda$$
那么$w=0$这个点就会变成极值点，参数就会变稀疏。
反之如果是L2正则，如果在添加正则项以前$w=0$不是$L$的极值点，那么增加正则项后依旧不会是极值点（因为$w=0$处的梯度不会变）