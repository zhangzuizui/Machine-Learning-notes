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
这个问题其实还蛮复杂的，靠谱的解释很少， 不过这篇[博文](https://blog.csdn.net/zouxy09/article/details/24971995)不错，虽然感觉也略难懂。
L1：得到$w$的稀疏矩阵(特征选择)，增加数据的可解释性，也有能防止模型过拟合的作用；
L2：L2范数不但可以防止过拟合，还可以让我们的优化求解变得稳定和快速。
研究了很久都没有找到想到的这样的答案：就是类似于在防止过拟合的性能上L1与L2的对比。首先可以肯定的是L1 L2都能在一定程度上防止模型过拟合。但是到底谁的效果好？为什么在想要对模型做防止模型过拟合处理的时候，普遍选择的是L2而不是L1？上面的博文里在讲L2的地方隐隐有说了那种感觉，大概就是因为增加了L2正则以后，使解从无穷多个变成了一个（让矩阵满秩），这或许是防止模型过拟合的关键一点，然后还有一个就是对于使用梯度下降类的算法，加快了模型的收敛速度；而L1正则对于防止模型过拟合上面的贡献看起来像是在做特征选择的时候顺带的。所以需要做特征选择的时候就用L1，需要用于梯度下降/防止模型过拟合的时候就用L2?

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