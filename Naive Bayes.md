# Naive Bayes
朴素贝叶斯应该算生成模型吧？因为通过对样本的统计，我们可以得到$X$和$Y$的联合概率分布。
## 推导过程
大概解释下各符号含义：
$C_k$指$Y$有$1,2,3,...,K$类
$x$指需要预测的样本
$x^{(i)}$指需要预测的样本的第$i$个特征，每个样本一共有$n$个特征
$a_{il}$指第$i$个样本的第$l$个取值，一共有$S_i$个取值

$$
    \begin{aligned}
        P(Y=C_k|x)&=\frac{P(x|Y=C_k)P(Y=C_k)}{\sum_{k=1}^KP(x|Y=C_k)P(Y=C_k)} \\
        &=\frac{P(Y=C_k)\prod_{i=1}^n P(x^{(j)}=a_{jl}|Y=C_k)}{\sum_{k=1}^KP(Y=C_k)\prod_{i=1}^n P(x^{(j)}=a_{jl}|Y=C_k)}
    \end{aligned}
$$
其实比较明显的是，分母是个常数，因此有
$$P(Y=C_l|x)=P(Y=C_k)\prod_{i=1}^n P(x^{(i)}=a_{il}|Y=C_k)$$
然后用最大似然估计或者贝叶斯估计求$P(Y)$和$P(X|Y)$就是了，下面直接上贝叶斯估计
$$
    \begin{aligned}
        P(Y=C_k) &= \frac{\sum_{i=1}^nI(Y=C_k)+\lambda}{N+K\lambda} \\
        P(x^{(j)}=a_{jl}|Y=C_k)&=\frac{\sum_{i=1}^nI(x^{(j)}=a_{jl},Y=C_k)+\lambda}{\sum_{i=1}^nI(Y=C_k)+S_j\lambda}
    \end{aligned}
$$
训练的时候上面所有的这些$P$都算好，来了个需要预测的数据算一波看对应哪个$C_k$的时候最大就行了。一般$\lambda$取值为1，此时被称为拉普拉斯平滑，目的是为了让分母不为0