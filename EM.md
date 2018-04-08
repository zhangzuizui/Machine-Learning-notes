# EM
EM算法是用于含有隐变量的概率模型参数的极大似然估计或极大后验概率估计。
## EM算法的推导
这些公式的推导过程trick极多。。我是几乎记不住。难受啊。
设模型的参数为$\theta$，观测为$Y$，那么我们的对数似然函数是
$$L(\theta)=\log P(Y|\theta)$$
但是！如果该模型中含有隐变量，那么就无法直接进行计算，需要通过完全数据间接计算，设隐变量为$Z$
$$
    \begin{aligned}
        L(\theta)&=\log \int_ZP(Y,Z|\theta)dZ\\
        &=\log\int_ZP(Y|Z,\theta)P(Z|\theta)dZ
    \end{aligned}
$$
跟$IIS$类似的，考虑用迭代的方法求参数，那么有
$$
    \begin{aligned}
        L(\theta)-L(\theta^{(i)})&=\log\int_ZP(Y,Z|\theta)dZ-\log P(Y|\theta^{(i)}) \\
        &=\log\int_ZP(Y|Z,\theta)P(Z|\theta)dZ-\log P(Y|\theta^{(i)}) \\
        &=\log\int_ZP(Z|Y,\theta^{(i)})\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})}dZ-\log P(Y|\theta^{(i)}) \\
        &\geq \int_ZP(Z|Y,\theta^{(i)})\log\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})}dZ-\log P(Y|\theta^{(i)})\quad \text{(詹森不等式)} \\
        &=\int_ZP(Z|Y,\theta^{(i)})\log\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})}dZ-\int_ZP(Z|Y,\theta^{(i)})\log P(Y|\theta^{(i)})dZ \\
        &=\int_ZP(Z|Y,\theta^{(i)})\log\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})} dZ \\
        &=\int_ZP(Z|Y,\theta^{(i)})\log\frac{P(Y,Z|\theta)}{P(Z,Y|\theta^{(i)})} dZ \\
        &=B(\theta,\theta^{(i)})
    \end{aligned}
$$
可以知道$B(\theta,\theta^{(i)})$是$L(\theta)-L(\theta^{(i)})$的下界，并且由$B$的公式不难知道$B(\theta^{(i)},\theta^{(i)})=0$，因此如果存在$\theta^{(i+1)}$使$B$函数增大，那么同样的$\theta^{(i+1)}$也有机会使$L(\theta)-L(\theta^{(i)})$增大，于是有
$$
    \begin{aligned}
        \theta^{(i+1)}&=\arg\underset{\theta}\max B(\theta,\theta^{(i)}) \\
        &=\arg\underset{\theta}\max \int_ZP(Z|Y,\theta^{(i)})\log\frac{P(Y,Z|\theta)}{P(Z,Y|\theta^{(i)})} dZ \\
        &=\arg\underset{\theta}\max \int_ZP(Z|Y,\theta^{(i)})\log P(Y,Z|\theta)dZ-\int_ZP(Z|Y,\theta^{(i)})\log P(Z,Y|\theta^{(i)})dZ \\
        \text{省去常数项}&=\arg\underset{\theta}\max \int_ZP(Z|Y,\theta^{(i)})\log P(Y,Z|\theta)dZ
    \end{aligned}
$$
其实现在就得到EM算法的核心$Q$函数了
$$Q(\theta,\theta^{(i)})=\int_ZP(Z|Y,\theta^{(i)})\log P(Y,Z|\theta)$$
这个东西咋一看，其实就是$\log P(Y,Z|\theta)$对于$Z$的条件概率分布$P(Z|Y,\theta^{(i)})$下的期望嘛，
$$Q(\theta,\theta^{(i)})=E_Z(\log P(Y,Z|\theta)|Y,\theta^{(i)})$$
然后自然的也能得到EM算法的迭代流程了：
1. 初始化$\theta^{(0)}$
2. 计算$Q(\theta,\theta^{(i)})$
3. 求$\theta^{(i+1)}$
4. 2，3循环起来
需要注意的是EM算法的效果受初始参数的影响很大，可常用的方法是多设置几个初始值，然后选最好的。

## 高斯混合模型
emmm懒得写了，对高斯混合模型参数的求解使用EM算法其实就是，
**参数**是选择各高斯模型的概率和各高斯模型的参数
**隐变量**是观察到的某组观测数据来自于哪个高斯模型
然后带到$Q$函数里算就是了。