#Optimal method
## IIS
IIS的思路是，对于原优化问题$L(w_1,w_2,...,w_n)$找到一个诸如这样的新的方法$w_i\leftarrow w_i+\delta_i$进行更新，IIS适用于最大熵模型的优化。
因此有
$$L(w+\delta)-L(w)=blabla$$
通过不断的优化此函数的下界，以求解函数下界最大时候的$\delta$值的方法来进行迭代，这个推导还蛮麻烦的。。感觉要使用到它的情况极少？具体推导参考李航《统计学习方法》P89。

## 牛顿法
对于函数$f(x)$在$x=x^{(k)}$处泰勒二阶展开，得到
$$f(x)=f(x^{(k)})+f'(x)(x-x^{(k)})+\frac{1}{2}f''(x)(x-x^{(k)})^2$$
将它变换一下，就有
$$f(x)=f(x^{(k)})+g_k^T(x-x^{(k)})+\frac{1}{2}(x-x^{(k)})^TH_k(x-x^{(k)})$$
其中$H_k$是海森矩阵
$$H(x)=[\frac{\partial^2 f}{\partial x_i\partial x_j}]_{n\times n}$$
在点$x^{(k)}$的值
然后讨论一下在什么情况下$f(x)$有极值，$f(x)$有极值的必要条件是$\nabla f(x)=0$，也就是说当$H(x)$是正定矩阵或者负定矩阵时，函数有极小/极大值。
设每次迭代从$x^{(k)}$开始，然后有
$$\nabla f(x^{(k+1)})=0$$
因为
$$\nabla f(x)=g_k+H_k(x-x^{(k)})$$
所以
$$g_k+H_k(x^{(k+1)}-x^{(k)})=0$$
这样其实就得到了求解$x^{(k+1)}$的方程
$$x^{(k+1)}=-H_k^{-1}g_k+x^{(k)}$$
牛顿法的思路就是对海森矩阵求逆矩阵，进而实现更新$x$再更新$g$，$H$的循环。

## 拟牛顿法
拟牛顿法的出现主要是因为牛顿法在迭代过程中需要求解海森矩阵的逆阵，这极其影响模型的收敛速度。
现在考虑这个式子
$$\nabla f(x)=g_k+H_k(x-x^{(k)})$$
易知$\nabla f(x)$其实就是$g$，因此当$x=x^{(k+1)}$时，有
$$g_{k+1}-g_k=H_k(x^{(k+1)}-x^{(k)})$$
记住这个式子，这个式子被称为是拟牛顿条件，这是我认为牛顿法与拟牛顿法最大的不同之处。牛顿法的做法是先算$x^{(k)}$时的海森矩阵，再求其逆阵，然后计算$x^{(k+1)}$，而拟牛顿法的思路是，想办法先找到新的$g$和$x$再计算出当前状态的"海森矩阵"（这里打引号的原因是，实际上以这种方式计算出来的并不是海森矩阵了，这一点很容易想到，因为在推导过程中是通过对原函数做泰勒展开进行的，每多做一次处理都会有精度上的损失，因此拟牛顿法只是对海森矩阵（或海森矩阵的逆矩阵的近似求解））。

在牛顿法的推导过程中，我们有推出这个公式
$$x^{(k+1)}=-H_k^{-1}g_k+x^{(k)}$$
如果海森矩阵是正定的，那么可以保证牛顿法的搜索方向是下降的方向（每次迭代的目标都是奔着$\nabla f(x)=0$去的，当然下降了。。）所以我们其实可以写成这个样子（我这里说的其实很不严谨，并且感觉李航在书上写的也特别的简单，具体的证明还是得网上再找资料，不过大概的了解拟牛顿法的话这样就够用了）
$$x=-\lambda H_k^{-1}g_k+x^{(k)}$$
将这个带入到$f(x)$里面，并在$x^{(k)}$处泰勒一阶展开，能得到
$$f(x)=f(x^{(k)})-\lambda g_k^TH_k^{-1}g_k$$
这个式子就特别棒了，因为$H_k$是正定矩阵，所以$g_k^TH_k^{-1}g_k>0$，于是当$\lambda$足够小的时候，就一定有$f(x)<f(x^{(k)})$，也就是说$f(x)$在一直减小。

整理一下思路，假设在迭代到某一步的时候，我们计算出了此时的$H_k^{-1}$，$x^{(k)}$，$g_k$，现在需要做的就是找$x^{(k+1)}$，怎么找？通过对$\lambda$做线性搜索，求$\lambda_k$
$$f(x^{(k)}-\lambda_k H_k^{-1}g_k)=\underset{\lambda\geq0}\min f(x^{(k)}-\lambda H_k^{-1}g_k)$$
因为之前提到过当$\lambda$足够小的时候，一定能使$f(x)$下降，所以找到这个足够小的$\lambda$就行了，然后就能得到$$x^{(k+1)}=x^{(k)}-\lambda_k H_k^{-1}g_k$$
$x^{(k+1)}$都有了，通过求导数就能得到$g_{k+1}$
现在的任务就是求解$H_{k+1}^{-1}$了

### DFP与BFGS
DFS和BFGS是拟牛顿法的两种不同实现，先接着上面的内容来说DFP，DFP的思想是在于不停的近似求解$H_{k}^{-1}$，现在我们称其为$G_k$，因为需要想办法迭代的计算$G_k$，所以很自然的就有
$$G_{k+1}=G_k+\Delta G_k$$
不妨设
$$G_{k+1}=G_k+P_k+Q_k$$
根据之前的拟牛顿条件，我们知道
$$H_{k}^{-1}(g_{k+1}-g_k)=x^{(k+1)}-x^{(k)}$$
其中$g_{k+1}$和$x^{(k+1)}$我们都已经通过计算得到了，现在将这个式子进行变换
$$G_ky_k=\delta_k$$
结合上面的式子
$$G_{k+1}y_k=G_ky_k+P_ky_k+Q_ky_k$$
为了使之满足拟牛顿条件，可以使
$$P_ky_k=\delta_k$$
$$Q_ky_k=-G_ky_k$$
比如取
$$P_k=\frac{\delta_k\delta_k^T}{\delta_k^Ty_k}$$
$$Q_k=-\frac{G_ky_ky_k^TG_k}{y_k^TG_ky_k}$$

BFGS是目前效果最好的拟牛顿法，它的思路是既然可以用$G_k$求海森矩阵逆阵的近似矩阵，那么自然也可以用$B_k$去拟合海森矩阵，这一点看起来有点奇怪，最终还不是要求一个$B_k^{-1}$？不然怎么求$x^{(k+1)}$？，其实是因为使用Sherman-Morrison公式可以将$B_k$转化为$G_k$，当然这样的转化求得的$G_k$与DFP中直接求的$G_k$是不等价的，所以BFGS的效果最好。

对于牛顿法我只了解到了这里，对于之前提到的一些疑问与我不清楚的地方可自行探究。