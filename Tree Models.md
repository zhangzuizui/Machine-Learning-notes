# Tree Models
## Decision Tree
决策树是一个典型的判别模型，通过学习已知数据的条件概率 $ P(Y|X) $ 来对未知数据做预测，看这篇笔记之前需要先看一哈Entropy的基础知识。
### ID3
设当前被分到某结点的数据集为$D$，数据维度为$K$维，即数据共有$K$个特征，用$A$表示特征。根据信息增益公式：
$$g(D,A)=H(D)-H(D|A)$$
找到使信息增益最大的特征$A_g$，如果$g(D,A)$小于$\epsilon$,则停止分裂，否则根据$A_g=a_i$将$D$分割为$D_i$(即分裂出$k$个子结点)，每个结点的值为$D_i$中label个数最多的那一个。

### C4.5
C4.5与ID3类似，唯一不同的是特征选择的依据从信息增益变为了信息增益比，即：
$$g_R(D,A) = \frac{g(D,A)}{H_A(D)}$$

### 决策树减枝
剪枝的目的是增强决策树的泛化能力。决策树的损失函数为：
$$C_\alpha(T)=\sum_{t=1}^{|T|}N_tH_t(T)+\alpha|T|$$
其中$|T|$指决策树中叶子结点的个数，$N_t$表示第$t$个结点的样本数，$H_t(T)$是这个东西：
$$H_t(T)=-\sum_{k=0}^n\frac{|N_{tk}|}{|N_t|}log\frac{|N_{tk}|}{|N_t|}$$
其中，$N_{tk}$表示结点$t$中$k$类样本的数量。
剪枝时从叶结点递归的向上遍历，只要剪枝后$C_\alpha(T)$减少，那么就进行剪枝。
## CART
给定数据集：
$$D=\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$$
### 回归树
回归树的生成实质上用的是最小二乘法，在树的叶子结点分裂时，采用启发式算法来选择最优分裂情况，具体如下：

令$j$为用以切分的$x$的特征，$s$为用以切分的切分点，目标是：
$$min_{j,s}[min_{c_1}\sum_{x_i\in{R_1(s,j)}}(y_i-c_1)^2+min_{c_2}\sum_{x_i\in{R_2(s,j)}}(y_i-c_2)^2]$$
```python
# X是一个m维的向量，意思是数据的特征数
for j in range(m): 
    # 对某一维特征中的所有样本，按x进行排序, 
    # 其中x = [[x_1, y_1], [x_2, y_2]...]
    x = sort(X[j]) 
    # n指样本容量
    for s in range(n): 
        # 对于排序后的每一个x的值，
        # 可以将数组x划分为R_1, R_2两部分
        R_1 = x[:s+1]
        R_2 = x[s+1:]
        # c为区域内y的均值
        c_1 = sum([y[1] for y in R_1])
        c_2 = sum([y[1] for y in R_2])
        # 选出最小val对应的s和j进行分裂即可
        val = sum([(v-c_1)**2 for [y[1] for y in R_1]])+
              sum([(v-c_2)**2 for [y[1] for y in R_2]])
```

### 分类树
在分类问题中，使用基尼指数来选择最优特征，
$$Gini(p)=1-\sum_{k=1}^Kp_k^2$$
其中，样本点一共有$K$类，$p_k$是第$k$类样本点的概率。

对于给定的样本集合$D$，其基尼系数为：
$$Gini(D)=1-\sum_{k=1}^K(\frac{|C_k|}{|D|})^2$$
其中$|C_k|$是第$k$类的样本数。

在特征$A$下的基尼系数为：
$$Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)$$
其中，根据特征$A$的取值，将样本$D$划分为$D_1$和$D_2$（同样是启发式方法）。

### CART剪枝
与前面提到的剪枝方法类似的，CART剪枝也有以下损失函数：
$$C_\alpha(T)=\sum_{t=1}^{|T|} N_{t}H_t(T)+\alpha|T|$$
令$C(T)=\sum_{t=1}^{|T|} N_{t}H_t(T)$有：
$$C_\alpha(T)=C(T)+\alpha|T|$$
对于结点$t$，以$t$为单结点的损失函数是：
$$C_\alpha(t)=C(t)+\alpha$$
以$t$为根结点的子树$T_t$的损失函数是：
$$C_\alpha(T_t)=C(T_t)+\alpha|T_t|$$
易知当$C_\alpha(t)=C_\alpha(T_t)$时，有：
$$\alpha=\frac{C(t)-C(T_t)}{|T_t|-1}$$
即只要$\alpha=\frac{C(t)-C(T_t)}{|T_t|-1}$那么就有$C_\alpha(t)=C_\alpha(T_t)$，但是因为$t$比$T_t$的结点数少，所以$t$比$T_t$更可取，于是对$T_t$剪枝。

因此，对于树$T_0$中的每一个结点计算
$$g(t)=\frac{C(t)-C(T_t)}{|T_t|-1}$$
以表示剪枝后整体损失函数减少的程度，每次剪枝剪去$g(t)$最小的那个结点，然后将剪枝后的树$T_k$作为下一次迭代时的$T_0$直到$T_k$是由根节点及两个叶子结点构成的树，最后采用交叉验证法从$K$棵树中选择最优的那一个。

### RandomForest
#### 建模过程
没太多好说的，实际上就是“采样版”的CART，遵循两个采样原则
1. 每次建树时，有放回的从所有观测数据中进行随机采样
2. 每次做结点分裂时，随机选择一个特征进行分裂
#### 特征选取
OOB: out of bag?应该是这个，即袋外数据，比如每次建树时抽样1/3的数据，那么剩下的大约2/3的数据就是这棵树的OOB
1. 计算树的袋外数据误差$erroob_{t1}$（OOB的存在是随机森林的一大优势，每棵树自带test case）
2. 对所有样本的某特征引入随机噪声，计算$erroob_{t2}$
3. $score = \frac{1}{|T|}\sum_{t=1}^{|T|}erroob_{t1}-erroob_{t2}$ 

**顺便提一下GBDT和Xgboost的特征重要性判断**
GBDT和Xgboost每个结点分裂的时候，都是根据某特征进行分类的，把最终模型的每类特征进行分裂时减少的损失分别求和，然后进行比较即可得到特征的重要性排名。


## Boosting Algorithm
提升算法这个说法，来源于对弱学习算法和强学习算法的思考，容易知道的是弱学习方法的学习容易，而强学习算法难。将弱学习算法“提升”为强学习算法的算法，谓之提升算法。

### AdaBoost
考虑对于一个二分类的数据集
$$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$$
建模$M$次并将$M$个模型组合起来。

1. 先初始化训练数据的权重分布
$$D_1=(w_{11},w_{1i},...,w_{1N}), w_{1i}=\frac{1}{N},i=1,2,...,N$$
2. 对$m=1,2,...,M$
2.1 设定规则将数据集分类预测，
$$G_m(x):\chi\to\{-1,+1\}$$
2.2 计算分类误差率
$$e_m=P(G_m(x)\neq y_i)=\sum_{i=1}^Nw_{mi}I(G_m(x)\neq y_i)$$
2.3 计算$G_m(x)$的系数
$$\alpha_m=\frac{1}{2}log\frac{1-e_m}{e_m}$$
2.4 更新权重分布
$$w_{m+1,i} = \frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i))$$
其中，$Z_m$是规范化因子
$$Z_m=\sum_{i=1}^Nw_{mi}exp(-\alpha_my_iG_m(x_i))$$
3. 构建分类器线性组合
$$f(x)=\sum_{i}^M\alpha_mG_m(x)$$
得到
$$G(x)=sign(G_m(x))$$
实际上AdaBoost的训练过程就是不断，找当前分类器下分类误差最小的判断方式，更新权重分布，算系数然后求新的分类器如此循环的过程。

### Boosting Tree
在提升树模型中，我们要求得的最终分类器是
$$f_M(x)=\sum_{m=1}^MT(x;\Theta_m)$$
使用前向分布算法，有
$$f_{m}(x)=f_{m-1}(x)+T(x;\Theta_{m})$$
决策树训练的目标就是
$$\Theta_m={\arg\underset{\Theta_m}\min}L(y_i,f_{m-1}(x)+T(x;\Theta_m)$$

对于回归问题，可以用平方误差做误差函数，即
$$ 
    \begin{aligned}
        &L(y_i,f_{m-1}(x)+T(x;\Theta_m) \\
        &=[y_i-f_{m-1}(x)-T(x;\Theta_m)]^2 \\
        &=[r-T(x;\Theta_m)]^2\\
    \end{aligned}
$$
其中，
$$r=y_i-f_{m-1}(x)$$
也就是说$T(x;\Theta_m)$只用一直去拟合当前模型的残差$r$即可。

### GBDT
GBDT既然叫梯度提升树，那么明显的就是对梯度的拟合了。

BUT：WHY？为什么不停的拟合梯度就能得到最终模型？推导如下
与提升树类似的，有
$$f_{m}(x)=f_{m-1}(x)+T(x;\Theta_{m})$$
那么将损失函数一阶泰勒展开
$$
    \begin{aligned}
        L(y_i,f_m(x))&=L(y_i,f_{m-1}(x)+T(x;\Theta_{m}))\\
        &\approx L(y_i,f_{m-1}(x))+[\frac{\partial L(y_i,f(x))}{\partial f(x)}]_{f(x)=f_{m-1}(x)}T(x;\Theta_m)
    \end{aligned}
$$
令，
$$r=-[\frac{\partial L(y_i,f(x))}{\partial f(x)}]_{f(x)=f_{m-1}(x)}$$
易知$r$就是梯度的负方向，所以$T$不断的拟合$r$，往梯度的负方向搜寻即可。

### Xgboost
Xgboost实际上就是GBDT的升级版，首先重新定义Xgboost的目标函数
$$Obj(t)=\sum_{i=1}^nL(y_i,\hat{y}^{t-1}+f_t(x_i))+\Omega(f_t)+constant$$
其中$f_t(x)$是我们要拟合的树,正则项$\Omega(f_t)$为
$$\Omega(f_t)=\gamma T+\frac{1}{2}\lambda \sum_{j=1}^nw_j^2$$
其中$T$表示叶子结点的数量，$w_j$表示叶子结点的权重。
对$Obj(t)$二阶泰勒展开
$$Obj(t)\approx \sum_{i=1}^n[L(y_i,\hat{y}^{t-1})+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)+constant$$
其中$g_i$是$L(y_i,\hat{y}^{t-1})$对$\hat{y}^{t-1}$的一阶导数，$h_i$是$L(y_i,\hat{y}^{t-1})$对$\hat{y}^{t-1}$的二阶导数.
因为$L(y_i,\hat{y}^{t-1})$是已知的，且对于当前需要拟合的树来说是固定值，拟合的目标函数有
$$
    \begin{aligned}
        Obj(t) &\approx \sum_{i=1}^n[L(y_i,\hat{y}^{t-1})+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)+C\\
        &=\sum_{i=1}^n[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)+C\\
        &=\sum_{j=1}^T[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in{I_j}}h_i)w_j^2] + \gamma T +\frac{1}{2}\lambda\sum_{j=1}^Tw_j^2+C\\
        &=\sum_{j=1}^T[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in{I_j}}h_i+\lambda)w_j^2] + \gamma T+C\\
        &=\sum_{j=1}^T[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2] + \gamma T+C\\
    \end{aligned}
$$
其中$G_j=\sum_{i\in I_j}g_i$,$H_j=\sum_{i\in I_j}h_i$
这个式子是一个典型的凸函数，对$w$求导并使其值等于$0$可以得到
$$w_j=-\frac{G_j}{H_j+\lambda}$$
将其带入目标函数即可得到
$$Obj(t)=-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T+C$$
所以说，在新树的建立过程中，叶子节点进行分裂时，跟CART类似的，对于单节点树$t$，有
$$C(t)=-\frac{1}{2}\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}+\gamma+C$$
对于以$t$为根的树$T_t$
$$C(T_t)=-\frac{1}{2}\frac{G_L^2}{H_L+\lambda}-\frac{1}{2}\frac{G_R^2}{H_R+\lambda}+2\gamma+C$$
$$
    \begin{aligned}
        Gain &= C(t) - C(T_t)\\
        &=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{1}{2}\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma
    \end{aligned}
$$
当Gain大于阈值时即可分裂。
这里说一下，为什么在之前的公式推导过程中，$f_t(x_i)$可以用$w_j$代替。因为一个样本在进入决策树后，最终会落到某个叶结点上，根据该叶结点的情况也就是其权重进行判断，最终输出某个值，因此可以用$w$替换表示。
#### xgboost加速
1. 预排序
    计算Gain时，对某特征预排序后的数据能更快的进行左孩子与右孩子的数据的切分
2. 近似直方图
    可以理解为把连续数据离散化，当样本容量特别大的情况下不可能在结点分裂时把所有的分裂方式都试一遍以求最大Gain，因此将数据“离散化”后，相当于减少了样本容量，然后做近似求解。
