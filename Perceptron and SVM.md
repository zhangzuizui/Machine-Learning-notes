# Perceptron and SVM
以为两个模型都与用超平面划分数据有关，因此放在一起讨论
## Perceptron
### 感知机普通形式
感知机的实质是，假设数据是线性可分的（即存在超平面$S$将将数据的正实例点和数据的负实例点完全正确的划分到其两侧）情况下，找到这个超平面$S$。
感知机的学习目标是
$$L(w,b)=-\sum_{x_i\in M}y_i(wx_i+b)$$
解释：
容易知道的是，点到超平面的距离是
$$\frac{1}{||w||}|wx_i+b|$$
对于误分类数据来说$wx_i+b>0$时，$y_i<0$，$wx_i+b<0$时，$y_i>0$，即误分类数据到超平面的距离为
$$-\frac{1}{||w||}y_i(wx_i+b)$$
在不考虑$L2$正则，即几何间隔，转而考虑函数间隔的情况下，得到$L(w,b)$
利用梯度下降进行求解
$$
    \begin{aligned}
        w&:=w+\alpha y_i x_i \\
        b&:=b+\alpha y_i
    \end{aligned}
$$
注意，其中的$(x_i,y_i)$仅只被误分类的数据。

### 感知机对偶形式
引入变量$n_i \in (n_1,n_2,...,n_N)$，其中$N$指样本容量，$n_i$代表第$i$个样本被误分类的次数，初始全部置$0$。
由上面梯度下降的式子可知，$w$和$b$增加的过程，是每次误分类后对误分类数据以某种方式进行累计的过程，所以当$w$和$b$初始为0时有
$$
    \begin{aligned}
        w&=\alpha\sum_{i=1}^Nn_iy_ix_i \\
        b&=\alpha\sum_{i=1}^Nn_iy_i
    \end{aligned}
$$
感知机的模型就变为
$$f(x)=sign[\alpha\sum_{i=1}^Nn_iy_i(x_ix+1)]$$
式中有$x_ix$啊，就是对偶所在了，每次训练前先计算一个关于$x$的$Gram$矩阵，以方便运算。
然后明显的，因为每次迭代，实际上是对误分类数据对应的$n_i$不断的做$+1$操作，所以迭代过程为：
1. 已知有模型$f_{m-1}(x)$
2. $n_i += 1 \space if \space y_if_{m-1}(x_i)\leq0$
3. 更新所有$n_i$，得到$f_m(x)$

## Support vector machines
与感知机类似的，支持向量机的模型表达式为
$$f(x)=sign(wx+b)$$
其与感知机最大的不同之处在于，SVM的训练目的不光是为了找到超平面将数据划分为两个部分，还要使两个部分中，离超平面距离最近的点的间隔最大。
### 函数间隔与几何间隔
**函数间隔**
$$\hat \gamma_i = y_i(wx_i+b)$$
那么离超平面最近的点的函数间隔就是
$$\hat \gamma = \underset{i\in1,...,N}{min} \hat\gamma_i$$
由函数间隔的表达式可以知道，如果我们将$w$和$b$成同比例的放大或者缩小，完全不影响模型的分类效果，因此有
**几何间隔**
$$\gamma_i = y_i(\frac{w}{||w||}x_i+\frac{b}{||w||})$$
在这里我们令$\hat \gamma = ||w||\gamma$以对$w$和$b$进行约束，使模型只有一个最优解。
### 线性可分SVM与硬间隔最大
硬间隔最大的含义就是支持向量到划分超平面的距离最大，线性可分指存在超平面使划分的数据到超平面的距离均大于或等于其函数间隔
于是我们要求解的目标就是
$$\gamma = \underset{i\in1,...,N}{min}\gamma_i$$
对于SVM，我们要做的就是在约束条件下，找到能使$\gamma$达到最大值的参数$w$和$b$
$$
    \begin{aligned}
        &\underset{w,b}{max}\quad \gamma \\
        &s.t.\quad y_i(\frac{w}{||w||}x_i+\frac{b}{||w||})\geq \gamma,\quad i\in 1,2,3,...,N
    \end{aligned}
$$
其等价于
$$
    \begin{aligned}
        &\underset{w,b}{max}\quad \frac{\hat\gamma}{||w||} \\
        &s.t.\quad y_i(wx_i+b)\geq \hat\gamma,\quad i\in 1,2,3,...,N
    \end{aligned}
$$
因为函数间隔的放大与缩小并不影响求解的结果，因此令$\hat \gamma=1$，上式等价于
$$
    \begin{aligned}
        &\underset{w,b}{max}\quad \frac{1}{||w||} \\
        &s.t.\quad y_i(wx_i+b)\geq 1,\quad i\in 1,2,3,...,N
    \end{aligned}
$$
最后再对上式进行转换，把求最大值问题转化为求最小值问题
$$
    \begin{aligned}
        &\underset{w,b}{min}\quad \frac{1}{2}||w||^2 \\
        &s.t.\quad y_i(wx_i+b)-1\geq 0,\quad i\in 1,2,3,...,N
    \end{aligned}
$$
这是一个典型的凸二次规划(*convex quadratic programming*)问题，所谓凸优化，指的是具有以下形式的最优化问题
$$
    \begin{aligned}
        &\underset{w}{min}\quad f(w) \\
        &s.t.\quad c_i(w)\leq0,\quad i\in 1,2,3,...,k \\
        &\quad \quad \; h_i(w)=0, \quad i\in 1,2,3,...,l
    \end{aligned}
$$
其中目标函数$f(w)$和约束函数$c_i(w)$都是$R^n$上的连续可微凸函数，$h_i(w)$是$R^n$上的仿射函数。
当$f(w)$是二次函数且$c_i(w)$是仿射函数时，上述凸优化问题就成为凸二次规划问题。
#### SVM的对偶算法
因为SVM求解的问题是一个凸二次规划问题，因此引入广义拉格朗日函数进行求解
$$
    \begin{aligned}
        L(w,b,\alpha)&=\frac{1}{2}||w||^2+\sum_{i=1}^N\alpha_i[1-(y_i(wx_i+b))] \\
        &=\frac{1}{2}||w||^2-\sum_{i=1}^N\alpha_iy_i(wx_i+b)+\sum_{i=1}^N\alpha_i
    \end{aligned}
$$
此时原始问题是
$$\underset{w,b}{min}\underset{\alpha:\alpha_i\geq0}{max}L(w,b,\alpha)$$
因为该拉格朗日函数满足*KKT*条件，因此转化为求原问题的对偶问题
$$\underset {\alpha:\alpha_i\geq0}{max}\underset{w,b}{min}L(w,b,\alpha)$$

$$
    \begin{aligned}
        \frac{\partial L}{\partial w}&=w-\sum_{i=1}^N\alpha_iy_ix_i=0 \\
        \frac{\partial L}{\partial b}&=-\sum_{i=1}^N\alpha_iy_i=0
    \end{aligned}
$$
得
$$
    \begin{aligned}
        w&=\sum_{i=1}^N\alpha_iy_ix_i \\
        &\sum_{i=1}^N\alpha_iy_i=0
    \end{aligned}
$$
带入原问题得到
$$
    \begin{aligned}
        L(w,b,\alpha)&=\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i\cdot x_j-\sum_{i=1}^N\alpha_iy_i(\sum_{j=1}^N(\alpha_jy_jx_j)\cdot x_i+b)+\sum_{i=1}^N\alpha_i \\
        &=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i\cdot x_j+\sum_{i=1}^N\alpha_i
    \end{aligned}
$$
即
$$\underset{w,b}{min}L(w,b,\alpha)=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i\cdot x_j+\sum_{i=1}^N\alpha_i$$
求$\underset{\alpha}{max}L(w,b,\alpha)$对$\alpha$求极大即是对偶问题
$$
    \begin{aligned}
        &\underset{\alpha}{max}\;-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i\cdot x_j+\sum_{i=1}^N\alpha_i \\ 
        &\begin{aligned}
            s.t.\quad&\sum_{i=0}^N\alpha_iy_i=0 \\
            & \alpha_i\geq0,\quad i=1,2,...,N
        \end{aligned}
    \end{aligned}
$$
将求极大转化为求极小，这个问题等价于
$$
    \begin{aligned}
        &\underset{\alpha}{min}\;\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i\cdot x_j-\sum_{i=1}^N\alpha_i \\ 
        &\begin{aligned}
            s.t.\quad&\sum_{i=0}^N\alpha_iy_i=0 \\
            & \alpha_i\geq0,\quad i=1,2,...,N
        \end{aligned}
    \end{aligned}
$$
假设该问题的解$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$，$\alpha_j>0$所对应的点$x_j$即为SVM的支持向量
那么容易得到
$$w^*=\sum_{i=1}^N\alpha_iy_ix_i$$
再根据*KKT*条件
$$\alpha_i^*(1-y_i(wx_i+b)))=0$$
得($b^*$随便取一个支持向量计算即可)
$$b^*=y_j-\sum_{i=1}^N\alpha_iy_ix_i\cdot x_j$$

**注意**
存在这样一个情况，就是解出来的$\alpha$中存在$\alpha_i<0$，此时这个解释不满足*KKT*条件的，因此得从满足*KKT*条件的情况下，从边缘情况即置某个$\alpha_i=0$重新找$\alpha$

### 线性支持向量机及软间隔最大
对于实际数据来说，数据通常并不是线性可分的，而是有一些特异点，在去掉这些特异点之后，整体数据就会变为线性可分。于是在线性不可分的情况下就不能使用对线性可分数据建模的方法找到划分超平面。因此引入了松弛变量$\xi_i\geq0$，意指允许点到划分超平面的函数距离加上$\xi_i$大于等于 1，这样约束条件就变为
$$y_i(wx_i+b)\geq 1-\xi_i$$
对于目标函数来说，因为松弛变量的存在，所以需要对松弛变量进行惩罚，因此目标函数变为（对于为什么惩罚项是以这种形式添加，后面看了合页损失函数后就能理解了）
$$\frac{1}{2}||w||^2+C\sum_i^N\xi_i$$
这里$C>0$是一个惩罚参数，$C$越大代表分类器越倾向于将训练集中的所有数据都尽可能的分类正确(为了使引入松弛变量的样本尽可能少，引入的值尽可能小)，这样容易导致过拟合。而$C$越小则表示分类器越不在乎分类错误，使得分类器性能变差。

引入松弛变量后，线性不可分的线性支持向量机的学习问题就变成了如下的凸二次规划问题
$$
    \begin{aligned}
        &\underset{w,b,\xi}{min}\quad \frac{1}{2}||w||^2+C\sum_i^N\xi_i \\
        &\begin{aligned}
            s.t.&\quad y_i(wx_i+b)\geq1-\xi_i,\quad i\in 1,2,...,N \\
            &\quad \xi_i \geq0,\quad i\in 1,2,...,N
        \end{aligned}
    \end{aligned}
$$
之后的计算方法与线性可分支持向量机相同，先引入拉格朗日乘子构成拉格朗日函数
$$
    \begin{aligned}
        L(w,b,\xi,\alpha,\mu)=\frac{1}{2}||w||^2+C\sum_i^N\xi_i+\sum_{i=1}^N\alpha_i(1-\xi_i-y_i(wx_i+b))-\sum_{i=0}^N\mu_i\xi_i
    \end{aligned}
$$
由
$$
    \begin{aligned}
        \frac {\partial L(w,b,\xi,\alpha,\mu)}{\partial w}&=w-\sum_{i=1}^N\alpha_iy_ix_i=0 \\
        \frac {\partial L(w,b,\xi,\alpha,\mu)}{\partial b}&=-\sum_{i=1}^N\alpha_iy_i=0 \\
        \frac {\partial L(w,b,\xi,\alpha,\mu)}{\partial \xi_i}&=C-\alpha_i-\mu_i=0
    \end{aligned}
$$
得
$$
    \begin{aligned}
        w=\sum_{i=1}^N\alpha_iy_ix_i \\
        \sum_{i=1}^N\alpha_iy_i=0 \\
        C-\alpha_i-\mu_i=0
    \end{aligned}
$$
带入到拉格朗日函数中，有
$$
    \begin{aligned}
        \underset{w,b,\xi}{min}L(w,b,\xi,\alpha,\mu)=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^N\alpha_i
    \end{aligned}
$$
再对$\alpha$求最大即可得到对偶问题
$$
    \begin{aligned}
        &\underset{\alpha}{min}\;\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i \\
        &\begin{aligned}
            s.t.\quad&\sum_{i=1}^N\alpha_iy_i=0 \\
            &C-\alpha_i-\mu_i=0 \\
            &\alpha_i\geq0 \\
            &\mu_i\geq0, \quad i\in 1,2,...,N
        \end{aligned}
    \end{aligned}
$$
把上面约束条件中的最后三个合一，能够得到对$\alpha_i$的约束
$$0\leq\alpha_i\leq C$$
这里对$\alpha$，$\xi$和$C$的关系进行一波讨论，上面拉格朗日函数的*KKT*条件为

$$
    \begin{aligned}
        \frac {\partial L(w^*,b^*,\xi^*,\alpha^*,\mu^*)}{\partial w^*}&=w^*-\sum_{i=1}^N\alpha_i^*y_ix_i=0 \\
        \frac {\partial L(w^*,b^*,\xi^*,\alpha^*,\mu^*)}{\partial b^*}&=-\sum_{i=1}^N\alpha_i^*y_i=0 \\
        \frac {\partial L(w^*,b^*,\xi^*,\alpha^*,\mu^*)}{\partial \xi_i^*}&=C-\alpha^*-\mu^*=0 \\
        \alpha_i^*(1-\xi_i^*-y_i^*&(w^*x_i+b^*))=0 \\
        \mu_i^*\xi^*_i&=0 \\
        1-\xi_i^*-y_i^*(&w^*x_i+b^*) \leq 0 \\
        \xi_i^*&\geq0 \\
        \alpha_i^*&\geq0 \\
        \mu_i^*&\geq0,\quad i\in 1,2,...,N
    \end{aligned}
$$
容易知道，
1. 当拉格朗日乘子$\alpha_i=0$时，对应的点不是SVM的支持向量；
2. 当$\alpha_i<C$时，有$\mu_i>0$，此时$\xi_i=0$，即对应的的点刚好在离划分超平面距离为1的地方；
3. 当$\alpha_i=C$时，$\mu_i=0$，若$1\geq\xi_i>0$，此时对应的点离划分超平面的距离为$1-\xi_i$；若$\xi_i>1$，则对应的点在划分超平面误分类的那一册。

并且可以看出$b^*$的解不唯一，是一个区间（求出支持向量后，划分超平面可以在一定范围内平移）。与线性可分支持向量机相同的，取$0<\alpha_i<C$，$w^*$和$b^*$可按下式求得
$$w^*=\sum_{i=1}^N\alpha_iy_ix_i$$
$$b^*=y_j-\sum_{i=1}^N\alpha_iy_ix_i\cdot x_j$$
由于$b^*$的值不唯一，所以实际上取所有符合条件的样本点的均值。
#### 合页损失函数
除了用凸二次规划解求解决策函数$f(x)=sign(w^*x+b^*)$外，线性支持向量机还有一种解释，就是最小化合页损失函数：
$$\sum_{i=1}^N[1-y_i(w_ix+b)]_++\lambda||w||^2$$
左式是合页损失函数的经验风险
$$L(y(wx+b))=[1-y(wx+b)]_+$$
其中
$$[z]_+ = 
    \begin{cases}
        z,\quad z>0\\
        0,\quad z\leq0
    \end{cases}
$$
类似的，感知机的损失函数实质是
$$[-y(wx+b)]_+$$
可以发现的是，对于感知机的损失来说，合页损失函数不但要求样本被正确的分类，并且要置信度足够高时损失才为0，对学习有更高的要求。

然后可以得到的是，合页损失函数的优化问题为
$$\underset{w,b}{min}\sum_{i=1}^N[1-y_i(w_ix+b)]_++\lambda||w||^2$$
若取$\xi_i=[1-y_i(w_ix+b)]_+$，则有
$$\underset{w,b}{min}\sum_{i=1}^N\xi_i+\lambda||w||^2$$
再令$\lambda=\large \frac{1}{2C}$
就得到了凸二次规划问题的表达式
$$\underset{w,b}{min}\quad\frac{1}{C}(\frac{1}{2}||w||^2+C\sum_{i=1}^N\xi_i)$$

### 非线性支持向量机与核函数
对于非线性数据，无法找到超平面将原始数据划分为两个部分，因此引入了核函数，通过核函数把原数据从欧式空间$R^n$或离散空间映射到一个特征空间$H$(希尔伯特空间)，这就是核技巧。（注欧式空间与希尔伯特空间的具体含义不清楚，个人理解就是把原数据通过一个函数进行转换，不一定是升维，但一般就表现来说看起来都是通过升维从而使得数据变得可分）通过核技巧使得数据在特征空间内呈近似线性可分（用线性不可分SVM求解）。

对于一个函数来说，在将原始数据所在空间的任意数据进行映射后得到的新的Gram矩阵是一个半正定矩阵，则称这个函数是核函数，或是正定核（至于为什么得是正定核，或者准确的说是半正定核，大概是只有当他是半正定核的时候，对与映射到希尔伯特空间后的点，其两点之间的距离才都为正，就是有点类似于实数域的感觉。具体的推论看不太懂，总之就是mercer定理）。
$$K=[K(x_i,x_j)]_{m\times m}$$
这个定义在构造核函数时很有用，但是对于具体函数来说，检测其是否为核函数并不容易，因此在实际问题中往往使用已有的核函数。

#### 三种常用核函数
1. 多项式核函数
$$K(x,z)=(x\cdot z+1)^p$$
2. 高斯核函数
$$K(x,z)=\exp(-\frac{||x-z||^2}{2\sigma^2})$$
为什么高斯核函数可以把向量映射到无穷维？
参看这篇quora的[回答](https://www.quora.com/Why-does-the-RBF-radial-basis-function-kernel-map-into-infinite-dimensional-space-mentioned-many-times-in-machine-learning-lectures)
就如果原数据是三维$<x_1, x_2, x_3>$ 那么通过某种映射我们可以得到
$<x_1x_1, x_1x_2, x_1x_3........>$一共九维。 然后求内积（难受吗，难受啊，核函数表示很难受啊，好不容易映射到了那么多维，一求内积就变成一个数回到解放前）
现在来看为什么说高斯核能把数据映射到无穷维。
因为
$$K(\overrightarrow{X_1},\overrightarrow{X_2})=\exp(-\frac{||\overrightarrow{X_1}-\overrightarrow{X_2}||^2}{2\sigma^2})$$
把平方乘出来，展开，然后只看其中的一项
$$K'(\overrightarrow{X_1},\overrightarrow{X_2})=\exp(-\frac{\overrightarrow{X_1}\overrightarrow{X_2}}{\sigma^2})$$
接着有个很厉害的地方就是，$e^x$这个东西，可以泰勒展开，于是上面的式子就变成了
$$K'(\overrightarrow{X_1},\overrightarrow{X_2})=\sum_{n=0}^{+\infty}\frac{(\overrightarrow{X_1}\overrightarrow{X_2})^n}{\sigma^2n!}$$
想一下，这个$K$是什么？不就是内积吗，再看看这形式，是不是无穷维的向量的内积？所以说，高斯核将特征映射到了无穷维，不过很难受的是管你几维，求了内积就剩下一数。
为什么把原数据映射到无穷维后不会发生维度灾难？
因为我们一直在求的其实是内积，即映射前求欧式空间的内积（参看gram矩阵），映射后求的是希尔伯特空间的内积，这个时候我们并没有关心特征的维度是多少，关注的重点是放在特征的内积上）。
    **核函数的选择**
    下面是吴恩达的见解：
    1. 如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM
    2. 如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM+Gaussian Kernel
    3. 如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况

    **高斯核的一大优点**
    懒得写公式了，大概的就能看出来，不过我也说不太清楚，就是如果有两组特征，他们在欧式空间中的距离十分的近，假设是个二维或者三维图，随便杵两个很近的点，我们很难对他俩进行划分（假设他俩的label不一样），但是用高斯核映射到希尔伯特空间后，通过核函数可以发现，因为是$e$的负多少多少次幂，幂约小，其值越大（越接近1），他俩在欧式空间里约相近，在希尔伯特空间的这种”距离”就会被放大，从而变为近乎线性可分。顺便的，并不是使用了高斯核数据就一定线性可分了，松弛变量$/xi$还是很重要。
3. 字符串核函数（表示看不懂

#### 使用核技巧后，SVM的对偶形式求解
$$
    \begin{aligned}
        &\underset{\alpha}{min}\quad \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^n\alpha_i \\
        &\begin{aligned}
            s.t.\quad &\sum_{i=1}^n\alpha_iy_i=0
            &0\geq\alpha_i\geq C,\quad i\in 1,2,...,n
        \end{aligned}
    \end{aligned}
$$
求得最优解$\alpha^*=(\alpha_1^*,...,\alpha_n^*)$
选择$0<\alpha_j^*<C$求解
$$b^*=y_j-\sum_{i=1}^n\alpha_i^*K(x_i,x_j)$$
然后得到决策函数
$$f(x)=sign(\sum_{i=1}^n\alpha_i^*y_iK(x,x_i)+b^*)$$

### 序列最小最优算法SMO
用常规方法求解划分超平面的复杂度太高（O(n^2)?）因此使用迭代的方法对其求解，这个方法就是SMO
考虑一下凸二次规划问题
$$
    \begin{aligned}
        &\underset{\alpha}{min}\quad \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^n\alpha_i \\
        &\begin{aligned}
            s.t.\quad &\sum_{i=1}^n\alpha_iy_i=0 \\
            &0\leq\alpha_i\leq C,\quad i\in 1,2,...,n
        \end{aligned}
    \end{aligned}
$$
若固定除了$\alpha_1$和$\alpha_2$外的其他所有变量，那么就可以得到$\alpha_1$和$\alpha_2$这两个变量之间的关系，即
$$\alpha_1=-y_1sum_{i=2}^n\alpha_iy_i$$
即只要确定了$\alpha_2$那么就能通过约束条件求$\alpha_1$

整个SMO包括两个部分：求解两个变量二次规划的解析方法和选择变量的启发式方法
#### 两个变量二次规划的求解方法
首先将原本的最优化问题改写为
$$
    \begin{aligned}
    & \\
        &\begin{aligned}
        \underset{\alpha_1,\alpha_2}{min}\quad W(\alpha_1,\alpha_2)=&\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2-\\
            &\begin{aligned}
                &(\alpha_1+\alpha_2)+y_1\alpha_1\sum_{i=3}^ny_i\alpha_iK_{i1}+y_2\alpha_2\sum_{i=3}^ny_i\alpha_iK_{i2}
            \end{aligned} 
        \end{aligned} \\
        &\begin{aligned}
            s.t.\quad &\alpha_1y1+\alpha_2y2=-\sum_{i=3}^ny_i\alpha_i=\zeta \\
        &0\leq\alpha_i\leq C,\quad i\in 1,2
    \end{aligned}
    \end{aligned}
$$

在$y_1\not=y_2$时，有
$$\alpha_2^{old}-\alpha_1^{old}=k$$
假设已经得到$\alpha_1^{new}$
那么有
$$\alpha_2^{new}=\alpha_1^{new}+k$$
因为
$$
    \begin{aligned}
        0\leq\alpha_1^{new}\leq C \\
        0\leq\alpha_2^{new}\leq C
    \end{aligned}
$$
所以
$$max(0,\alpha_2^{old}-\alpha_1^{old})\leq\alpha_2^{new}\leq min(C,C+\alpha_2^{old}-\alpha_1^{old})$$
同理，当$y_1=y_2$
$$max(0,\alpha_2^{old}+\alpha_1^{old}-C)\leq\alpha_2^{new}\leq min(C,\alpha_2^{old}+\alpha_1^{old})$$

令
$$L\leq\alpha_2^{new}\leq H$$
设未考虑约束时$\alpha_2$的最优解为$\alpha_2^{new,unc}$

啊不想写了，一大坨公式要输入。
推导的大致流程如下，细节参考李航《统计机器学习》P127
之前不是有个$W(\alpha_1,\alpha2)$的式子吗，
令
$$\alpha_1y_1+\alpha_2y_2=\zeta$$
那么就有
$$\alpha_1=y_1(\zeta-\alpha_2y_2)$$
将这个式子带入到$W$中，就变成了一个只含有$\alpha_2$一个变量的式子
我们计算$\alpha_2$的目的是要看它的值为多少时，能使$W$最小，因此求$w$对$\alpha_2$的偏导数，并使之为零即可
最后能得到一个关于$\alpha_2$的方程，将$\zeta=\alpha_1^{old}y_1+\alpha_2^{old}y_2$带入到方程中即可得到
$$\alpha_2^{new,unc}=\alpha_2^{old}+\frac{y_2(E_2-E_1)}{\eta}$$
其中，记
$$g(x)=\sum_{i=1}^n\alpha_iy_iK(xi,x)+b$$
令
$$E_i=g(x_i)-y_i=(\sum_{j=1}^n\alpha_jy_jK(x_j,x_i)+b)-y_i, \quad i=1,2$$
当$i=1,2$时，$E_i$为函数$g(x)$对输入$x_i$的预测值与真实值$y_i$之差。
上面提到过，$\alpha_2$实际是有一个约束范围的，因此
$$ \alpha_2^{new}= 
    \begin{cases} 
        H, & \text {$\alpha_2^{new,unc}> H$} \\ 
        \alpha_2^{new,unc}, & \text{$L\leq\alpha_2^{new,unc}\leq H$} \\
        L, & \text {$\alpha_2^{new,unc}<L$}
    \end{cases} 
$$
最后由$\alpha_2^{new}$求得$\alpha_1^{new}$
$$\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})$$

#### 变量的选择方法
**第一个变量选择**
第一个样本选择违反*KKT*条件最严重的点，即对每一个样本点检查其是否满足*KKT*条件（具体的解释可返回上面查看$\alpha$，$C$与$\xi$的关系部分）
$$
    \begin{aligned}
        \alpha_i = 0&\iff y_ig(x_i)\geq 1 \\
        0<\alpha_i<C&\iff y_ig(x_i)=1 \\
        \alpha_i=C&\iff y_ig(x_i)\leq 1
    \end{aligned}
$$

在检验过程中，先检验在间隔边界上的支持向量点，如果都满足，再检验其他点。
**第二个变量选择**
第二个变量选择的标准的希望有足够大的变化。
根据之前得到的结果，可以知道的是$\alpha_2^{new}$的值与$E_1-E_2$关系很大，则若$E_1$为负数，则找正最大的$E_2$，反之。
如果这样得到的结果还没有使目标函数有足够的变化（$W$的变化没有超过变化的阈值范围？），就遍历在间隔边界上的支持向量点，依次将其作为$\alpha_2$试用直到找到合适的$\alpha_2$，或者如果依旧找不到，那么重新选择$\alpha_1$。
