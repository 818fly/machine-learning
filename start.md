## 什么是机器学习

机器学习是计算机从数据中学习的总称，机器可以学习许多不同的方法（“算法”），这些算法可以分为有监督、无监督和强化算法。
您提供给机器学习算法的数据可以是输入输出对，也可以只是输入，监督学习算法需要输入输出对（即它们需要输出），无监督学习只需要输入数据（而不是输出）。

一般来说，监督算法是这样工作的：你给它一个示例输入，然后是相关的输出；你重复上面的步骤很多次；最终，该算法会在输入和输出之间找到一个模式；现在，你可以给它一个全新的输入，它会为你预测输出。

一般来说，无监督算法是这样工作的：你给它一个示例输入（没有相关的输出），你重复上面的步骤很多次，最终，该算法将您的输入聚类成组，现在，你可以给它一个全新的输入，算法会预测它属于哪个集群。

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_17-17-19.png)

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_18-46-04.png)

在这张无监督的图片中，将数据传递给计算机，我们的计算机不会知道这个是猫，那个是狗，它只会将含有公共点的图片放在一起

强化学习：有一个智能体在某种互动环境中学习，基于奖励和处罚

## 监督学习

将特征向量输入到模型中，模型吐出一个输出，这个就是我们的预测

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_18-54-09.png)

## 特征

- 定性的分类数据名词

我有一个数据集，我们的输入可能来自美国、印度、加拿大和法国，现在我们如何让我们的计算机识别，我们必须做一些热编码的问题，即如果匹配则设置为1

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-17_22-41-39.png)

**分类问题：二分类与多分类**，多分类意味着在多个之间进行预测，二分类问题是在两个之间进行预测

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_19-17-58.png)

还有一个称为回归，这意味着我们预测的是连续值，因此我们不只是试图预测不同的类别，而是试图计算出数字，例如：明天的以太币的价格，明天的温度或者房子的价格，我们试图用我们数据集的不同特征预测一个尽可能接近真实值的数字。

##  监督学习数据集

特征：怀孕次数、不同血糖水平、血压、皮肤厚度、胰岛素、BIM、年龄、是否患有糖尿病结果。

每一行都是一个样本，每一列表示的是它的测量值，outcome是输出的标签。我们的输入的值称为特征向量（怀孕次数、不同血糖水平、血压、皮肤厚度、胰岛素、BIM、年龄 ）。所有的特征向量的组合就形成特征矩阵，下图的X，目标输出的那一列是我们的目标向量，下图的Y

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_19-41-33.png)

我们把把它用巧克力进行比喻。

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-18_16-10-42.png)

每一行的数据将输入到我们的模型中，我们的模型将做出某种预测，将模型预测出来的巧克力与标签中的实际y值进行比较，微调模型直到与真实值相近，这个过程就是训练的过程。

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-18_16-17-40.png)

注意：我们不是把我们整块巧克力放入我们的模型中进行训练，如果这样做了，我们怎么知道我们的模型在我们没有见过的新数据上做得很好？

所以在实际中，我们会将我们的数据分为三个不同的数据集类型：训练数据集、验证数据集和测试数据集，它们的分配可以是6：2：2；8：1：1取决于你总体的数据量的多少

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-18_16-23-48.png)

我们将这个训练数据集放入到我们的模型中，得出一个预测的结果向量，将这个预测结果与真实的结果进行比较，得到loss，根据这个loss，我们对模型进行调整 

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-18_16-27-29.png)

我们需要损失值最小的C模型

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_20-13-53.png)

然后我们使用模型C，用测试集进行测试，测试集的目的就是用于最终的检查，以检查所选的模型的泛化程度，这个结果是模型的最终训练表现 

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_20-18-34.png)

### 深入LOSS

LOSS就是你的预测值与实际标签之间的差异，我们用函数的方式来表达损失，第一种函数：

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_20-30-12.png)

第二种函数：

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_20-34-25.png)

tips：我们需要知道的是loss会随着性能的提高而减少

### 其他衡量模型的标准

准确率

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-15_20-41-10.png)

### 代码引入

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cols = ['fLength','fWidth','fSize','fConc',' fConc1','fAsym','fM3Long',' fM3Trans',' fAlpha','fDist','class']
df = pd.read_csv('magic04.data',names=cols)
# 将分类中的原始数据的g、h变为1、0
df['class'] = (df['class']=='g').astype(int)
```

【插入】给一个列的数据根据分类画出直方图

```python
# 画出class为1和0在每一个col下面的直方图
# density 可以将同一个面板下的不同的直方图进行标准化，从而可以进行比较
# label标签值表示是图例，与plt.legend()配套使用
for label in cols[:-1]:
    plt.hist(df[df['class']==1][label], color = 'blue', label = 'gamma' , alpha = 0.7, density =True)
    plt.hist(df[df['class']==0][label], color = 'red', label = 'hadron' , alpha = 0.8, density =True)
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend()
    plt.show()
```

### 样本数据划分为三部分

```python
# 将原始的数据分为训练集、测试集和验证集
#60%训练集、30%测试集【从60%转到90%意味着有30%的数据用于测试】、10%验证集
train,valid,test=np.split(df.sample(frac=1),[int(len(df)*0.6),int(len(df)*0.8)])  
```

### 将样本的三部分数据进行处理函数

```python
# 就一个特征向量来说，每一个特征值的值之间的大小相差很大，有时会影响我们的结果，所以我们需要对数据集中的数据进行缩放
# 我们需要导入标准标量
#  preprocessing 预处理  StandardScaler  缩放器
from sklearn.preprocessing  import StandardScaler
# 当你导入的时候发现报出提示没有imblearn模块
# 你需要以管理员的身份进行运行Anconada，然后执行 conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
def sclar_datasets(dataframe,oversample=False):
    # 让我们假设这些列将是
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values  # 它得到是一维的数组，在合并的时候，需要都是一样的维数才可以合并
    # 所以需要对y进行重型
    y= np.reshape(y,(len(y),1))
    # 创建一个标量
    scaler = StandardScaler()
    # 将原始的数据重新拟合转换
    X = scaler.fit_transform(X)
    # 如果是过采样
    if oversample:
        # 设置随机采样器
        ros = RandomOverSampler()
        # 重新采样X和y
        X, y = ros.fit_resample(X,y)
        y= np.reshape(y,(len(y),1))
    # 拿一个数组与另一个数组进行水平堆叠
    data = np.hstack((X,y))
    return data,X,y
```

### 调用这个处理函数

```python
# 过度采样的目的就是将训练集中的class为1的和class为0的个数弄成相等的个数
# 这样训练的结果才可靠，没有偏见性，达到均匀的目的，如果我们不采用oversample的结果可能class为1和0的个数不一样
train,x_train,y_train = sclar_datasets(train,oversample = True)
# 验证集与数据集是不用将versample设置为True的，因为在这两个测试机中我不关系平衡性，只需要得到的随机数据即可
valid,x_valid,y_valid = sclar_datasets(valid,oversample = False)
test,x_test,y_test = sclar_datasets(test,oversample = False)
```

> 数据准备好了，接下来就是将这个数据放在模型上测试了
>
> 接下来学习的第一个模型就是KNN

### K-Nearest Neighbors(k-最近邻算法)

K最邻近算法是一种简单的监督学习算法，它可以用来被解决分类和回归问题

解释该算法：

1、假如我们需要将给定的一个点看它在三个分类中的哪一个，使用k最邻近算法

2、需要计算给定的点到所有点之间的距离（常用euclidean（欧几里得）距离）

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-18_18-18-38.png)

3、将给定的点与它的邻近点的距离进行低到高排序

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-18_18-28-30.png)

4、对于分类问题，如果k=5，则选择距离这个点的最近的5个点，则发现有3个点是黄色，2个点是灰色，则这个新的点预测值就是黄色，我们选择的k值需要一直是奇数值，避免邻近的点不会出现两侧都相等从而选不出是哪一个类别的情况。

说明：k值在这里控制过拟合和欠拟合之间的平衡。通过交叉验证和学习曲线可以找到k的最佳值。小的k值通常会导致偏差偏低，但方差较大，而较大的k通常会导致较高的偏差，但方差较小，所以找到这两点之间的平衡点很重要。

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-18_18-53-05.png)

对于回归问题，也是得到我这个新点在k个最邻近的点，它试图是找到k个最邻近的几个点的平均值，即我们只需要返回最近的k个邻居标签的平均值作为预测

### 另一个视频说的KNN算法

将新的数据点贴上我周围大多数人的标签，为了获得“周围”这个概念，我们需要做的第一件事就是定义一个距离函数，在二维图中一般使用欧几里得距离【即两个点的直接连点距离】。在KNN算法中我们看到一个K，这个K表示我们使用多少个邻居来判断标签是什么。一般的k值取决于你的数据集的大小。

### 代码实现KNN

```python
# 导入K邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 如果我们需要从sklern学习指标中导入分类报告。【metric指标】
from sklearn.metrics import classification_report
```

首先我们的分类器设置为1个

```python
# 首先我们的分类器设置为1个
knn_model = KNeighborsClassifier(n_neighbors=1)
# 使用模型拟合
knn_model.fit(X_train,y_train)
```

使用测试集放入模型，让其进行预测

```python
y_pred = knn_model.predict(X_test)
```

得到模型的报告

```python
print(classification_report(y_test,y_pred))
```

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_16-35-31.png)

### （Native bayes）朴素贝叶斯

朴素贝叶斯工作原理：我们收到来自家人和朋友的正常消息，我们同时也收到了垃圾邮件，这些垃圾邮件通常是诈骗或是未经请求的广告，我们想过滤掉垃圾邮件，所以我们要做的第一件事就是对来自朋友和家人正常消息中出现的所有单词制作直方图，我们可以使用直方图来计算看每个单词的概率，假设我们统计的单词有dear、friend、lunch和money，当在正常消息中的时候，我们统计出dear出现的次数是8，普通消息中我们统计的总单词数是17，得到了一些概率：

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_18-35-10.png)

我们在垃圾邮件中也统计出dear、friend、lunch、money这些单词的概率

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_18-39-50.png)

仅仅保留概率。说明：我们计算出单个单词的概率而不是连续东西的概率，这些概率也被称为可能性。得到这些单词在正常邮箱和在垃圾邮箱中出现的可能性，如下图：

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_18-44-00.png)

情景一：如果我们收到了一个新的消息“dear friend"，我们想确定他是正常的消息或者是垃圾消息。不考虑任何消息它里面说了什么，它是正常邮件的概率是：因为在12条消息中有8条消息是正常的消息，所以我们的初始猜测是正常邮件的概率是8/12=0.67。

> **这个事先被人们就可以判断为是好的邮件或者是坏的邮件称为先验概率**

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_18-50-08.png)

为了得到"dear friend"在正常邮件中的分数：是正常邮件的概率乘在正常邮件中dear的概率乘friend在正常邮件中的概率，得到分数0.09

下面对垃圾邮箱进行分析：

不考虑任何消息它里面说了什么，它是span邮件的概率是：

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_19-01-09.png)

为了得到"dear friend"在垃圾邮件中的分数，现在我们将初始猜测乘以dear单词出现在垃圾邮件中的概率以及friend出现在垃圾邮件中的概率，得到分数0.01。最后通过分数的比较0.09>0.01，我们得出结论”dear friend'是正常邮件

情景二：对于收到的消息”Lunch Money Money Money Money"，我们想确定他是正常的消息或者是垃圾消息。在正常消息中的分数是0.67x0.18x(0.06)^4；在垃圾邮件中的分数是：0x(0.57)^4=0。现在问题出现了，只要回复的邮件中含有Lunch，而不管Money含有多少个结果都是0，这样都会判断为正常的邮件，这显然是不对的，为了解决这个问题，我们一般是用这种办法进行解决：

为了不会出现0的情况，我们的做法是，假设给每一个单词都手动增加一个，下图的黑色就是增加的单词

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_20-15-11.png)

在增加的情况下，我们重新计算每一个单词出现的概率

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_20-19-26.png)

注意：为每一个单词增加一个数量不会，不会改变正常邮件和垃圾邮件的原始猜想，因为对每一个单词添加计数不会改变训练数据集中正常邮件的数量或垃圾邮件的数量。现在我们再一次计算出”Lunch Money Money Money Money"在正常邮件和垃圾邮件中的分数，如下图所示：

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_20-24-34.png)

由此：可以确定它是垃圾邮件的概率大一些

### 谈论朴素贝叶斯为什么是朴素的？

因为它对待词序不同的词都有一样的分数，即不管“dear friend"还是”friend dear"得到的概率都是一样的

![](https://cdn.jsdelivr.net/gh/818fly/myImg/Snipaste_2023-04-19_20-28-57.png)



