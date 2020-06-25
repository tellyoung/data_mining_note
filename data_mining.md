[toc]
# 数学概念

## 概率论

### 方差
1. 标准差
	标准差是方差的算术平方根

## 线性代数

### 矩阵分解


# 特征工程


## 数值特征

### 归一化
1. 线性归一化
    x  -  最小值  /  最大值  -  最小值
2. 零均值归一化(Z-score Normalization)
将原始数映射到均值为0 、标准差为l 的分布上
      x - μ / σ     均值为μ 、标准差为σ

![image-20200623164253690](C:\Users\yuty\AppData\Roaming\Typora\typora-user-images\image-20200623164253690.png)
- 在学习速率相同的情况下，X1的更新速度会大于X2， 需要较多的迭代才能找到最优解。
- 如果将X1 和X2 归一化到相同的数值区间后，优化目标的等值图会变成图中的圆形， X1 和X2的更新速度变得更为一致，更快地通过梯度下降找到最优解。

## 类别特征

### 独热编码 (onehot encoding)

### 序号编码（ Ordinal Encoding ）
  高中低    321 保留大小信息

### 二进制编码（ Binary Encoding)
  先用序号编码, 再用二进制表示

### 频数编码（count encoding）
  统计类别出现的次数

### labelcount编码
  根据类别在训练集中的频次排序类别（升序或降序）,用序号表示类别

## 组合特征
  用降维的方法来减小两个高维特征组合后需要学习的参数



## 图像处理

## 文本处理

### word2vec
sentences它是一个list，size表示输出向量的维度。
window：
min_count：用于字典阶段，词频少于min_count次数的单词会被丢弃掉，默认为5

```python
embedding = Word2Vec.load('../res/w2v_all.model')
embedding_dim = embedding.vector_size #向量长度 


model.save('word2vec.model')

model.wv.save_word2vec_format('word2vec.vector')
model = Word2Vec.load('word2vec.model')

#Compute the Word Mover's Distance between two documents.
#计算两个文档的相似度——词移距离算法
model.wv.wmdistance()
 
# Compute cosine similarity between two sets of words.
# 计算两列单词之间的余弦相似度——也可以用来评估文本之间的相似度
model.wv.n_similarity(ws1, ws2)
 
#Compute cosine similarities between one vector and a set of other vectors.
#计算向量之间的余弦相似度
model.wv.cosine_similarities(vector_1, vectors_all)
 
#Compute cosine similarity between two words.
#计算2个词之间的余弦相似度
model.wv.similarity(w1, w2)
 
#Find the top-N most similar words.
# 找出前N个最相似的词
model.wv.most_similar(positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None)

```


### countvec

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



```
### TF - IDF
>TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率).
>TF-IDF是一种统计方法，用以评估一字词对于一个文件集的重要程度或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章.

- 词频 (term frequency, TF) 指的是某一个给定的词语在该文件中出现的次数。这个数字通常会被归一化(一般是词频除以文章总词数), 以防止它偏向长的文件。

  ![img](http://www.ruanyifeng.com/blogimg/asset/201303/bg2013031504.png)

- 逆向文件频率 (inverse document frequency, IDF) IDF的主要思想是：**如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。**某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。
某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。

![img](http://www.ruanyifeng.com/blogimg/asset/201303/bg2013031506.png)

![img](http://www.ruanyifeng.com/blogimg/asset/201303/bg2013031507.png)



### Data Preprocess
```
<PAD>  ：無意義，將句子拓展到相同長度
<BOS>  ：Begin of sentence, 開始字元
<EOS>  ：End of sentence, 結尾字元
<UNK>  ：沒有出現在字典裡的詞
```









## 基础概念

### 二范数
向量范数：向量x的2范数是 x中各个元素平方之和再开根号；



## hyperparameters
In contrast to model **parameters** which are learned during training, model **hyperparameters** are set by the data scientist ahead of training and control implementation aspects of the model. 

Early stopping means training until the validation error does not decrease for a specified number of iterations.

## 集成学习

## 交叉熵（Cross Entropy）
>一般用来求目标与预测值之间的差距
>

1. 信息量
越不可能的事件发生了，我们获取到的信息量就越大
$I(x)=−log(p(x))$
对数均为自然对数, 自然对数是以常数e为底数的对数

2. 熵
熵用来表示所有信息量的期望
$H(X)=−∑p(xi)log(p(xi))$

3. 相对熵（KL散度）
同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x), 可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异
$DKL(p||q)=∑p(xi)log(p(xi)q(xi))$
DKL的值越小，表示q分布和p分布越接近

4. 交叉熵
      $$DKL(p||q) = ∑p(xi)log(p(xi)) − ∑p(xi)log(q(xi))$$
                 $$ = −H(p(x)) + [−∑p(xi)log(q(xi))]$$

$$H(p,q)=−∑p(xi)log(q(xi))$$
在机器学习中，我们需要评估label和predicts之间的差距，使用KL散度刚刚好，即DKL(y||y^)，由于KL散度中的前一部分−H(y)不变，故在优化过程中，只需要关注交叉熵就可以了。所以一般在机器学习中直接用用交叉熵做loss，评估模型



---

# deeplearning

## 感知机算法

## BP神经网络
(Back Propogation)后向传播

## 多层神经网络(DNN)

### 随机梯度下降(SGD) 
1. Learning Rate 的选择
2. 泰勒证明GD


### 激活函数

### 归一化

### 初始化(W, b)
1. 梯度消失和梯度爆炸
2. Batch Normalization

### 目标函数选择
1. softmax
2. 交叉熵
3. 正则项

### 参数更新
>出现的问题

1. AdaGrad
2. RMSProp
3. SGD+Momentum
4. Adam

## 卷积神经网络(CNN)
1. LENET-5


## 神经网络的改进
1. AlexNet
	- 池化
	- 随机丢弃
Dropout说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征，如图1所示。

![img](https://pic2.zhimg.com/80/v2-5530bdc5d49f9e261975521f8afd35e9_720w.jpg)


2. VGGNet
	- 连续多次卷积
3. GoogLeNet
	- 多个小卷积代替大卷积
4. ResNet


## RNN
Vanilla RNN matrix + ReLU

## LSTM
1. 如何解决梯度消失问题


# 广告

# NLP

## 概念
1. 词典
>统计语料库中所有的单词, 每个单词的出现次数


建立词典 vocabulary

## Word Embedding

### one hot

### bag of words (BoW)


### TF- IDF

## Word2Vec

## Seq2Seq
>[1] Cho et al., 2014, learning phase representations using RNN Encoder-decoder for statistical machine translation.
>[2] Sutskever et al, 2014, Sequence to sequence learning with neural networks.

把一个语言序列翻译成另一种语言序列，整个处理过程是通过使用深度神经网络( LSTM (长短记忆网络)，或者RNN (递归神经网络)
![img](https://upload-images.jianshu.io/upload_images/9312194-434209d541580dc3.png?imageMogr2/auto-orient/strip|imageView2/2/w/580/format/webp) 

- 接收输入序列"A B C EOS ( EOS=End of Sentence，句末标记)", 在这个过程中每一个时间点接收一个词或者字，并在读取的EOS时终止接受输入，最后输出一个向量作为输入序列的语义表示向量W，这一过程也被称为编码(Encoder)过程


- 第二个神经网络接收到第一个神经网络产生的输出向量后输出相应的输出语义向量，并且在这个时候每一个时刻输出词的概率都与前一个时刻的输出有关系，模型会将这些序列一次映射为"W X Y Z EOS"，这一过程也被称为解码 (Decoder)过程

![img](https://upload-images.jianshu.io/upload_images/9312194-8cd0fe14adae019f.png?imageMogr2/auto-orient/strip|imageView2/2/w/717/format/webp)


## attention
https://www.cnblogs.com/strangewx/p/10316413.html




## BERT



## 语言模型
语言模型就是用来计算一个句子的概率的模型，也就是判断一句话是否是人话的概率
![img](https://pic2.zhimg.com/80/v2-e8e7c61133d1b23e4d869352aae0c455_720w.png)

1. 參数空间过大：条件概率P(wn|w1,w2,..,wn-1)的可能性太多，无法估算
2. 数据稀疏严重：对于非常多词对的组合，在语料库中都没有出现，依据最大似然估计得到的概率将会是0。

马尔科夫假设：随意一个词出现的概率只与它前面出现的有限的一个或者几个词有关。

- 如果一个词的出现与它周围的词是独立的，那么我们就称之为unigram也就是一元语言模型

  ![img](https://pic3.zhimg.com/80/v2-dfb6d0be8fa42f803d45e27cb02acf5e_720w.png)

- 如果一个词的出现仅依赖于它前面出现的一个词，那么我们就称之为bigram

  ![img](https://pic1.zhimg.com/80/v2-f0e63faeed0dbde5219a3e09778e5b0c_720w.png)

- 假设一个词的出现仅依赖于它前面出现的两个词，那么我们就称之为trigram

  ![img](https://pic3.zhimg.com/80/v2-4c4b2b156e248bc0dea8812b2b5f0002_720w.png)

- 一般来说，N元模型就是假设当前词的出现概率只与它前面的N-1个词有关

  

在实践中用的最多的就是bigram和trigram了，高于四元的用的非常少，由于训练它须要更庞大的语料，并且数据稀疏严重，时间复杂度高，精度却提高的不多。





# python

## print
```python
print('the x is : {}'.format(x))

```

## enumerate
```python
    >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    >>> list(enumerate(seasons))
    [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
    >>> list(enumerate(seasons, start=1))       # 下标从 1 开始
    [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

## flatten()函数用法

>flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用！
```python
>>> from numpy import *
>>> a=array([[1,2],[3,4],[5,6]])
>>> a
array([[1, 2],
    [3, 4],
    [5, 6]])
>>> a.flatten() #默认按行的方向降维
array([1, 2, 3, 4, 5, 6])
>>> a.flatten('F') #按列降维
array([1, 3, 5, 2, 4, 6]) 
>>> a.flatten('A') #按行降维
array([1, 2, 3, 4, 5, 6])
```


# numpy


# pandas
	data = pd.DataFrame()

## 列名
	X_.add_suffix('_suffix')

## 

## 分割训练 / 测试 
```python
question_ids_train = set(pd.Series(question_ids).sample(frac=0.8))
question_ids_valid = set(question_ids).difference(question_ids_train)
X_train = X[X.question_id.isin(question_ids_train)]
X_valid = X[X.question_id.isin(question_ids_valid)]

```

## pandas简化内存

```

```

## groupby
```python
df = df.groupby(['col_name'])
res = df.get_group('col_value')
```

## 聚合
nunique、mean、max、min、std、count



## 压缩矩阵

```python
>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
```
（1）data表示数据，为[1, 2, 3, 4, 5, 6]
（2）shape表示矩阵的形状
（3）indices表示对应data中的数据，在压缩后矩阵中各行的下标，如：数据1在某行的0位置处，数据2在某行的2位置处，数据6在某行的2位置处。
（4）indptr表示压缩后矩阵中每一行所拥有数据的个数，如：[0 2 3 6]表示从第0行开始数据的个数，0表示默认起始点，0之后有几个数字就表示有几行，第一个数字2表示第一行有2 - 0 = 2个数字，因而数字1，2都第0行，第二行有3 - 2 = 1个数字，因而数字3在第1行，以此类推。





# lightGBM

## 参数
- 'objective': 'binary', #定义的目标函数  'objective': 'multiclass',
- 'num_class': 11, # 多分类类别数, 0,1,2, ...
- "lambda_l1": 0.1,             #l1正则
- 'lambda_l2': 0.001,     	#l2正则
- "nthread": -1,              	#线程数量，-1表示全部线程，线程越多，运行的速度越快 
- 'metric': {'binary_logloss',  'auc'},                    #评价函数选择
- 'device': 'gpu' 	             #如果安装的事gpu版本的lightgbm,可以加快运算
- "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
- "feature_fraction": 0.9,  #提取的特征比率
- "bagging_fraction": 0.8, #不进行重采样的情况下随机选择部分数据
- max_depth     #树的最大深度 当模型过拟合时,可以考虑首先降低 max_depth
- "bagging_freq": 1, #bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
- early_stopping_round, 默认为0, type=int, 如果一个验证集的度量在 early_stopping_round 数量次的循环中没有提升, 将停止训练
- verbose_eval：迭代多少次打印




```python

params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'binary', #定义的目标函数
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,             #l1正则
          # 'lambda_l2': 0.001,     #l2正则
          "verbosity": -1,
          "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},  ##评价函数选择
          "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }

                  
                  

```

    params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'binary', #定义的目标函数
         # 'max_depth': -1,
          'learning_rate': 0.03,
          #"min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          #"bagging_freq": 1,
          #"bagging_fraction": 0.8,
          #"bagging_seed": 11,
          "lambda_l1": 0.1,         #l1正则
          # 'lambda_l2': 0.001,     #l2正则
          #"verbosity": -1,
          "nthread": -1,      #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},  ##评价函数选择
          "random_state": 2020, #随机数种子，可以防止每次运行的结果不一致
          #'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算


## metric




# kears


## np_utils
>处理标签为onehot
```python
    from keras.utils import np_utils
    N_CLASSES = 3
    label = [0,0,0,1,1,1,2,2,2]
    train_label = np_utils.to_categorical(label, N_CLASSES)
    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 1.]], dtype=float32)

```

## 参数

models.Sequential，用来一层一层一层的去建立神经层；
layers.Dense 意思是这个神经层是全连接层。
layers.Activation 激励函数。
optimizers.RMSprop 优化器采用 RMSprop，加速神经网络训练方法


# pytorch

> forward pass -> compute loss -> back to compute gradients -> update weight


## tensor
```python
torch.tensor([5.5, 3, 6, 9]) # 从数据直接直接构建tensor
torch.empty(5, 3)      # 构造一个未初始化的5x3矩阵
torch.rand(5, 3)       # 构建一个随机初始化的矩阵
torch.ones(5, 3)
torch.zeros(5, 3, dtype=torch.long)  # 构建一个全部为0，类型为long的矩阵
torch.randn_like(x, dtype=torch.float) # override dtype!result has the same size

x.size() 得到tensor的形状

x + y
torch.add(x, y)
torch.add(x, y, out=result) # result = torch.empty(5, 3)
y.add_(x) # 会改变y
# 任何in-place的运算都会以``_``结尾。 举例来说：``x.copy_(y)``, ``x.t_()``, 会改变 ``x``

x.view(16) # reshape一个tensor
x.view(-1, 8)

x.item() # 只有一个元素的tensor 返回数值

# >>>>>-----和numpy的转换-----<<<<<<
a = torch.ones(5)
b = a.numpy()
b = torch.from_numpy(a)
```

## 矩阵运算
```python
x.dot(y) # x, y 矩阵相乘

```

## CUDA

```python
# run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)              # or just use strings ``.to("cuda")``
    z = x + y
    print(z.to("cpu", torch.double))  
    # ``.to`` can also change dtype together!

```
## autograd
```python
# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 

```



## torch
1. ```torch.multinomial(input, num_samples,replacement=False, out=None) -> LongTensor```
>对input的每一行做 n_samples 次取值，输出的张量是每一次取值时input张量对应行的下标。
输入是一个input张量，一个取样数量，和一个布尔值replacement

- input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。
- n_samples是每一行的取值次数
- replacement指的是取样时是否是有放回的取样，True是有放回，False无放回。

2. `torch.unsqueeze() and torch.squeeze()` 
torch.unsqueeze(2) 第二个维度加一 , 第二维度行变列. (128,100) -> (128,100,1)
torch.unsqueeze() 增加一维
torch.squeeze() 压扁,减少一维

## torch.nn
1. `torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)`


2. ` torch.nn.functional`
>import torch.nn.functional as F

3. `nn.Dropout(p=0.5, inplace=False)`
```python
# 一个矩阵, 每行一半变为0
res = torch.nn.Dropout(p=0.5, inplace=False)

drop = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = drop(input)
```

##  torchtext

```python
import torchtext
from torchtext.vocab import Vectors

# TorchText给我们增加了两个特殊的token，<unk>表示未知的单词，<pad>表示padding
TEXT = torchtext.data.Field(lower=True)
TEXT.vocab.itos[:10]       # 字典中的单词list
TEXT.vocab.stoi['<pad>']   # 字典单词对应的idx (word2idx) -> dict
```

## base模版

- nn

```python
import torch
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. 
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad() # model.zero_grad() 

    # Backward pass: compute gradient of the loss with respect to all the learnable parameters of the model. 
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so we can access its gradients like we did before.
    optimizer.step()
    #with torch.no_grad():
    #    for param in model.parameters():
    #        param -= learning_rate * param.grad
```


- nn.Module

```python
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return a Tensor of output data.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
```





# tensorflow




```python


```



```python


```



```python


```



```python


```















