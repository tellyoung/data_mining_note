# daily

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




# 特征工程

## 数值特征

## 类别特征
### 独热编码 (onehot encoding)
### 标签编码 (label encoding)
>标签编码直接将类别转换为数字
>


### 频数编码（count encoding）
```python
def count_encode(X, categorical_features, normalize=False):
    '''
    类别出现的次数,来表示类别
    '''
    print('Count encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        X_[cat_feature] = X[cat_feature].astype('object').map(X[cat_feature].value_counts())
        if normalize:
            X_[cat_feature] = X_[cat_feature] / np.max(X_[cat_feature])
            
    X_ = X_.add_suffix('_count_encoded')
    if normalize:
        X_ = X_.astype(np.float32)
        X_ = X_.add_suffix('_normalized')
    else:
        X_ = X_.astype(np.uint32)
        
    return X_

train_count = count_encode(train, ['col1', 'col2'])
```

### labelcount编码
根据类别在训练集中的频次排序类别（升序或降序）,用序号表示类别
```python
def labelcount_encode(X, categorical_features, ascending=True):
    print('LabelCount encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = X[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # 升序
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            # 降序
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        X_[cat_feature] = X[cat_feature].map(
            labelcount_dict)
    X_ = X_.add_suffix('_labelcount_encoded')
    if ascending:
        X_ = X_.add_suffix('_ascending')
    else:
        X_ = X_.add_suffix('_descending')
    X_ = X_.astype(np.uint32)
    return X_

train_lc_subreddit = labelcount_encode(X_train, ['subreddit'])

```

### 目标编码 （target encoding）




## 文本处理

### word2vec


sentences它是一个list，size表示输出向量的维度。
window：
min_count：用于字典阶段，词频少于min_count次数的单词会被丢弃掉，默认为5

```python
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



# pytorch

>CLASS torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
num_embeddings (int) – size of the dictionary of embeddings

embedding_dim (int) – the size of each embedding vector

padding_idx (int, optional) – If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.

max_norm (float, optional) – If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.

norm_type (float, optional) – The p of the p-norm to compute for the max_norm option. Default 2.

scale_grad_by_freq (boolean, optional) – If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.

sparse (bool, optional) – If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.

```python



```



# 基础概念

## 二范数
向量范数：向量x的2范数是 x中各个元素平方之和再开根号；



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


```python


```



```python


```



```python


```



```python


```















