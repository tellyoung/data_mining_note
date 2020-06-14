# daily

# python

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

## pandas简化内存

```

```



# word2vec


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


# countvec

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



```

















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
- 'lambda_l2': 0.001,     #l2正则
- "nthread": -1,           #线程数量，-1表示全部线程，线程越多，运行的速度越快 
- 'metric': {'binary_logloss',  'auc'},  ##评价函数选择
- 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
- "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
- "feature_fraction": 0.9,  #提取的特征比率
- "bagging_fraction": 0.8, 不进行重采样的情况下随机选择部分数据
- max_depth     #树的最大深度 当模型过拟合时,可以考虑首先降低 max_depth
- "bagging_freq": 1, #bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
- early_stopping_round, 默认为0, type=int, 也称early_stopping_rounds, early_stopping。如果一个验证集的度量在 early_stopping_round 循环中没有提升, 将停止训练
- verbose_eval：迭代多少次打印


## metric




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


































