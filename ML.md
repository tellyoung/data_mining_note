[toc]

---

# ML

## hyperparameters
In contrast to model **parameters** which are learned during training, model **hyperparameters** are set by the data scientist ahead of training and control implementation aspects of the model. 

Early stopping means training until the validation error does not decrease for a specified number of iterations.

## 集成学习



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
2. VGGNet
	- 连续多次卷积
3. GoogLeNet
	- 多个小卷积代替大卷积
4. ResNet