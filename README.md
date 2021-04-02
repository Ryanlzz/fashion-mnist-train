# fashion-mnist-train

> 详细测试结果参考目录下的fashion_mnist.docm

### 三种机器学习分类器：
- 随机森林
- KNN
- 朴素贝叶斯

### 卷积网络结构
> 两层卷积层，一个全连接层

### 数据集
> fashion-mnist

### 机器学习工具
> sk-learn

### 深度学习框架
> pytorch(GPU)

### 运行代码流程
 
#### 1.获取数据集
> 解压fashion_mnist目录下的fashion_mnist_data.zip，共4个文件，为训练集、测试集、训练集标签、测试集标签。

#### 2.数据可视化和标签制作
> 运行make_data.py，可以在fashion_mnist下得到训练集和测试集的图片文件、训练集和测试集标签。

#### 3.机器学习分类器测试
> 运行train_minst.py，可以测试三种不同机器学习分类器的性能。

#### 4.深度学习卷积网络测试
> 运行fashion_mnist_cnn.py,可以改变超参数LR，EPOCH，BATCH_SIZE来调节准确率。

### 其他工具
- KNN.py 找到KNN中的最佳参数k
- Visualization_module.py 可视化网络结构
- fashion_mnist_load.py 机器学习分类器数据集准备
- fashion_mnist_data_ready.py 卷积神经网络数据集准备
- plt_roc.py 绘制机器学习分类器性能的ROC曲线
- select_optimizers.py 测试深度学习下不同优化器的性能
- tensorboard.py 数据集可视化工具。