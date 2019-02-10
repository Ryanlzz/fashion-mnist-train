from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import metrics
import fashion_mnist_load as mnist_load

# iris = load_iris()
# X = iris.data
# y = iris.target

k_range = range(1,10)
k_scores = []

train_x, train_y, test_x, test_y = mnist_load.get_data()

for k in k_range:
    start_time = time.time()
    print('k ========================= %f' % k)
    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=8)
    # 交叉验证
    # loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
    # scores = cross_val_score(knn, X, y, cv=4,scoring='accuracy') # for classification
    knn.fit(train_x,train_y)
    predict = knn.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    k_scores.append(accuracy)
    print('training took %fs!' % (time.time() - start_time))
plt.figure()
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
