# coding=gbk

import time
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
import fashion_mnist_load as mnist_load
import plt_roc as pr


# Multinomial Naive Bayes
def multinomial_naive_bayes(train_x,train_y):
    print('************* Multinomial Naive Bayes ************')
    # ���������˹ƽ�� �Ƿ�Ҫ�����������
    model = MultinomialNB(alpha=1,fit_prior=True)
    model.fit(train_x,train_y)
    return model

# Random Forest
def random_forest(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    print('************* Random Forest ************')
    # �������ã����ĸ�������� gini(���᲻����)����entropy(��Ϣ����)
    model = RandomForestClassifier(n_estimators=100,max_depth=50,criterion='entropy',n_jobs=10)
    model.fit(train_x, train_y)
    return model


# KNN
def knn(train_x, train_y):
    print('************* KNN ************')
    # k�ĸ��� uniform�Ǿ��ȵ�Ȩ�� distance�ǲ����ȵ�Ȩ��
    model = KNeighborsClassifier(n_neighbors=4,weights='distance',n_jobs=10)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':

    # �������ݼ�
    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = mnist_load.get_data()

    # ����ǩ��ֵ��
    # train_y = label_binarize(train_y,classes=[0,1,2,3,4,5,6,7,8,9])
    # test_y = label_binarize(test_y,classes=[0,1,2,3,4,5,6,7,8,9])

    # # �������͸�Ϊfloat32�������ȸ�����
    # train_x = train_x.astype('float32')
    # test_x = test_x.astype('float32')
    # # ���ݹ�һ��
    # train_x /= 255
    # test_x /= 255

    start_time = time.time()

    # psum1 = 0
    # asum2 = 0
    # for i in range(10,50,10):
    #     print('k ================================ %f' % i)
    model = multinomial_naive_bayes(train_x, train_y)

    print('training took %fs!' % (time.time() - start_time))

    start_time = time.time()

    # ����ģ����Ԥ�⣬����Ԥ����
    predict = model.predict(test_x)

    print('predict took %fs!' % (time.time() - start_time))

    print(classification_report(test_y, predict))

    precision = metrics.precision_score(test_y, predict,average='macro')
    recall = metrics.recall_score(test_y, predict,average='macro')
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))
    pr.plt_roc(model,test_x,test_y)
    #     psum1 += precision
    #     asum2 += accuracy
    #
    # print('ave precision = %.2f%%' % ((psum1 / 10) * 100))
    # print('ave accuracy = %.2f%%' % ((asum2 / 10) * 100))
