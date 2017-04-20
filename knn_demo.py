# coding: utf-8
# coding: utf-8
import os
import copy
from skimage import io
import numpy as np
import random
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


def load_training_set(path):
    """

    :return: 
    """
    document = dict()
    all = list()
    count = 0
    for parent, dirnames, filenames in os.walk(path):
        for image_name in filenames:  # 输出文件信息
            document["name"] = image_name
            document["category_1"] = image_name[0]
            document["category_2"] = image_name[1]
            document["category_3"] = image_name[2]
            document["random"] = count
            document["img"] = io.imread(os.path.join(path, image_name)).reshape((1, -1))[0]
            count += 1
            all.append(copy.deepcopy(document))
            print(count)
    return all


def _make_knn_model(num=200000, training_set=None, n_neighbors=10, test_num=400):
    if training_set is None:
        training_set = load_training_set()
    # 将样本切割一下做交叉验证
    test_set = training_set[300000:]
    training_set = training_set[0:300000]

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=8)
    X = np.empty([num, 4624])
    y = list()

    for i in range(0, num):
        r = int(random.random() * num)
        X[i] = training_set[r]["img"]
        y.append(training_set[r]["name"][0:3])
    print("样本抽取完成")

    neigh.fit(X, y)
    print("y分类器（三个数字作为整体）构建完成")
    joblib.dump(neigh, "knn_0.model")

    print("随机抽取4个进行验证")
    correct = 0
    for i in range(0, test_num):
        r = int(random.random() * test_num)

        # prediction4 = neigh.predict_proba(test_set[r]["img"].reshape(1, -1)).max()
        answer = test_set[r]["name"][0:3]
        prediction4 = neigh.predict(test_set[r]["img"].reshape(1, -1))
        if prediction4 == answer:
            correct += 1
            print("Correct", answer)
        else:
            print("Wrong", answer)
    print(correct / test_num)
    joblib.dump(neigh, "knn_0.model")
    return neigh


def predict(path, image_name=""):
    knn_model = joblib.load('knn_0.model')
    X = np.empty([1, 4624])
    X[0] = io.imread(os.path.join(path, image_name)).reshape((1, -1))[0]
    prediction = knn_model.predict(np.array(X))
    print(prediction)

if __name__ == "__main__":
    ts = load_training_set(path="/home/emptyset/BmpMoban")
    _make_knn_model(num=200000, training_set=ts, n_neighbors=10, test_num=300)
