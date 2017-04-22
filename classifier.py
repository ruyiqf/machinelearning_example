#coding:utf-8
import time
import os
import numpy as np
import pandas as pd
import collections
import random
import pickle
from skimage import io
from sklearn import metrics

class NormalClassifier(object):

    def __init__(self):
        self.label_file_map = collections.defaultdict(list)

    def Sampling(self, path, number):
        """Random extracting samples under path
        :path: Samples dir
        :number: Number of extracing samples in every label
        """
        allfiles = os.listdir(path)
        for image_name in allfiles:
            number_label = image_name.split('.')[0].split('_')[0]
            self.label_file_map[number_label].append(os.path.join(path, image_name))
        
        # 将样本均匀随机抽样切割成训练集合和测试集合
        training_set = collections.defaultdict(list)
        testing_set = collections.defaultdict(list)
        for label in self.label_file_map:
            file_list = self.label_file_map[label]
            training_set[label] = [file_list[random.randint(0,len(file_list)-1)] for i in range(number)] 
            testing_set[label] = set(file_list) - set(training_set[label])

        train_x, train_y = self._generate_data_label_pair(len(training_set)*number, 68*68, training_set)
        test_total_num = 0
        for elt in testing_set:
            test_total_num += len(testing_set[elt])
        test_x, test_y = self._generate_data_label_pair(test_total_num, 68*68, testing_set)
        return (train_x, train_y, test_x, test_y)

    def Sampling4Test(self, path, number):
        allfiles = os.listdir(path)
        for image_name in allfiles:
            number_label = image_name.split('.')[0].split('_')[0]
            self.label_file_map[number_label].append(os.path.join(path, image_name))
        testing_set = collections.defaultdict(list)
        for label in self.label_file_map:
            file_list = self.label_file_map[label]
            testing_set[label] = [file_list[random.randint(0,len(file_list)-1)] for i in range(number)] 
        test_x, test_y = self._generate_data_label_pair(len(testing_set)*number, 68*68, testing_set)
        return (test_x, test_y)
        
    def _generate_data_label_pair(self, row_num, column_num, dataset):
        x = np.empty([row_num, column_num])
        y = list()
        i = 0
        for label in dataset:
            file_list = dataset[label]
            for image_file in file_list:
                x[i] = io.imread(image_file).reshape((1,-1))[0]
                y.append(label)
                i+=1
        return (x, y)
        
    def NaiveBayesClassifier(self, train_x, train_y):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=1)
        model.fit(train_x, train_y)    
        return model

def main():
    nc = NormalClassifier()
    print('********************%s*******************\n'%'NaiveBayes')
    """
    start_time = time.time()
    train_x, train_y, test_x, test_y = nc.Sampling('./BmpMoban', 200)
    print('Build date set time:%fs!'%(time.time() - start_time))
    print('Begin training')
    start_time = time.time()
    model = nc.NaiveBayesClassifier(train_x, train_y)
    pickle.dump(model, open('nc_model', 'wb'))
    print('Training time:%fs!'%(time.time() - start_time))
    """
    test_x, test_y = nc.Sampling4Test('./BmpMoban', 10) 
    model = pickle.load(open('nc_model','rb'))
    predict = model.predict(test_x)
    predict = list(predict)
    cnt = 0
    for elt in predict:
        if elt == test_y[predict.index(elt)]:
            cnt += 1
    print(float(cnt/len(predict)))
    #precision = metrics.precision_score(test_y, predict)
    #recall = metrics.recall_score(test_y, predict)
    #print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    #accuracy = metrics.accuracy_score(test_y, predict)
    #print('accuracy: %.2f%%' % (100 * accuracy))

if __name__ == '__main__':
    main()
