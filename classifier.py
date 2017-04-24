#coding:utf-8
import time
import os
import numpy as np
import pandas as pd
import collections
import random
import pickle
from skimage import io
from sklearn.externals import joblib
from sklearn import metrics

class NormalClassifier(object):

    def __init__(self):
        self.label_file_map = collections.defaultdict(list)
        self.classifier = {'NB':self.NaiveBayesClassifier,
                           'LR':self.LogisticRegressionClassifier,
                           'RF':self.RandomForestClassifier,
                           'DT':self.DecisionTreeClassifier,
                           'SVM':self.SvmClassifier,
                           'SVMCV':self.SvmCrossValidationClassifier,
                           'GBDT':self.GradientBoostingClassifier}

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

    def LogisticRegressionClassifier(self, train_x, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2')
        model.fit(train_x, train_y)
        return model

    def RandomForestClassifier(self, train_x, train_y):    
        from sklearn.ensemble import RandomForestClassifier    
        model = RandomForestClassifier(n_estimators=8)    
        model.fit(train_x, train_y)    
        return model    

    def DecisionTreeClassifier(self, train_x, train_y):    
        from sklearn import tree    
        model = tree.DecisionTreeClassifier()    
        model.fit(train_x, train_y)    
        return model

    def GradientBoostingClassifier(self, train_x, train_y):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(train_x, train_y)
        return model

    def SvmClassifier(self, train_x, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        model.fit(train_x, train_y)
        return model
    
    def SvmCrossValidationClassifier(self, train_x, train_y):    
        from sklearn.grid_search import GridSearchCV
        from sklearn.svm import SVC 
        model = SVC(kernel='rbf', probability=True)    
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
        grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)    
        grid_search.fit(train_x, train_y)    
        best_parameters = grid_search.best_estimator_.get_params()    
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
        model.fit(train_x, train_y)    
        return model    

    def GenerateOneModel(self, model_name, train_x, train_y):
        """Generate training model
        :model_name: []
        :train_x: training data
        :train_y: training label
        """
        print('Begin training %s modle'%model_name)
        start_time = time.time()
        model = self.classifier[model_name](train_x, train_y)
        joblib.dump(model, model_name+'_model')
        print('Trained %s model time:%fs!'%(model_name, time.time()-start_time))

    def PredictSet(self, path, model_name):
        allfiles = os.listdir(path)
        x = np.empty([len(allfiles), 68*68])
        i = 0
        file_index_map = collections.defaultdict(str)
        ret = collections.defaultdict(str)
        for elt in allfiles:
            x[i] = io.imread(os.path.join(path, elt)).reshape((1,-1))[0]
            file_index_map[i] = elt
            i+=1
        model = joblib.load(model_name+'_model')
        predict = model.predict(x)
        for i in range(len(predict)):
            ret[file_index_map[i]] = predict[i]
        return ret

    def ValidateSet(self, path, number, model_name):
        test_x, test_y = self.Sampling4Test(path, number)
        model = joblib.load(model_name+'_model')
        predict = model.predict(test_x)
        test_y = np.array(test_y, dtype=np.int)
        predict = predict.astype(int)
        for i in range(len(predict)):
            if predict[i] == test_y[i]:
                print(predict[i])
        precision = metrics.precision_score(test_y, predict, average='micro')
        recall = metrics.recall_score(test_y, predict, average='micro')
        accuracy = metrics.accuracy_score(test_y, predict)
        print('precision: %.2f%%' % (100 * precision))
        print('recall: %.2f%%' % (100 * recall))
        print('accuracy: %.2f%%' % (100 * accuracy))

"""
Testing code
"""
def main():
    nc = NormalClassifier()
    start_time = time.time()
    train_x, train_y, test_x, test_y = nc.Sampling('./BmpMoban', 500)
    print('Build date set time:%fs!'%(time.time() - start_time))

    print('Build NB model')
    start_time = time.time()
    nc.GenerateOneModel('NB', train_x, train_y)
    print('Finish NB time:%fs!'%(time.time() - start_time))

    print('Build LR model')
    start_time = time.time()
    nc.GenerateOneModel('LR', train_x, train_y)
    print('Finish LR time:%fs!'%(time.time() - start_time))

    print('Build RF model')
    start_time = time.time()
    nc.GenerateOneModel('RF', train_x, train_y)
    print('Finish RF time:%fs!'%(time.time() - start_time))
    
    print('Build DT model')
    start_time = time.time()
    nc.GenerateOneModel('DT', train_x, train_y)
    print('Finish DT time:%fs!'%(time.time() - start_time))

    print('Build SVM model')
    start_time = time.time()
    nc.GenerateOneModel('SVM', train_x, train_y)
    print('Finish SVM time:%fs!'%(time.time() - start_time))

    print('Build SVMCV model')
    start_time = time.time()
    nc.GenerateOneModel('SVMCV', train_x, train_y)
    print('Finish SVMCV time:%fs!'%(time.time() - start_time))
    
    print('Build GBDT model')
    start_time = time.time()
    nc.GenerateOneModel('GBDT', train_x, train_y)
    print('Finish GBDT time:%fs!'%(time.time() - start_time))

if __name__ == '__main__':
    main()
