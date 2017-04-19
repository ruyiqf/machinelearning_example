#coding:utf-8

import collections
import operator

from math import log

class DecideTree(object):

    def __init__(self):
        pass

    def create_data_set(self):
        data_set = [[1,1,'yes'],
                    [1,1,'yes'],
                    [1,0,'no'],
                    [0,1,'no'],
                    [0,1,'no']]
        labels = ['no surfacing', 'flippers']
        return data_set, labels

    def cal_shannon_ent(self, dataset):
        num_entries = len(dataset)
        label_counts = collections.defaultdict(int)
        for feat_vec in dataset:
            current_label = feat_vec[-1]
            label_counts[current_label] += 1
        shannon_ent = 0.0
        for key in label_counts:
            #Calculate probability
            prob = float(label_counts[key]) / num_entries
            shannon_ent -= prob * log(prob,2)
        return shannon_ent

    def split_data_set(self, dataset, axis, value):
        ret_data_set = []
        for feat_vec in dataset:
            if feat_vec [axis] == value:
                reduce_feat_vec = feat_vec[:axis]
                reduce_feat_vec.extend(feat_vec[axis+1:])
                ret_data_set.append(reduce_feat_vec)
        return ret_data_set

    def choose_best_feat_to_split(self, dataset):
        num_features = len(dataset[0]) - 1
        base_entropy = self.cal_shannon_ent(dataset)
        best_info_gain = .0
        best_feature = -1
        for i in range(num_features):
            feat_list = [example[i] for example in dataset]
            unique_vals = set(feat_list)
            new_entropy = .0
            for value in unique_vals:
                sub_data_set = self.split_data_set(dataset, i, value)
                prob = len(sub_data_set) / float(len(dataset))
                new_entropy += prob * self.cal_shannon_ent(sub_data_set)
            info_gain = base_entropy - new_entropy
            if (info_gain > best_info_gain):
                best_info_gain = info_gain
                best_feature = i
        return best_feature
    
    def majority_cnt(self, class_list):
        class_count = collections.defaultdict(int)
        for vote in class_list:
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    def create_tree(self, dataset, labels):
        class_list = [example[-1] for example in dataset]
        
        #The elements in list are the same
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(dataset[0]) == 1:
            return self.majority_cnt(class_list)

        #Recursive procedure
        best_feat = self.choose_best_feat_to_split(dataset)
        best_feat_label = labels[best_feat]
        mytree = {best_feat_label:{}}
        del(labels[best_feat])
        feat_values = [example[best_feat] for example in dataset]
        unique_vals = set(feat_values)
        for value in unique_vals:
            #copy labels to sublabels
            sublabels = labels[:]
            mytree[best_feat_label][value] = self.create_tree(self.split_data_set(dataset, best_feat, value),
                                                             sublabels)
        return mytree
            
"""
Testing code
"""
def main():
    dt = DecideTree()
    dat, labels = dt.create_data_set()
    #print(dt.choose_best_feat_to_split(dat))
    print(dt.create_tree(dat, labels))

if __name__ == '__main__':
    main()
    
