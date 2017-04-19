#coding: utf-8
import os
import sys
import uuid
import random
import kNN
import numpy as np

class SimuDateData(object):

    def __init__(self):
        self.number = 1000

    def work(self):
        miles = list()
        gametimes = list()
        icecream = list()
        labels = list()

        for i in range(self.number):
            miles.append(random.randint(500,3000))
            gametimes.append(random.random())
            icecream.append(random.randint(10,150))

        normMat, ranges, minVals = kNN.autoNorm(np.array(miles))
        miles_normal = list(normMat[0])
        normMat, ranges, minVals = kNN.autoNorm(np.array(gametimes))
        gametimes_normal = list(normMat[0])
        normMat, ranges, minVals = kNN.autoNorm(np.array(icecream))
        icecream_normal = list(normMat[0])


        for i in range(self.number):
            maxval = max(miles_normal[i], gametimes_normal[i], icecream_normal[i])
            if maxval == miles_normal[i]: labels.append(1)
            elif maxval == gametimes_normal[i]: labels.append(2)
            else: labels.append(3)

        with open('datingtest', 'w') as f:
            for i in range(self.number):
                record = str(miles[i]) + \
                         '\t' + str(gametimes[i]) + \
                         '\t' + str(icecream[i]) + \
                         '\t' + str(labels[i]) + '\n'
                f.write(record)

def main():
    sdd = SimuDateData()
    sdd.work()

if __name__ == '__main__':
    main()
