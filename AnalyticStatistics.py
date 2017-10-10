# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 18:12:53 2017

@author: Smau2
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import csv
import os
import sys

numTrials = 25;
numNu = 6;
trialIndex = 1;
nuIndex = 6;

def csv_reader(path):
    """
    Read data from CSV file path
    """
    if sys.version_info < (3, 0):
        with open(path, "rb") as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            data = np.zeros((numNu, 2, numTrials))
            rowCount = 0
            titles = next(reader)
            for row in reader:
                sensitivity = float(row[4])/(float(row[4])+float(row[5]))
                specificity = float(row[2])/(float(row[2])+float(row[3]))
                trialNumber = int(row[1])
                data[rowCount][0][trialNumber-1] = sensitivity
                data[rowCount][1][trialNumber-1] = specificity
                rowCount = (rowCount+1) % numNu
            return data;
    else:
        with open(path, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            data = np.zeros((numNu, numTrials, 2))
            rowCount = 0
            titles = next(reader)
            for row in reader:
                sensitivity = float(row[4])/(float(row[4])+float(row[5]))
                specificity = float(row[2])/(float(row[2])+float(row[3]))
                trialNumber = int(row[1])
                data[rowCount][trialNumber-1][0] = sensitivity
                data[rowCount][trialNumber-1][1] = specificity
                rowCount = (rowCount+1) % numNu
            return data;
                
                
#Just playing around with the ErrorBar and ROC plots.

src = os.getcwd()
path = os.path.join(src, 'trial_results.csv')
trialdata = csv_reader(path)

x = []
y = []
yerr = []
for i in range (0, numNu*2):
    x.append(i)
    y.append(np.mean(trialdata[i//2][i%2]))
    yerr.append(np.std(trialdata[i//2][i%2]))
print x
print y
print yerr
'''
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
plt.plot(trialdata[1][0],trialdata[1][1],'ro')
'''

plt.figure()
plt.errorbar(x,y,yerr,fmt='o',color='r', capsize=5)
plt.show()