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

numSamples = 100
numTrials = 500
numNu = 5
trialIndex = 1
nuIndex = 6

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
                accuracy = (float(row[2]) + float(row[4])) / (numSamples/2)
                trialNumber = int(row[1])
                data[rowCount][0][trialNumber - 1] = accuracy
                rowCount = (rowCount+1) % numNu
            return data;
    else:
        with open(path, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            data = np.zeros((numNu, 1, numTrials))
            rowCount = 0
            titles = next(reader)
            for row in reader:
                accuracy = (float(row[2]) + float(row[4])) / (numSamples/2)
                trialNumber = int(row[1])
                data[rowCount][0][trialNumber - 1] = accuracy
                rowCount = (rowCount+1) % numNu
            return data;
                
                
#Just playing around with the ErrorBar and ROC plots.

src = os.getcwd()
path = os.path.join(src, 'trial_results.csv')
trialdata = csv_reader(path)

print(trialdata)
print()

x = []
y = []
yerr = []

# nu-vals for the plot
nu_vals = [0.1, 0.2, 0.3, 0.4, 0.5]

fig = plt.figure()
ax = fig.add_subplot(111)    # The big subplot

for j in range (0, 1):
    for i in range(0, numNu):
        x.append((i+1)*0.10)
        y.append(np.mean(trialdata[i][j], dtype=np.float64))
        print('Result: %d: %f' % (j, y[i]))
        yerr.append(np.std(trialdata[i][j]))
    plt.errorbar(x,y,yerr,fmt='o', capsize=5)
    x = []
    y = []
    yerr = []

'''
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
plt.plot(trialdata[1][0],trialdata[1][1],'ro')
'''

# plt.figure()
ax.set_title('Guassian Noise, SD=8')
ax.set_ylim(.1,1.1)
ax.set_xlabel('Nu')
ax.set_ylabel('Accuracy')
plt.show()
