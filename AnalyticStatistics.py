# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 18:12:53 2017

@author: Smau2
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt
from sklearn import svm, metrics

#Just playing around with the ErrorBar and ROC plots.
x = 1
y = 2
yerr = 1

plt.figure()
plt.errorbar(x,y,yerr,fmt='o',color='r')