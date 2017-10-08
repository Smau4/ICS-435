# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 18:12:53 2017

@author: Smau2
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import stat

#Just playing around with the ErrorBar and ROC plots.
x = [0.1, 0.2, 0.3]
y = [2, 4, 3]
yerr = 1

plt.figure()
plt.errorbar(x,y,yerr,fmt='o',color='r', capsize=5)
plt.show()