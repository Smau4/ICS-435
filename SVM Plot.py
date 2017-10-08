print(__doc__)

import numpy as np
import math as math
import matplotlib.pyplot as plt
from sklearn import svm, metrics
# Benchmark contains Glenn's dataset generation functions
# from benchmark import Benchmark
from database import Database
import os
import csv
import sys


def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    if sys.version_info < (3, 0):
        with open(path, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                writer.writerow(line)
    else:
        with open(path, "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                writer.writerow(line)

#Choose to generate meshgrid or generate a confusion matrix
meshgrids = False

point_id = 1
result_id = 1

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def generate_points_and_labels(x, y, num):
    """ Generate a set of data points and labels
    
    Parameters
    ----------
    x: Length of the x data values (e.g. x=5 will generate values from -5 to 5)
    y: Length of the y data values (e.g. y=10 will generate values from -10 to 10)
    num: Number of training samples to generate """
    
    data = np.zeros((num,2))
    labels = []
    
    #Generates a random decision boundary slope centered at origin
    slope = math.tan(math.pi*np.random.random()-math.pi/2); 
    
    #Generate points and labels
    for i in range(0, num):
        global point_id
        label = 0
        x_point = np.random.random()*2*x-x
        y_point = np.random.random()*2*y-y
        data[i][0]=x_point
        data[i][1]=y_point
        if data[i][0] > data[i][1]*slope:
            label = 1
        else:
            label = 0
        labels.append(label)
        trial_data_records.append([point_id, trial_i, x_point, y_point, label])
        point_id = point_id + 1
    return data, labels


"""
-------------------------------------
main
-------------------------------------
"""

num_samples = 100
num_trials = 25
if meshgrids == True:
    num_trials = 1

trial_records = [['trial_id', 'param_varied']]
trial_data_records = [['point_id', 'trial_id', 'x_point', 'y_point', 'label']]
trial_results_records = [['result_id', 'trial_id', 'true_neg', 'false_neg', 'true_pos', 'false_pos', 'nu_val']]
src = os.getcwd()

for trial_i in range(num_trials):
    trial_records.append([trial_i, 'nu'])
    print('Trial %d\n' % trial_i)

    # Generate Glenn's linear data ###################
    #  a = Benchmark.generate_linear(100, 0.001, 2)
    #  print()
    # X contains data points and y contains labels
    #  X = a[0]
    #  y = a[1]

    #  print(X)
    #  print(y)
    ##################################################

    data, labels = generate_points_and_labels(10, 10, num_samples)
    sd = 2.0
    
    # Generates gaussian and poisson noise.
    # gaussian_noise - error generated from gaussian curve w/ mean = 0 and sd 
    # poisson noise - poisson noise generates a curver with mean = lam and sd = sqrt(lam).  To make this comparable
    # with gaussian_noise, we shift the poisson noise by lam.
    
    gaussian_noise = np.random.normal(scale=sd, size=(num_samples, 2))
    poisson_noise = np.random.poisson(lam=sd*sd,size=(num_samples, 2)) - np.ones((num_samples, 2))*sd*sd
    
#    Checking the variance of each set    
    gaussian_sum = 0
    poisson_sum = 0
#    for point in gaussian_noise:
#        for coordinate in point:
#            gaussian_sum = gaussian_sum + coordinate * coordinate
#            
#    for point in poisson_noise:
#        for coordinate in point:
#            poisson_sum = poisson_sum + coordinate * coordinate
#    
#    print (gaussian_sum/200)
#    print (poisson_sum/200)

    # Add in noise
    X = data + poisson_noise
    y = labels

    # We create instances of SVM with different values of nu and fit out data. We do not scale our
    # data since we want to plot the support vectors
    kernelFunc = 'linear'

    models = (svm.NuSVC(nu=0.1, kernel=kernelFunc),
              svm.NuSVC(nu=0.2, kernel=kernelFunc),
              svm.NuSVC(nu=0.3, kernel=kernelFunc),
              svm.NuSVC(nu=0.4, kernel=kernelFunc),
              svm.NuSVC(nu=0.5, kernel=kernelFunc),
              svm.NuSVC(nu=0.6, kernel=kernelFunc))
    #models = (clf.fit() for clf in models)
    models = (clf.fit(X[:num_samples // 2], y[:num_samples // 2]) for clf in models)

    # title for the plots
    titles = ('0.1',
              '0.2',
              '0.3',
              '0.4',
              '0.5',
              '0.6')
    
    if meshgrids == True:
        
    # Set-up 2x2 grid for plotting.
        fig, sub = plt.subplots(2,3)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)


    # Process Glenn's linear data ###################
    #  X0 = []
    #  X1 = []
    #  for array_point in X:
    #      X0.append(array_point[0])
    #      X1.append(array_point[1])

    #  X0 = np.array(X0)
    #  X1 = np.array(X1)
    #  print()
    #  print(X0)
    #  print(X1)
    ##################################################

        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
    
        for clf, title, ax in zip(models, titles, sub.flatten()):
            plot_contours(ax, clf, xx, yy,
            cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)

        plt.show()
    
    else:
        for clf, title in zip(models, titles):
            expected = y[num_samples // 2:]
            predicted = clf.predict(X[num_samples // 2:])
            print("Confusion matrix nu=%s:\n%s" % (title, metrics.confusion_matrix(expected, predicted)))

        print('\n')

path = os.path.join(src, 'trials.csv')
csv_writer(trial_records, path)
path = os.path.join(src, 'trial_data.csv')
csv_writer(trial_data_records, path)

    
    
