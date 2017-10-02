print(__doc__)

import numpy as np
import math as math
import matplotlib.pyplot as plt
from sklearn import svm, metrics
# Benchmark contains Glenn's dataset generation functions
from benchmark import Benchmark
import sqlite3
from sqlite3 import Error
import os


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print('SQLite version: %s' % sqlite3.version)
        print()
        return conn
    except Error as e:
        print(e)

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


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
        data[i][0]=np.random.random()*2*x-x
        data[i][1]=np.random.random()*2*y-y
        if data[i][0] > data[i][1]*slope:
            labels.append(1)
        else:
            labels.append(0)
    return data, labels

if __name__ == '__main__':
    database_name = "svmsqlite.db"

    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS projects (
                                            id integer PRIMARY KEY,
                                            name text NOT NULL,
                                            begin_date text,
                                            end_date text
                                        ); """

    sql_create_tasks_table = """CREATE TABLE IF NOT EXISTS tasks (
                                        id integer PRIMARY KEY,
                                        name text NOT NULL,
                                        priority integer,
                                        status_id integer NOT NULL,
                                        project_id integer NOT NULL,
                                        begin_date text NOT NULL,
                                        end_date text NOT NULL,
                                        FOREIGN KEY (project_id) REFERENCES projects (id)
                                    );"""

    # create a database connection
    conn = create_connection(os.getcwd() + database_name)
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_projects_table)
        # create tasks table
        # create_table(conn, sql_create_tasks_table)
    else:
        print("Error! cannot create the database connection.")

num_samples = 100

for i in range(25):
    print('Trial %d\n' % i)

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

    gaussian_noise = np.random.normal(scale=3.0, size=(num_samples, 2))
    poisson_noise = np.random.poisson(size=(num_samples, 2))

    # Add in noise
    X = data + gaussian_noise
    y = labels

    # We create instances of SVM with different values of nu and fit out data. We do not scale our
    # data since we want to plot the support vectors
    kernelFunc = 'linear'

    models = (svm.NuSVC(nu=0.1, kernel=kernelFunc),
              svm.NuSVC(nu=0.3, kernel=kernelFunc),
              svm.NuSVC(nu=0.5, kernel=kernelFunc),
              svm.NuSVC(nu=0.6, kernel=kernelFunc))
    #models = (clf.fit() for clf in models)
    models = (clf.fit(X[:num_samples // 2], y[:num_samples // 2]) for clf in models)

    # title for the plots
    titles = ('0.1',
              '0.3',
              '0.5',
              '0.6')

    # Set-up 2x2 grid for plotting.
    #  fig, sub = plt.subplots(2, 2)
    #  plt.subplots_adjust(wspace=0.4, hspace=0.4)


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

    #  X0, X1 = X[:, 0], X[:, 1]
    #  xx, yy = make_meshgrid(X0, X1)

    for clf, title in zip(models, titles):
        expected = y[num_samples // 2:]
        predicted = clf.predict(X[num_samples // 2:])
        print("Confusion matrix nu=%s:\n%s" % (title, metrics.confusion_matrix(expected, predicted)))

    print('\n')
    #for clf, title, ax in zip(models, titles, sub.flatten()):
    #    plot_contours(ax, clf, xx, yy,
    #                  cmap=plt.cm.coolwarm, alpha=0.8)
    #    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #    ax.set_xlim(xx.min(), xx.max())
    #    ax.set_ylim(yy.min(), yy.max())
    #    ax.set_xlabel('X-axis')
    #    ax.set_ylabel('Y-axis')
    #    ax.set_xticks(())
    #    ax.set_yticks(())
    #    ax.set_title(title)

    #plt.show()
