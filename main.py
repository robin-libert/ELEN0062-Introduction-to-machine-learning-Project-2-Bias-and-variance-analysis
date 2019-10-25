# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:09:04 2019

@author: USER
"""

import numpy as np
from matplotlib import pyplot as plt

def make_data(n_sample):    
    X = np.zeros((n_sample, 2))
    y = np.random.randint(low = -1, high = 1, size = n_sample)
    y[y == 0] = 1
    
    std_dev = 0.1
    neg_mean = 1
    pos_mean = 2
    
    neg_normal_distrib = np.random.normal(neg_mean, std_dev, n_sample)
    pos_normal_distrib = np.random.normal(pos_mean, std_dev, n_sample)
    uniform_distrib = np.random.uniform(0, 2*np.pi, n_sample)
    
    for i in range(n_sample):
        if y[i] == -1:
            X[i][0] = neg_normal_distrib[i] * np.cos(uniform_distrib[i])
            X[i][1] = neg_normal_distrib[i] * np.sin(uniform_distrib[i])
        else:
            X[i][0] = pos_normal_distrib[i] * np.cos(uniform_distrib[i])
            X[i][1] = pos_normal_distrib[i] * np.sin(uniform_distrib[i])
    return X, y

def plot_datas(X, y):
    plt.title("Datas question 1")
    plt.xlabel("X_0")
    plt.ylabel("X_1")

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    
        
if __name__ == "__main__":
    X, y = make_data(1000)
    plot_datas(X, y)
