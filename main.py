# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:09:04 2019

@author: USER
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def f(x):
    return np.sin(x) * np.exp(-((np.square(x))/16)) + (1/10) * np.random.normal(0,1,np.size(x))

def make_data(n_sample, n_xj):
    X = np.zeros((n_sample, n_xj+1))
    xr = np.random.uniform(-10, 10, n_sample)
    X[:,0] = xr
    y = f(xr)
    return X, y

def linear_regression(X_training, y_training, X_testing, y_testing=None, plot=False):
    reg = LinearRegression().fit(X_training,y_training)
    prediction = reg.predict(X_testing)
    if plot == True and y_testing is None:
        print('You have to enter the y_testing output in parameters in order to plot a graph !')
    if plot == True and y_testing is not None:
        plt.scatter(X_testing, y_testing,  color='black')
        plt.plot(X_testing, prediction, color='blue', linewidth=3)
        plt.savefig("images/examples/linear_regression.pdf")
        plt.show()
        plt.close()
    return prediction
    
def knn(X_training, y_training, X_testing, y_testing=None, neighbors=1, plot=False):
    knn = KNeighborsRegressor(n_neighbors=neighbors)
    knn = knn.fit(X_training, y_training)
    prediction = knn.predict(X_testing)
    if plot == True and y_testing is None:
        print('You have to enter the y_testing output in parameters in order to plot a graph !')
    if plot == True and y_testing is not None:
        plt.scatter(X_testing, y_testing,  color='black')
        plt.scatter(X_training, y_training,  color='red')
        line = np.linspace(-10, 10, np.size(X_testing)).reshape(-1, 1)
        plt.plot(line, knn.predict(line), color='blue', linewidth=1)
        plt.axis('tight')
        plt.savefig("images/examples/knn.pdf")
        plt.show()
        plt.close()
    return prediction

def linear_estimator(x, N, N_ls, xj = 0):
    hb = np.ones(N)
    noise = np.ones(N)
    y_ls = np.ones(N)
    for i in range(N):
        hb[i] = f(x)
        X, y = make_data(N_ls, xj)
        reg = LinearRegression()
        reg.fit(X,y)
        y_ls[i] = reg.predict([[x]])
    hb = np.mean(hb)
    for i in range(N):
        noise[i] = np.square(f(x) - hb)
    noise = np.mean(noise)
    y_ls_mean = np.mean(y_ls)
    bias = np.square(hb - y_ls_mean)
    variance = np.mean(np.square(y_ls-y_ls_mean))
    error = bias + variance + noise
    return bias, variance, noise, error

def linear_mean_estimator(N, N_ls, xj):
    x = np.arange(-10, 10, 0.1)
    bias = np.ones(np.size(x))
    variance = np.ones(np.size(x))
    noise = np.ones(np.size(x))
    error = np.ones(np.size(x))
    for i in range(np.size(x)):
        bias[i], variance[i], noise[i], error[i] = linear_estimator(x[i], N, N_ls, xj=xj)
    return np.mean(bias), np.mean(variance), np.mean(noise), np.mean(error)
    
def non_linear_estimator(x,N,N_ls,complexity=1, xj=0):
    hb = np.ones(N)
    noise = np.ones(N)
    y_ls = np.ones(N)
    for i in range(N):
        hb[i] = f(x)
        X, y = make_data(N_ls, xj)
        reg = KNeighborsRegressor(n_neighbors=complexity)
        reg.fit(X,y)
        y_ls[i] = reg.predict([[x]])
    hb = np.mean(hb)
    for i in range(N):
        noise[i] = np.square(f(x) - hb)
    noise = np.mean(noise)
    y_ls_mean = np.mean(y_ls)
    bias = np.square(hb - y_ls_mean)
    variance = np.mean(np.square(y_ls-y_ls_mean))
    error = bias + variance + noise
    return bias, variance, noise, error

def non_linear_mean_estimator(N, N_ls, xj, complexity=1):
    x = np.arange(-10, 10, 0.1)
    bias = np.ones(np.size(x))
    variance = np.ones(np.size(x))
    noise = np.ones(np.size(x))
    error = np.ones(np.size(x))
    for i in range(np.size(x)):
        bias[i], variance[i], noise[i], error[i] = non_linear_estimator(x[i], N, N_ls, complexity=complexity, xj=xj)
    return np.mean(bias), np.mean(variance), np.mean(noise), np.mean(error)

def plot_error(N, N_ls, xj=0, complexity=1):
    x = np.arange(-10, 10, 0.1)
    bias = np.ones(np.size(x))
    variance = np.ones(np.size(x))
    noise = np.ones(np.size(x))
    error = np.ones(np.size(x))
    bias1 = np.ones(np.size(x))
    variance1 = np.ones(np.size(x))
    noise1 = np.ones(np.size(x))
    error1 = np.ones(np.size(x))
    for i in range(np.size(x)):
        bias[i], variance[i], noise[i], error[i] = linear_estimator(x[i], N, N_ls, xj=xj)
        bias1[i], variance1[i], noise1[i], error1[i] = non_linear_estimator(x[i], N, N_ls, complexity=complexity, xj=xj)
    fig = plt.figure()
    noise_fig = fig.add_subplot(121)
    noise_fig.plot(x,noise)
    noise1_fig = fig.add_subplot(122)
    noise1_fig.plot(x,noise1)
    noise_fig.set(title='Noise linear', xlabel='x')
    noise1_fig.set(title='Noise non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/error_on_one_sample/noise_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(121)
    bias_fig.plot(x,bias)
    bias1_fig = fig.add_subplot(122)
    bias1_fig.plot(x,bias1)
    bias_fig.set(title='Bias linear', xlabel='x')
    bias1_fig.set(title='Bias non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/error_on_one_sample/bias_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(121)
    variance_fig.plot(x,variance)
    variance1_fig = fig.add_subplot(122)
    variance1_fig.plot(x,variance1)
    variance_fig.set(title='Variance linear', xlabel='x')
    variance1_fig.set(title='Variance non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/error_on_one_sample/variance_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(121)
    error_fig.plot(x,error)
    error1_fig = fig.add_subplot(122)
    error1_fig.plot(x,error1)
    error_fig.set(title='Error linear', xlabel='x')
    error1_fig.set(title='Error non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/error_on_one_sample/error_plot.pdf")
    plt.show()
    plt.close()
    
def plot_linear_mean_by_size(minSize, maxSize, step, N=100, xj=0):
    size = int(np.abs(minSize-maxSize)/step)
    bias = np.ones(size)
    variance = np.ones(size)
    noise = np.ones(size)
    error = np.ones(size)
    counter = 0
    for i in range(minSize, maxSize, step):
        bias[counter], variance[counter], noise[counter], error[counter] = linear_mean_estimator(N, i, xj)
        counter += 1
    fig = plt.figure()
    noise_fig = fig.add_subplot(111)
    noise_fig.plot(range(minSize, maxSize, step), noise)
    noise_fig.set(title='Noise linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/noise_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(range(minSize, maxSize, step),bias)
    bias_fig.set(title='Bias linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/bias_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(range(minSize, maxSize, step),variance)
    variance_fig.set(title='Variance linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/variance_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(range(minSize, maxSize, step),error)
    error_fig.set(title='Error linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/error_linear_plot.pdf")
    plt.show()
    plt.close()
    
def plot_non_linear_mean_by_size(minSize, maxSize, step, N=100, xj=0, complexity=1):
    size = int(np.abs(minSize-maxSize)/step)
    bias = np.ones(size)
    variance = np.ones(size)
    noise = np.ones(size)
    error = np.ones(size)
    counter = 0
    for i in range(minSize, maxSize, step):
        bias[counter], variance[counter], noise[counter], error[counter] = non_linear_mean_estimator(N, i, xj, complexity=complexity)
        counter += 1
    fig = plt.figure()
    noise_fig = fig.add_subplot(111)
    noise_fig.plot(range(minSize, maxSize, step), noise)
    noise_fig.set(title='Noise non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/noise_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(range(minSize, maxSize, step),bias)
    bias_fig.set(title='Bias non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/bias_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(range(minSize, maxSize, step),variance)
    variance_fig.set(title='Variance non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/variance_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(range(minSize, maxSize, step),error)
    error_fig.set(title='Error non linear', xlabel='x')
    fig.tight_layout()
    plt.savefig("images/mean_values/error_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
if __name__ == "__main__":

    n_data = 300
    n_training = int(n_data/10)
    X, y = make_data(n_data, 0)
    X_training, y_training = X[0:n_training], y[0:n_training]
    X_testing, y_testing = X[n_training:n_data], y[n_training:n_data]
    
    prediction_linear_regression = linear_regression(X_training, y_training, X_testing, y_testing, plot=True)
    prediction_knn = knn(X_training, y_training, X_testing, y_testing, neighbors=1, plot=True)
    
    #plot_error(100,1000) 
    #plot_linear_mean_by_size(100,1000,100)
    #plot_non_linear_mean_by_size(100,1000,100)
    
