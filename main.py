# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:09:04 2019

@author: Leloup Emeline, Libert Robin
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

def f(x):
    """
        Function wich takes x as an input and return the desired output given in our project assesment
    """
    return np.sin(x) * np.exp(-((np.square(x))/16)) + (1/10) * np.random.normal(0,1,np.size(x))

def make_data(n_sample, n_xj):
    """
        n_sample: the number of objects sample we want
        n_xj: the number of irrelevant variables for each objects
        return our datas where X is the input matrix and y the output matrix
    """
    X = np.zeros((n_sample, n_xj+1))
    xr = np.random.uniform(-10, 10, n_sample)
    X[:,0] = xr
    for i in range(n_xj):
        X[:,i+1] = np.random.uniform(-10,10, n_sample)
    y = f(xr)
    return X, y

def linear_regression(X_training, y_training, X_testing, y_testing=None, alpha=0, plot=False):
    reg = Ridge(alpha).fit(X_training,y_training)
    prediction = reg.predict(X_testing)
    if plot == True and y_testing is None:
        print('You have to enter the y_testing output in parameters in order to plot a graph !')
    if plot == True and y_testing is not None:
        plt.scatter(X_testing[:,0], y_testing, color='black')
        plt.scatter(X_training[:,0], y_training,  color='red')
        Z = np.copy(X_testing)
        Z = Z[Z[:,0].argsort()]
        plt.plot(Z, reg.predict(Z), linewidth=1, color='blue')
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
        plt.scatter(X_testing[:,0], y_testing,  color='black')
        plt.scatter(X_training[:,0], y_training,  color='red')
        Z = np.copy(X_testing)
        Z[:,0] = np.sort(Z[:,0])
        plt.plot(Z[:,0], knn.predict(Z), color='blue', linewidth=1)
        plt.axis('tight')
        plt.savefig("images/examples/knn.pdf")
        plt.show()
        plt.close()
    return prediction

def linear_estimator(x, N, N_ls, complexity=0, xj = 0):
    """
        x: the desired x we want to compute
        N: the number of sample we use
        N_ls: the size of our samples
        [optional] complexity: the size of the alpha parameter for the linear regression
        [optional] xj: the number of irrelevant variables
        return the squared bias, the variance, the noise and the error at a given x
    """
    hb = np.ones(N)
    noise = np.ones(N)
    y_ls = np.ones(N)
    for i in range(N):
        hb[i] = f(x[0])
        X, y = make_data(N_ls, xj)
        reg = Ridge(alpha=complexity)
        reg = reg.fit(X,y)
        y_ls[i] = reg.predict([x])
    hb = np.mean(hb)
    for i in range(N):
        noise[i] = np.square(f(x[0]) - hb)
    noise = np.mean(noise)
    y_ls_mean = np.mean(y_ls)
    bias = np.square(hb - y_ls_mean)
    variance = np.mean(np.square(y_ls-y_ls_mean))
    error = bias + variance + noise
    return bias, variance, noise, error

def linear_mean_estimator(N, N_ls, xj, complexity=0):
    """
        N: the number of sample we use
        N_ls: the size of our samples
        [optional] xj: the number of irrelevant variables
        [optional] complexity: the size of the alpha parameter for the linear regression
        return the mean squared bias, the mean variance, the mean noise and the mean error at a given x
    """
    x0 = np.arange(-10, 10, 0.1)
    x = np.ones((len(x0), xj+1))
    x[:,0] = x0
    for i in range(xj):
        x[:,i+1] = np.random.uniform(-10, 10, len(x0))
    bias = np.ones(len(x))
    variance = np.ones(len(x))
    noise = np.ones(len(x))
    error = np.ones(len(x))
    for i in range(len(x)):
        bias[i], variance[i], noise[i], error[i] = linear_estimator(x[i], N, N_ls, complexity=complexity, xj=xj)
    return np.mean(bias), np.mean(variance), np.mean(noise), np.mean(error)
    
def non_linear_estimator(x,N,N_ls,complexity=1, xj=0):
    """
        x: the desired x we want to compute
        N: the number of sample we use
        N_ls: the size of our samples
        [optional] complexity: the size of the number of neighbors of K-NN
        [optional] xj: the number of irrelevant variables
        return the squared bias, the variance, the noise and the error at a given x
    """
    hb = np.ones(N)
    noise = np.ones(N)
    y_ls = np.ones(N)
    for i in range(N):
        hb[i] = f(x[0])
        X, y = make_data(N_ls, xj)
        reg = KNeighborsRegressor(n_neighbors=complexity)
        reg = reg.fit(X,y)
        y_ls[i] = reg.predict([x])
    hb = np.mean(hb)
    for i in range(N):
        noise[i] = np.square(f(x[0]) - hb)
    noise = np.mean(noise)
    y_ls_mean = np.mean(y_ls)
    bias = np.square(hb - y_ls_mean)
    variance = np.mean(np.square(y_ls-y_ls_mean))
    error = bias + variance + noise
    return bias, variance, noise, error

def non_linear_mean_estimator(N, N_ls, xj, complexity=1):
    """
        N: the number of sample we use
        N_ls: the size of our samples
        [optional] xj: the number of irrelevant variables
        [optional] complexity: the size of the number of neighbors of K-NN
        return the mean squared bias, the mean variance, the mean noise and the mean error at a given x
    """
    x0 = np.arange(-10, 10, 0.1)
    x = np.ones((len(x0), xj+1))
    x[:,0] = x0
    for i in range(xj):
        x[:,i+1] = np.random.uniform(-10, 10, len(x0))
    bias = np.ones(len(x))
    variance = np.ones(len(x))
    noise = np.ones(len(x))
    error = np.ones(len(x))
    for i in range(len(x)):
        bias[i], variance[i], noise[i], error[i] = non_linear_estimator(x[i], N, N_ls, complexity=complexity, xj=xj)
    return np.mean(bias), np.mean(variance), np.mean(noise), np.mean(error)

def plot_error(N, N_ls, xj=0, neighbors=1, alpha=0):
    """
        Plot the noise, the squared bias, the variance and the error for the linear and the non linear method.
        N: the number of sample we considered for our computation
        N_ls: the size of the sample
        [optional]xj: the number of irrelevant variables we add to our data
        [optional]neighbors: the number of neighbors for the non linear method
        [optional]alpha: the size of the alpha for the regression
    """
    x0 = np.arange(-10, 10, 0.1)
    x = np.ones((len(x0), xj+1))
    x[:,0] = x0
    for i in range(xj):
        x[:,i+1] = np.random.uniform(-10, 10, len(x0))
    bias = np.ones(len(x))
    variance = np.ones(len(x))
    noise = np.ones(len(x))
    error = np.ones(len(x))
    bias1 = np.ones(len(x))
    variance1 = np.ones(len(x))
    noise1 = np.ones(len(x))
    error1 = np.ones(len(x))
    for i in range(len(x)):
        bias[i], variance[i], noise[i], error[i] = linear_estimator(x[i], N, N_ls, complexity=alpha, xj=xj)
        bias1[i], variance1[i], noise1[i], error1[i] = non_linear_estimator(x[i], N, N_ls, complexity=neighbors, xj=xj)
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
    bias_fig.set(title='Squared Bias linear', xlabel='x')
    bias1_fig.set(title='Squared Bias non linear', xlabel='x')
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
    
def plot_linear_mean_by_size(minSize, maxSize, step, N=100, xj=0, complexity=0):
    """
        Plot the mean noise, squared bias, variance and error for the linear method, for a given number of computation.
        minSize: the minimum size we allowed for our sample of data
        maxSize: the maximum size we allowed for ou sample of data
        step: the number of steps in order not to consider all different size of data between minSize and maxSize. (The computation would be too long)
        [optional]N: the number of sample we considered to compute the mean
        [optional]xj: the number of irrelevant variables
        [optional]complexity: the size of alpha
    """
    size = int(np.abs(minSize-maxSize)/step)
    bias = np.ones(size)
    variance = np.ones(size)
    noise = np.ones(size)
    error = np.ones(size)
    counter = 0
    for i in range(minSize, maxSize, step):
        bias[counter], variance[counter], noise[counter], error[counter] = linear_mean_estimator(N, i, xj, complexity)
        counter += 1
    fig = plt.figure()
    noise_fig = fig.add_subplot(111)
    noise_fig.plot(range(minSize, maxSize, step), noise)
    noise_fig.set(title='Noise linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/noise_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(range(minSize, maxSize, step),bias)
    bias_fig.set(title='Squared Bias linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/bias_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(range(minSize, maxSize, step),variance)
    variance_fig.set(title='Variance linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/variance_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(range(minSize, maxSize, step),error)
    error_fig.set(title='Error linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/error_linear_plot.pdf")
    plt.show()
    plt.close()
    
def plot_non_linear_mean_by_size(minSize, maxSize, step, N=100, xj=0, complexity=1):
    """
        Plot the mean noise, squared bias, variance and error for the non linear method, for a given number of computation.
        minSize: the minimum size we allowed for our sample of data
        maxSize: the maximum size we allowed for ou sample of data
        step: the number of steps in order not to consider all different size of data between minSize and maxSize. (The computation would be too long)
        [optional]N: the number of sample we considered to compute the mean
        [optional]xj: the number of irrelevant variables
        [optional]complexity: the number of neighbors for K-NN
    """
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
    noise_fig.set(title='Noise non linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/noise_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(range(minSize, maxSize, step),bias)
    bias_fig.set(title='Squared Bias non linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/bias_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(range(minSize, maxSize, step),variance)
    variance_fig.set(title='Variance non linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/variance_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(range(minSize, maxSize, step),error)
    error_fig.set(title='Error non linear', xlabel='size')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_size/error_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
def plot_linear_mean_by_complexity(complexity, N=100, N_ls=1000, xj=0):
    """
        Plot the mean noise, squared bias, variance and error for the linear method, for a given number of computation.
        complexity: a list containing float which are the desired size of alpha we want to test
        [optional]N: the number of sample we considered to compute the mean
        [optional]N_ls: the size of the samples we tested
        [optional]xj: the number of irrelevant variables
    """
    size = len(complexity)
    bias = np.ones(size)
    variance = np.ones(size)
    noise = np.ones(size)
    error = np.ones(size)
    counter = 0
    for i in complexity:
        bias[counter], variance[counter], noise[counter], error[counter] = linear_mean_estimator(N, N_ls, xj=xj, complexity=i)
        counter += 1
    fig = plt.figure()
    noise_fig = fig.add_subplot(111)
    noise_fig.plot(complexity, noise)
    noise_fig.set(title='Noise linear', xlabel='alpha')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/noise_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(complexity,bias)
    bias_fig.set(title='Squared Bias linear', xlabel='alpha')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/bias_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(complexity,variance)
    variance_fig.set(title='Variance linear', xlabel='alpha')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/variance_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(complexity,error)
    error_fig.set(title='Error linear', xlabel='alpha')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/error_linear_plot.pdf")
    plt.show()
    plt.close()
    
def plot_non_linear_mean_by_complexity(complexity, N=100, N_ls=1000, xj=0):
    """
        Plot the mean noise, squared bias, variance and error for the linear method, for a given number of computation.
        complexity: a list containing float which are the desired number of neighbors we want to test
        [optional]N: the number of sample we considered to compute the mean
        [optional]N_ls: the size of the samples we tested
        [optional]xj: the number of irrelevant variables
    """
    size = len(complexity)
    bias = np.ones(size)
    variance = np.ones(size)
    noise = np.ones(size)
    error = np.ones(size)
    counter = 0
    for i in complexity:
        bias[counter], variance[counter], noise[counter], error[counter] = non_linear_mean_estimator(N, N_ls, xj=xj, complexity=i)
        counter += 1
    fig = plt.figure()
    noise_fig = fig.add_subplot(111)
    noise_fig.plot(complexity, noise)
    noise_fig.set(title='Noise non linear', xlabel='number of neighbors')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/noise_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(complexity,bias)
    bias_fig.set(title='Squared Bias non linear', xlabel='number of neighbors')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/bias_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(complexity,variance)
    variance_fig.set(title='Variance non linear', xlabel='number of neighbors')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/variance_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(complexity,error)
    error_fig.set(title='Error non linear', xlabel='number of neighbors')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_complexity/error_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
def plot_linear_mean_by_irrelevant_variables(xj, N_ls=1000, N=100, complexity=0):
    """
        Plot the mean noise, squared bias, variance and error for the linear method, for a given number of computation.
        xj: a list containing int which are the number of irrelevant variables
        [optional]N: the number of sample we considered to compute the mean
        [optional]N_ls: the size of the samples we tested
        [optional]complexity: the size of alpha for the regression
    """
    size = len(xj)
    bias = np.ones(size)
    variance = np.ones(size)
    noise = np.ones(size)
    error = np.ones(size)
    counter = 0
    for i in xj:
        bias[counter], variance[counter], noise[counter], error[counter] = linear_mean_estimator(N, N_ls, i, complexity)
        counter += 1
    fig = plt.figure()
    noise_fig = fig.add_subplot(111)
    noise_fig.plot(xj, noise)
    noise_fig.set(title='Noise linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/noise_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(xj,bias)
    bias_fig.set(title='Squared Bias linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/bias_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(xj,variance)
    variance_fig.set(title='Variance linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/variance_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(xj,error)
    error_fig.set(title='Error linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/error_linear_plot.pdf")
    plt.show()
    plt.close()
    
def plot_non_linear_mean_by_irrelevant_variables(xj, N_ls=1000, N=100, complexity=1):
    """
        Plot the mean noise, squared bias, variance and error for the linear method, for a given number of computation.
        xj: a list containing int which are the number of irrelevant variables
        [optional]N: the number of sample we considered to compute the mean
        [optional]N_ls: the size of the samples we tested
        [optional]complexity: the number of neighbors for K-NN
    """
    size = len(xj)
    bias = np.ones(size)
    variance = np.ones(size)
    noise = np.ones(size)
    error = np.ones(size)
    counter = 0
    for i in xj:
        bias[counter], variance[counter], noise[counter], error[counter] = non_linear_mean_estimator(N, N_ls, i, complexity)
        counter += 1
    fig = plt.figure()
    noise_fig = fig.add_subplot(111)
    noise_fig.plot(xj, noise)
    noise_fig.set(title='Noise non linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/noise_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    bias_fig = fig.add_subplot(111)
    bias_fig.plot(xj,bias)
    bias_fig.set(title='Squared Bias non linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/bias_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    variance_fig = fig.add_subplot(111)
    variance_fig.plot(xj,variance)
    variance_fig.set(title='Variance non linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/variance_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
    fig = plt.figure()
    error_fig = fig.add_subplot(111)
    error_fig.plot(xj,error)
    error_fig.set(title='Error non linear', xlabel='Number irrelevant variables')
    fig.tight_layout()
    plt.savefig("images/mean_values_by_irrelevant_variables/error_non_linear_plot.pdf")
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    #some code in order to visualize our data
    #the size of the testing set equals 900
    #the size of the training set is 100
    #plot our models and our predictions
    n_data = 1000
    n_training = int(n_data/10)
    X, y = make_data(n_data, 0)
    X_training, y_training = X[0:n_training], y[0:n_training]
    X_testing, y_testing = X[n_training:n_data], y[n_training:n_data]
    
    
    prediction_linear_regression = linear_regression(X_training, y_training, X_testing, y_testing, plot=True)
    prediction_knn = knn(X_training, y_training, X_testing, y_testing, neighbors=5, plot=True)
    
    #uncomment a function in order to plot and save the result
    #each function takes several minutes to compute, particulari if you increase the paramaters values
    
    #plot_error(200,1000) 
    #plot_linear_mean_by_size(100,1100,100)
    #plot_non_linear_mean_by_size(100,1100,100)
    #plot_linear_mean_by_complexity([0,2,5,10,30,100,200,300,400,500,600,700,800,900,1000])
    #plot_non_linear_mean_by_complexity([1,2,5,10,30,100,200,300,400,500,600,700,800,900,1000])
    #plot_linear_mean_by_irrelevant_variables([0,1,5,10,20,30,40,50,60,70,80,90,100])
    #plot_non_linear_mean_by_irrelevant_variables([0,1,5,10,20,30,40,50,60,70,80,90,100])
    
