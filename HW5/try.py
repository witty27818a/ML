import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# read data lines from file and return two ndarrays "X" and "y",
# each with the shape (N,1) (N being the total number of data)
def read_file(path: str):
    X = []
    y = []
    with open(path, 'r') as file:
        for line in file.readlines():
            X.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    return np.array(X).reshape(-1, 1), np.array(y).reshape(-1, 1)

# rational quadratic kernel
'''
3 parameters: "length" is the length scale. "alpha" is the scale-mixture (alpha > 0), "sigma2" is the overall variance.

The kernel formula is: k(x1, x2) = sigma2 * [1 + (||x1 - x2||^2) / (2 * alpha * length^2)] ^ (-alpha)
'''
def kernel(X1, X2, params):
    alpha = params[0]
    length = params[1]
    sigma2 = params[2]
    return sigma2 * ((1 + cdist(X1, X2, metric = 'sqeuclidean') / (2 * alpha * (length ** 2))) ** (-alpha))

# training session
'''
the marginal likelihood P(y) follows a multivariate distribution
with mean vector = 0 and covariance matrix = C, where the element in C:
C(xn, xm) = k(xn, xm) + beta_inv * delta(n, m), where delta(n, m) = 1 if n = m, 0, otherwise.
k(xn, xm) is the kernel function between xn and xm.
In matrix from: C = K + beta_inv * I, where K is the gram matrix consisted of kernel functions
and I is the identity matrix.
'''
def training(X, beta_inv, params):
    mu = np.zeros(X.shape) # mean vector = 0
    cov = kernel(X, X, params) + beta_inv * np.identity(X.shape[0]) # covariance matrix C = K + beta_inv * I
    # since in testing session, we only need the inverse of covariance matrix (from formula)
    # we return the inverse of covariance matrix instead of the original one.
    cov_inv = np.linalg.inv(cov)

    return mu, cov_inv

# testing session
'''
When doing prediction, the condition distribution P(y*|y) of the prediction given the result of training data, y
is a Gaussian Distribution, with the mean: mu = k(x, x*)^T * cov_inv * y
the variance: k(x*, x*) + beta_inv * I - k(x, x*)^T * cov_inv * k(x, x*),
where k(.,.) denotes the kernel function and x is the training data, x* is the testing data.
I is the identity matrix with the size of testing data.
'''
def testing(X, y, beta_inv, params, cov_inv, n):
    # initialization part for testing data. Set them all as column vectors.
    X_test = np.linspace(-60, 60, n).reshape(-1, 1)
    y_test = np.empty(n).reshape(-1, 1) # initialization for prediction results.
    y_test_Upper = np.empty_like(y_test) # the upper bound
    y_test_Lower = np.empty_like(y_test) # the lower bound

    kxxs = kernel(X, X_test, params) # k(x, x*)
    kxsxs = kernel(X_test, X_test, params) # k(x*, x*)
    
    y_test = np.linalg.multi_dot([kxxs.T, cov_inv, y]) # mu = k(x, x*)^T * cov_inv * y
    # use multi_dot to accelerate matrix multiplication when the matrixes become very large.
    # the variance = k(x*, x*) + beta_inv  * I - k(x, x*)^T * cov_inv * k(x, x*)
    # to draw out upper bound and lower bound, what we need is the standard deviation.
    var = np.diag(kxsxs + beta_inv * np.identity(X_test.shape[0]) - np.linalg.multi_dot([kxxs.T, cov_inv, kxxs]))
    std = np.sqrt(var).reshape(-1, 1)
    y_test_Upper = y_test + 1.96 * std
    y_test_Lower = y_test - 1.96 * std

    return X_test, y_test, y_test_Upper, y_test_Lower

# draw out the results
def drawout(X_train, y_train, X_test, y_test, y_test_Upper, y_test_Lower, params):
    for i in range(len(params)):
        plt.subplot(311+i)
        plt.xlim(-60, 60) # x-axis range from -60 to 60
        plt.plot(X_test.ravel(), y_test[i].ravel(), color = 'black') # prediction mean
        plt.plot(X_test.ravel(), y_test_Lower[i].ravel(), color = 'red') # lower bound
        plt.plot(X_test.ravel(), y_test_Upper[i].ravel(), color = 'red') # upper bound
        plt.fill_between(X_test.ravel(), y_test_Upper[i].ravel(), y_test_Lower[i].ravel(), facecolor = 'pink')
        plt.scatter(X_train, y_train, color = 'blue') # scatter plot of the training data
        plt.title("Gaussian Process\nchange sigma2") # Set title
        plt.text(-55, -4, f'sigma2 = {params[i]}', fontsize = 12, color = 'green')
    plt.tight_layout()
    plt.savefig("GP_sigma2.png")

if __name__ == '__main__':
    # read data
    X_train, y_train = read_file('./data/input.data')
    # initial parameters setting
    beta_inv = 1/5 # beta=5 is the precision of the white noise of data. The overall variance of white noise is 0.2.
    params = np.array([1.0, 1.0, 1.0]) # parameters for rational quadratic kernel, alpha and length only, since sigma2 is set to 0.2.
    n = 1000 # the number of testing data. The testing data are segmented from interval [-60, 60] evenly.

    y_test = [None, None, None]
    y_test_Upper = [None, None, None]
    y_test_Lower = [None, None, None]
    scale = [0.1, 1, 10]

    for i in range(len(scale)):
        params[2] = scale[i]
        # training
        mu, cov_inv = training(X_train, beta_inv, params)
        # testing
        X_test, y_test[i], y_test_Upper[i], y_test_Lower[i] = testing(X_train, y_train, beta_inv, params, cov_inv, n)
    
    # drawing out result
    drawout(X_train, y_train, X_test, y_test, y_test_Upper, y_test_Lower, scale)