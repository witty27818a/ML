import argparse
from random_generator import polynomial
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-b', help = "the precision for initial prior, reciprocal of the variance", type = float, default = 1)
parser.add_argument('-n', help = "the order/basis number of the polynomial basis linear model", type = int, default = 4)
parser.add_argument('-a', help = "the variance of the random noise added", type = float, default = 1)
parser.add_argument('-w', help = "the weights vector", type = float, nargs = '+', default = [1,2,3,4]) # values after the flag would be collected as a list.
args = parser.parse_args()

count = 1

def design_matrix(x, n, X):
    '''
    x: the input data vector
    n: the basis number of the polynomial basis linear model.
    X: the previous design matrix (i.e., without the new data point x
    '''
    # Use numpy "vstack" function to append a new row under the last line of "X".
    # And return the new design matrix
    
    newrow = []
    for i in range(n):
        newrow.append(x ** i)
    newrow = np.array(newrow)
    
    X = np.vstack((X, newrow))
    return X

if __name__ == "__main__":
    b = args.b
    n = args.n
    a = args.a
    w = args.w # a list
    file = open("result_bayesianLR.txt", "a")
    assert n == len(w) # dummy check
    
    # all data
    x = []
    y = []
    # first 10 data points
    x_10 = []
    y_10 = []
    # first 50 data points
    x_50 = []
    y_50 = []
    
    # the first iteration
    # call "polynomial" to generate a data point
    newx, newy = polynomial(n, a, w)
    x.append(newx)
    y.append(newy)
    # set design matrix of the first iteration, it's a 1 by n matrix (a row vector) now.
    X = np.array([[newx ** i for i in range(n)]])
    
    # the prior mean for the first iteration is a 0 vector
    prior_mean = np.array([0.0] * n).reshape(-1, 1) # make it a column vector.
    # the prior variance for the first iteration, which can be set to any value.
    # Here, we set it as the same as the identity matrix times the reciprocal of the precision b given.
    prior_var = np.zeros((n, n))
    for i in range(n):
        prior_var[i][i] += (1 / b)
    
    # the posterior mean vector and covariance matrix
    '''
    In the first iteration, the posterior covariance matrix is: ((1 / a) * X^T * X + b * I)^(-1)
    the posterior mean vector is: (1 / a) * posterior covariance matrix * X^T * y, where
    '''
    precision_matrix = (1 / a) * np.matmul(X.T, X) + b * np.identity(n) # the posterior precision matrix Lambda
    post_var = np.linalg.inv(precision_matrix) # the posterior covariance matrix
    post_mean = (1 / a) * np.matmul(np.matmul(post_var, X.T), np.array(y)) # the posterior mean vector.
    
    '''
    Initialize the variables that store the posterior mean and variance
    after 10 datapoints and 50 datapoints.
    '''
    post_mean_10 = np.zeros_like(post_mean)
    post_var_10 = np.zeros_like(post_var)
    post_mean_50 = np.zeros_like(post_mean)
    post_var_50 = np.zeros_like(post_var)
    
    
    '''
    The predictive distribution of the new observation is a normal distribution, with:
    predict_mean = post_mean^T * X^T
    predict_var = a + X * post_var * X^T
    '''
    # Note that we only calculate the predictive distribution for the current new data x, for each iteration.
    predict_mean = np.matmul(post_mean.T, X[-1, :].T)
    predict_var = a + np.matmul(np.matmul(X[-1, :], post_var), X[-1, :].T)
    
    # print out some messages and write them out to a text file.
    print("(n = {}) add data point: ({}, {}):".format(count, newx, newy))
    file.write("(n = {}) add data point: ({}, {}):\n".format(count, newx, newy))
    print("")
    file.write("\n")
    print("Posterior mean: {}".format(post_mean))
    file.write("Posterior mean: {}\n".format(post_mean))
    print("")
    file.write("\n")
    print("Posterior covariance:\n{}".format(post_var))
    file.write("Posterior covariance:\n{}\n".format(post_var))
    print("")
    file.write("\n")
    print("Predictive distribution ~ N({}, {})". format(predict_mean, predict_var))
    file.write("Predictive distribution ~ N({}, {})\n". format(predict_mean, predict_var))
    print("--------------------------------------------------------------------------")
    file.write("--------------------------------------------------------------------------\n")
    
    count += 1
    
    # the second iteration and thereafter
    while True:
        if count > 500:
            print("Oops! Fail to coverge!")
            file.write("Oops! Fail to coverge!\n")
            break
            
        
        # call "polynomial" to generate a data point
        newx, newy = polynomial(n, a, w)
        x.append(newx)
        y.append(newy)
        
        # set the design matrix
        X = design_matrix(newx, n, X)
        
        # copy the previous posterior parameters as the current prior parameters.
        prior_mean = deepcopy(post_mean)
        prior_var = deepcopy(post_var)
        
        '''
        In the latter iterations, the posterior parameters are different, by online learning.
        Note that the prior mean would likely be non-zero vector,
        and the prior covariance would likely be non-diagonal matrix.
        After some calculating and comparing processes, we obtain:
        posterior precision matrix is: (1 / a) * X^T * X + S
        the posterior covariance matrix would be its inverse.
        posterior mean vector is: posterior covariance matrix * ((1 / a) * X^T * y + S * m), where
        S = the inverse of prior covariance matrix, m = prior mean
        '''
        m = deepcopy(prior_mean)
        S = np.linalg.inv(prior_var)
        precision_matrix = (1 / a) * np.matmul(X.T, X) + S
        post_var = np.linalg.inv(precision_matrix)
        post_mean = np.matmul(post_var, (1 / a) * np.matmul(X.T, np.array(y)) + np.matmul(S, m))
        
        '''
        The predictive distribution of the new observation is a normal distribution, with:
        predict_mean = post_mean^T * X^T
        predict_var = a + X * post_var * X^T
        These are the same.
        '''
        predict_mean = np.matmul(post_mean.T, X[-1, :].T)
        predict_var = a + np.matmul(np.matmul(X[-1, :], post_var), X[-1, :].T)
        
        # print out some messages and write them out to a text file.
        print("(n = {}) add data point: ({}, {}):".format(count, newx, newy))
        file.write("(n = {}) add data point: ({}, {}):\n".format(count, newx, newy))
        print("")
        file.write("\n")
        print("Posterior mean: {}".format(post_mean))
        file.write("Posterior mean: {}\n".format(post_mean))
        print("")
        file.write("\n")
        print("Posterior covariance:\n{}".format(post_var))
        file.write("Posterior covariance:\n{}\n".format(post_var))
        print("")
        file.write("\n")
        print("Predictive distribution ~ N({}, {})". format(predict_mean, predict_var))
        file.write("Predictive distribution ~ N({}, {})\n". format(predict_mean, predict_var))
        print("--------------------------------------------------------------------------")
        file.write("--------------------------------------------------------------------------\n")
        
        # every 10 and 50 datapoints, save them other places as well.
        # for later drawing the results.
        if count == 10:
            x_10 = deepcopy(x)
            y_10 = deepcopy(y)
            post_mean_10 = deepcopy(post_mean)
            post_var_10 = deepcopy(post_var)
        
        if count == 50:
            x_50 = deepcopy(x)
            y_50 = deepcopy(y)
            post_mean_50 = deepcopy(post_mean)
            post_var_50 = deepcopy(post_var)
            
        # converge condition
        if np.linalg.norm(prior_mean - post_mean, 2) < 1e-3:
            break
        
        count += 1
        
    '''Visualization'''
    plt.figure(figsize = (10, 10))
    # ground truth
    plt.subplot(221)
    plt.title("Ground Truth")
    plt.xlim(-2.0, 2.0) # there should be 95% within the normal
    real_x = np.linspace(-2.0, 2.0, 40) # evenly split into 40 segments.
    # calculate the y for each x. it's a polynomial with coefficients as the initial parameter vector w.
    real_y = np.array([np.polyval(np.poly1d(np.flip(w)), i) for i in real_x])
    plt.plot(real_x, real_y, c = 'k')
    plt.plot(real_x, real_y + np.sqrt(a), c = 'r') # the upper bound
    plt.plot(real_x, real_y - np.sqrt(a), c = 'r') # the lower bound
    
    # predict result
    plt.subplot(222)
    plt.title("Final Result")
    plt.xlim(-2.0, 2.0)
    result_x = np.linspace(-2.0, 2.0, 40)
    # use the posterior mean for the paramter vector w.
    result_y = np.array([np.polyval(np.poly1d(np.flip(post_mean.flatten())), i) for i in result_x])
    plt.scatter(x, y, c = 'b')
    deviate = np.zeros_like(result_y)
    '''
    the variance of the predictive distribution is: predict_var = a + X * post_var * X^T
    '''
    # Actually, this is very weird. we should +-1 a standard deviation, not a variance.
    # different units!
    for i in range(len(result_x)):
        # for every x, form a row vector which is a design matrix.
        temp_X = np.array([[result_x[i] ** j for j in range(n)]]).reshape(1, -1)
        # calculate the variance of the predictive distribution of this x.
        temp_var = a + np.matmul(np.matmul(temp_X, post_var), temp_X.T)
        # temp_var = np.sqrt(a + np.matmul(np.matmul(temp_X, post_var), temp_X.T))

        deviate[i] += temp_var
    plt.plot(result_x, result_y, c = 'k')
    plt.plot(result_x, result_y + deviate, c = 'r') # the upper bound
    plt.plot(result_x, result_y - deviate, c = 'r') # the lower bound
    
    # predict result (10 data points)
    # the whole processes are the same of those above.
    plt.subplot(223)
    plt.title("Result after 10")
    plt.xlim(-2.0, 2.0)
    result_y = np.array([np.polyval(np.poly1d(np.flip(post_mean_10.flatten())), i) for i in result_x])
    plt.scatter(x_10, y_10, c = 'b')
    deviate = np.zeros_like(result_y)
    '''
    the variance of the predictive distribution is: predict_var = a + X * post_var_inv * X^T
    '''
    for i in range(len(result_x)):
        temp_X = np.array([[result_x[i] ** j for j in range(n)]])
        temp_var = a + np.matmul(np.matmul(temp_X, post_var_10), temp_X.T)
        # temp_var = np.sqrt(a + np.matmul(np.matmul(temp_X, post_var_10), temp_X.T))
        deviate[i] += temp_var
    plt.plot(result_x, result_y, c = 'k')
    plt.plot(result_x, result_y + deviate, c = 'r') # the upper bound
    plt.plot(result_x, result_y - deviate, c = 'r') # the lower bound
    
    # predict result (50 data points)
    # the whole processes are the same of those above.
    plt.subplot(224)
    plt.title("Result after 50")
    plt.xlim(-2.0, 2.0)
    result_y = np.array([np.polyval(np.poly1d(np.flip(post_mean_50.flatten())), i) for i in result_x])
    plt.scatter(x_50, y_50, c = 'b')
    deviate = np.zeros_like(result_y)
    '''
    the variance of the predictive distribution is: predict_var = a + X * post_var_inv * X^T
    '''
    for i in range(len(result_x)):
        temp_X = np.array([[result_x[i] ** j for j in range(n)]])
        temp_var = a + np.matmul(np.matmul(temp_X, post_var_50), temp_X.T)
        # temp_var = np.sqrt(a + np.matmul(np.matmul(temp_X, post_var_50), temp_X.T))
        deviate[i] += temp_var
    plt.plot(result_x, result_y, c = 'k')
    plt.plot(result_x, result_y + deviate, c = 'r') # the upper bound
    plt.plot(result_x, result_y - deviate, c = 'r') # the lower bound
    
    plt.tight_layout() # so that each subplots won't bump each other.
    plt.savefig("Result_BayesianLR.png")
    plt.show()
    
    file.close()