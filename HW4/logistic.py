import numpy as np
from math import exp, isinf
import argparse
from random_generator import normal
import matplotlib.pyplot as plt

def design_matrix(data):
    '''
    the design matrix should have a shape (N, 3), where "N" is the total number of observations, and
    3 for 3 parameters of constant term, x value and y value.
    The input "data" has a shape (N, 2) now. We need to append "1" in the first place of each row.
    '''
    ones = np.ones((data.shape[0], 1))
    return np.concatenate((ones, data), axis = 1)

def sigmoid(rawpred):
    '''
    input: raw prediction for each observation. "rawpred" is a n by 1 column vector.
    We need to pass each component to sigmoid function: sigma(x) = 1 / 1 + exp(-x).
    Thus, we get a new vector, which is the prediction result after sigmoid function.
    '''
    def sigma_function(x):
        '''
        When x is a large negative value, -x would become a large positive value, which
        makes e^(-x) become very very big and cause overflow error.
        To overcome this, when x is negative, we need to change the way we write the sigmoid function.
        '''
        if x >= 0:
            return 1 / (1 + exp(-x))
        else:
            return 1 - 1 / (1 + exp(x))
    
    sigma = np.vectorize(sigma_function) # vectorized version of sigmoid function.
    # apply the vectorized sigmoid function to the raw prediction and return.
    return sigma(rawpred)

def create_D(X, w):
    '''X is the design matrix and w is the current weight vector.'''
    '''D is diagonal matrix defined as:
    D[i][i] = -exp(-xi * w) / (1 + exp(-xi * w))^2, 0 for all other non-diagonal elements.'''
    D = np.identity(X.shape[0]) # initialize with identity matrix
    for i in range(D.shape[0]):
        exp_term = np.exp((-1) * np.matmul(X[i], w))
        # if the value of xi * w is a large negative value, then the exponential value would be very big
        if isinf(exp_term):
            exp_term = np.exp(1e3) # set an upper bound
        try:
            D[i][i] = -exp_term / ((1 + exp_term) ** 2)
        except:
            D[i][i] = -0.25 # set exp_term as 1
    
    return D

def confusion_matrix(ground_truth, prediction):
    # compute and show the confusion matrix, as well as the sensitivity and specificity
    '''
    True Positive, False Positive, True Negative and False Negative counts
    grounp 0 as negative and group 1 as positive
    '''
    TP = np.count_nonzero(np.logical_and(ground_truth, prediction)) # true positive count
    TN = np.count_nonzero(np.logical_not(np.logical_or(ground_truth, prediction))) # true negative count
    FP = np.count_nonzero(np.logical_and(np.logical_xor(ground_truth, prediction), prediction)) # false positive count
    FN = np.count_nonzero(np.logical_and(np.logical_xor(ground_truth, prediction), ground_truth)) # false negative count

    print("Confusion Matrix: ")
    file.write("Confusion Matrix: \n")
    print("\t\tReal 1\t\tReal 0")
    file.write("\t\tReal 1\t\tReal 0\n")
    print("Predicted 1\t{}\t\t{}".format(TP, FP))
    file.write("Predicted 1\t{}\t\t{}\n".format(TP, FP))
    print("Predicted 0\t{}\t\t{}".format(FN, TN))
    file.write("Predicted 0\t{}\t\t{}\n".format(FN, TN))
    print("Sensitivity (cluster 1, successfully predicted): {}".format(TP / (TP + FN)))
    file.write("Sensitivity (cluster 1, successfully predicted): {}\n".format(TP / (TP + FN)))
    print("Specificity (cluster 0, successfully predicted): {}".format(TN / (TN + FP)))
    file.write("Specificity (cluster 0, successfully predicted): {}\n".format(TN / (TN + FP)))
    print("\n\n")
    file.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help = "Number of data points for each group", type = int, default = 50)
    parser.add_argument("-mx1", help = "The mean of x values in group 1", type = float, default = 1)
    parser.add_argument("-vx1", help = "The variance of x values in group 1", type = float, default = 2)
    parser.add_argument("-my1", help = "The mean of y values in group 1", type = float, default = 1)
    parser.add_argument("-vy1", help = "The variance of y values in group 1", type = float, default = 2)
    parser.add_argument("-mx2", help = "The mean of x values in group 2", type = float, default = 10)
    parser.add_argument("-vx2", help = "The variance of x values in group 2", type = float, default = 2)
    parser.add_argument("-my2", help = "The mean of y values in group 2", type = float, default = 10)
    parser.add_argument("-vy2", help = "The variance of y values in group 2", type = float, default = 2)
    args = parser.parse_args()
    
    epoch = 1000 # how many iterations.
    lr = 0.1 # the learning rate

    n = args.n
    mx1 = args.mx1
    vx1 = args.vx1
    my1 = args.my1
    vy1 = args.vy1
    mx2 = args.mx2
    vx2 = args.vx2
    my2 = args.my2
    vy2 = args.vy2

    # the output file
    file = open("logistic.txt", "a")

    data = [] # data points

    # generate n data points from the first group
    for i in range(n):
        x = normal(mx1, vx1)
        y = normal(my1, vy1)
        data.append([x, y])
    
    # generate n data points from the second group
    for i in range(n):
        x = normal(mx2, vx2)
        y = normal(my2, vy2)
        data.append([x, y])
    
    data = np.array(data) # make the data an ndarray

    # the ground truth is that the first n observations are in group 1 while the latter n observations are in group 1.
    z = np.array([0] * n + [1] * n).reshape(-1, 1) # the binary ground truth z (which group)
    x = data[:, 0] # the x values for each observations
    y = data[:, 1] # the y values for each observations

    w = np.zeros((3, 1), dtype = float) # initial parameter vector with 0 vector

    X = design_matrix(data) # make design matrix

    # Gradient Descent method
    '''
    Following is the iterating processes:
    w(n+1) = w(n) + lr * grad(logL), where w(n) is the current weights and w(n+1) is the next weights.
    logL is the log likelihood, and grad(logL) is its gradient w.r.t. weight vector w. lr is the learning rate.
    Furthermore, grad(logL) = X^T * (z - sigmoid(X * w)), one can see this as X^T * the deviation between truth and predicted
    X is the design matrix and z is the ground truth binary vector, standing for the group number of each observation.
    w is the current weight vector. sigmoid is the sigmoid function.

    The iteration processes end when all epochs are finished, or the gradient is small enough.
    the threshold is a hyperparameter, set to 10^(-1). If each observation has sigmoid(xw) deviate from z for about 10^(-2)
    then the 2-norm of gradient would be about (100 observations * [10^(-2)]^2)^(0.5) = 10^(-1) long.
    '''
    # threshold to early stop
    threshold = 1e-1
    w_gradDes = w.copy()
    
    for i in range(epoch):
        grad_logL = np.matmul(X.T, np.subtract(z, sigmoid(np.matmul(X, w_gradDes))))
        w_gradDes += lr * grad_logL
        if np.linalg.norm(w_gradDes, ord = 2) <= threshold:
            break
    
    # make the final prediction by applying the weight vector on the design matrix
    # Note that they are still raw values now, we would turn them into binary for classification purpose later.
    raw_prediction = np.matmul(X, w_gradDes)
    
    # the binary prediction: 0 for group 0 and 1 for group 1
    prediction_gd = np.array([1 if raw_prediction[i] > 0.5 else 0 for i in range(raw_prediction.shape[0])])

    print("Gradient Descent Method: ")
    file.write("Gradient Descent Method: \n")
    print("w: ")
    file.write("w: \n")
    print(w_gradDes)
    file.write(str(w_gradDes) + '\n')
    print("")
    file.write('\n')

    # compute and show the confusion matrix, as well as the sensitivity and specificity
    confusion_matrix(z.ravel(), prediction_gd) # now, these two vectors are 1d with the shape (100,) if n = 50.

    print("---------------------------------------------------------------------")
    file.write("---------------------------------------------------------------------\n")

    # Newton's method
    '''
    The iteration process for the Newton's method will be: w(n+1) = w(n) - lr * hessian(logL)^(-1) * grad(logL),
    where w(n) is the current weights and w(n+1) is the next weights.
    logL is the log likelihood, and grad(logL) is its gradient w.r.t. weight vector w. lr is the learning rate.
    Furthermore, grad(logL) = X^T * (z - sigmoid(X * w)), one can see this as X^T * the deviation between truth and predicted
    X is the design matrix and z is the ground truth binary vector, standing for the group number of each observation.
    w is the current weight vector. sigmoid is the sigmoid function.
    hessian(logL) is the Hessian matrix of the log likelihood w.r.t. weight vector w.

    The iteration processes end when all epochs are finished, or the change "hessian(logL)^(-1) * grad(logL)" is small enough.
    the threshold is a hyperparameter, set to 10^(-1), which as the same as that in gradient descent method.

    The Hessian matrix is hessian(logL) = X^T * D * X, where X is the design matrix and D is diagonal matrix defined as:
    D[i][i] = -exp(-xi * w) / (1 + exp(-xi * w))^2, 0 for all other non-diagonal elements.
    '''

    w_newton = w.copy()

    for i in range(epoch):
        grad_logL = np.matmul(X.T, np.subtract(z, sigmoid(np.matmul(X, w_newton))))
        # w_gradDes += lr * grad_logL
        D = create_D(X, w_newton)
        # hessian(logL) = X^T * D * X
        hessian_logL = np.matmul(np.matmul(X.T, D), X)

        if np.linalg.matrix_rank(hessian_logL) == hessian_logL.shape[0]: # if the Hessian is full rank, i.e. invertible
            w_newton -= lr * np.matmul(np.linalg.inv(hessian_logL), grad_logL) # w(n+1) = w(n) - hessian(logL)^(-1) * grad(logL)
        else:
            w_newton += lr * grad_logL # if the hessian matrix is not invertible, use the gradient descent method instead.

        if np.linalg.norm(w_newton, ord = 2) <= threshold:
            break
    
    # make the final prediction by applying the weight vector on the design matrix
    # Note that they are still raw values now, we would turn them into binary for classification purpose later.
    raw_prediction = np.matmul(X, w_newton)
    
    # the binary prediction: 0 for group 0 and 1 for group 1
    prediction_nt = np.array([1 if raw_prediction[i] > 0.5 else 0 for i in range(raw_prediction.shape[0])])

    print("Newton's Method: ")
    file.write("Newton's Method: \n")
    print("w: ")
    file.write("w: \n")
    print(w_newton)
    file.write(str(w_newton) + '\n')
    print("")
    file.write('\n')

    # compute and show the confusion matrix, as well as the sensitivity and specificity
    confusion_matrix(z.ravel(), prediction_nt) # now, these two vectors are 1d with the shape (100,) if n = 50.

    '''visualization'''
    '''group 0 as blue, group 1 as red'''

    # ground truth
    plt.subplot(131)
    plt.scatter(x, y, c = z, cmap = 'RdBu')
    plt.title("Ground Truth")
    
    # gradient descent method
    plt.subplot(132)
    plt.scatter(x, y, c = prediction_gd, cmap = 'RdBu')
    plt.title("Gradient Descent")

    # Newton's method
    plt.subplot(133)
    plt.scatter(x, y, c = prediction_nt, cmap = 'RdBu')
    plt.title("Newton's method")

    plt.savefig("logistic.png")
    plt.show()

    file.close()