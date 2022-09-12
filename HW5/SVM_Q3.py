import numpy as np
from libsvm.svmutil import *
import csv
from scipy.spatial.distance import cdist

def read_file(X_train_path, y_train_path, X_test_path, y_test_path):
    # training data
    with open(X_train_path, 'r') as file:
        csv_reader = csv.reader(file)
        X_train = list(csv_reader)
        X_train = [[float(col) for col in row] for row in X_train]
    
    with open(y_train_path, 'r') as file:
        csv_reader = csv.reader(file)
        y_train = list(csv_reader)
        y_train = [int(label) for line in y_train for label in line]
    
    # testing data
    with open(X_test_path, 'r') as file:
        csv_reader = csv.reader(file)
        X_test = list(csv_reader)
        X_test = [[float(col) for col in row] for row in X_test]
    
    with open(y_test_path, 'r') as file:
        csv_reader = csv.reader(file)
        y_test = list(csv_reader)
        y_test = [int(label) for line in y_test for label in line]
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# the linear kernel function
def Linear(u, v):
    return np.matmul(u, v.T)

# the RBF kernel function
def RBF(u, v, gamma):
    return np.exp(-gamma * cdist(u, v, 'sqeuclidean'))

# the polynomial kernel function
def Polynomial(u, v, gamma):
    return np.power(gamma * np.matmul(u, v.T), 3)

# combine the linear kernel and the RBF kernel to generate a new precomputed kernel.
def linear_and_RBF(data1, data2, gamma):
    linear = Linear(data1, data2) # the linear kernel
    rbf = RBF(data1, data2, gamma) # the RBF kernel
    combinated = np.add(linear, rbf) # combine them up
    # we need to add an index column as the first column, from 1 to N, the total number of data1.
    # data1 = training data or testing data.
    X_kernel_ver = np.hstack((np.arange(1, data1.shape[0]+1).reshape(-1, 1), combinated))
    return X_kernel_ver

# combine the linear kernel and the polynomial kernel to generate a new precomputed kernel.
def linear_and_polynomial(data1, data2, gamma):
    linear = Linear(data1, data2) # the linear kernel
    polynomial = Polynomial(data1, data2, gamma) # the polynomial kernel
    combinated = np.add(linear, polynomial) # combine them up
    # we need to add an index column as the first column, from 1 to N, the total number of data1.
    # data1 = training data or testing data.
    X_kernel_ver = np.hstack((np.arange(1, data1.shape[0]+1).reshape(-1, 1), combinated))
    return X_kernel_ver

# train an SVM and make predictions
def SVM(X_train, y_train, X_test, y_test, gamma, nfold = 5):
    C = [0.001, 0.01, 0.1, 1, 10] # for RBF C = 10, while for linear C = 0.01
    max_acc = 0.0
    best_C = None
    '''
    Since the best value of parameter C is different between the RBF kernel and the linear kernel,
    we must tune the parameter by grid-searching again to find the best value.

    Note that both "X_train" and "X_test" are precomputed kernels.
    So, the kernel type should be set to precomputed kernel mode by "-t 4" in "svm_parameter".
    Also, in "svm_problem", the argument "isKernel" should be set to True for using precomputed kernel.
    '''
    # training
    problem = svm_problem(y_train, X_train, isKernel = True)
    for c in C:
        param_str = "-t 4 -c {} -g {} -q".format(c, gamma)
        # again, we use N-fold validation. Default N = 5 here.
        if nfold:
            param_str += " -v {}".format(nfold)
        param = svm_parameter(param_str)
        cv_acc = svm_train(problem, param)

        if cv_acc > max_acc:
            max_acc = cv_acc
            best_C = c
    # testing
    model = svm_train(problem, svm_parameter("-t 4 -c {} -g {} -q".format(best_C, gamma)))
    _, p_acc, _ = svm_predict(y_test, X_test, model)
    
    return float(np.round(p_acc[0], 2)), best_C # the accuracy and the new value of parameter C.

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = read_file("./data/X_train.csv", "./data/Y_train.csv", "./data/X_test.csv", "./data/Y_test.csv")

    # The best parameters from the previous question which can be obtained from SVM_Q2.py.
    # C = 10, gamma = 0.01 for RBF, C = 0.01 for linear
    # however, since there's two values for C, we need to tune the parameter "C" again, to find out the best suited one in this case.
    gamma = 0.01

    # calculate the precomputed kernels for both training data and testing data.
    X_train_kernel_ver = linear_and_RBF(X_train, X_train, gamma)
    X_test_kernel_ver = linear_and_RBF(X_test, X_train, gamma)
    
    # Use all the things to train an SVM and make predictions.
    # Before that, we will tune parameter C by grid-searching again.
    acc, C = SVM(X_train_kernel_ver, y_train, X_test_kernel_ver, y_test, gamma)
    print("The accuracy for linear kernel+RBF kernel is {}.\nThe corresponding parameters are gamma = {} and C = {}.".format(acc, gamma, C))

    # calculate the precomputed kernels for both training data and testing data.
    X_train_kernel_ver = linear_and_polynomial(X_train, X_train, gamma)
    X_test_kernel_ver = linear_and_polynomial(X_test, X_train, gamma)
    
    # Use all the things to train an SVM and make predictions.
    # Before that, we will tune parameter C by grid-searching again.
    acc, C = SVM(X_train_kernel_ver, y_train, X_test_kernel_ver, y_test, gamma)
    print("The accuracy for linear kernel+polynomial kernel is {}.\nThe corresponding parameters are gamma = {} and C = {}.".format(acc, gamma, C))