import numpy as np
import csv
from libsvm.svmutil import *
'''REMEMBER TO SET UP libsvm TO YOUR LIBRARY FIRST!!!'''

def read_file(X_train_path, y_train_path, X_test_path, y_test_path):
    # training data
    with open(X_train_path, 'r') as file:
        csv_reader = csv.reader(file) # now, it's a csvReader object
        X_train = list(csv_reader) # turn it into a 2d list, but now each element is still a string, not a float.
        X_train = [[float(col) for col in row] for row in X_train] # iterate through, turn each element into a float.
    
    with open(y_train_path, 'r') as file:
        csv_reader = csv.reader(file)
        y_train = list(csv_reader)
        y_train = [int(label) for line in y_train for label in line] # for labels, turn them into integer.
    
    # testing data
    with open(X_test_path, 'r') as file:
        csv_reader = csv.reader(file)
        X_test = list(csv_reader)
        X_test = [[float(col) for col in row] for row in X_test]
    
    with open(y_test_path, 'r') as file:
        csv_reader = csv.reader(file)
        y_test = list(csv_reader)
        y_test = [int(label) for line in y_test for label in line]
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test) # return as ndarrays

'''
Explanation part:
Let u be the feature map of some input vector x1, v be the feature map of another input vector x2.
Then the linear kernel = u' * v. The u-prime "u'" is the transpose of u.
    No parameter for this kernel.
the polynomial kernel = (gamma * u' * v + coef0)^degree
    There are 3 parameters for this kernel: gamma, coef0, degree (-g -r -d)
the RBF kernel = exp(-gamma * |u-v|^2)
    There are 1 parameter for this kernel: gamma (-g)
The default type of the SVM will be the C-SVC. So, every kernel has 1 additional parameter "C" for the penalty term. (-c)

However, in the first part of the SVM problem, we leave these parameters as default settings.
We simply compare these 3 kernels in the default settings.
'''

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = read_file("./data/X_train.csv", "./data/Y_train.csv", "./data/X_test.csv", "./data/Y_test.csv")
    acc = dict()

    print("Linear Kernel:")
    # read the problem. The first argument is the labels y while the second is the features X.
    problem = svm_problem(y_train, X_train)
    # set the parameters globally. "-t" to choose kernel type: 0 for linear, 1 for polynomial, 2 for RBF.
    # "-q" is to activate the quiet mode (no output for training).
    param = svm_parameter("-t 0 -q") # others: -c
    # train the SVM
    model = svm_train(problem, param)
    # predict and get the evaluation metrics. BY TESTING DATA
    '''This fundction will return 3 things:
    1. p_label: the prediction (label) for each data point.
    2. p_acc: it's a tuple, containing 3 things
        2.1 the accuracy (for classification)
        2.2 the MSE (for regression)
        2.3 the R^2 (for regression)
    3. p_vals: in classification case (-b 0), it's a list of decision values.
    If there are k classes, then every element in p_vals is a tuple containing the decision values
    from k(k-1)/2 binary-class SVM.

    We only need 2-1
    '''
    _, p_acc, _ = svm_predict(y_test, X_test, model)
    acc.setdefault("Linear", p_acc[0])

    print("Polynomial Kernel:")
    problem = svm_problem(y_train, X_train)
    param = svm_parameter("-t 1 -q") # others: -c -d -g -r
    model = svm_train(problem, param)
    _, p_acc, _ = svm_predict(y_test, X_test, model)
    acc.setdefault("Polynomial", p_acc[0])

    print("RBF Kernel:")
    problem = svm_problem(y_train, X_train)
    param = svm_parameter("-t 2 -q") # others: -c -g
    model = svm_train(problem, param)
    _, p_acc, _ = svm_predict(y_test, X_test, model)
    acc.setdefault("RBF", p_acc[0])
