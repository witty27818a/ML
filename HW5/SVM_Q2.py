import numpy as np
from libsvm.svmutil import *
import csv
import yaml

'''
Explanation part:
Let u be the feature map of some input vector x1, v be the feature map of another input vector x2.
Then the linear kernel = u' * v. The u-prime "u'" is the transpose of u.
    No parameter for this kernel.
the polynomial kernel = (gamma * u' * v + coef0)^degree
    There are 3 parameters for this kernel: gamma, coef0, degree (-g -r -d)
the RBF kernel = exp(-gamma * |u-v|^2)
    There are 1 parameter for this kernel: gamma (-g)
We will use C-SVC. So, every kernel has 1 additional parameter "C" for the penalty term. (-c)
'''

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

def Grid_Search(SVM_type, nfold = 5):
    problem = svm_problem(y_train, X_train)
    max_acc = 0.0 # maximum accuracy currently

    # candidate values for parameters
    C = [0.001, 0.01, 0.1, 1, 10] # default = 1
    gamma = [0.001, 1/784, 0.01, 0.1, 1, 10] # default= 1/# of feature = 1/784
    # as for coef0 and degree, we will leave them as default
    # this is simply because combination explosion. Considering all 4 parameters takes way too long.

    if SVM_type == 'Linear':
        param_str = "-t 0 -q"
        best_param = {"C":None}
        if nfold:
            param_str += " -v "+str(nfold)
        for c in C:
            param = svm_parameter(param_str+" -c "+str(c))
            cv_acc = svm_train(problem, param) # the output is the accuracy for cross-validation

            if cv_acc > max_acc:
                max_acc = cv_acc
                best_param.update({"C":c})

        # testing
        model = svm_train(problem, svm_parameter("-t 0 -q -c "+str(best_param["C"])))
        _, p_acc, _ = svm_predict(y_test, X_test, model)
    elif SVM_type == 'Polynomial':
        param_str = "-t 1 -q"
        best_param = {"C":None, "gamma":None, "degree":3, "coef0":0}
        if nfold:
            param_str += " -v "+str(nfold)
        for c in C:
            for g in gamma:
                param = svm_parameter(param_str+" -c "+str(c)+" -g "+str(g))
                cv_acc = svm_train(problem, param) # the output is the accuracy for cross-validation

                if cv_acc > max_acc:
                    max_acc = cv_acc
                    best_param.update({"C":c, "gamma":g})

        # testing
        model = svm_train(problem, svm_parameter("-t 1 -q -c "+str(best_param["C"])+" -g "+str(best_param["gamma"])))
        _, p_acc, _ = svm_predict(y_test, X_test, model)
    elif SVM_type == 'RBF':
        param_str = "-t 2 -q"
        best_param = {"C":None, "gamma":None}
        if nfold:
            param_str += " -v "+str(nfold)
        for c in C:
            for g in gamma:
                param = svm_parameter(param_str+" -c "+str(c)+" -g "+str(g))
                cv_acc = svm_train(problem, param) # the output is the accuracy for cross-validation

                if cv_acc > max_acc:
                    max_acc = cv_acc
                    best_param.update({"C":c, "gamma":g})

        # testing
        model = svm_train(problem, svm_parameter("-t 2 -q -c "+str(best_param["C"])+" -g "+str(best_param["gamma"])))
        _, p_acc, _ = svm_predict(y_test, X_test, model)
    else:
        raise ValueError("Kernel Not Available")
    
    return float(np.round(max_acc, 2)), float(np.round(p_acc[0], 2)), best_param

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = read_file("./data/X_train.csv", "./data/Y_train.csv", "./data/X_test.csv", "./data/Y_test.csv")
    result = dict()

    # Linear Kernel
    cv_acc, acc, best_param = Grid_Search("Linear")
    result.setdefault("Linear", {"validation accuracy":cv_acc, "testing accuracy":acc, "best parameters":best_param})
    cv_acc, acc, best_param = Grid_Search("Polynomial")
    result.setdefault("Polynomial", {"validation accuracy":cv_acc, "testing accuracy":acc, "best parameters":best_param})
    cv_acc, acc, best_param = Grid_Search("RBF")
    result.setdefault("RBF", {"validation accuracy":cv_acc, "testing accuracy":acc, "best parameters":best_param})

    print(yaml.dump(result, default_flow_style=False))