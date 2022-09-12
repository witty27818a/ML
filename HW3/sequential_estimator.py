from random_generator import normal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', help = "the mean of the source normal distribution.", type = float, default = 3.0)
parser.add_argument('-s', help = "the variance of the source normal distribution.", type = float, default = 5.0)
args = parser.parse_args()

'''convergence conditions'''
mean_limit = 1e-2
var_limit = 1e-2

if __name__ == "__main__":
    mean = args.m
    var = args.s
    
    # print out and write out results to a text file
    file = open("result_sequential.txt", "a")
    print("Data point source function: N({}, {})".format(mean, var))
    file.write("Data point source function: N({}, {})\n".format(mean, var))
    print("")
    file.write('\n')
    
    # the first data point before training.
    # randomly generate a value from normal as a initialization for mean
    mean_old = normal(mean, var)
    # as for the initialization for variance, since there is only 1 point now, should be 0.
    var_old = 0.0
    # the current data size
    n = 1
    
    print("(n = {}) data point: {}".format(n, mean_old))
    file.write("(n = {}) data point: {}\n".format(n, mean_old))
    
    while True:
        # a new data randomly generated from univariate normal with the given mean and variance.
        new = normal(mean, var)
        # add 1 to the current data size
        n += 1
        
        '''
        Let the current data size = n
        Let the new mean = mean_new, the old mean = mean_old,
        Let the new variance = var_new, the old variance = var_old
        Let the new data point = new
        '''
        
        # The following recursive function of mean and variance based on the sequential estimation
        # are based on "Welford's online algorithm"
        mean_new = mean_old + (new - mean_old) / n
        var_new = var_old + ((new - mean_old) * (new - mean_new) - var_old) / n
        
        # print out and write out results to a text file
        print("(n = {}) add data point: {}".format(n, new))
        file.write("(n = {}) add data point: {}\n".format(n, new))
        print("Mean = {}\t Variance = {}".format(mean_new, var_new))
        file.write("Mean = {}\t Variance = {}\n".format(mean_new, var_new))
        
        # recursively loop until the convergence conditions are satisfied.
        if abs(mean_new - mean_old) < mean_limit and abs(var_new - var_old) < var_limit:
            break
        else:
            mean_old = mean_new
            var_old = var_new
    
    file.close()