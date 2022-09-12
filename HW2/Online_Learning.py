import argparse

# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", help = "The path of input txt or csv file", type = str, default = "beta.txt")
parser.add_argument("-a", help = "The first parameter of beta distribution", type = int, default = 0)
parser.add_argument("-b", help = "The first parameter of beta distribution", type = int, default = 0)
parser.add_argument("-o", help = "The path of output txt file", type = str, default = "result_beta.txt")
args = parser.parse_args()

# count how many 1s and 0s in each single line of input data strings.
def count(binstr):
    ones = 0
    for i in range(len(binstr)):
        ones += int(binstr[i])
    return (ones, len(binstr) - ones) # return a tuple with the numbers of 1s and 0s

# process the input file
def process_file(fpath):
    #open the file
    with open(fpath, "r") as f:
        # read all lines from the file and store them in a list.
        data = f.readlines()
        # for each line, take out the string and discard the '\n' at the end.
        # Also, recompose the "data" list to a list of 2-item tuples, where the second items are also 2-item tuples.
        # The data structure look like this: (binary_string, (how_many_1s, how_many_0s))
        for i in range(len(data)):
            data[i] = (data[i].strip('\n'), count(data[i].strip('\n')))
        return data

# calculate the combination mCN
def C(N, m):
    numerator = 1
    denominator = 1

    '''an example to show the logic of following code:
    EX: 2C5 = (5*4) / (2*1)
    3C5 = 2C5 = (5*4) / (2*1)'''

    for i in range(N, N - m if m < N - m else m, -1):
        numerator *= i
    for i in range(m if m < N - m else N - m, 0, -1):
        denominator *= i
    return numerator / denominator

# for each binary string input, calculate the binomial likelihood,
# the parameters of prior and posterior beta distributions
# And then, print them (including the input binary string) out as well as write them to a result file.
def beta(index, data_tuple, a, b, outfile):
    # data structure of "data_tuple": (binary_string, (how_many_1s, how_many_0s))

    # open the result txt file.
    file = open(outfile, "a")

    '''first, let's print out the data, a binary string
    Also, write it to the result txt file'''
    print("Case {}: {}".format(index+1, data_tuple[0]))
    file.write("Case {}: {}\n".format(index+1, data_tuple[0]))

    '''next, print out the likelihood of binomial distribution.'''
    '''Not that the MLE of the probability parameter 'p' in binomial distribution is
    the number of successful trials / the number of total trials.
    that is, p_hat = m / N, where 'm' = the number of successful trials and 'N' = the number of total trials'''
    '''therefore, the binomial likelihood is NCm * p_hat^(m) * p_hat^(N - m)'''
    m = data_tuple[1][0]
    N = data_tuple[1][0] + data_tuple[1][1]
    p_hat = m / N # the mle of parameter 'p' in binomial distribution.
    likelihood = C(N, m) * (p_hat ** m) * ((1-p_hat) ** (N-m))
    print("Likelihood: {}".format(likelihood))
    file.write("Likelihood: {}\n".format(likelihood))

    '''then, print out the parameters for prior beta distribution.
    Also, write it to the result txt file'''
    print("Beta prior:\ta = {} b = {}".format(a, b))
    file.write("Beta prior:\ta = {} b = {}\n".format(a, b))

    '''finally, print out the parameters for posterior beta distribution,
    which happended to be the adding the number of successful trials, 'm' to 'a', and
    adding the number of failure trials, 'N' - 'm' to 'b'.'''
    '''Also, write it to the result txt file.'''
    print("Beta posterior: a = {} b = {}".format(a+m, b+N-m))
    file.write("Beta posterior: a = {} b = {}\n".format(a+m, b+N-m))
    
    # A blank line
    print('\n')
    file.write('\n')

    # close the result txt file.
    file.close()

    # return the new parameter pair
    return a+m, b+N-m

if __name__ == "__main__":
    # get the arguments
    fpath = args.i
    a = args.a
    b = args.b
    out = args.o

    # process the input file and get a list of tuples "data"
    # the data structure of data is: [(binary_string, (how_many_1s, how_many_0s)), ...]
    data = process_file(fpath)

    # for each binary string in "data", call "beta" to get
    # the binomial likelihood, parameters of prior and posterior beta distributions.
    # print them out and write them to a result text file
    # finally, return the parameters of posterior beta distributions as the prior of the next round.
    for i, d in enumerate(data):
        a, b = beta(i, d, a, b, out)
