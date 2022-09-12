import argparse
import numpy as np

# Set the precision to be displayed.
# Up to 4 decimal places, and suppress the scientific notation.
np.set_printoptions(precision = 4, suppress = True)

# parsing input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--train_image", help = "The path of raw training image file, format = .idx3-ubyte", type = str, default = "D:/HW2_ML/train-images.idx3-ubyte")
parser.add_argument("--train_label", help = "The path of raw training label file, format = .idx1-ubyte", type = str, default = "D:/HW2_ML/train-labels.idx1-ubyte")
parser.add_argument("--test_image", help = "The path of raw testing image file, format = .idx3-ubyte", type = str, default = "D:/HW2_ML/t10k-images.idx3-ubyte")
parser.add_argument("--test_label", help = "The path of raw testing label file, format = .idx1-ubyte", type = str, default = "D:/HW2_ML/t10k-labels.idx1-ubyte")
parser.add_argument("--option", help = "Two options, 0 for discrete or 1 for continuous", type = int, default = 0)
args = parser.parse_args()

# store all things about training data and testing data as 2 dictionaries
# each of them has 2 items: "image" and "label" with respective dictionaries.
# in the inner dictionaries, store items such as magic number, numbers of images, rows and columns, and each images' data/labels.
train = {"image":{}, "label":{}}
test = {"image":{}, "label":{}}
# there are 10 categories in this dataset
classnum = 10
# a variable that stores the error count.
error = 0

'''A function that process the file.'''
def process_file(train_image, train_label, test_image, test_label):
    '''the whole binary file is in "big endian" format.
    read in all things besides the actual images' data,
    such as magic numbers, total number of images, numbers of pixels in a row and a column.
    And then, these images' data
    
    As for label data, the whole process is same.
    read in magic numbers, total number of labels.
    And then, those labels'''

    # the images in training data.
    with open(train_image, 'rb') as f:
        train["image"]["magic"] = int.from_bytes(f.read(4), byteorder = 'big')
        train["image"]["number"] = int.from_bytes(f.read(4), byteorder = 'big')
        train["image"]["row"] = int.from_bytes(f.read(4), byteorder = 'big')
        train["image"]["col"] = int.from_bytes(f.read(4), byteorder = 'big')

        image = []
        for _ in range(train["image"]["number"]):
            data = []
            for _ in range(train["image"]["row"]):
                temp = []
                for _ in range(train["image"]["col"]):
                    temp.append(int.from_bytes(f.read(1), byteorder = 'big'))
                data.append(temp)
            image.append(data)
        image = np.array(image)
        train["image"]["image"] = image
    
    # the images in testing data.
    with open(test_image, 'rb') as f:
        test["image"]["magic"] = int.from_bytes(f.read(4), byteorder = 'big')
        test["image"]["number"] = int.from_bytes(f.read(4), byteorder = 'big')
        test["image"]["row"] = int.from_bytes(f.read(4), byteorder = 'big')
        test["image"]["col"] = int.from_bytes(f.read(4), byteorder = 'big')

        image = []
        for _ in range(test["image"]["number"]):
            data = []
            for _ in range(test["image"]["row"]):
                temp = []
                for _ in range(test["image"]["col"]):
                    temp.append(int.from_bytes(f.read(1), byteorder = 'big'))
                data.append(temp)
            image.append(data)
        image = np.array(image)
        test["image"]["image"] = image

    # the labels in training data.
    with open(train_label, 'rb') as f:
        train["label"]["magic"] = int.from_bytes(f.read(4), byteorder = 'big')
        train["label"]["number"] = int.from_bytes(f.read(4), byteorder = 'big')

        label = []
        for _ in range(train["label"]["number"]):
            label.append(int.from_bytes(f.read(1), byteorder = 'big'))
        label = np.array(label)
        train["label"]["label"] = label
    
    # the labels in testing data
    with open(test_label, 'rb') as f:
        test["label"]["magic"] = int.from_bytes(f.read(4), byteorder = 'big')
        test["label"]["number"] = int.from_bytes(f.read(4), byteorder = 'big')

        label = []
        for _ in range(test["label"]["number"]):
            label.append(int.from_bytes(f.read(1), byteorder = 'big'))
        label = np.array(label)
        test["label"]["label"] = label
     
'''normalize the probabilities so that they add up to 1.'''
def normalize(p):
    summation = np.sum(p)
    return p / summation

'''judge if the prediction is correct, and print the prediction.'''
def judge_and_print(postp, gt_label, n, option):
    print("figure ", n)
    print("Posterior (in log scale):")
    for c in range(classnum):
        # for each class, print out the posterior probability of each class
        print(c, ": ", postp[c])
    
    '''Since in the original posterior probability, it's always less than 1
    So, the logarithm value will always be negative.
    The bigger the original probability was, the less negative, i.e. the bigger it will become after the logarithm.
    However, after the normalization, which makes all values become positive. So, in this case.
    The biggest value becomes the smallest. Therefore, we need to choose the minimum as our prediction!
    '''
    predicted = np.argmin(postp) # get the index number which happened to be the exact prediction we wants
    print("Prediction: ", predicted, ", Ans: ", gt_label, "\n")

    # write out the complete results to a txt file
    if not option: # discrete mode
        with open("result_discrete.txt", "a") as f:
            f.write("figure "+str(n)+"\n")
            f.write("Posterior (in log scale):\n")
            for c in range(classnum):
                f.write(str(c)+": "+str(postp[c])+"\n")
            f.write("Prediction: "+str(predicted)+", Ans: "+str(gt_label)+"\n\n")
    else: # continuous mode
        with open("result_continuous.txt", "a") as f:
            f.write("figure "+str(n)+"\n")
            f.write("Posterior (in log scale): \n")
            for c in range(classnum):
                f.write(str(c)+": "+str(postp[c])+"\n")
            f.write("Prediction: "+str(predicted)+", Ans: "+str(gt_label)+"\n\n")

    if predicted == gt_label:
        return 0 # not an error prediction
    else:
        return 1 # is an error prediction

'''showing the classification results of each figure in testing data
by using the testing data to modify prior and obtain the posterior.
In discrete mode, each pixel in testing image is classified into 1 of the 32 bins.
and then, the log probability would be directly added up to the posterior probability of each label.
In continuous mode, instead of classfying each pixel, we would fit them with some Guassian distribution
and then again, the log probability would be added up to the posterior probability of each label.'''
def classify_testing_data(a, b, n, p, option):
    if not option: # discrete mode
        likelihood = a
        likelihood_marginal = b
        # Now, for the current testing figure, determine for each pixel
        # which of the 32 bins it should be classified in, by its values.

        # initialize a 28 * 28 1d array.
        curr_img = np.zeros(test["image"]["row"] * test["image"]["col"], dtype = int)
        for pixel in range(test["image"]["row"] * test["image"]["col"]):
            # classify each pixel into some bin. (8 gray scales as a bin.), and store the bin number into the 2d array above.
            curr_img[pixel] = test["image"]["image"][n][pixel // test["image"]["col"]][pixel % test["image"]["col"]] // 8
            for c in range(classnum):
                # if the bin count under the class and pixel was 0
                if not likelihood[c][pixel][curr_img[pixel]]:
                    # then, we need to set pseudo count
                    p[c] += np.log(float(1e-6 / likelihood_marginal[c][pixel]))
                else:
                    # Otherwise, we can directly use the count to calculate the log probability for the posteriori distribution.
                    p[c] += np.log(float(likelihood[c][pixel][curr_img[pixel]] / likelihood_marginal[c][pixel]))
    else: # continuous mode
        mean = a
        var = b
        # Now, for the current testing figure, for each pixel under each class,
        # fit a value from some Gaussian distribution based on the mean and variance

        # initialize a 28 * 28 1d array.
        curr_img = np.zeros(test["image"]["row"] * test["image"]["col"], dtype = float)
        for pixel in range(test["image"]["row"] * test["image"]["col"]):
            # copy the gray scale value of each pixel of the current tested image to "curr_img"
            curr_img[pixel] = test["image"]["image"][n][pixel // test["image"]["col"]][pixel % test["image"]["col"]]
            # and then for each class label,
            for c in range(classnum):
                # obtain a fitted likelihood for the value of the current pixel under current class(label), from a Gaussian distribution
                # Use the count to calculate the log probability
                fitted_likelihood = Gaussian_logged(curr_img[pixel], mean[c][pixel], var[c][pixel])
                # add the log probability to the posteriori distribution.
                p[c] += fitted_likelihood

    # normalization. So that the probabilities add up to 1.
    postp = normalize(p)

    return postp

'''The outline of discrete process'''
def discrete():
    '''training'''
    # initialiize
    prior = np.zeros(classnum, dtype = int) # prior distribution for 10 classes
    # 10 classes, 28*28 pixels, partitioned gray scale values into 32 bins (each 8 values a bin)
    likelihood = np.zeros((classnum, train["image"]["row"] * train["image"]["col"], 32)) # so, the likelihood array is a 3d array.

    # counting detailed likelihood
    for n in range(train["image"]["number"]):
        # for each image in training data
        # first, +1 count to the ground truth label in prior distribution.
        gt_label = train["label"]["label"][n]
        prior[gt_label] += 1

        # the detailed likelihood counts, for each class, each pixel, and each gray scale bins. Insert in the 3D numpy array above
        for pixel in range(train["image"]["row"] * train["image"]["col"]):
            pixel_value = train["image"]["image"][n][pixel // train["image"]["col"]][pixel % train["image"]["col"]]
            bin = pixel_value // 8
            likelihood[gt_label][pixel][bin] += 1
    
    # counting marginal likelihood for each class, each pixel.
    # That is, merging all bins' count into 1 total count, for each pixel, each class.
    likelihood_marginal = np.zeros((classnum, train["image"]["row"] * train["image"]["col"]), dtype = int)
    for i in range(classnum):
        for j in range(train["image"]["row"] * train["image"]["col"]):
            for k in range(32):
                likelihood_marginal[i][j] += likelihood[i][j][k]
    
    '''testing'''
    for n in range(test["image"]["number"]):
        p = np.zeros(classnum, dtype = float) # initialize
        # the probability of the prior distribution in training data (in log scale)
        for c in range(classnum):
            p[c] += np.log(float(prior[c]/train["image"]["number"]))
        
        # posteriori distribution (in log scale)
        postp = classify_testing_data(likelihood, likelihood_marginal, n, p, 0)
        
        # the ground truth label for the current testing figure
        gt_label = test["label"]["label"][n]
        # judge if this figure has the correction prediction, and print the prediction result
        global error
        error += judge_and_print(postp, gt_label, n, 0)

    # print the imagination of each digit(class) by this classifier.
    print_imagination(likelihood, 0)

    # the error rate.
    print("Error rate: ", float(error / test["image"]["number"]))
    with open("result_discrete.txt", "a") as file:
        file.write("Error rate: {}".format(float(error / test["image"]["number"])))

'''print the imagination of each digit(class) by this classifier.
Also, write them into a text file.'''
def print_imagination(likelihood, option):
    if not option: # discret mode
        f = open("result_discrete.txt", "a")
    else: # continuous mode
        f = open("result_continuous.txt", "a")
    
    print("Imagination of numbers in Bayesian Classifier:\n")
    f.write("Imagination of numbers in Bayesian Classifier:\n")

    for c in range(classnum):
        print(c, ":")
        f.write(str(c)+":\n")
        for i in range(train["image"]["row"]):
            for j in range(train["image"]["col"]):
                # print out 0 as white, 1 as black pixel
                if not option: # discrete mode
                    # with gray scale values range from 0~127 (bins 0 ~ 15), the classifier expects them to be white(0)
                    # On the other hand, values range from 128~255 (bins 16 ~ 31) are expected to be black(1)
                    white = 0
                    for bin in range(32):
                        if bin < 16:
                            white += likelihood[c][i * train["image"]["col"] + j][bin]
                        else:
                            white -= likelihood[c][i * train["image"]["col"] + j][bin]
                    if white > 0:
                        # this pixel is expected to be white.
                        print("0", end = " ")
                        f.write("0 ")
                    else:
                        # this pixel is expected to be black.
                        print("1", end = " ")
                        f.write("1 ")
                else: # continuous mode
                    # if the mean of the gray scale value of the pixel under the label is less than 128.
                    # the classifier expects it to be white(0); black(1), otherwise.
                    if likelihood[c][i * train["image"]["col"] + j] < 128:
                        # this pixel is expected to be white.
                        print("0", end = " ")
                        f.write("0 ")
                    else:
                        # this pixel is expected to be black.
                        print("1", end = " ")
                        f.write("1 ")
            print("\n")
            f.write("\n")
        print("\n")
        f.write("\n")

    f.close()

'''input a gray scale value, and return the likelihood of the value under some Gaussian distribution with given mean and variance. (in log scale)'''
def Gaussian_logged(gray_value, mean, var):
    return np.log(1.0 / np.sqrt(2.0 * np.pi * var)) - (gray_value - mean) ** 2.0 / (2.0 * var)

'''The outline of continuous process'''
def continuous():
    '''training '''
    # initialize a 1d array for prior distribution.
    prior = np.zeros(classnum, dtype = float) # 10 classes
    # for each class, each pixel: initialization
    square = np.zeros((classnum, train["image"]["row"] * train["image"]["col"]), dtype = float) # the square value
    mean = np.zeros_like(square) # the mean of Gaussian distribution
    var = np.zeros_like(square) # the variance of Gaussian distribution

    # counting the prior for each class and the square value for each pixel

    # Also, set the mean of a pixel under some label as a total counting for now.
    # We will proportionalize them, to make them become mean number later.
    for n in range(train["label"]["number"]):
        gt_label = train["label"]["label"][n]
        prior[gt_label] += 1
        for pixel in range(train["image"]["row"] * train["image"]["col"]):
            square[gt_label][pixel] += (train["image"]["image"][n][pixel // train["image"]["col"]][pixel % train["image"]["col"]] ** 2)
            mean[gt_label][pixel] += train["image"]["image"][n][pixel // train["image"]["col"]][pixel % train["image"]["col"]]

    # calculate the mean and variance of the Gaussian Distribution, fitted for each pixel, each class
    for c in range(classnum):
        for pixel in range(train["image"]["row"] * train["image"]["col"]):
            mean[c][pixel] = float(mean[c][pixel] / prior[c])
            var[c][pixel] = float(square[c][pixel] / prior[c]) - float(mean[c][pixel] ** 2) # Use the formula var(X) = E(X^2) - (E(X))^2
            # pseudo count for the variance, since variance cannot be too small.
            # If the variance is too small, it will cause the gaussian distribution or any variance-related calculations become diverged.
            # And thus make poor prediction. (only 56% accuracy rate)
            if var[c][pixel] < 1000:
                var[c][pixel] = 1000

    '''testing'''
    for n in range(test["image"]["number"]):
        # the probability of the prior distribution in training data (in log scale)
        # normalize the prior so that it sums up to 1, and then transform it into log scaled.
        p = np.zeros(classnum, dtype = float) # initialze
        for c in range(classnum):
            p[c] += np.log(float(prior[c]/train["image"]["number"]))
        
        # posteriori distribution (in log scale)
        postp = classify_testing_data(mean, var, n, p, 1)
        
        # the ground truth label for the current testing figure
        gt_label = test["label"]["label"][n]
        # judge if this figure has the correction prediction, and print the prediction result
        global error
        error += judge_and_print(postp, gt_label, n, 1)

    # print the imagination of each digit(class) by this classifier.
    print_imagination(mean, 1)

    # the error rate.
    print("Error rate: ", float(error / test["image"]["number"]))
    with open("result_continuous.txt", "a") as file:
        file.write("Error rate: {}".format(float(error / test["image"]["number"])))

if __name__ == '__main__':
    # input parameters, including some paths.
    train_image = args.train_image
    train_label = args.train_label
    test_image = args.test_image
    test_label = args.test_label
    option = args.option
    
    # process the input files. 4 files:
    # images files and labels files for training data and testing data, respectively.
    process_file(train_image, train_label, test_image, test_label)

    # Do the classification based on the toggle option.
    if option == 0:
        print("Toggle option: discrete mode")
        discrete()
    elif option == 1:
        print("Toggle option: continuous mode")
        continuous()
    else:
        raise TypeError("Please do not enter values besides 0 or 1!")