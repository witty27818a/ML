import numpy as np
import argparse
import numba as nb
from scipy.optimize import linear_sum_assignment

# there are 10 categories in this dataset
classnum = 10
# total pixel number
pixelnum = 28 * 28 # 784

epochs = 15 # total epochs (iterations)

'''A function that process the file.'''
def process_file(image_path, label_path):
    '''the whole binary file is in "big endian" format.
    read in all things besides the actual images' data,
    such as magic numbers, total number of images, numbers of pixels in a row and a column.
    And then, these images' data
    
    As for label data, the whole process is the same.
    read in magic numbers, total number of labels.
    And then, those labels'''

    # the images
    with open(image_path, 'rb') as f:
        _ = int.from_bytes(f.read(4), byteorder = 'big')
        number = int.from_bytes(f.read(4), byteorder = 'big')
        row = int.from_bytes(f.read(4), byteorder = 'big')
        col = int.from_bytes(f.read(4), byteorder = 'big')

        images = []
        for _ in range(number):
            image = []
            for _ in range(row):
                temp = []
                for _ in range(col):
                    # binarize the gray level value into 2 bins.
                    # if the value is less than 128, it's deemed as bin 0, bin 1 otherwise.
                    gray_value = int.from_bytes(f.read(1), byteorder = 'big')
                    if gray_value < 128:
                        temp.append(0)
                    else:
                        temp.append(1)
                image.append(temp)
            images.append(image)
        images = np.array(images)
    
    # the labels
    with open(label_path, 'rb') as f:
        _ = int.from_bytes(f.read(4), byteorder = 'big')
        _ = int.from_bytes(f.read(4), byteorder = 'big')

        label = []
        for _ in range(number):
            label.append(int.from_bytes(f.read(1), byteorder = 'big'))
        label = np.array(label)
    
    return number, row, col, images, label

'''EM Algorithms'''
# Expectation Step
'''
Remember that the responsibility for class i, say wi, is calculated as:
wi = P(Z = class i, x containing 784 pixels|lambda of class i, P of 784 pixels under class i )
= lambda of class i * (P of pixel j under class i)^(1 for pixel j) * (1 - P of pixel j under class i)^(0 for pixel j)
Note: what I used in code has different notations. Z is the responsibility matrix.
Lambda is of the same definition as that from the text book.'
As for P, please refer to the definition in the initialization part.
'''
@nb.jit(nopython = True)
def E(Lambda, P, Z, n, col, images):
    for image in range(n):
        for c in range(classnum):
            Z[image, c] = 1
            for i in range(pixelnum):
                if images[image, i // col, i % col] == 1:
                    Z[image, c] *= P[c][i]
                else:
                    Z[image, c] *= (1 - P[c][i])
            Z[image, c] *= Lambda[c]
        
        normalize = np.sum(Z[image, :])
        if normalize != 0: # prevent from divide by 0
            Z[image, :] /= normalize
        # try if possible:
        # if normalize < 1e-10:
        #     normalize = 1e-10
        # Z[image, :] /= normalize
    
    return Z

# Maximization Step
'''
Update the parameters with their MLE-like things (responsibilities would be used).
for Lambda: summation responsibilities wi of class i over all 60000 images, then divided by n, the total number of images.
for P: for each row say class i, summation responsibilities wi of class i multiplying the bin value (0 or 1) over all 60000 images,
then divided by summation of responsibilities wi of class i over all 60000 images.
Again, note that the notations used in codes might be different from those appeared in the text book.
'''
@nb.jit(nopython = True)
def M(Lambda, P, Z, n, col, images):
    # To maximize Lambda, summate responsibilities of class i over all 60000 images, then divided by 60000.
    all_res = np.sum(Z, axis = 0)
    Lambda = all_res / n
    
    for c in range(classnum):
        # for those elements = 0 in all_res, set them to 1
        # all_res is the responsibilites for each class, summated over all 60000 images
        if all_res[c] == 0:
            all_res[c] = 1

    # To maximize P, for each row say class i, summate responsibilities wi of class i multiplying the bin value (0 or 1) over all 60000 images,
    # then divided by summation of responsibilities wi of class i over all 60000 images.
    # So step 1, the denominator, which is the summation responsibilities of the class i over all images.
    # we would directly divided the responsibilites matrix by the denominator.
    for image in range(n):
        for c in range(classnum):
            Z[image, c] /= all_res[c]

    # and then step 2, the nominator
    for c in range(classnum):
        for i in range(pixelnum):
            '''
            The probability that the ith pixel is of bin-1 under class c is updated by
            summating the result of ith pixel of this image (0 or 1)
            multiplying the probability of this image being class c.
            Say, we are considering digit 7, and we are looking at pixel 123. Then,
            We will look at the pixels 123 of all images, if for some image, the pixel is a 1, then
            add the corresponding prob. of digit 7 of the image. Add 0, otherwise.
            '''
            # Reset the (c, i) entry of the Probability matrix "P"
            P[c, i] = 0
            # Do as those described in the above orange section.
            # We can directly update with the nominator part, since
            # the denominator part has been considered in the last step.
            for image in range(n):
                P[c, i] += images[image, i // col, i % col] * Z[image, c]
    
    return Lambda, P

# print out the imagination of each digit.
def imagination(P, col, file):
    for c in range(classnum):
        print("class: ", c)
        file.write("class: {}\n".format(c))
        for i in range(pixelnum):
            if P[c][i] >= 0.5:
                print(1, end = ' ')
                file.write("1 ")
            else:
                print(0, end = ' ')
                file.write("0 ")
            
            if i % col == (col - 1):
                print("") # change line
                file.write("\n")
        print("") # change line
        file.write("\n")

# print out the imagination of each digit, after we have the matching relations between clusters and digits.
def final_imagination(P, col, relations, file):
    # transform the persepective of relations. from "cluster to digit", to "digit to cluster"
    relations_d2c = []
    for digit in range(classnum):
        relations_d2c.append(np.where(relations == digit))
    for c in range(classnum):
        print("digit: ", c)
        file.write("digit: {}\n".format(c))
        cluster = relations_d2c[c]
        for i in range(pixelnum):
            if P[cluster, i] >= 0.5:
                print(1, end = ' ')
                file.write("1 ")
            else:
                print(0, end = ' ')
                file.write("0 ")
            if i % col == (col - 1):
                print("")
                file.write("\n")
        print("")
        file.write("\n")

# Get the ground truth distribution. That is, Under each class, how likely a specific pixel would be painted black.
@nb.jit(nopython = True)
def get_gt_distribution(images, labels, n, col):
    gt_P = np.zeros((classnum, pixelnum)) # shape: (10, 784)
    
    label_count = np.zeros(classnum)
    for i in labels:
        label_count[i] += 1
    
    # count for each pixel, how many times it's painted as black, under each class
    for image in range(n):
        gt = labels[image]
        for i in range(pixelnum):
            if images[image, i // col, i % col] == 1:
                gt_P[gt, i] += 1
    # normalize
    for c in range(classnum):
        for i in range(pixelnum):
            gt_P[c, i] /= label_count[c]
    
    return gt_P

# Do the matching between clusters and digits by Hungarian Algorithm.
def matching(P, gt_P):
    # Set the cost matrix
    cost = np.zeros((classnum, classnum))
    # Calculate the 2-norm between the 784-d vectors of each cluster and each gt digit.
    for i in range(classnum):
        for j in range(classnum):
            cost[i, j] = np.linalg.norm(P[i] - gt_P[j])
    
    # Do the algorithm and get the relation
    _, relations = linear_sum_assignment(cost)

    return relations

# Calculate the confusion matrix and sensitivity, specificity for each digit.
# Return the total error count and all confusion matrixes.
# Note that "relations" would transfer cluster to digit.
@nb.jit(nopython = True)
def confusion_and_error(predicted_cluster, labels, n, relations):
    error = 0
    # confusion matrix: shape = (10, 4), each row is the TP, FP, TN, FN counts for each digit
    confusion_matrix = np.zeros((classnum, 4), dtype = np.uint)

    # for each image
    for image in range(n):
        # based on the predicted result and the ground truth label of this image
        # update the confusion matrixes for each digit.
        for c in range(classnum):
            # Case: the current class is the ground truth. Then, it should be positive.
            if c == labels[image]:
                # TP case
                if c == relations[predicted_cluster[image]]:
                    confusion_matrix[c][0] += 1
                # FN case
                else:
                    confusion_matrix[c][3] += 1
                    error += 1
            # Case: the current class is not the gt. Then, it should be negative.
            else:
                # FP case
                if c == relations[predicted_cluster[image]]:
                    confusion_matrix[c][1] += 1
                # TN case
                else:
                    confusion_matrix[c][2] += 1

    return error, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help = "The path of training image data from MNIST dataset.", type = str, default = "train-images.idx3-ubyte")
    parser.add_argument("--label", help = "The path of labels of training data from MNIST dataset.", type = str, default = "train-labels.idx1-ubyte")
    args = parser.parse_args()
    
    image_path = args.image
    label_path = args.label

    '''process the input file. 2 files: images file and label file.'''
    n, row, col, images, labels = process_file(image_path, label_path)
    # now, all pixel values are binary.
    # The shape of "images" should be (60000, 28, 28) and that of "labels" is (60000, )

    '''initial parameters setting'''
    # the responsibilities of each class, for each image, shape: (60000, 10)
    Z = np.zeros((n, classnum), dtype = 'float64')
    
    # Lambda, parameters related with the probability that each class appears.
    # shape: (10,)
    Lambda = np.random.uniform(size = classnum).astype('float64')
    # normalize so that it sums up to 1.
    Lambda /= np.sum(Lambda)
    
    # P, parameters of joint possibilities.
    # For the (i,j)th entry, it denotes that under the ith class
    # the probability that the jth pixel is of bin-1 (i.e., black pixel).
    # shape: (10, 784)
    P = np.random.uniform(size = (classnum, pixelnum)).astype('float64')
    P = (P * 8 + 1) / 10 # Scale it so that it scales from 0.1 ~ 0.9, not too big or too small.

    # save the paramters initialization
    np.save('P.npy', P)
    np.save('Lambda.npy', Lambda)

    # Load the initial parameters that generated good results.
    # P = np.load('P.npy')
    # Lambda = np.load('Lambda.npy)
    
    # result file to output.
    file = open("EM_results.txt", "a")

    for e in range(epochs):
        # previous lambda
        P_prev = P.copy()
        print("No. of Iteration: {}".format(e+1))
        file.write("No. of Iteration: {}\n".format(e+1))
        Z = E(Lambda, P, Z, n, col, images) # expectation step
        Lambda, P = M(Lambda, P, Z, n, col, images) # maximization step
        diff = np.linalg.norm(P - P_prev) # the difference between the "P" before and after.

        imagination(P, col, file) # print out the imagination of each digit.
        print("Difference: {}".format(diff))
        file.write("Difference: {}\n".format(diff))
        print("------------------------------------------------------\n")
        file.write("------------------------------------------------------\n\n")
    
    # By the responsibility matrix Z, we can set the max value of each row as the predicted cluster for each image.
    predicted_cluster = np.argmax(Z, axis = 1)
    '''
    To do the matching between clusters and digits, we need to first know the ground truth distribution.
    That is, Under each class, how likely a specific pixel would be painted black. Basically the same concept as "P", thus the same shape.
    '''
    gt_P = get_gt_distribution(images, labels, n, col)

    '''
    Now, for the matching part. We would use the linear_sum_assignment from Scipy,
    which is in fact the Hungarian Algorithm.
    '''
    relations = matching(P, gt_P) # the true digit of each cluster.

    # Print the final imagination for each digit.
    print("final imaginations:")
    file.write("final imaginations:\n")
    final_imagination(P, col, relations, file)

    error, confusion_matrix = confusion_and_error(predicted_cluster, labels, n, relations)
    # For each digit, print out the final confusion matrix.
    for c in range(classnum):
        print("Confusion Matrix: digit {}".format(c))
        file.write("Confusion Matrix: digit {}\n".format(c))
        print("\t\t\tIs digit {}\tNot digit {}".format(c, c))
        file.write("\t\t\tIs digit {}\tNot digit {}\n".format(c, c))
        print("Predicted: is  {}\t{}\t\t{}".format(c, confusion_matrix[c][0], confusion_matrix[c][1]))
        file.write("Predicted: is  {}\t{}\t\t{}\n".format(c, confusion_matrix[c][0], confusion_matrix[c][1]))
        print("Predicted: not {}\t{}\t\t{}".format(c, confusion_matrix[c][3], confusion_matrix[c][2]))
        file.write("Predicted: not {}\t{}\t\t{}\n".format(c, confusion_matrix[c][3], confusion_matrix[c][2]))
        print("Sensitivity: {}".format(confusion_matrix[c][0] / (confusion_matrix[c][0] + confusion_matrix[c][3])))
        file.write("Sensitivity: {}\n".format(confusion_matrix[c][0] / (confusion_matrix[c][0] + confusion_matrix[c][3])))
        print("Specificity: {}".format(confusion_matrix[c][2] / (confusion_matrix[c][2] + confusion_matrix[c][1])))
        file.write("Specificity: {}\n".format(confusion_matrix[c][2] / (confusion_matrix[c][2] + confusion_matrix[c][1])))
        print("------------------------------------------------------\n")
        file.write("------------------------------------------------------\n\n")

    print("Total iteration to converge: {}".format(epochs))
    file.write("Total iteration to converge: {}\n".format(epochs))
    print("Total error rate: {}".format(error / n))
    file.write("Total error rate: {}\n".format(error / n))
    
    file.close()