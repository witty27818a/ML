import numpy as np
import os, re
import argparse
from PIL import Image
from scipy.spatial.distance import cdist

def read_files(path, size):
    # images array, settings names array, labels(integer) array
    data, settings, labels = [], [], []
    for pgm in os.listdir(path):
        image = Image.open(os.path.join(path, pgm))
        image = image.resize((size, size), resample = Image.ANTIALIAS) # use filters when resizing
        image = np.array(image).ravel().astype(np.float64)

        label = int(re.search(r"subject(\d+)", pgm, flags = re.I).group(1))

        data.append(image)
        settings.append(pgm.split('.')[1]+'_'+str(label))
        labels.append(label)
    return (np.asarray(data), np.asarray(settings), np.asarray(labels)) # list of numpy arrays.

def face_recognition(train, train_labels, test, test_labels, K, output_dir, method, params):
    file = open(os.path.join(output_dir, "{}.txt".format(method)), mode = 'a')
    if method == 'kernel_PCA':
        print("parameters used: gamma = {}, coeff = {}, degree = {}\n".format(params[0], params[1], params[2]))
        file.write("parameters used: gamma = {}, coeff = {}, degree = {}\n\n".format(params[0], params[1], params[2]))
    else:
        print("parameters used: gamma = {}\n".format(params[0]))
        file.write("parameters used: gamma = {}\n\n".format(params[0]))

    # train: (135, 2500) test: (30, 2500)
    distances = [] # initialize a matrix that stores tuples
    '''the (i,j) element in distances is (a,b), where a is the squared Euclidean distance for ith testing data
    and jth training data, and b is the label of the jth training data.'''
    for i in range(test.shape[0]): # 30
        test_dist = [] # initialize a vector that stores the distance tuples between this testing data and all training data
        for j in range(train.shape[0]): # 135
            sqdist = cdist(test[i].reshape(1, -1), train[j].reshape(1, -1), metric = 'sqeuclidean').item()
            test_dist.append((sqdist, train_labels[j]))
        test_dist.sort(key = lambda x: x[0]) # sort by sqdist, min first
        distances.append(test_dist)
    distances = np.asarray(distances)

    # do KNN and get prediction results as well as accuracy
    correct = 0
    for i in range(test.shape[0]):
        test_dist = distances[i] # distances between the testing images and all training images.
        KNN_labels = np.asarray([x[1] for x in test_dist[:K]]) # pick out the labels of the k nearest neighbors
        candidate_labels, counts = np.unique(KNN_labels, return_counts = True) # get counts of all unique labels in KNN
        prediction = candidate_labels[np.argmax(counts)]
        if prediction == test_labels[i]:
            correct += 1
    print("accuracy = {:>.3f} ({}/{})".format(correct / test.shape[0], correct, test.shape[0]))
    file.write("accuracy = {:>.3f} ({}/{})\n".format(correct / test.shape[0], correct, test.shape[0]))
    print('\n')
    file.write('\n')
    file.close()

def compute_kernel(data, kernel_type, params = None):
    # you can tune the hyper-parameters in discussion section
    if kernel_type == 'linear':
        return np.matmul(data, data.T)
    elif kernel_type == 'polynomial':
        gamma = 5
        coeff = 10
        degree = 2
        if params is not None:
            gamma, coeff, degree = params
        return np.power(gamma * np.matmul(data, data.T) + coeff, degree)
    else:
        gamma = 1e-7
        if params is not None:
            gamma = params[0]
        return np.exp(-gamma * cdist(data, data, metric = 'sqeuclidean'))

def kernelPCA(data, PCnum, kernel, params = None):
    K = compute_kernel(data, kernel, params) # get the Gram matrix
    N = K.shape[0] # the total data size
    # we need to center the data in feature space, so that we can continue to conduct eigen decomposition
    oneN = np.full((N, N), fill_value = 1 / N, dtype = np.float64) # square matrix of N order with every element = 1/N
    Kc = K - np.matmul(oneN, K) - np.matmul(K, oneN) + np.linalg.multi_dot([oneN, K, oneN])
    
    eigvalues, eigvectors = np.linalg.eigh(Kc)
    for i in range(eigvectors.shape[1]):
        eigvectors[:, i] /= np.linalg.norm(eigvectors[:, i])

    maxK_idx = np.argsort(eigvalues)[::-1][:PCnum]
    W = eigvectors[:, maxK_idx].real

    all_kernel_proj = np.matmul(Kc, W)

    return all_kernel_proj

def kernelLDA(data, labels, dim, kernel, params = None):
    classes = np.unique(labels)
    K = compute_kernel(data, kernel, params)
    N = K.shape[0] # 165
    mu = np.mean(K, axis = 0) # mean vector, dim=165
    # initialize within-class scatter kernel version ker_Sw and between-class scatter kernel version ker_Sb
    ker_Sw = np.zeros((N, N), dtype = np.float64)
    ker_Sb = np.zeros_like(ker_Sw, dtype = np.float64)

    # calculate ker_Sw and ker_Sb
    for c in classes:
        K_c = K[np.where(labels == c)[0], :]
        mu_c = np.mean(K_c, axis = 0)
        Nk = K_c.shape[0] # size of the cluster c.
        I = np.identity(Nk)
        oneNk = np.full((Nk, Nk), fill_value = 1 / Nk, dtype = np.float64)
        # this part needs some derivations, look at https://zhuanlan.zhihu.com/p/92359921
        ker_Sw += np.linalg.multi_dot([K_c.T, I - oneNk, K_c]) # 和參考文章相反，因為K_c取法問題
        ker_Sb += Nk * np.matmul((mu_c - mu).T, (mu_c - mu))
    
    eigvalues, eigvectors = np.linalg.eigh(np.matmul(np.linalg.pinv(ker_Sw), ker_Sb))
    for i in range(eigvectors.shape[1]):
        eigvectors[:, i] /= np.linalg.norm(eigvectors[:, i])

    maxq_idx = np.argsort(eigvalues)[::-1][:dim]
    W = eigvectors[:, maxq_idx].real

    all_kernel_proj = np.matmul(K, W)

    return all_kernel_proj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = 'the input path of Yale Face database folders. 2 folders "Training" and "Testing" containing some pgm files in the path', type = str, default = "D:\\HW7_ML_Yale_Face_Database")
    parser.add_argument("--output", help = "the path of output root folder", type = str, default = ".\\output\\observation_and_discussion")
    parser.add_argument("--size", help = "the size to used in opening images. size x size images.", type = int, default = 50)
    parser.add_argument("--dim", help = "number of principal components to be used, or the dimension after mapping in LDA", type = int, default = 25)
    parser.add_argument("--seed", help = "random seed for reproducing results", type = int, default = 100)
    args = parser.parse_args()
    np.random.seed(args.seed) # set seeds for reproduction and don't need to calculate eigen decomposition 2nd time.

    train_path = os.path.join(args.input, 'Training')
    test_path = os.path.join(args.input, 'Testing')
    if not os.path.exists(train_path):
        raise OSError("Cannot open or find the training folder!")
    if not os.path.exists(test_path):
        raise OSError("Cannot open or find the testing folder!")
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    size = args.size # 50, so (50, 50) images
    train_data,  train_settings, train_labels = read_files(train_path, size)
    test_data, test_settings, test_labels = read_files(test_path, size)
    # combining all data, setting names and labels in training and testing
    all_data = np.vstack((train_data, test_data)) #(165, 2500)
    all_labels = np.hstack((train_labels, test_labels)) #(165,)

    '''part 3: face recognition by kernel PCA and kernel LDA. use KNN to cluster images'''
    '''kernel PCA'''
    pc_number = args.dim # 25
    K = 5
    kernel = 'polynomial'
    for gamma in [0.2, 1, 5]:
        for coeff in [0, 10]:
            for degree in [2, 3, 4]:
                params = [gamma, coeff, degree]

                all_kernel_proj = kernelPCA(all_data, pc_number, kernel, params)
                # training data
                train_kernel_proj = all_kernel_proj[:train_data.shape[0]]
                # testing data
                test_kernel_proj = all_kernel_proj[train_data.shape[0]:]

                face_recognition(train_kernel_proj, train_labels, test_kernel_proj, test_labels, K, args.output, 'kernel_PCA', params)
    
    '''kernel LDA'''
    q = args.dim # 25
    K = 7
    kernel = 'RBF'
    for gamma in [1e-10, 1e-7, 1e-4]:
        params = [gamma]

        all_kernel_proj = kernelLDA(all_data, all_labels, q, kernel, params)
        # training data
        train_kernel_proj = all_kernel_proj[:train_data.shape[0]]
        # testing data
        test_kernel_proj = all_kernel_proj[train_data.shape[0]:]

        face_recognition(train_kernel_proj, train_labels, test_kernel_proj, test_labels, K, args.output, 'kernel_LDA', params)