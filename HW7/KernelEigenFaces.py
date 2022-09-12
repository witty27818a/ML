import numpy as np
import os, re
import argparse
from PIL import Image
import matplotlib.pyplot as plt
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

def PCA(data, PCnum):
    mu = np.mean(data, axis = 0) # mean vector, dim=size*size=50*50=2500
    demeaned = data - mu
    cov = np.cov(demeaned.T)

    eigvalues, eigvectors = np.linalg.eigh(cov)
    # normalization
    for i in range(eigvectors.shape[1]):
        eigvectors[:, i] /= np.linalg.norm(eigvectors[:, i])
    np.save("D:\\HW7_ML_params\\eigval_pca.npy", eigvalues)
    np.save("D:\\HW7_ML_params\\eigvec_pca.npy", eigvectors)
    eigvalues = np.load("D:\\HW7_ML_params\\eigval_pca.npy")
    eigvectors = np.load("D:\\HW7_ML_params\\eigvec_pca.npy")

    maxK_idx = np.argsort(eigvalues)[::-1][:PCnum] # max 25 eigvalues
    W = eigvectors[:, maxK_idx].real # discard imaginary part

    return mu, W

def LDA(data, labels, dim):
    d = data.shape[1] # original dimension in data space, 2500
    classes = np.unique(labels) # unique labels, which are subject numbers in this case.
    mu = np.mean(data, axis = 0) # mean vector, dim=2500
    # initialize within-class scatter Sw and between-class scatter Sb
    Sw = np.zeros((d, d), dtype = np.float64)
    Sb = np.zeros_like(Sw, dtype = np.float64)

    # calculate Sw and Sb, by summating over each class (a subject/person is deemed as a cluster/class).
    for c in classes:
        data_c = data[np.where(labels == c)[0], :] # use index 0 since numpy where returns a tuple
        mu_c = np.mean(data_c, axis = 0) # mean vector of the class c
        Sw += np.matmul((data_c - mu_c).T, (data_c - mu_c))
        Sb += data_c.shape[0] * np.matmul((mu_c - mu).T, (mu_c - mu))
    
    # since Sw might be non-invertible if the data size, n < the dimension of data space, d,
    # which is True in this case. (165 < 2500). So, we use the pseudo inverse, by "pinv"
    # solve the eigen problem of matrix Sw^(-1)*Sb.
    
    eigvalues, eigvectors = np.linalg.eigh(np.matmul(np.linalg.pinv(Sw), Sb))
    # normalization
    for i in range(eigvectors.shape[1]):
        eigvectors[:, i] /= np.linalg.norm(eigvectors[:, i])
    np.save("D:\\HW7_ML_params\\eigval_lda.npy", eigvalues)
    np.save("D:\\HW7_ML_params\\eigvec_lda.npy", eigvectors)
    eigvalues = np.load("D:\\HW7_ML_params\\eigval_lda.npy")
    eigvectors = np.load("D:\\HW7_ML_params\\eigvec_lda.npy")

    maxq_idx = np.argsort(eigvalues)[::-1][:dim] # max 25 eigvalues
    W = eigvectors[:, maxq_idx].real # discard imaginary part

    return W

def visualization(data, settings, output_dir, size, W, mu = None):
    # data = (10, 2500)
    if mu is None:
        mu = np.zeros(data.shape[1]) # 2500
    demeaned = data - mu
    proj = np.matmul(demeaned, W) # (10, 25), each row a proj vector.
    reconstr = np.matmul(proj, W.T) + mu # (10, 2500), each row a reconstructed image
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    dim = W.shape[1] # 25

    # 25 eigenfaces/fisherfaces
    if int((dim ** 0.5) + 0.5) ** 2 == dim: # if the PC number in PCA or dimension in LDA is a square number
        plt.clf()
        for i in range(dim): # 25
            plt.subplot(int(dim ** 0.5 + 0.5), int(dim ** 0.5 + 0.5), i+1) # 5 * 5
            plt.imshow(W[:, i].reshape((size, size)), cmap = 'gray')
            plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'all.png'), bbox_inches = 'tight')
    for i in range(dim): # 25
        plt.clf()
        plt.imshow(W[:, i].reshape((size, size)), cmap = 'gray')
        plt.savefig(os.path.join(output_dir, '{}.png'.format(i+1)))

    # 10 reconstructed faces
    if reconstr.shape[0] == 10:
        plt.clf()
        for i in range(data.shape[0]):
            plt.subplot(2, 5, i+1)
            plt.imshow(reconstr[i].reshape((size, size)), cmap = 'gray')
            plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'reconstructions.png'), bbox_inches = 'tight')
    for i in range(reconstr.shape[0]):
        plt.clf()
        plt.imshow(reconstr[i].reshape((size, size)), cmap = 'gray')
        plt.savefig(os.path.join(output_dir, settings[i] + "_reconstructed.png"))

def face_recognition(train, train_labels, test, test_labels, K, output_dir, method, kernel = None):
    file = open(os.path.join(output_dir, "{}.txt".format(method)), mode = 'a')
    if kernel is None:
        print("method used: {}, clustering by: KNN\n".format(method))
        file.write("method used: {}, clustering by: KNN\n\n".format(method))
    else:
        print("method used: {}, clustering by: KNN, kernel type: {}\n".format(method, kernel))
        file.write("method used: {}, clustering by: KNN, kernel type: {}\n\n".format(method, kernel))

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
    for k in K:
        correct = 0
        for i in range(test.shape[0]):
            test_dist = distances[i] # distances between the testing images and all training images.
            KNN_labels = np.asarray([x[1] for x in test_dist[:k]]) # pick out the labels of the k nearest neighbors
            candidate_labels, counts = np.unique(KNN_labels, return_counts = True) # get counts of all unique labels in KNN
            prediction = candidate_labels[np.argmax(counts)]
            if prediction == test_labels[i]:
                correct += 1
        print("K = {:>2}, accuracy = {:>.3f} ({}/{})".format(k, correct / test.shape[0], correct, test.shape[0]))
        file.write("K = {:>2}, accuracy = {:>.3f} ({}/{})\n".format(k, correct / test.shape[0], correct, test.shape[0]))
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

def kernelPCA(data, PCnum, kernel):
    K = compute_kernel(data, kernel) # get the Gram matrix
    N = K.shape[0] # the total data size
    # we need to center the data in feature space, so that we can continue to conduct eigen decomposition
    oneN = np.full((N, N), fill_value = 1 / N, dtype = np.float64) # square matrix of N order with every element = 1/N
    Kc = K - np.matmul(oneN, K) - np.matmul(K, oneN) + np.linalg.multi_dot([oneN, K, oneN])
    
    eigvalues, eigvectors = np.linalg.eigh(Kc)
    for i in range(eigvectors.shape[1]):
        eigvectors[:, i] /= np.linalg.norm(eigvectors[:, i])
    np.save("D:\\HW7_ML_params\\eigval_kernel_pca_" + kernel + ".npy", eigvalues)
    np.save("D:\\HW7_ML_params\\eigvec_kernel_pca_" + kernel + ".npy", eigvectors)
    eigvalues = np.load("D:\\HW7_ML_params\\eigval_kernel_pca_" + kernel + ".npy")
    eigvectors = np.load("D:\\HW7_ML_params\\eigvec_kernel_pca_" + kernel + ".npy")

    maxK_idx = np.argsort(eigvalues)[::-1][:PCnum]
    W = eigvectors[:, maxK_idx].real

    all_kernel_proj = np.matmul(Kc, W)

    return all_kernel_proj

def kernelLDA(data, labels, dim, kernel):
    classes = np.unique(labels)
    K = compute_kernel(data, kernel)
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
    np.save("D:\\HW7_ML_params\\eigval_kernel_lda_" + kernel + ".npy", eigvalues)
    np.save("D:\\HW7_ML_params\\eigvec_kernel_lda_" + kernel + ".npy", eigvectors)
    eigvalues = np.load("D:\\HW7_ML_params\\eigval_kernel_lda_" + kernel + ".npy")
    eigvectors = np.load("D:\\HW7_ML_params\\eigvec_kernel_lda_" + kernel + ".npy")

    maxq_idx = np.argsort(eigvalues)[::-1][:dim]
    W = eigvectors[:, maxq_idx].real

    all_kernel_proj = np.matmul(K, W)

    return all_kernel_proj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = 'the input path of Yale Face database folders. 2 folders "Training" and "Testing" containing some pgm files in the path', type = str, default = "D:\\HW7_ML_Yale_Face_Database")
    parser.add_argument("--output", help = "the path of output root folder", type = str, default = ".\\output")
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
    all_settings = np.hstack((train_settings, test_settings)) #(165,)
    all_labels = np.hstack((train_labels, test_labels)) #(165,)

    '''part 1: use PCA and LDA to show first 25 eigenfaces/fisherfaces. Randomly pick 10 images to show their reconstruction.'''
    random_idx = np.random.choice(all_data.shape[0], 10)
    data_picked = all_data[random_idx] # data_picked = (10, 2500)
    settings_picked = all_settings[random_idx] # (10,)

    '''PCA: eigen faces'''
    pc_number = args.dim # 25
    mu, W = PCA(all_data, pc_number) # mu = (2500,) W = (2500, 25)
    visualization(data_picked, settings_picked, os.path.join(args.output, 'eigenfaces'), size, W, mu)
    
    '''LDA: fisher faces'''
    q = args.dim # 25
    W = LDA(all_data, all_labels, q)
    visualization(data_picked, settings_picked, os.path.join(args.output, 'fisherfaces'), size, W)

    '''part 2: face recognition by PCA and LDA. use KNN to cluster images'''
    '''PCA'''
    pc_number = args.dim # 25
    K = np.arange(1, 17, 2) # cluster numbers to be tried: 1, 3, 5, ..., 15
    mu, W = PCA(all_data, pc_number) # mu = (2500,) W = (2500, 25)
    # demean training data and testing data, and get their projections, respectively.
    train_proj = np.matmul(train_data - mu, W)
    test_proj = np.matmul(test_data - mu, W)
    # do the face recognition and print+write out the accuracy
    face_recognition(train_proj, train_labels, test_proj, test_labels, K, args.output, 'PCA')

    '''LDA'''
    q = args.dim # 25
    K = np.arange(1, 17, 2) # cluster numbers to be tried: 1, 3, 5, ..., 15
    W = LDA(all_data, all_labels, q) # W = (2500, 25)
    # demean training data and testing data, and get their projections, respectively.
    train_proj = np.matmul(train_data, W)
    test_proj = np.matmul(test_data, W)
    # do the face recognition and print+write out the accuracy
    face_recognition(train_proj, train_labels, test_proj, test_labels, K, args.output, 'LDA')

    '''part 3: face recognition by kernel PCA and kernel LDA. use KNN to cluster images'''
    '''kernel PCA'''
    pc_number = args.dim # 25
    K = np.arange(1, 17, 2)
    kernels = ['linear', 'polynomial', 'RBF'] # 3 different kernels
    for kernel in kernels:
        all_kernel_proj = kernelPCA(all_data, pc_number, kernel)
        # training data
        train_kernel_proj = all_kernel_proj[:train_data.shape[0]]
        # testing data
        test_kernel_proj = all_kernel_proj[train_data.shape[0]:]

        face_recognition(train_kernel_proj, train_labels, test_kernel_proj, test_labels, K, args.output, 'kernel_PCA', kernel)
    
    '''kernel LDA'''
    q = args.dim # 25
    K = np.arange(1, 17, 2)
    kernels = ['linear', 'polynomial', 'RBF']
    for kernel in kernels:
        all_kernel_proj = kernelLDA(all_data, all_labels, q, kernel)
        # training data
        train_kernel_proj = all_kernel_proj[:train_data.shape[0]]
        # testing data
        test_kernel_proj = all_kernel_proj[train_data.shape[0]:]

        face_recognition(train_kernel_proj, train_labels, test_kernel_proj, test_labels, K, args.output, 'kernel_LDA', kernel)