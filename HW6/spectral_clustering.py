import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from scipy.spatial.distance import cdist
import warnings
import pickle

warnings.filterwarnings("ignore")

def load_image(path: str):
    try:
        img = Image.open(path)
        img = img.convert('RGB') # make sure that it's converted to RGB order
    except:
        raise IOError("Failed to open the image {} !".format(path))
    
    w, h = img.size # 100*100

    image = np.array(img.getdata()).reshape((w*h, 3)) # 10000 * 3
    coords = np.empty((0, 2))
    for i in range(h):
        for j in range(w):
            coords = np.append(coords, [[i,j]], axis = 0)
    '''
    coordinates be like:
    [[0, 0],[0, 1],...,[0, 99],[1, 0],...,[99, 99]]
    '''
    
    return image, coords, w, h

def save_parameters(output_path: str, L, D, W, Lsym, Dsym, normalized: bool):
    if normalized:
        parameters = {
            'L': L,
            'D': D,
            'W': W,
            'Lsym': Lsym,
            'Dsym': Dsym
        }
    else:
        parameters = {
            'L': L,
            'D': D,
            'W': W
        }
    file = open(output_path, "wb")
    pickle.dump(parameters, file)
    file.close()

def load_parameters(output_path: str, normalized: bool):
    with open(output_path, "rb") as file:
        parameters = pickle.load(file)
    
    L = parameters['L']
    D = parameters['D']
    W = parameters['W']
    if normalized:
        Lsym = parameters["Lsym"]
        Dsym = parameters["Dsym"]
        return L, D, W, Lsym, Dsym
    return L, D, W

def RBF_kernel_mix(image, coords, gamma_s, gamma_c):
    dist_coords = cdist(coords, coords, "sqeuclidean") # 10000*10000
    dist_colors = cdist(image, image, "sqeuclidean")

    RBF_coords = np.exp(-gamma_s * dist_coords)
    RBF_colors = np.exp(-gamma_c * dist_colors)
    mix_kernel = np.multiply(RBF_coords, RBF_colors)
    
    return mix_kernel

def Laplacian_and_Degree(W, n):
    D = np.zeros_like(W)
    L = np.zeros_like(W)

    # for i in range(n):
    #     for j in range(n):
    #         D[i, i] += W[i, j]
    D = np.diag(np.sum(W, axis = 1))
    
    L = D - W

    return D, L

def Normalized_Laplacian_and_Degree(D, L):
    Dsym = np.diag(np.reciprocal(np.diag(np.sqrt(D)))) # Dsym = D^{-1/2}
    Lsym = np.linalg.multi_dot([Dsym, L, Dsym]) # Lsym = D^{-1/2}*L*D^{-1/2}

    return Dsym, Lsym

def get_initial_center(K, n, U, mode):
    if mode == "random":
        initial_cluster = U[list(random.sample(range(0, n), K)), :]
    elif mode == "normal":
        initial_cluster = np.empty((K, K))
        mean_vector = np.mean(U, axis = 0) # the mean of all 10000 pixels
        std_vector = np.std(U, axis = 0) # the standard deviation of all 10000 pixels.
        # sample the value of each dimension from the normal distribution from the corresponding mean and standard deviation.
        for feature in range(K):
            initial_cluster[:, feature] = np.random.normal(mean_vector[feature], std_vector[feature], K)
    elif mode == "kmeans++":
        clusters = []
        clusters.append(random.randint(0, n-1))
        cluster_num = 1
        while (cluster_num < K):
            dist = np.empty((n, cluster_num)) # initialize an array to store the minimum distance of each point to all current clusters.
            for i in range(n):
                for c in range(cluster_num):
                    dist[i, c] = np.linalg.norm(U[i, :]-U[clusters[c], :])
            dist_min = np.min(dist, axis = 1) # the shortest distance for each point w.r.t some cluster in the current "clusters".
            # note that we are selecting the furthest point from all the current clusters, randomly sampled, using the "roulette method".
            sum_rand = np.sum(dist_min) * np.random.rand() # summation times a random floatining point number between 0 and 1.
            for i in range(n):
                sum_rand -= dist_min[i]
                if sum_rand <= 0:
                    clusters.append(i)
                    break
            cluster_num += 1
        initial_cluster = U[clusters, :]
    else:
        raise ValueError("Unknown clustering mode: {}!".format(mode))
    
    return initial_cluster

def visualize(n, w, h, clusters, epoch, output_path, mode: str):
    # color list for drawing different clusters. 8 colors at most.
    colors = []
    for r in [0, 255]:
        for g in [0, 255]:
            for b in [0, 255]:
                colors.append([r, g, b])
    colors = np.array(colors)

    # the image
    img = np.empty((h, w, 3))
    for i in range(n):
        img[i // w, i % w, :] = colors[clusters[i], :]
    
    # turn to an image and save the image
    img = Image.fromarray(np.uint8(img))
    # img.save(os.path.join(output_path, mode + "_%03d.png" % epoch))

    return img

def visualize_eigenspaces(U, clusters, output_path, mode: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    for i in np.arange(3):
        ax.scatter(U[:, 0][clusters == i], U[:, 1][clusters == i], U[:, 2][clusters == i])
    
    ax.set_xlabel('eigenvector dim 1')
    ax.set_ylabel('eigenvector dim 2')
    ax.set_zlabel('eigenvector dim 3')

    plt.savefig(os.path.join(output_path, mode + '_eigenspaces.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help = "path of the input image", type = str, default = "data/image1.png")
    parser.add_argument("--o", help = "path of the output directory", type = str, default = "output_spectral_clustering/output1")
    parser.add_argument("--k", help = "number of clusters", type = int, default = 3)
    parser.add_argument("--s", help = "parameter gamma_s of the mix RBF kernel", type = float, default = 0.001)
    parser.add_argument("--c", help = "parameter gamma_c of the mix RBF kernel", type = float, default = 0.001)
    parser.add_argument("--m", help = "mode to use in initial clustering, 3 available: random, normal, kmeans++", type = str, default = "kmeans++")
    parser.add_argument("--n", help = "mode to use in spectral clustering, 2 available: 0 for unnormalized, 1 for normalized", type = int, default = 0)
    parser.add_argument("--epoch", help = "how many epochs at most", type = int, default = 20)
    args = parser.parse_args()

    gamma_s = args.s
    gamma_c = args.c
    K = args.k
    mode = args.m
    normalized = args.n
    epoch = args.epoch
    output_path = args.o
    output_dir = (args.o).split(sep = "/")[0]
    output_dir_img = (args.o).split(sep = "/")[1]
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError("Failed to construct the output directory {} !".format(output_dir))
    if not os.path.exists(os.path.join(output_dir, output_dir_img)):
        try:
            os.mkdir(os.path.join(output_dir, output_dir_img))
        except:
            raise OSError("Failed to construct the output directory {} !".format(output_dir_img))

    if normalized == 0: # unnormalized spectral clustering, ratio cut used
        image, coords, w, h = load_image(args.i)
        image_list = []
        n = image.shape[0]

        # W = RBF_kernel_mix(image, coords, gamma_s, gamma_c) # the kernel output is the "W" matrix in text book.
        # D, L = Laplacian_and_Degree(W, n) # construct Laplacian matrix and degree matrix

        # save and load the parameters
        # save_parameters(os.path.join(output_path, "ratio_cut.pkl"), L, D, W, None, None, False)
        L, D, W = load_parameters(os.path.join(output_path, "ratio_cut.pkl"), False)

        # By Rayleight quotient, the optimum indicator vector f is just the eigenvector from the eigen problem
        # L*f = lambda*f. So, we are going to solve the eigen pairs for L.

        # eigenvalues, eigenvectors = np.linalg.eig(L)
        
        # save and load the eigen pairs
        # np.save(os.path.join(output_path, "ratio_eigenvalues.npy"), eigenvalues)
        # np.save(os.path.join(output_path, "ratio_eigenvectors.npy"), eigenvectors)
        eigenvalues = np.load(os.path.join(output_path, "ratio_eigenvalues.npy"))
        eigenvectors = np.load(os.path.join(output_path, "ratio_eigenvectors.npy"))

        # sort the eigenvalues, from small to big
        sorted_idx = np.argsort(eigenvalues)
        # get those none null eigenvalues. create a mask
        masked = np.where(eigenvalues[sorted_idx] > 0, True, False)
        # mask the zeros and get the indexes or non zero eigenvalues from small to big
        sorted_idx = sorted_idx[masked]

        # contruct U matrix in text book, which is H matrix in text book. We only need the first K eigenvectors.
        U = eigenvectors[:, sorted_idx[0:K]]
        # first, apply initial clustering
        # there are 3 modes available, random, normal and kmeans++
        # default to "kmeans++".
        initial_centers = get_initial_center(K, n, U, mode)

        # next, we do the kernel k-means
        threshold = 1e-9
        for e in range(1, epoch+1):
            # initialize a final cluster array
            final_clusters = np.empty(n, dtype = np.uint8)

            # E-step: calculate the distance and classify
            for i in range(n):
                # find the closest cluster center for each point.
                dist = []
                for k in range(K):
                    dist.append(np.linalg.norm(U[i, :] - initial_centers[k, :]))
                dist = np.array(dist)
                final_clusters[i] = np.argmin(dist)

            # M-step: calculate the new centers.
            final_centers = np.zeros_like(initial_centers)
            for k in range(K):
                masked = np.where(final_clusters == k, True, False) # get the mask which picks out the data points belonging to cluster k
                final_centers[k] = np.sum(U[masked, :], axis = 0)
                if np.sum(masked) > 0: # if there's at least 1 point belonged to this cluster
                    final_centers[k] /= np.sum(masked)
                
            # visualize the current result
            img = visualize(n, w, h, final_clusters, e, output_path, "ratio")
            image_list.append(img)

            # check if the clustering process converged
            if (np.linalg.norm(final_centers - initial_centers) < threshold):
                break
            
            # otherwise, reset the initial cluster as the result cluster now.
            initial_centers = final_centers
        # make gif
        image_list[0].save(os.path.join(output_path, "ratio_GIF.gif"), save_all = True, append_images = image_list[1:], duration = 200, loop = 0)

        # visualize the eigenspaces only if the cluster number K = 3
        if K == 3:
            visualize_eigenspaces(U, final_clusters, output_path, "ratio")

    elif normalized == 1: # normalized spectral clustering, normal cut used
        image, coords, w, h = load_image(args.i)
        image_list = []
        n = image.shape[0]

        # W = RBF_kernel_mix(image, coords, gamma_s, gamma_c)
        # D, L = Laplacian_and_Degree(W, n)

        # calculate the normalized Laplacian and degree matrixes
        # Dsym, Lsym = Normalized_Laplacian_and_Degree(D, L)

        # save_parameters(os.path.join(output_path, "normal_cut.pkl"), L, D, W, Lsym, Dsym, True)
        L, D, W, Lsym, Dsym = load_parameters(os.path.join(output_path, "normal_cut.pkl"), True)

        # By Rayleight quotient, the optimum indicator vector f is just the eigenvector from the eigen problem
        # Lsym*f = lambda*f. So, we are going to solve the eigen pairs for Lsym.

        # eigenvalues, eigenvectors = np.linalg.eig(Lsym)

        # save and load the eigen pairs
        # np.save(os.path.join(output_path, "normal_eigenvalues.npy"), eigenvalues)
        # np.save(os.path.join(output_path, "normal_eigenvectors.npy"), eigenvectors)
        eigenvalues = np.load(os.path.join(output_path, "normal_eigenvalues.npy"))
        eigenvectors = np.load(os.path.join(output_path, "normal_eigenvectors.npy"))

        # sort the eigenvalues, from small to big
        sorted_idx = np.argsort(eigenvalues)
        # get those none null eigenvalues. create a mask
        masked = np.where(eigenvalues[sorted_idx] > 0, True, False)
        # mask the zeros and get the indexes or non zero eigenvalues from small to big
        sorted_idx = sorted_idx[masked]

        # contruct U matrix, which is T matrix in text book. We only need the first K eigenvectors.
        U = eigenvectors[:, sorted_idx[0:K]]
        # construct U_norm matrix, which is the normalized version of U.
        norm = np.linalg.norm(U, axis = 1).reshape(-1, 1) # L2 norm for each rows.
        U_norm = U / norm

        # first, apply initial clustering
        # there are 3 modes available, random, normal and kmeans++
        # default to "kmeans++".
        initial_centers = get_initial_center(K, n, U_norm, mode)

        # next, we do the kernel k-means
        threshold = 1e-9
        for e in range(1, epoch+1):
            # initialize a final cluster array
            final_clusters = np.empty(n, dtype = np.uint8)

            # E-step: calculate the distance and classify
            for i in range(n):
                # find the closest cluster center for each point.
                dist = []
                for k in range(K):
                    dist.append(np.linalg.norm(U_norm[i, :] - initial_centers[k, :]))
                dist = np.array(dist)
                final_clusters[i] = np.argmin(dist)

            # M-step: calculate the new centers.
            final_centers = np.zeros_like(initial_centers)
            for k in range(K):
                masked = np.where(final_clusters == k, True, False) # get the mask which picks out the data points belonging to cluster k
                final_centers[k] = np.sum(U_norm[masked, :], axis = 0)
                if np.sum(masked) > 0: # if there's at least 1 point belonged to this cluster
                    final_centers[k] /= np.sum(masked)

            # visualize the current result
            img = visualize(n, w, h, final_clusters, e, output_path, "normal")
            image_list.append(img)

            # check if the clustering process converged
            if (np.linalg.norm(final_centers - initial_centers) < threshold):
                break
            
            # otherwise, reset the initial cluster as the result cluster now.
            initial_centers = final_centers
        # make gif
        image_list[0].save(os.path.join(output_path, "normal_GIF.gif"), save_all = True, append_images = image_list[1:], duration = 200, loop = 0)

        # visualize the eigenspaces only if the cluster number K = 3
        if K == 3:
            visualize_eigenspaces(U_norm, final_clusters, output_path, "normal")
    else:
        raise ValueError("Unknown mode selected! Please input either 0 or 1 for mode!")