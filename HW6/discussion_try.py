import argparse
import os
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
import random
import warnings

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

def RBF_kernel_mix(image, coords, gamma_s, gamma_c):
    dist_coords = cdist(coords, coords, "sqeuclidean") # 10000*10000
    dist_colors = cdist(image, image, "sqeuclidean")

    RBF_coords = np.exp(-gamma_s * dist_coords)
    RBF_colors = np.exp(-gamma_c * dist_colors)
    mix_kernel = np.multiply(RBF_coords, RBF_colors)
    
    return mix_kernel

def get_initial_center(K, n, kernel, mode):
    if mode == "random":
        initial_cluster = kernel[list(random.sample(range(0, n), K)), :]
    elif mode == "normal":
        initial_cluster = np.empty((K, n))
        mean_vector = np.mean(kernel, axis = 0) # the mean of all 10000 pixels
        std_vector = np.std(kernel, axis = 0) # the standard deviation of all 10000 pixels.
        # sample the value of each dimension from the normal distribution from the corresponding mean and standard deviation.
        for feature in range(n):
            initial_cluster[:, feature] = np.random.normal(mean_vector[feature], std_vector[feature], K)
    elif mode == "kmeans++":
        clusters = []
        clusters.append(random.randint(0, n-1))
        cluster_num = 1
        while (cluster_num < K):
            dist = np.empty((n, cluster_num)) # initialize an array to store the minimum distance of each point to all current clusters.
            for i in range(n):
                for c in range(cluster_num):
                    dist[i, c] = np.linalg.norm(kernel[i, :]-kernel[clusters[c], :])
            dist_min = np.min(dist, axis = 1) # the shortest distance for each point w.r.t some cluster in the current "clusters".
            # note that we are selecting the furthest point from all the current clusters, randomly sampled, using the "roulette method".
            sum_rand = np.sum(dist_min) * np.random.rand() # summation times a random floatining point number between 0 and 1.
            for i in range(n):
                sum_rand -= dist_min[i]
                if sum_rand <= 0:
                    clusters.append(i)
                    break
            cluster_num += 1
        initial_cluster = kernel[clusters, :]
    else:
        raise ValueError("Unknown clustering mode: {}!".format(mode))
    
    return initial_cluster

def visualize(n, w, h, clusters):
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

    return img

def Kernel_Kmeans(output_path, image, coords, w, h, gamma_s, gamma_c, K, mode, epoch):
    n = image.shape[0] # the number of pixels, 10000 in this case

    # find out the mixed kernels
    kernel = RBF_kernel_mix(image, coords, gamma_s, gamma_c)

    # first, apply initial clustering
    # there are 3 modes available, random, normal and kmeans++
    # default to "kmeans++".
    initial_centers = get_initial_center(K, n, kernel, mode)
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
                dist.append(np.linalg.norm(kernel[i, :] - initial_centers[k, :]))
            dist = np.array(dist)
            final_clusters[i] = np.argmin(dist)

        # M-step: calculate the new centers.
        final_centers = np.zeros_like(initial_centers)
        for k in range(K):
            masked = np.where(final_clusters == k, True, False) # get the mask which picks out the data points belonging to cluster k
            final_centers[k] = np.sum(kernel[masked, :], axis = 0)
            if np.sum(masked) > 0: # if there's at least 1 point belonged to this cluster
                final_centers[k] /= np.sum(masked)
            
        # visualize the current result
        img = visualize(n, w, h, final_clusters)

        # check if the clustering process converged
        if (np.linalg.norm(final_centers - initial_centers) < threshold):
            img.save(os.path.join(output_path, "s_{}_c_{}.png".format(gamma_s, gamma_c)))
            break
        
        # otherwise, reset the initial cluster as the result cluster now.
        initial_centers = final_centers
    img.save(os.path.join(output_path, "s_{}_c_{}_no_conv.png".format(gamma_s, gamma_c)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help = "path of the input image", type = str, default = "data/image1.png")
    parser.add_argument("--o", help = "path of the output directory", type = str, default = "output_kernel_kmeans/output1")
    parser.add_argument("--k", help = "number of clusters", type = int, default = 4)
    parser.add_argument("--s", help = "parameter gamma_s of the mix RBF kernel", type = float, default = 0.001)
    parser.add_argument("--c", help = "parameter gamma_c of the mix RBF kernel", type = float, default = 0.001)
    parser.add_argument("--m", help = "mode to use in initial clustering, 3 available: random, normal, kmeans++", type = str, default = "kmeans++")
    parser.add_argument("--epoch", help = "how many epochs at most", type = int, default = 20)
    args = parser.parse_args()

    gamma_s = args.s
    gamma_c = args.c
    K = args.k
    mode = args.m
    epoch = args.epoch
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
    image, coords, w, h = load_image(args.i)
    for s in [0.01, 0.001, 0.0001]:
        for c in [0.01, 0.001, 0.0001]:
            gamma_s = s
            gamma_c = c
            Kernel_Kmeans(args.o, image, coords, w, h, gamma_s, gamma_c, K, mode, epoch)