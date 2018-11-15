import numpy as np
import os
from sklearn.decomposition import IncrementalPCA
from os import listdir
from os.path import isfile, join
import copy
import torch
import time

def get_PCA(src_dir, tar_dir, k):
    '''
    Assuming IPCA object hasn't been trained yet.

    src_dir: directory with all the original features *only*
    tar_dir: directory to place PCA reduced features
    k: reduced dimension of each vector
    '''

    onlyfiles = [f for f in listdir(src_dir) if isfile(join(src_dir, f))]
    onlyfiles.sort()

    tot_im = len(onlyfiles)

    bs = k
    total_bs = tot_im // bs
    rem = tot_im % bs

    ipca = IncrementalPCA(n_components=bs)

    print("Starting PCA Training")
    
    temp = np.zeros((bs, 512*512), dtype=float)
    for i in range(total_bs):
        for j in range(bs):
            nm = onlyfiles[i*bs+j]
            temp[j,:] = np.load(src_dir + nm).flatten()
        ipca.partial_fit(temp)
    del temp

    print("Getting PCA Features")

    temp = np.zeros((512*512), dtype=float)
    for i in range(tot_im):
        nm = onlyfiles[i]
        temp[:] = np.load(src_dir + nm).flatten()
        np.save(tar_dir + onlyfiles[i], ipca.transform(temp.reshape(1,-1)))
    del temp

    
def normalize(x, mean, k):
    if k == None:
        return (x - mean)
    return np.divide((x - mean), np.absolute(x - mean) ** k)

def pdist(sample_1, sample_2, norm=2):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.abs(distances_squared)
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return inner

def get_nearest_neighbours(src_dir, tar_dir, k):
    '''
    src_dir: directory with all the features stored as numpy files
    tar_dir: directory to store the txt file listing all the neighbours
    k: number of neighbours
    '''

    image_names = listdir(src_dir)
    image_names.sort()
    num_of_images = len(image_names)

    print(len(image_names), "images found.")
    
    # TODO: To parallelize
    mean = None
    for i, image_name in enumerate(image_names):
        if i % 1000 == 0:
            print(i)
        features = np.load(src_dir + image_name)
        if type(mean) == type(None):
            mean = copy.deepcopy(features)
        mean = ((i * mean) + features) / (1 + i)

    print("Calculated the mean of all features!")

    X = []
    for i, image_name in enumerate(image_names):
        if (i % 100 == 0):
            print(i)
        features = np.load(src_dir + image_name)
        features = normalize(features, mean, None)
        X.append(features)

    X = np.array(X)

    print("Loaded all features after normalizing ->", X.shape)
    
    BATCH_SIZE = 1000
    sources = []
    B = torch.from_numpy(X.astype(np.float32))

    top15 = open(tar_dir + "Top15.txt","w+")
    bottom15 = open(tar_dir + "Bottom15.txt","w+")

    begin = time.time()
    for i in range(0, num_of_images, BATCH_SIZE):
        sources = []
        for j in range(BATCH_SIZE):
            if (i + j >= num_of_images):
                break
            sources.append(X[i+j])
        sources = np.array(sources)
        print("Initialized batch with ", sources.shape)

        A = torch.from_numpy(sources.astype(np.float32))

        d = pdist(A, B)
        print("Calculated distances!")
        neighbours = d.data.numpy().argsort(axis = -1)
        print("Sorted distances!")

        n = len(sources)
        for img_id in range(n):
            top15.write(image_names[i + img_id].strip('.npy') + ', ')
            bottom15.write(image_names[i + img_id].strip('.npy') + ', ')
            for j in range(k):
                top15.write(image_names[neighbours[img_id, j + 1]].strip('.npy') + ', ')
                bottom15.write(image_names[neighbours[img_id, num_of_images-1-j]].strip('.npy') + ', ')
            top15.write('\n')
            bottom15.write('\n')
        print("Elapsed: ", time.time() - begin)
        print("ETA: ", ((time.time() - begin) * (num_of_images - i - n)) / (i + n), " seconds")
    top15.close()
    bottom15.close()
