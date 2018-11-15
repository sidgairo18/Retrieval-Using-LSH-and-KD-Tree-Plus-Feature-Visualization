import numpy as np
import os
from sklearn.decomposition import IncrementalPCA
from os import listdir
from os.path import isfile, join

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
