import numpy as np
import os
from sklearn.decomposition import IncrementalPCA

from os import listdir
from os.path import isfile, join
plc = '/scratch/mohsin/final_features/'
onlyfiles = [f for f in listdir(plc) if isfile(join(plc, f))]
onlyfiles = onlyfiles[:10000]
print(len(onlyfiles))

total_bs = 10
bs = 1000

#
#with h5py.File('feature_db.h5', 'w') as hf:
#    for i in range(total_bs):
#        nd = np.zeros((100,512*512),dtype=float) 
#        print(i, 'a')
#        for j in range(bs):
#            nm = onlyfiles[i*bs+j]
#            data_tw = np.load('features/numpy/' + nm).flatten()
#            nd[j,:] = data_tw
#        hf.create_dataset(str(i), data=nd)

# IPCA 
ipca = IncrementalPCA(n_components=bs, batch_size=bs)

#for i in range(0, n//chunk_size):
temp = np.zeros((bs, 512*512), dtype=float)
for i in range(total_bs):
    print(i,'b')
    for j in range(bs):
        nm = onlyfiles[i*bs+j]
        temp[j,:] = np.load(plc + nm).flatten()
    ipca.partial_fit(temp)
del temp

temp = np.zeros((512*512), dtype=float)
for i in range(8913, len(onlyfiles)):
    print(i,'c')
    nm = onlyfiles[i]
    temp[:] = np.load(plc+ nm).flatten()
#    print(temp.shape)
    np.save('/scratch/mohsin/final_features_pca/' + onlyfiles[i], ipca.transform(temp.reshape(1,-1)))
#    del temp
del temp
