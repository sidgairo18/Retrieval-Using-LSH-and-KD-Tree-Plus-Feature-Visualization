import h5py
import numpy as np
import sh
import os
from sklearn.decomposition import IncrementalPCA

def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def finale():
    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
        key= lambda x: -x[1])[:10]: print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('features/numpy/') if isfile(join('features/numpy/', f))]

total_bs = 100
bs = 100

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

finale()

#for i in range(0, n//chunk_size):
temp = np.zeros((bs, 512*512), dtype=float)
for i in range(total_bs):
    print(i,'b')
    for j in range(bs):
        nm = onlyfiles[i*bs+j]
        data_tw = np.load('features/numpy/' + nm).flatten()
        temp[j,:] = data_tw
        del data_tw
    ipca.partial_fit(temp)
    finale()
del temp

temp = np.zeros((512*512), dtype=float)
for i in range(len(onlyfiles)):
    print(i,'c')
    nm = onlyfiles[i]
    temp[:] = np.load('features/numpy/'+ nm).flatten()
#    print(temp.shape)
    np.save('small_features/' + onlyfiles[i], ipca.transform(temp.reshape(1,-1)))
#    del temp
del temp
