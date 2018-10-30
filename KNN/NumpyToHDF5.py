import numpy as np
import h5py

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('features/numpy/') if isfile(join('features/numpy/', f))]

with h5py.File('feature_db.h5', 'w') as hf:
    for i in range(len(onlyfiles)):
        print (i)
        nm = onlyfiles[i]
        data_tw = np.load('features/numpy/' + nm)
        hf.create_dataset(nm.strip('.npy'), data=data_tw)
