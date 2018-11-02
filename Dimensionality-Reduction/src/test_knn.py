import cv2
import pdb
import numpy as np
from extract_conv4_3 import get_style_features
import random

def my_func():
    x = get_style_features()

    #ids = range(5000)
    #random.shuffle(ids)

    #names_id = {}
    #f = open('image_names_list_vgg_conv.txt')
    #for i, line in enumerate(f):
        #line = line.strip()
        #names_id[i] = line

    #x2 = []
    #for i in ids:
        #x2.append(x[i,:])
    #x2 = np.asarray(x2)

    #return x, x2, names_id

def compute_distances(X, Y):                                            
    print ("Computing Distances")
    num_test = X.shape[0]                                               
    num_train = Y.shape[0]                                              
    dists = np.zeros((num_test, num_train))                             
    dists = np.sqrt((X**2).sum(axis=1, keepdims=True) + (Y**2).sum(axis=1) - 2 * X.dot(Y.T))
    return dists

def get_knn(dists, k=1):        
    fact = -1
    ## fact = 1 for top K
    ## fact = -1 for bottom K
                                                                        
    knns = []                                                           
                                                                        
    for i in range(dists.shape[0]):                                     
        l = list(np.argsort(fact*dists[i,:]))[:k]                            
        knns.append(l)
                                                                        
    knns = np.asarray(knns)                                             
    return knns
