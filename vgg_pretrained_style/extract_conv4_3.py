import tensorflow as tf
#from scipy import misc
from os import listdir
from os.path import isfile, join
import argparse
import utils
import numpy as np
import pickle
import time
import os

import pdb

def get_style_features():
    model_path = './Data/vgg16.tfmodel'
    split = 'train'
    data_dir = './Data'
    batch_size = 1
    
    vgg_file = open(model_path)
    vgg16raw = vgg_file.read()
    vgg_file.close()
    
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(vgg16raw)
    
    print ("VGG done successfully")

    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })

    graph = tf.get_default_graph()

    for opn in graph.get_operations():
        print ("Name", opn.name, opn.values())


    #image_id_list = [img_id for img_id in image_ids]
    image_names1 = os.listdir('./Images/')
    image_names1.sort()
    print "No of images", len(image_names1)

    image_names = []
    no_of_images = len(image_names1)
    for i in range(no_of_images):
        im = image_names1[i]
        image_names.append(im)


    print ("Images extracted", no_of_images)

    image_id_list = []

    for i in range(len(image_names)):
        image_id_list.append(i)

    print ("Total Images", len(image_id_list))
    
    
    sess = tf.Session()
    #fc7 = np.ndarray( (len(image_id_list), 4096 ) )
    conv4_3 = np.ndarray( (len(image_id_list),  512*512) )
    idx = 0

    from_start = time.clock()

    image_name_list = []

    while idx < len(image_id_list):
        start = time.clock()
        image_batch = np.ndarray( (batch_size, 224, 224, 3 ) )

        count = 0
        for i in range(0, batch_size):
            if idx >= len(image_id_list):
                    break
            image_name_list.append(image_names[idx])
            #image_file = join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, image_id_list[idx]) )
            image_file = "Images/"+image_names[idx]
            print "Image name", image_file
            image_batch[i,:,:,:] = utils.load_image_array(image_file)
            idx += 1
            count += 1
        
        
        feed_dict  = { images : image_batch[0:count,:,:,:] }
        #fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
        conv4_3_tensor = graph.get_tensor_by_name("import/conv4_3/Relu:0")
        #fc7_batch = sess.run(fc7_tensor, feed_dict = feed_dict)
        conv4_3_batch = sess.run(conv4_3_tensor, feed_dict = feed_dict)
        conv4_3_batch = conv4_3_batch.reshape((1,28*28,512))
        conv4_3_batch = np.matmul(conv4_3_batch[0,:,:].T, conv4_3_batch[0,:,:])
        temp = np.ndarray((1,512*512))
        temp[0,:] = conv4_3_batch.reshape(512*512)
        conv4_3_batch = temp
        #fc7[(idx - count):idx, :] = fc7_batch[0:count,:]
        conv4_3[(idx - count):idx, :] = conv4_3_batch[0:count,:]

        end = time.clock()
        print "Time for batch 1 photos", end - start
       # print "Hours For Whole Dataset" , (len(image_id_list) * 1.0)*(end - start)/60.0/60.0/10.0
        print "Time Elapsed:", (from_start)/60, "Minutes"

        print "Images Processed", idx

    #np.savetxt('/scratch/sid_imp/conv4_3_features_vgg16.txt', conv4_3)

    f = open('image_names_list_vgg_conv.txt', 'w')
    for name in image_name_list:
        f.write(name+'\n')
    f.close()

    return conv4_3


if __name__ == '__main__':
    get_style_features()
