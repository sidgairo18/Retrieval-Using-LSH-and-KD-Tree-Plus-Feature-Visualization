#################Getting Features and Then PCA Functions################

1. get_features('<dataset_directory>', lambda1, lambda2, lambda3)
    * Features => F1, F2, F3 (Different Gram Matrix Features from conv2_1, conv3_1,  conv4_1)
    * Normalization Step 1
    * Run PCA on F1, F2, F3
    * Normanlization Step 2
    * Here lamda1, lambda2 and lambda3 are parameters each between 0 and 1 to include the weight/contribution for each distance D1, D2, D3

2. get_PCA('<features_matrix>' or '<dataset_directory>', '<destination_directory>', k)
    * k - Number of Dimensions to reduce to.
    * (Currently assuming the IPCA training has to be done. Could add one more parameter for location of IPCA object if it has already been trained.)

3. get_nearest_neighbours(''<reduced_features_directory>', '<destination_directory>', k)
    * k - get number of nearest neighbours

Also add here the dependencies required.

#################Getting Features and Then PCA Functions################


#################Feature Visualization##################################

1. get_visualization(X, labels, I, num):
    * X - Feature vector matrix (Each row is a data point)
    * labels - is a dictionary to get labels for each data point.
    * I - Image vector matrix (Each row is an image vector for that data point)
    * num - Number of PCA components used to reduce data before applying tSNE.
#################Feature Visualization##################################
