from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import os
import time
import heapq
from scipy import spatial

featurePath = './small_features/'
neighbourPath = './small_neighbours/'

try:
	torch.cuda.set_device(0)
except:
	pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on " + str(device))

#fileNames = os.listdir('./features/numpy/')
fileNames = os.listdir(featurePath)
fileNames.sort()

def getDistance(x, y):
	return np.linalg.norm(x-y)
	# x = x.flatten()
	# y = y.flatten()
	# return 1 - spatial.distance.cosine(x, y)

### UNCOMMENT THIS LATER ###
# numOfImages = len(fileNames)

k = 10
numOfImages = 1000

print("No of images", numOfImages)

n = 100
begin = time.time()
for i in range(n):
	filename = fileNames[i]
	print(i, filename)
	features = np.load(featurePath + filename)

	maxHeap = []

	for j in range(numOfImages):
		if i == j:
			continue
		# print(j)
		features2 = np.load(featurePath + fileNames[j])
		d = getDistance(features, features2)
		if len(maxHeap) < k:
			heapq.heappush(maxHeap, (-d, j))
		else:
			top = heapq.heappop(maxHeap)
			if (d < -top[0]):
				heapq.heappush(maxHeap, (-d, j))
			else:
				heapq.heappush(maxHeap, top)

	# print(len(maxHeap))
	neighbours = []
	for j in range(k-1, -1, -1):
		top = heapq.heappop(maxHeap)
		neighbours.append(top[1])
		# print(top)
		# file.write(fileNames[maxHeap[j][1]])
	file = open(neighbourPath + filename.strip('.npy') + '.txt', 'w')
	neighbours = neighbours[::-1]
	for j in range(len(neighbours)):
		file.write(fileNames[neighbours[j]] + '\n')
	file.close()

	# print(i)
	print("ETA: ", ((time.time() - begin) * (n - i - 1)) / (i + 1), " seconds")

print("Extracted all neighbours successfully!")
