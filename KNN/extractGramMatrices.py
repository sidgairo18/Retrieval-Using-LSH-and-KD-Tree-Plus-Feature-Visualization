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


try:
	torch.cuda.set_device(0)
except:
	pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on " + str(device))


# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def getGramMatrix(self):
    	return self.target

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style losses :
style_layers_default = ['conv_9']

def getGramMatrices(img, cnn=cnn, normalization_mean=cnn_normalization_mean,
				    normalization_std=cnn_normalization_std,
				    style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    gram_matrices = None
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(img).detach()
            style_loss = StyleLoss(target_feature)
            return style_loss.getGramMatrix()

    return gram_matrices

image_names = os.listdir('/scratch/bam_subset/')
image_names.sort()

### UNCOMMENT THIS LATER ### 
# numOfImages = len(image_names)
numOfImages = 10000

print("No of images", numOfImages)

begin = time.time()
for i in range(0, numOfImages):
    if i % 10 == 0:
	    print(i)
    image = image_names[i]
    # print(image)
    img = image_loader('/scratch/bam_subset/' + image)
    a, b, c, d = img.size()
    if (b > 3):
        img = img[:, :3, :, :]
    else:
        img = img[:, :1, :, :]

    G = getGramMatrices(img)


	# torch.save(G, './features/numpy/' + image.strip('.jpg') + '.pt')

	# Convert to numpy array
    G = G.cpu().data.numpy()
    np.save('./features/numpy/' + image.strip('.jpg').strip('.png'), G)

    if i % 10 == 0:
        print("ETA: ", ((time.time() - begin) * (numOfImages - i - 1)) / (i + 1), " seconds")

print("Extracted all features successfully!")
