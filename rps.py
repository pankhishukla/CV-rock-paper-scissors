import torch #It is a pytorch library
#Pytorch library is used for building and training neural networks

import torch.nn as nn #Importing the neural network models
import torch.optim as optim #These are used for importing optimization models
from torchvision import datasets, transforms #this contains the datasets and image transformations
from torch.utils.data import DataLoader #Dataloader helps load data in batches during training

#Image Processing
transform = transforms.Compose([ #This is combining multiple image transformations into a single pipeline
    transforms.Resize((64, 64)), #Resizing every image to 64x64 pixels so the CNN receives same dimensions 
    transforms.toTensor() #Converts the image to the pytorch tensor and scale the pixel values from 0-255 and 0-1
])

#Loading the dataset from the folder
