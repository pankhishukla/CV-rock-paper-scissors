import torch #It is a pytorch library
#Pytorch library is used for building and training neural networks
import torch.nn as nn #Importing the neural network modules such as Conv2d, Linear

import torch.nn as nn #Importing the neural network models
import torch.optim as optim #These are used for importing optimization models
from torchvision import datasets, transforms #this contains the datasets and image transformations
from torch.utils.data import DataLoader #Dataloader helps load data in batches during training

#Image Processing
transform = transforms.Compose([ #This is combining multiple image transformations into a single pipeline
    transforms.Resize((64, 64)), #Resizing every image to 64x64 pixels so the CNN receives same dimensions 
    transforms.ToTensor() #Converts the image to the pytorch tensor and scale the pixel values from 0-255 and 0-1
])

#Loading the dataset from the folder
dataset = datasets.ImageFolder( #Image folder reads the images and assigns labels automatically
    "dataset", #path to the dataset directory
    transform = transform #Applying the transformations defined above to every image
)

#Creating the data loader
train_loader = DataLoader( #DataLoader prepares the dataset for training
    dataset, #Dataset that we just loaded
    batch_size = 32, #Number of images processed at once for training
    shuffle = True #Shuffle the images at each epoch so that the model doesn't learn order patterns
)

#Checking the dataset information
print("Total images: ",len(dataset)) 
print("Classes: ", dataset.classes) #This prints the class labels detected from the folder name (rock, paper, scissors)

#Inspecting 1 batch
for images, labels in train_loader: #Looping through the data loader to retrieve a batch of images and labels
    print("Batch Image Shape: ", images.shape) #Printing the tensor shape of the image batch (batch_Size, channels, height, width)
    print("Batch Labels: ", labels) #printing the numeric labels corresponding to rock/paper/scissors
    break #Stopping after the first batch because we jsut want to inspect the data

#Defining the CNN
class CNN(nn.Module): #Defining a class for our CNN model that inherits from nn.Module

    def __init__(self): #Constructer function where we define all functions
        super().__init__() #Intitializing the parent class so pytorch can track layers and parameters

        #Convulational layer 1
        self.conv1 = nn.Conv2d(
            in_channels = 3, #Input has 3 channels(RGB image)
            out_channels = 16, #We will create 16 feature maps(filters)
            kernal_size = 3 #Each filter is 3x3 in size
        )

        #Convulational layer 2
        self.conv2 = nn.Conv2d(
            in_channels = 16, #Takes the output from the previous layer (16 feature maps)
            out_channels = 32,
            kernal_size = 3
        )

        #Pooling layer
        self.pool = nn.MaxPool2d(
            kernal_size = 2, #Taking a 2x2 region
            stride = 2 #Jumping 2 steps each time for sliding
        )

        #Fully COnnected Layer
        self.fc1 = nn.Linear(
            32 * 14 * 14, #Input size after conv + pooling
            128 #Number of neurons in the hidden layer
        )

        #Output Layer
        self.fc2 = nn.Linear(
            128, #Input from the previous layers
            3 #Output neurons (rock, paper, scissors)
        )

        #Activation function
        self.relu = nn.ReLU() #ReLU introduces non-linearity so model can learn complex patterns

        def forward(self, x): #Defines how the data flows through the network






