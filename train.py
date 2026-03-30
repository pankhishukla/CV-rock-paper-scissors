from model import CNN

import torch #It is a pytorch library
#Pytorch library is used for building and training neural networks
import torch.nn as nn #Importing the neural network modules such as Conv2d, Linear

import cv2 #OpenCV for web cam
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

#Calculting the loss function
loss_function = nn.CrossEntropyLoss() #Measures how wrong the model's predictions are for classfication

#Optimizer
optimizer = optim.Adam(
    model.parameters(), #Passing all the model parameters (weights) to the optimizer so that it can update them
    lr = 0.001 #Learning rate controls how big each update step is
)

#Training Loop
for epoch in range(30): #Looping over the dataset (1 loop = 1 epoch)
    correct = 0 #Counts corrects predictions
    total = 0 #total samples

    for images, labels in train_loader: #Get a batch of images and their correct labels
        outputs = model(images) #Forward pass -> making all the predictions
        
        _, predicted = torch.max(outputs, 1) #Getting the predicted class (index of the highest score)
        #Like outputs provide the list of all the raw scores and torch.max picks the highest value index

        total += labels.size(0) #Total samples
        correct += (predicted == labels).sum().item() #count matches

        loss = loss_function(
            outputs, #This is the predictions done by the model
            labels #This is the actual labels
        )

        optimizer.zero_grad() #Clearing the previous gradients this is important to avoid accumulations
        loss.backward() #Backpropagation -> compute gradients (how to improve weights)
        optimizer.step() #Updating weights using gradients

    print("Epoch: ", epoch, "Loss: ", loss.item())

torch.save(model.state_dict(), "model.pth")
    