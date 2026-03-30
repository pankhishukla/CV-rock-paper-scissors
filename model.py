#Defining the CNN
class CNN(nn.Module): #Defining a class for our CNN model that inherits from nn.Module
    #This class only defines what the model is

    def __init__(self): #Constructer function where we define all functions 
        #Defining all the layers
        super().__init__() #Intitializing the parent class so pytorch can track layers and parameters

        #Convulational layer 1
        self.conv1 = nn.Conv2d(
            in_channels = 3, #Input has 3 channels(RGB image)
            out_channels = 16, #We will create 16 feature maps(filters)
            kernel_size = 3 #Each filter is 3x3 in size
        )

        #Convulational layer 2
        self.conv2 = nn.Conv2d(
            in_channels = 16, #Takes the output from the previous layer (16 feature maps)
            out_channels = 32,
            kernel_size = 3
        )

        #Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size = 2, #Taking a 2x2 region
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
        #x is the input image batch
        
        #First convolution -> activation(reLU) -> pooling
        x = self.pool(self.relu(self.conv1(x))) 
        #self.conv1(x) -> This applies filters to detect the basic patterns (edges, corners, textures)
        #Output is (batch, num_filters, new_height, new_width)
        #seld.relu(self.conv1(x)) -> Now the activation function removes the negative values and introduces non-linearity(allows complex pattern learning)
        #self.pool(self.relu(self.conv1(x))) -> Now we are just downscaling (downsampling) This reduces the computation

        x = self.pool(self.relu(self.conv2(x)))
        #Now the model is learning the complex shapes, combinations of all the earlier features
        #Layer 1 - edges
        #Layer 2 - shapes(fingers)

        x = x.view(x.size(0), -1) #x.size(0) -> This keeps the batch size same and flattens everything else
        #So if we got the output as (32, 32, 14, 14)-> (batch, num_filters, new_height, new_width)
        #With this step we will get (32, 32*14*14) -> (32,6272)
        #We flatten because the fully connected layers want 1D vectors and not 2D vectors

        x = self.relu(self.fc1(x))
        #This is now mixing all the extracted features. 
        #Eg. if these edges + this shape = then it is rock
        #Relu just removes the negative values and provided non-linearity(allows learning complex patterns)

        x = self.fc2(x) #Output shape will be (batch_size, 3). 3 is rock, paper, size
        return x
        
#Initializing the model
model = CNN() #Creating a instance of the CNN model so that we can train it

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