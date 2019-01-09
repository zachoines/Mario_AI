# Importing the libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import random

# Importing the packages for OpenAI and Mario
from gym import wrappers
import gym

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()

        # first input to the network will have one channel of black and white images, with 32 feature detectors
        # With a feature detector that is 5 * 5 kernel
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3)
        # Now we have 32 inputs to the next layer, which are the output feature detectors from the previous network
        # Then we utilize a new kernel size to narrow down features
        self.convolution2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.convolution4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        # now we flaten the pixels
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 64, 64)), out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = number_actions)

        self.init_weights(self.convolution1)
        self.init_weights(self.convolution2)
        self.init_weights(self.convolution3)
        self.init_weights(self.convolution4)
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)
    
        

    def init_weights(self, m):
        #torch.nn.init.normal_()
        m.bias.data.fill_(0.01)
        torch.nn.init.kaiming_uniform_(m.weight)
        
    # Get the number of neurons in the flattening layer
    def count_neurons(self, image_dim):
        # here we create a fake image in order find the number of neurons (Channels, hight, width)
        # randomly generate values representing an image
        x = torch.rand(1, *image_dim)
        
        # Apply convolutions to the input image
        # Apply max pooling to the resulting feature maps with kernel size of 3 and stride of 2
        # Then activate the neurons using ReLU function, reducing linearity in the pooled feature maps
        x = F.max_pool2d(F.relu(self.convolution1(x)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution2(x)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution3(x)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution4(x)), 3, 2)

        # put all the pixels into one large array of pixels, in what is called the flattening layer
        return x.data.view(1, -1).size(1)

    # Forward propagtion function for our NN
    def forward(self, state):
        # Apply convolutions to the input image
        # Apply max pooling to the resulting feature maps with kernel size of 3 and stride of 2
        # Then activate the neurons using ReLU function, reducing linearity in the pooled feature maps

        x = F.max_pool2d(F.relu(self.convolution1(state)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution2(x)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution3(x)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution4(x)), 3, 2)

        # Flatten the third convolutional
        x = x.view(x.size(0), -1)

        # Fully connected hidden layer neural network. Utilizing a linear transmission of the data, then break up the linearity with ReLU
        x = F.relu(self.fc1(x))


        # Output layer with Q-Values
        x = self.fc2(x)
        return x


# Making the body

class SoftmaxBody(nn.Module):
    # T for temperature, scales the probabilities. Function establishes and calculates the probabilities of the output layer
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
    
    # Forward the ourput signal of the NN to the body of the simulation
    def forward(self, outputs): 
        # Grab our softmax probability distribution, generated from the output layers Q-Value's. One per action. 
        # Temp allows for a more random selection of actions, instead of just choosing the highest Q-Value action
        probs = F.softmax(outputs * self.T, dim=None)   
        # Select from our probability distribution an action to play using a Multinomial Distribution
        actions = probs.multinomial(1)
        return actions

# Making the AI
class AI:
    # Brain is he neural network and the body are the actions generated with softmax
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    # Inputs are the images from the simulation
    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        # Feed state to brain, uses the forward function from the CNN class
        output = self.brain(input)
        # Get our determined action, uses the forward function from the SoftmaxBody class
        actions = self.body(output)
        # convert from torch tensor to numpy array
        return actions.data.numpy()