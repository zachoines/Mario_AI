# Importing the libraries
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
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Other Classes
from Wrappers import preprocess

# TODO::Build the CNN

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        # first input to the network will have one channel of black and white images, with 32 feature detectors
        # With a feature detector that is 5 * 5 kernel
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        # Now we have 32 inputs to the next layer, which are the output feature detectors from the previous network
        # Then we utilize a new kernel size to narrow down features
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        # now we flaten the pixels
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
    # Get the number of neurons in the flattening layer
    def count_neurons(self, image_dim):
        # here we create a fake image in order find the number of neurons (Channels, hight, width)
        # randomly generate values representing an image
        x = torch.rand(1, *image_dim)
        # Apply convolutions to the input image
        # Apply max pooling to the resulting feature maps with kernel size of 3 and stride of 2
        # Then activate the neurons using ReLU function, reducing linearity in the pooled feature maps
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # put all the pixels into one large array of pixels, in what is called the flattening layer
        return x.data.view(1, -1).size(1)

    # Forward propagtion function for our NN
    def forward(self, x):

        # Apply convolutions to the input image
        # Apply max pooling to the resulting feature maps with kernel size of 3 and stride of 2
        # Then activate the neurons using ReLU function, reducing linearity in the pooled feature maps
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        # Flatten the third convolutional
        x = x.view(x.size(0), -1)

        # Fully connected hidden layer neural network. Utilizing a linear transmission of the data, then break up the linearity with ReLU
        x = F.relu(self.fc1(x))

        # Output layer with Q-Values
        x = self.fc2(x)
        return x


# TODO::Build the forward pass 

# TODO::Build he Agent

# TODO::Build simulation driver




# Main 
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
#env = preprocess.GrayScaleImage(env, width = 80, height = 80, grayscale = True)
# env = wrappers.Monitor(env, "Super_Mario_AI/videos", force = True)
#env.write_upon_reset = True

done = True 
for step in range(10):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close() 

