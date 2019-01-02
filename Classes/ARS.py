# Importing the libraries
import os
import numpy as np
import torch
from torch import random

# Importing the packages for OpenAI and Mario
from gym import wrappers
import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

# Other Classes
from Wrappers import preprocess

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
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 128, 128)), out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = number_actions)

        self.init_weights(self.convolution1)
        self.init_weights(self.convolution2)
        self.init_weights(self.convolution3)
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

        # put all the pixels into one large array of pixels, in what is called the flattening layer
        return x.data.view(1, -1).size(1)

# ARS V2 State normalization
class  Normalizer():
    def __init__(self, numPerceptInputs):
        # Input vector
        self.n = np.zeros(numPerceptInputs)
        self.mean = np.zeros(numPerceptInputs)
        self.mean_diff = np.zeros(numPerceptInputs)
        self.var = np.zeros(numPerceptInputs)

    
    def observe(self, x):
        # Online calculation of the mean
        lastMean = self.mean.copy()
        
        self.n += 1.

        # Online calculation of the mean
        lastMean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        # Online calculation of the variance
        self.mean_diff += (x - lastMean) * (x - self.mean)
        # variance
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    # Observe a state and normalize by a mean and standard deviation computed online
    def normalize(self, inputs):
        observedMean = self.mean
        # Standard deviation
        observedSTD = np.sqrt(self.var)
        observedMean.clip(min = 1e-2)
        return (inputs - observedMean) / observedSTD



class Policy():
    # TODO:: Init our weights for all our networks
    def __init__(self, inputSize, outputSize):
        # Matrix of weights for our perceptron
        self.theta = np.zeros((outputSize, inputSize))

    # TODO:: V2 ARS weight perturbations. (states are normalized before passed in)
    def evaluate(self, input, delta=None, direction=None):

    # TODO:: Sample perturbations based on a gaussian distribution
    def samplePerturbations(self):
        # For all the convolutional and linear networks, 

    # TODO:: Implementation of the method of finite differences
    def update():

# Class for the Hyperparameters of AI
class HyperParam():
    def __init__(self, numSteps = 1000, episodeLength = 1000, learningRate = 0.02, numDirections = 16, numBestDirections = 16, noise = 0.03, seed = 1, environmentName = ''):
        self.numSteps = numSteps
        self.episodeLength = episodeLength
        self.learningRate = learningRate
        self.numDirections = numDirections
        self.numBestDirections = numBestDirections
        # Make sure the number of best directions is less then max directions
        assert self.numBestDirections <= self.numBestDirections
        self.noise = noise
        self.seed = seed
        self.environmentName = environmentName


# Explore the environment in one purterbation direction. Return the accumulated reward.
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    numPlays = 0.
    sumRewards = 0
    while not done and numPlays < hp.episodeLength:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        # We perform action in the environment
        state, reward, done, _ = env.step(action)
        # take care of outliers. forces all the rewards between 1 and -1. Removes bias.
        reward = max(min(reward, 1), -1)
        sumRewards += reward
        numPlays += 1
    return sumRewards


##################
###### Main ######
##################

### Here is a list of the available environments ###

# SuperMarioBros-v0, SuperMarioBros-v1, SuperMarioBros-v2, 
# SuperMarioBros-v3, SuperMarioBrosNoFrameskip-v0, SuperMarioBrosNoFrameskip-v1, 
# SuperMarioBrosNoFrameskip-v2, SuperMarioBrosNoFrameskip-v3, SuperMarioBros2-v0, 
# SuperMarioBros2-v1, SuperMarioBros2NoFrameskip-v0, SuperMarioBros2NoFrameskip-v1


# Get and build our test environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env = preprocess.GrayScaleImage(env, height = 128, width = 128, grayscale = True)
env = wrappers.Monitor(env, "./Super_Mario_AI/videos", force = True, write_upon_reset=True)

# Initialize our policy as a perceptron or matrix of weights
numberInputs = env.observation_space.shape
numberOutputs = env.action_space.n
policy = Policy(numberInputs, numberOutputs)

# Initialize our normalizer
normalizer = Normalizer(numberInputs)

# Training
train(env, policy, normalizer, hp)