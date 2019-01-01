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
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

# Other Classes
from Wrappers import preprocess
from Classes.experience_replay import NStepProgress, ReplayMemory

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()

        # first input to the network will have one channel of black and white images, with 32 feature detectors
        # With a feature detector that is 5 * 5 kernel
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5)
        # Now we have 32 inputs to the next layer, which are the output feature detectors from the previous network
        # Then we utilize a new kernel size to narrow down features
        self.convolution2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        # now we flaten the pixels
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 128, 128)), out_features = 96)
        self.fc2 = nn.Linear(in_features = 96, out_features = number_actions)

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

    # Forward propagtion function for our NN
    def forward(self, state):
        # Apply convolutions to the input image
        # Apply max pooling to the resulting feature maps with kernel size of 3 and stride of 2
        # Then activate the neurons using ReLU function, reducing linearity in the pooled feature maps

        x = F.max_pool2d(F.relu(self.convolution1(state)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution2(x)), 3, 2)
        x = F.max_pool2d(F.relu(self.convolution3(x)), 3, 2)

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


# A sliding average
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        # Make sure we are passing a list of rewards
        if isinstance(rewards, list):
            # Cat the two lists together
            self.list_of_rewards += rewards
        else:
            # append to the list if a single element
            self.list_of_rewards.append(rewards)
        # make sure there is no more than n elements 
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    # return the average
    def average(self):
        return np.mean(self.list_of_rewards)


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
number_actions = env.action_space.n

# Construct/Load a model
cnn = CNN(number_actions)
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
loss = nn.MSELoss()
last_epoch = 1

# If there is a previous save 
if os.path.exists("./Super_Mario_AI/Model/model.pth"):  
    checkpoint = torch.load("./Super_Mario_AI/Model/model.pth")
    cnn.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

cnn.eval()

# Build the body
softmax_body = SoftmaxBody(T = .80) # Temperature value dictates exploration
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = NStepProgress(env = env, ai = ai, n_step = 10)
memory = ReplayMemory(n_steps = n_steps, capacity = 100000)
    
# Implementing Eligibility Trace or N_Step Sarsa Learning
def eligibility_trace(batch):
    # Decay factor
    gamma = 0.99
    inputs = [] 
    targets = []
    for series in batch:
        # Create a tensor with the first transition in the series., along with the last transition in the series
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))

        # for entry in series:
        #     input.append(entry)
        output = cnn(input)
        # Return zero if we are at the terminal state
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()

        # Create an iterator to iterate in reverse through our n_step history, collecting rewards
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        # Get the state of te first step in our n_step transition
        state = series[0].state
        # The target associated with the input state of first transition. 
        # Contains the Q-Values predicted from the NN for the first transition.
        target = output[0].data
        # Assign to the action played the accumulated rewards for the n_step transition
        target[series[0].action] = cumul_reward
        # Build an array of the beginning states of the transitions. 
        # The network has been trained for 10 steps, so inputting the first is sufficient 
        inputs.append(state)
        # Build an array of the Q-Values predicted for each of the 10 step transitions.
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

ma = MA(100)


# Training the AI
start = last_epoch
for epoch in range(start, 1000):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, Q_Values = eligibility_trace(batch)
        inputs, Q_Values = Variable(inputs), Variable(Q_Values)

        # Input all our beginning transition states, getting a series of predicted Q_Values
        predicted_Qs = cnn(inputs)

        # Update our gradients
        loss_error = loss(predicted_Qs, Q_Values)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    
    # Save the Model 
    torch.save({
        'epoch': epoch,
        'model_state_dict': cnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, "./Super_Mario_AI/Model/model.pth")


# Closing the Mario environment
env.close() 


