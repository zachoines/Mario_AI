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
from Classes.CNN import CNN, SoftmaxBody, AI

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
optimizer = optim.Adam(cnn.parameters(), lr = 0.005)
loss = nn.MSELoss()
last_epoch = 1

# Build the body
softmax_body = SoftmaxBody(T = .7) # Temperature value dictates exploration
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = NStepProgress(env = env, ai = ai, n_step = 5)
memory = ReplayMemory(n_steps = n_steps, capacity = 10000)

# If there is a previous save 
if os.path.exists("./Super_Mario_AI/Model/model.pth"):  
    checkpoint = torch.load("./Super_Mario_AI/Model/model.pth")
    cnn.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    memory.buffer = checkpoint['memory']

cnn.eval()

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
for epoch in range(start, 10000):
    memory.run_steps(100)
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
        'loss': loss,
        'memory': memory.buffer
        }, "./Super_Mario_AI/Model/model.pth")


# Closing the Mario environment
env.close() 


