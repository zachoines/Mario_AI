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
from Classes.CNN import CNN, SoftmaxBody, AI
from Wrappers import preprocess

# Get and build our test environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env = preprocess.GrayScaleImage(env, height = 128, width = 128, grayscale = True)
env = wrappers.Monitor(env, "./Super_Mario_AI/videos", force = True, write_upon_reset=True)
number_actions = env.action_space.n

# Construct/Load a model
cnn = CNN(number_actions)

# If there is a previous save 
if os.path.exists("./Super_Mario_AI/Model/model.pth"):  
    checkpoint = torch.load("./Super_Mario_AI/Model/model.pth")
    cnn.load_state_dict(checkpoint['model_state_dict'])

cnn.eval()

# SuperMarioBros-v0, SuperMarioBros-v1, SuperMarioBros-v2, 
# SuperMarioBros-v3, SuperMarioBrosNoFrameskip-v0, SuperMarioBrosNoFrameskip-v1, 
# SuperMarioBrosNoFrameskip-v2, SuperMarioBrosNoFrameskip-v3, SuperMarioBros2-v0, 
# SuperMarioBros2-v1, SuperMarioBros2NoFrameskip-v0, SuperMarioBros2NoFrameskip-v1

##################
###### Main ######
##################
softmax_body = SoftmaxBody(T = .80) # Temperature value dictates exploration
ai = AI(brain = cnn, body = softmax_body)

state = env.reset()
reward = 0.0
is_done = False
while not is_done:
    
    # Feed the state into the network, return the suggested action
    action = ai(np.array([state]))[0][0]
    # Apply the action in the environment
    next_state, r, is_done, _ = env.step(action)
    env.render()
    state = next_state
    r = .01 * r
    reward += r

print("Reward: %s", str(reward))