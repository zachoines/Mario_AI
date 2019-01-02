# Experience Replay

# Importing the libraries
import numpy as np
import random
from collections import namedtuple, deque

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

# Making the AI progress on several (n_step) steps

class NStepProgress:
    # Constructor
    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
    
    # An iterable for running through the simulation
    def __iter__(self):
        state = self.env.reset()
        # This data structure provides fast access and building of our iterable ( O(1) )
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(np.array([state]))[0][0]
            # Feed the state into the network, return the suggested action
            if random.random() < .05:
                action = random.randint(0, self.env.action_space.n - 1)

            # Apply the action in the environment
            next_state, r, is_done, _ = self.env.step(action)
            # self.env.render()
            reward += r
            # Build our N-Step history
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            # Shrink history if too long
            while len(history) > self.n_step + 1:
                history.popleft()
            # Return the Build history
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                history.clear()
    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps

# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_steps, capacity = 10000):
        self.capacity = capacity
        self.n_steps = n_steps # Iterable of the NStepProgressClass
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size): # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs+1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            # Use NN to generate a "n" size tuple of ('Step', ['state', 'action', 'reward', 'done']) entries 
            # that has n_steps from the environment      
            entry = next(self.n_steps_iter) # 11 consecutive steps
            # Now take this n_step history and build a history buffer that has a certain number od n_step samples
            self.buffer.append(entry) 
            samples -= 1
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
