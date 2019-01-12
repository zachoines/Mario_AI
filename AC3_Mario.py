# import Tensor flow and other scientific packages
import time, random, threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
import tensorflow.keras.layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

# Importing the packages for OpenAI and MARIO
import gym
from gym import wrappers
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Import other classes
from Wrappers import preprocess

#-- constants
# TODO::Place these globals into their own hyperparameter class
 
ENV = 'SuperMarioBros-v0'

RUN_TIME = 1800
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 10
GAMMA_N = GAMMA ** N_STEP_RETURN

MIN_BATCH = 128
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

FRAME_NUM = 100
HIGHT = 64
WIDTH = 64
CHANNELS = 1

STEP_SIZE = 10
TIME_STEPS = FRAME_NUM
TEMPERATURE = .7


#---------
class MarioBrain:

	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
		self.session = tf.Session()
		K.set_session(self.session)
		K.manual_variable_initialization(True)

		self.model = self._build_model()
		self.graph = self._build_graph(self.model)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

		self.default_graph.finalize()	# avoid modifications

	def _build_model(self):

		# Define Convolution 1
		self.conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = [5, 5], padding = "valid", activation = "relu", name = "conv1")
		self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), name = "maxPool1")
		
		# define Convolution 2
		self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3, 3], padding = "valid", activation = "relu", name = "conv2")
		self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name = "maxPool2")
		
		# define Convolution 3
		self.conv3 = tf.keras.layers.Conv2D(filters = 96, kernel_size = [2, 2], padding = "valid", activation = "relu", name = "conv3")
		self.maxPool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name = "maxPool3")
		
		# Flatten the output for LSTM Hidden Layer
		# TODO:: Debate the usefullness of a fully connected hidden layer
		self.flattened = tf.keras.layers.Flatten(name = "flattening_layer")
		self.hiddenLayer = tf.keras.layers.Dense(512, activation = "relu", name = "hidden_layer")

		# define LSTM layer
		self.lstmLayer = tf.keras.layers.LSTM(128, name = "LSTM_layer")

		# Output Layer consisting of an Actor and a Critic
		self._value = tf.keras.layers.Dense(1, activation = "relu", name = "value_layer")
		self._policy = tf.keras.layers.Dense(NUM_ACTIONS, activation = "softmax", name = "policy_layer")

		# Define the shape of out input layer
		input_def = tf.keras.layers.Input(shape = (HIGHT, WIDTH, CHANNELS), name = "input_layer")

		# Define the forward pass for the convolutional and hidden layers
		conv1_out = self.conv1(input_def)
		maxPool1_out = self.maxPool1(conv1_out)
		conv2_out = self.conv2(maxPool1_out)
		maxPool2_out = self.maxPool2(conv2_out)
		conv3_out = self.conv3(maxPool2_out)
		maxPool3_out = self.maxPool3(conv3_out)
		flattened_out = self.flattened(maxPool3_out)
		hidden_out = self.hiddenLayer(flattened_out)

		def reshape_layer(x):
			reshape = tf.expand_dims(x, 0, 'reshape_layer')
			return reshape

		reshaped_hidden_out = tf.keras.layers.Lambda(reshape_layer)(hidden_out)
		
		# Now enter this network into a LSTM NETWORK 
		lstm_output = self.lstmLayer(reshaped_hidden_out)

		# Actor and the Critic outputs
		out_value = self._value(lstm_output)
		out_actions = self._policy(lstm_output)

		# Final model
		model = tf.keras.Model(inputs = [input_def], outputs = [out_actions, out_value])

		model._make_predict_function()	# have to initialize before threading

		return model

	def _build_graph(self, model):

		s_t = tf.placeholder(tf.float32, shape = (None, HIGHT, WIDTH, CHANNELS))
		a_t = tf.placeholder(tf.float32, shape = (None, NUM_ACTIONS))
		r_t = tf.placeholder(tf.float32, shape = (None, 1)) # not immediate, but discounted n step reward
		
		p, v = model(s_t)

		log_prob = tf.log( tf.reduce_sum(p * a_t, axis = 1, keep_dims=True) + 1e-10)
		advantage = r_t - v

		loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
		loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
		entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
		# .RMSPropOptimizer(LEARNING_RATE, decay=.99)
		optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
		minimize = optimizer.minimize(loss_total)

		return s_t, a_t, r_t, minimize

	def optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return

		with self.lock_queue:
			if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
				return 									# we can't yield inside lock

			s, a, r, s_, s_mask = self.train_queue
			self.train_queue = [ [], [], [], [], [] ]

		s = (np.array(s))
		a = np.vstack(a)
		r = np.vstack(r)
		s_ = (np.array(s_))

		s_mask = np.vstack(s_mask)

		if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
		v = None
		if s_.shape == (HIGHT, WIDTH, CHANNELS):
			s_ = np.expand_dims(s_, axis=0)
			v = self.predict_v(s_)
		else:
			v = self.predict_v(s_)

		if s.shape == (HIGHT, WIDTH, CHANNELS):
			s = np.expand_dims(s, axis=0)
	
			
		r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
		
		s_t, a_t, r_t, minimize = self.graph
		self.session.run(minimize, feed_dict = {s_t: s, a_t: a, r_t: r})

	def train_push(self, s, a, r, s_):
		with self.lock_queue:
			self.train_queue[0].append(s)
			self.train_queue[1].append(a)
			self.train_queue[2].append(r)

			if s_ is None:
				self.train_queue[3].append(NONE_STATE)
				self.train_queue[4].append(0.)
			else:	
				self.train_queue[3].append(s_)
				self.train_queue[4].append(1.)

	def predict(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return  p, v

	def predict_p(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
		return p

	def predict_v(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return v

#---------
frames = 0
class Agent:
	def __init__(self, temperature):
		self.memory = []	# used for n_step return
		self.R = 0.
		self.temperature = temperature

	def sample(self, softmax):
		EPSILON = 10e-16 # to avoid taking the log of zero
		
		(np.array(softmax) + EPSILON).astype('float64')
		preds = np.log(softmax) / self.temperature
		
		exp_preds = np.exp(preds)
		
		preds = exp_preds / np.sum(exp_preds)
		
		probas = np.random.multinomial(1, preds, 1)
		return probas[0]


	# Select action from a softmax probability vector at different temperatures
	def act(self, s):	
		p = MarioBrain.predict_p(s)[0]
		a = self.sample(p)
		return a

	
	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
		a_cats[a] = 1 


		# Dont add states that are in invalid formate
		self.memory.append( (s, a_cats, r, s_) )

		self.R = ( self.R + r * GAMMA_N ) / GAMMA

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				MarioBrain.train_push(s, a, r, s_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)		

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			MarioBrain.train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)	
	
	# possible edge case - if an episode ends in <N steps, the computation is incorrect
		
#---------
class Environment(threading.Thread):
	stop_signal = False

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		threading.Thread.__init__(self)
		self.render = render

		# Make the super mario gym environment and apply wrappers
		self.env = gym.make(ENV)
		self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
		self.env = preprocess.GrayScaleImage(self.env, height = HIGHT, width = WIDTH, grayscale = True)
		# self.env = wrappers.Monitor(self.env, "./Super_Mario_AI/videos", force = True, write_upon_reset=True)
		self.agent = Agent(TEMPERATURE)

	def runEpisode(self):
		s = self.env.reset()
		R = 0
		while True:         
			time.sleep(THREAD_DELAY) # yield 

			if self.render: self.env.render()

			a = self.agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done: # terminal state
				s_ = None


			self.agent.train(s, a, r, s_)

			s = s_
			R += r

			if done or self.stop_signal:
				break

		print("Total R:", R)

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True

#---------
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			MarioBrain.optimize()

	def stop(self):
		self.stop_signal = True

#-- main
env_test = Environment(render = True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

MarioBrain = MarioBrain()	# MarioBrain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
	o.start()

for e in envs:
	e.start()

time.sleep(RUN_TIME)

for e in envs:
	e.stop()
for e in envs:
	e.join()

for o in opts:
	o.stop()
for o in opts:
	o.join()

print("Training finished")
env_test.run()