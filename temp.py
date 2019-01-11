
		# Define Convolution 1
		self.conv1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3, 3], padding = "valid", activation = tf.nn.elu, data_format='channels_first', name = "conv1")
		self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), name = "maxPool1")
		
		# define Convolution 2
		self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3, 3], padding = "valid", activation = tf.nn.elu, data_format='channels_first', name = "conv2")
		self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name = "maxPool2")
		
		# define Convolution 3
		self.conv3 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3, 3], padding = "valid", activation = tf.nn.elu, data_format='channels_first', name = "conv3")
		self.maxPool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name = "maxPool3")
		
		# Flatten the output for LSTM Hidden Layer
		# TODO:: Debate the usefullness of a fully connected hidden layer
		self.flattened = tf.keras.layers.Flatten(name = "flattening_layer")
		self.hiddenLayer = tf.keras.layers.Dense(1026, activation = tf.nn.elu, name = "hidden_layer")

		# define LSTM layer
		self.lstmLayer = tf.keras.layers.LSTM(128, name = "LSTM_layer")

		# Output Layer consisting of an Actor and a Critic
		self._value = tf.keras.layers.Dense(1, activation = tf.nn.elu, name = "value_layer")
		self._policy = tf.keras.layers.Dense(NUM_ACTIONS, activation = tf.nn.softmax, name = "policy_layer")

		# Define the shape of out input layer
		input_def = tf.keras.layers.Input(shape = (STEP_SIZE, CHANNELS, HIGHT, WIDTH), name = "input_layer")

		# Define the forward pass for the convolutional and hidden layers
		x = tf.keras.layers.TimeDistributed(self.conv1)(input_def)
		x = tf.keras.layers.TimeDistributed(self.maxPool1)(x)
		x = tf.keras.layers.TimeDistributed(self.conv2)(x)
		x = tf.keras.layers.TimeDistributed(self.maxPool1)(x)
		x = tf.keras.layers.TimeDistributed(self.conv3)(x)
		x = tf.keras.layers.TimeDistributed(self.maxPool3)(x)
		x = tf.keras.layers.TimeDistributed(self.flattened)(x)
		x = tf.keras.layers.TimeDistributed(self.hiddenLayer)(x)
		
		# Now enter this network into a LSTM NETWORK 
		lstm_output = self.lstmLayer(x)

		# Actor and the Critic outputs
		out_value = self._value(lstm_output)
		out_actions = self._policy(lstm_output)

		# Final model
		model = tf.keras.Model(inputs=[input_def], outputs=[out_actions, out_value])

		model._make_predict_function()	# have to initialize before threading

		return model