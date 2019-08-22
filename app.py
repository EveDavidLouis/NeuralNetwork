import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#np.random.seed(100)
np.set_printoptions(precision=1)

class Layer:

	def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):

		self.weights = weights if weights is not None else np.random.standard_normal((n_input,n_neurons))/n_neurons**.5
		self.activation = activation
		self.bias = bias if bias is not None else np.zeros(n_neurons)

	def activate(self, x):

		r = np.dot(x, self.weights) + self.bias
		self.last_activation = self._apply_activation(r)
		return self.last_activation

	def _apply_activation(self, r):
	
		# In case no activation function was chosen
		if self.activation is None:
			return r

		# tanh
		if self.activation == 'tanh':
			return np.tanh(r)

		# sigmoid
		if self.activation == 'sigmoid':
			return 1 / (1 + np.exp(-r))

		return r

	def apply_activation_derivative(self, r):

		if self.activation is None:
			return r

		if self.activation == 'tanh':
			return 1 - r ** 2

		if self.activation == 'sigmoid':
			return r * (1 - r)

		return r

class NeuralNetwork:

	def __init__(self):

		self._layers = []

	def add_layer(self, layer):

		self._layers.append(layer)

	def feed_forward(self, X):

		for layer in self._layers:
			X = layer.activate(X)

		return X

	def predict(self, X):

		ff = self.feed_forward(X)
		#return ff

		# One row
		if ff.ndim == 1:
			return np.argmax(ff)

		# Multiple rows
		return np.argmax(ff, axis=1)

	def backpropagation(self, X, y, learning_rate):
   
		# Feed forward for the output
		output = self.feed_forward(X)

		# Loop over the layers backward
		for i in reversed(range(len(self._layers))):
			layer = self._layers[i]

			# If this is the output layer
			if layer == self._layers[-1]:
				layer.error = y - output
				# The output = layer.last_activation in this case
				layer.delta = layer.error * layer.apply_activation_derivative(output)
			else:
				next_layer = self._layers[i + 1]
				layer.error = np.dot(next_layer.weights, next_layer.delta)
				layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

		# Update the weights
		for i in range(len(self._layers)):
			layer = self._layers[i]
			# The input is either the previous layers output or X itself (for the first hidden layer)
			input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
			layer.weights += layer.delta * input_to_use.T * learning_rate
			layer.bias += layer.delta * learning_rate
	
	def train(self, X, y, learning_rate, max_epochs):

		mses = []

		for i in range(max_epochs):
			for j in range(len(X)):
				self.backpropagation(X[j], y[j], learning_rate)
			if i % 10 == 0:
				mse = np.mean(np.square(y - self.feed_forward(X)))
				mses.append(mse)
				print('Epoch: #%s, MSE: %f' % (i, float(mse)))

		return mses

	def display(self):

		for i,l in enumerate(self._layers):
			print ('Layer ',i,'\n#########')
			print ('w',i,'\n',l.weights,'\n---------')
			print ('b',i,'\n',l.bias,'\n---------')
	
	@staticmethod
	def accuracy(y_pred, y_true):
		return (y_pred == y_true).mean()
		
layer_sizes = (2,3,1)

nn = NeuralNetwork()
nn.add_layer(Layer(2, 20, 'sigmoid'))
nn.add_layer(Layer(20, 1, 'sigmoid'))

# Define dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
 
# Train the neural network
errors = nn.train(X, y, 0.1, 9999)
# Plot changes in mse
plt.plot(errors)
plt.title('MeanSquareError')
plt.xlabel('Epoch x10')
plt.ylabel('MSE')
plt.figure(1).canvas.set_window_title('Dyn ML NN')
plt.show()