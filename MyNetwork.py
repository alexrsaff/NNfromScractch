import _pickle as pickle
import numpy as np
import gzip
import random
import matplotlib.pyplot as plt

class DataLoader:
	def load_data(self):
		f = gzip.open('./data/mnist.pkl.gz', 'rb')
		training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
		f.close()
		return (training_data, validation_data, test_data)

	def load_and_reshape(self):
		training_data, validation_data, test_data = self.load_data()
		training_samples = []
		for sample in training_data[0]:
			training_samples.append(np.reshape(sample, (784, 1)))
		training_targets = []
		for target in training_data[1]:
			training_targets.append(self.one_hot(target))
		training_data = zip(training_samples, training_targets)
		validation_samples = []
		for sample in validation_data[0]:
			validation_samples.append(np.reshape(sample, (784, 1)))
		validation_data = zip(validation_samples, validation_data[1])
		test_samples = []
		for sample in test_data[0]:
			test_samples.append(np.reshape(sample, (784, 1)))
		test_data = zip(test_samples, test_data[1])
		return (list(training_data), list(validation_data), list(test_data))

	def one_hot(self,hot_spot):
		arr = np.zeros((10, 1))
		arr[hot_spot] = 1.0
		return(arr)

class Helpers:
	def sigmoid(self, input):
		return 1.0/(1.0+np.exp(-input))

	def sigmoid_prime(self, input):
		return self.sigmoid(input)*(1-self.sigmoid(input))

class Network(object):
	def __init__(self, layer_sizes):
		self.num_layers = len(layer_sizes)
		self.layer_sizes = layer_sizes
		self.helpers = Helpers()
		self.biases = []
		for layer in range(len(layer_sizes)-1):
			self.biases.append(np.random.randn(layer_sizes[layer+1],1))
		self.weights = []
		for layer in range(len(layer_sizes)-1):
			self.weights.append(np.random.randn(layer_sizes[layer+1],layer_sizes[layer]))

	def predict(self,input):
		for layer in range(len(self.layer_sizes)-1):
			input = self.helpers.sigmoid(np.dot(self.weights[layer], input) + self.biases[layer])
		return input
	
	def stochasticGradientDescent(self, training_data, epochs, batch_size, learning_rate, test_data=None):
		if(test_data != None):
			test_length = len(test_data)
		train_length = len(training_data)
		for epoch in range(epochs):
			random.shuffle(training_data)
			batches = []
			for index in range(0,train_length,batch_size):
				batches.append(training_data[index:index+batch_size])
			for batch in batches:
				self.update(batch,learning_rate)
			if(test_data!= None):
				print("Epoch {0}: {1} / {2}".format(epoch, self.score(test_data), test_length))
			else:
				print("Epoch {0} complete".format(epoch))
	
	def update(self,batch,learning_rate):
		bias_changes = []
		for bias in self.biases:
			bias_changes.append(np.zeros(bias.shape))
		weight_changes = []
		for weight in self.weights:
			weight_changes.append(np.zeros(weight.shape))
		for sample, target in batch:
			delta_bias_changes, delta_weight_changes = self.backpropagate(sample, target)
			for bias in range(len(bias_changes)):
				bias_changes[bias] = bias_changes[bias] + delta_bias_changes[bias]
			for weight in range(len(weight_changes)):
				weight_changes[weight] = weight_changes[weight]+delta_weight_changes[weight]
		for weight in range(len(self.weights)):
			self.weights[weight] = self.weights[weight]-(learning_rate/len(batch))*weight_changes[weight]
		for bias in range(len(self.biases)):
			self.biases[bias] = self.biases[bias]-(learning_rate/len(batch))*bias_changes[bias]

	def score(self, test_data):
		results = []
		for sample, target in test_data:
			results.append((np.argmax(self.predict(sample)), target))
		count = 0
		for sample, target in results:
			if(sample==target):
				count=count+1
		return(count)

	def cost_derivative(self, output_activations, target):
		return(output_activations - target)

	def backpropagate(self, sample, target):
		bias_changes = []
		for bias in self.biases:
			bias_changes.append(np.zeros(bias.shape))
		weight_changes = []
		for weight in self.weights:
			weight_changes.append(np.zeros(weight.shape))
		activation = sample
		activations = [sample]
		zs = []
		for layer in range(len(self.biases)):
			z = np.dot(self.weights[layer], activation)+self.biases[layer]
			zs.append(z)
			activation = self.helpers.sigmoid(z)
			activations.append(activation)
		delta = self.cost_derivative(activations[-1], target) * \
			self.helpers.sigmoid_prime(zs[-1])
		bias_changes[-1] = delta
		weight_changes[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = self.helpers.sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			bias_changes[-l] = delta
			weight_changes[-l] = np.dot(delta, activations[-l-1].transpose())
		return (bias_changes, weight_changes)

	def validate(self,validation_data):
		while(1):
			sample = random.sample(validation_data, k = 1)[0]
			print("\nLabel:", list(sample)[1])
			print("Prediction: ", np.argmax(net.predict(list(sample)[0])))
			plt.imshow(np.reshape(list(sample)[0],(28,28)), cmap = "Greys")
			plt.show()
			key = input("\nPress enter for next sample, or x to exit: ")
			if(key == "x"):
				break

Loader = DataLoader()
training_data, validation_data, test_data = Loader.load_and_reshape()

net = Network([784, 32, 10])
net.stochasticGradientDescent(training_data, epochs = 30, batch_size = 10, learning_rate = 3.0, test_data=test_data)
net.validate(validation_data)