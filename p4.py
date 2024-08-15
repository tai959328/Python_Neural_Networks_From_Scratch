import numpy as np

# why batch calculation? calculate in parallel. This is we tend to do NN training on GPUs instead of CPUs.
# CPU typically have 4-8 cores, compared to typical GPU having hundrends to thousands of cores
# Another reason is batches help with generalization. This seems to be related to batching samples of the data
# to help with the speed and stabililty of learning. If the batch size is too small, there is is volatility in
# the learning while if the batch size is too large, there would be overfitting thus poor out of sample performance

inputs = [[1, 2, 3, 2.5],
		  [2.0, 5.0, -1.0, 2.0],
		  [-1.5, 2.7, 3.3, -0.8]]

weights = [	[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# output = np.dot(weights,inputs) + biases 
# this would not work b/c it's (3,4) shape multipled by a (3,4) shape. Therefore need to transpose.
# we would transpose the variable weights, and the resulting shape would be (3,3)

# output = np.dot(weights,np.array(inputs).T) + biases 
# # this would be incorrect because we are not multiplying inputs with corresponding weights
# # said another way, we are multiplying the same set of weights across all sets of inputs
# # and we instead should be multiplying the same set of inputs across all sets of weights
# print(np.array(weights))
# print(np.array(inputs).T)
# print(output)



output = np.dot(inputs,np.array(weights).T) + biases 
print(output)


# now we want to add another layer

weights2 = [[0.1, -0.14, 0.5],
			[-0.5, 0.12, -0.33],
			[-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs,np.array(weights).T) + biases
print(layer1_outputs)

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)


# as we can see as we get more layers this coding style will get very unwieldy
# so let's write this code this more efficiently


# inputs become X, which is our input data
X = [ [1, 2, 3, 2.5],
	  [2.0, 5.0, -1.0, 2.0],
	  [-1.5, 2.7, 3.3, -0.8]]


np.random.seed(0)

class Layer_Dense:
		def __init__(self, n_inputs, n_neurons):
			self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
				# n_inputs here represent the number of features in each sample of input, so 4 in the previous examples
				# the order of n_inputs and n_neurons for the shape will prevent the need for transpose
			self.biases = np.zeros((1, n_neurons))
		def forward(self, inputs):
			self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)