inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [	inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
			inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
			inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

print(output)


# goal is to clean the above "list" code and transition it into more dynamic code with vectors and matrices
inputs = [1, 2, 3, 2.5]

weights = [	[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# instructor gave comments about why we have both weights and biases. The gist is that they provide different levers for fitting,
# similar to fitting y = mx + b where m (like weights) is a multiplier and where b is an offset (like biases)



layer_outputs = [] # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases): 
		# zip() combines two lists into a list of lists element-wise
		# i.e. 0th element of this zipped list is a list of both the 0th element of the weights list and the 0th element of the biases list
	neuron_output = 0 # output of given neuron
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output += n_input*weight
				# += operator provides a convenient way to add a value to an existing variable then assign the new value back to the same variable
				# for example: a = 4 then a += 5  would add 5 to a, and assign the result into a
	neuron_output += neuron_bias
	layer_outputs.append(neuron_output)

print(layer_outputs)



l = [1,5,6,2]
# list
# shape: (4, )
## type: a simple list in python is a 1D array in numpy (aka vector in mathematics)


lol = [ [1,5,6,2],
		[3,2,1,3]]
# list of list
# shape: (2, 4)
## type: 2D array in numpy (aka matrix in mathematics)
# arrays have to be homologous, meaning at each dimension they have to be the same size; e.g. list of lists needs the same number of elements in each list
# a matrix is simpy a rectangular array. A list of vectors is a matrix

lolol = [[[1,5,6,2],
		  [3,2,1,3]],
		 [[5,2,1,2],
		  [6,4,8,4]],
		 [[2,8,5,3],
		  [1,1,9,4]]]
# list of list of list
# shape: (3, 2, 4)
## type: 3D array in numpy
# this also needs to be homologous

# a tensor is an object that CAN be represented as an array, not just an array. In context of deep learning in programming, a tensor is represented as an array.
# this means we will work with tensors in the array form




import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(inputs,weights) + bias
print(output)
output = np.dot(weights,inputs) + bias
print(output)
# the output is the same but that's because of the vector shapes we are dealing it, the order will matter greatly with larger shapes



inputs = [1, 2, 3, 2.5]

weights = [	[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights,inputs) + biases
print(output)
# output = np.dot(inputs,weights) + biases would not work b/c like matrix multiplication shape (3,4) needs to be multiplied to shape (4,) resulting in shape (3,)
# which can then be added to shape (3,) of biases
