# print('hello world')

# to install numpy and matplotlib packages, go into the command prompt and use the following commands
# pip install numpy
# pip install matplotlib

import sys
import numpy as np
import matplotlib

print("Python:", sys.version)
print("Numpy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)


# a neuron has a unique connection to a previous neuron
inputs = [1.2, 5.1, 2.1] # these are outputs from the three neurons in the previous layer
weights = [3.1, 2.1, 8.7] # every input is going to have a unique weight associated to it
bias = 3 # every unique neuron has a unique bias 

# output of a neuron is to add up call the inputs times the weights plus the bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

# test update line for github