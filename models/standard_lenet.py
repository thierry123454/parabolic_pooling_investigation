from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import MaxPool2d
from torch import flatten
from morphology_package.src.morphological_torch.pooling_operations import ParabolicPool2D_V2, ParabolicPool2D
import torch

torch.manual_seed(0)

class LeNet_Standard(Module):
	# Pooling Method paramet!
	# Determine FC input featuress
	def __init__(self, numChannels, classes, fc_in_features=800):
		# call the parent constructor
		super(LeNet_Standard, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=fc_in_features, out_features=500)
		self.relu3 = ReLU()
		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		# print(x.shape)
		x = self.fc1(x)
		x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output
	
	def _calculate_num_features(self, x):
		# Calculate the number of features based on the shape of x
		with torch.no_grad():
			x = self.conv1(x)
			x = self.relu1(x)
			x = self.pool1(x)
			x = self.conv2(x)
			x = self.relu2(x)
			x = self.pool2(x)
		return x.shape[1] * x.shape[2] * x.shape[3]