import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import json

torch.manual_seed(0)

from models.parabolic_lenet import LeNet
from models.standard_lenet import LeNet_Standard

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 5
RUNS = 20

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the KMNIST dataset
trainData = KMNIST(root="data/kmnist", train=True, download=True,
	transform=ToTensor())
testData = KMNIST(root="data/kmnist", train=False, download=True,
	transform=ToTensor())

# calculate the train/validation split
numTrainSamples = int(len(trainData))

(trainData, _) = random_split(trainData, [numTrainSamples, 0], generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE

# Dictionary that contains all data
data = {}

first = True

def train_and_store(standard, standard_se, scale_space):
	global first

	accuracy = []
	avg_f1 = []
	avg_precision = []
	avg_recall = []
	times = []

	# Train model for one epoch to get training data in cache. This leads to
	# to the very first iteration not having a extraordinarily high time to
	# train.
	if first:
		model = LeNet(
			numChannels=1,
			classes=len(trainData.dataset.classes),
			ks=5).to(device)
		opt = Adam(model.parameters(), lr=INIT_LR)
		lossFn = nn.NLLLoss()
		model.train()
		for (x, y) in trainDataLoader:
			(x, y) = (x.to(device), y.to(device))
			pred = model(x)
			loss = lossFn(pred, y)
			opt.zero_grad()
			loss.backward()
			opt.step()
		first = False

	for r in range(RUNS):
		print(f"Run: {r}")

		# initialize the LeNet model
		if (standard):
			model = LeNet_Standard(
				numChannels=1,
				classes=len(trainData.dataset.classes)).to(device)
		else:
			model = LeNet(
				numChannels=1,
				classes=len(trainData.dataset.classes),
				ks=5 if not scale_space else 7,
				pool_std=standard_se,
				pool_init='scale_space' if scale_space else 'uniform').to(device)

		# initialize our optimizer and loss function
		opt = Adam(model.parameters(), lr=INIT_LR)
		lossFn = nn.NLLLoss()

		startTime = time.time()
		for _ in range(EPOCHS):
			model.train()

			for (x, y) in trainDataLoader:
				(x, y) = (x.to(device), y.to(device))
				pred = model(x)
				loss = lossFn(pred, y)
				opt.zero_grad()
				loss.backward()
				opt.step()
		totalTime = time.time() - startTime

		with torch.no_grad():
			model.eval()

			preds = []

			for (x, y) in testDataLoader:
				x = x.to(device)
				pred = model(x)
				preds.extend(pred.argmax(axis=1).cpu().numpy())
		
		class_report = classification_report(testData.targets.cpu().numpy(),
											 np.array(preds),
											 target_names=testData.classes,
											 output_dict=True)

		accuracy.append(class_report["accuracy"])
		avg_f1.append(class_report["macro avg"]["f1-score"])
		avg_precision.append(class_report["macro avg"]["precision"])
		avg_recall.append(class_report["macro avg"]["recall"])
		times.append(totalTime)
	
	uses_scale_space = "_scale_space" if scale_space else ""

	parabolic_key = "std_parabolic" + uses_scale_space if standard_se else "mp_parabola"

	data["standard" if standard else parabolic_key] = {
		"accuracy": accuracy,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	print(data)

# Train standard LeNet
train_and_store(True, False, False)

# Train LeNet with MP Parabolic Dilation
train_and_store(False, False, False)

# Train LeNet with Parabolic Dilation
train_and_store(False, True, False)

# Train LeNet with Parabolic Dilation with Scale Space
train_and_store(False, True, True)

with open("experiments/performance_max_pool_vs_mp_vs_parabolic_vs_parabolic_scale_space.json", "w") as outfile:
    json.dump(data, outfile)