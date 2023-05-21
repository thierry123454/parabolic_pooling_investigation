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

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 5
RUNS = 3

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the KMNIST dataset
print("[INFO] loading the KMNIST dataset...")
trainData = KMNIST(root="data/kmnist", train=True, download=True,
	transform=ToTensor())
testData = KMNIST(root="data/kmnist", train=False, download=True,
	transform=ToTensor())
# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData))

(trainData, _) = random_split(trainData, [numTrainSamples, 0], generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE

# Dictionary that contains all data
data = {}

def train_and_store(kernel_size):
	print(f"Starting experiment for kernel size: {kernel_size}")

	accuracy = []
	avg_f1 = []
	avg_precision = []
	avg_recall = []
	times = []

	for r in range(RUNS):
		print(f"Run: {r}")

		# initialize the LeNet model
		model = LeNet(
			numChannels=1,
			classes=len(trainData.dataset.classes),
			ks=kernel_size).to(device)

		# initialize our optimizer and loss function
		opt = Adam(model.parameters(), lr=INIT_LR)
		lossFn = nn.NLLLoss()

		startTime = time.time()
		for _ in range(EPOCHS):
			model.train()
			totalTrainLoss = 0
			trainCorrect = 0

			for (x, y) in trainDataLoader:
				(x, y) = (x.to(device), y.to(device))
				pred = model(x)
				loss = lossFn(pred, y)
				opt.zero_grad()
				loss.backward()
				opt.step()
				totalTrainLoss += loss
				trainCorrect += (pred.argmax(1) == y).type(
					torch.float).sum().item()
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
	
	data[kernel_size] = {
		"accuracy": accuracy,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	print(data[kernel_size])

kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

for ks in kernel_sizes:
	train_and_store(ks)

print(data)

with open("experiments/kernel_size_experiment.json", "w") as outfile:
    json.dump(data, outfile)