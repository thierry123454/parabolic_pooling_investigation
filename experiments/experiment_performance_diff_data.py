import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN
from torch.optim import Adam, SGD
from torch import nn

from torch.optim import lr_scheduler

from torchvision import models

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import json

torch.manual_seed(0)

from morphology_package.src.morphological_torch.pooling_operations import ParabolicPool2D_V2_TL, ParabolicPool2D_TL

# define training hyperparameters
INIT_LR = 0.01
BATCH_SIZE = 32
EPOCHS = 10
RUNS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CIFAR-10 Model
class CIFAR_10_CNN(nn.Module):
   # Credit: https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844

    def __init__(self):
        
        super(CIFAR_10_CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
    
# SVHN Model
class SVHN_CNN(nn.Module):
   # Credit: https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc

	def __init__(self):
		super(SVHN_CNN, self).__init__()
		self.conv_layer = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(32),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(0.3),
		
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(64),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(0.3),
		
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(128),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(0.3),
		)

		self.fc_layer = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10),
        )


	def forward(self, x):
		"""Perform forward."""

		# conv layers
		x = self.conv_layer(x)

		# flatten
		x = x.view(x.size(0), -1)

		# fc layer
		x = self.fc_layer(x)

		return x

def load_dataset(dataset):
	transformation = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

	if dataset == "CIFAR10":
		trainData = CIFAR10(root="data/CIFAR10", download=True, train=True,
			transform=transformation)
		testData = CIFAR10(root="data/CIFAR10", download=True, train=False,
			transform=transformation)
	elif dataset == "SVHN":
		trainData = SVHN(root="data/SVHN", download=True, split='train', transform=transformation)
		testData = SVHN(root="data/SVHN", download=True, split='test', transform=transformation)

	return trainData, testData

def train_and_eval_model(model, trainData, testData):
	# initialize the train, validation, and test data loaders
	trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
	testDataLoader = DataLoader(testData, shuffle=False, batch_size=BATCH_SIZE)

	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE

	# initialize our optimizer and loss function
	# opt = Adam(model.parameters(), lr=INIT_LR)
	opt = SGD(model.parameters(), lr=INIT_LR)
	lossFn = nn.CrossEntropyLoss()
	exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=4, gamma=0.1)

	accuracies = []
	avg_f1 = []
	avg_precision = []
	avg_recall = []
	times = []

	for r in range(RUNS):
		print(f"Run: {r + 1} / {RUNS}")
		
		startTime = time.time()
		for e in range(EPOCHS):
			print(f"Epoch: {e + 1} / {EPOCHS}")

			model.train()

			step = 0
			for (x, y) in trainDataLoader:
				(x, y) = (x.to(DEVICE), y.to(DEVICE))
				pred = model(x)
				loss = lossFn(pred, y)
				opt.zero_grad()
				loss.backward()
				opt.step()
				
				if (step % 100 == 0):
					print(f"Step {step} / {trainSteps}. Loss: {loss}")

				step += 1

			exp_lr_scheduler.step()
		totalTime = time.time() - startTime

		with torch.no_grad():
			model.eval()

			preds = []
			targets = []

			for (x, y) in testDataLoader:
				x = x.to(DEVICE)
				pred = model(x)
				preds.extend(pred.argmax(axis=1).cpu().numpy())
				targets.extend(y.numpy())

		class_report = classification_report(np.array(targets),
											 np.array(preds),
											 output_dict=True)
		
		accuracies.append(class_report["accuracy"])
		avg_f1.append(class_report["macro avg"]["f1-score"])
		avg_precision.append(class_report["macro avg"]["precision"])
		avg_recall.append(class_report["macro avg"]["recall"])
		times.append(totalTime)
	
	return accuracies, avg_f1, avg_precision, avg_recall, times


def load_model(dataset):
	return CIFAR_10_CNN().to(DEVICE) if dataset == "CIFAR10" else SVHN_CNN().to(DEVICE)

def run_experiment(dataset):
	# Store data in dict
	data = {}

	print("Running experiment for " + dataset)
	trainData, testData = load_dataset(dataset)

	# Default Model without Parabolic Dilation
	print("Standard Model")
	model = load_model(dataset)
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	data[dataset + "_standard"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# Default Model with MP Parabolic Dilation
	print("Model with MP Parabolic Dilation")
	model = load_model(dataset)
	channels = -1
	for i, feature in enumerate(model.conv_layer):
		if isinstance(feature, nn.Conv2d):
			channels = feature.out_channels
		# print(channels)
		if isinstance(feature, nn.MaxPool2d):
			model.conv_layer[i] = ParabolicPool2D_TL(channels, kernel_size=5, stride=2)
			# print(feature)

	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	data[dataset + "_mp_dilation"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# Default Model with Standard Parabolic Dilation
	print("Model with Standard Parabolic Dilation")
	model = load_model(dataset)
	channels = -1
	for i, feature in enumerate(model.conv_layer):
		if isinstance(feature, nn.Conv2d):
			channels = feature.out_channels
		# print(channels)
		if isinstance(feature, nn.MaxPool2d):
			model.conv_layer[i] = ParabolicPool2D_V2_TL(channels, kernel_size=5, stride=2, init='uniform')
			# print(feature)
	
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	data[dataset + "_std_dilation"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# Default Model with Standard Parabolic Dilation with SSI
	print("Model with Standard Parabolic Dilation with SSI")
	model = load_model(dataset)
	channels = -1
	for i, feature in enumerate(model.conv_layer):
		if isinstance(feature, nn.Conv2d):
			channels = feature.out_channels
		# print(channels)
		if isinstance(feature, nn.MaxPool2d):
			model.conv_layer[i] = ParabolicPool2D_V2_TL(channels, kernel_size=7, stride=2, init='scale_space')

	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	data[dataset + "_std_dilation_ssi"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}
	
	with open("experiments/performance_" + dataset + ".json", "w") as outfile:
		json.dump(data, outfile)

run_experiment("CIFAR10")
run_experiment("SVHN")