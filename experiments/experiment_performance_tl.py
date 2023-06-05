import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, Caltech101, SVHN
from torch.optim import Adam
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
INIT_LR = 0.1 # 1e-3 
BATCH_SIZE = 32
EPOCHS = 5 # 15
RUNS = 3 # 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(dataset):
	transformation = transforms.Compose([
				transforms.Resize(224),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

	if dataset == "CIFAR100":
		trainData = CIFAR100(root="data/cifar100", download=True, train=True,
			transform=transformation)
		testData = CIFAR100(root="data/cifar100", download=True, train=False,
			transform=transformation)
	elif dataset == "CalTech101":
		print("WIP")
		pass
		trainData = Caltech101(root="data/Caltech101", download=True, transform=transformation)
		testData = Caltech101(root="data/Caltech101", download=True, transform=transformation)
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
	opt = Adam(model.parameters(), lr=INIT_LR)
	lossFn = nn.CrossEntropyLoss()
	exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)

	accuracies = []
	avg_f1 = []
	avg_precision = []
	avg_recall = []
	times = []

	for r in range(RUNS):
		print(f"Run: {r} / {RUNS}")
		
		startTime = time.time()
		for e in range(EPOCHS):
			print(f"Epoch: {e} / {EPOCHS}")

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
				print(pred.argmax(axis=1).cpu().numpy())
				print(y.numpy())

		class_report = classification_report(np.array(targets),
											 np.array(preds),
											 output_dict=True)
		
		accuracies.append(class_report["accuracy"])
		avg_f1.append(class_report["macro avg"]["f1-score"])
		avg_precision.append(class_report["macro avg"]["precision"])
		avg_recall.append(class_report["macro avg"]["recall"])
		times.append(totalTime)
	
	return accuracies, avg_f1, avg_precision, avg_recall, times


def load_vgg16(num_classes):
	model = models.vgg16().to(DEVICE)

	# Deze op true?
	for param in model.parameters():
		param.requires_grad = True # False

	num_features = model.classifier[6].in_features
	features = list(model.classifier.children())[:-1] # Remove last layer
	features.extend([nn.Linear(num_features, num_classes)]) # Add our layer with 4 outputs
	model.classifier = nn.Sequential(*features).to(DEVICE) # Replace the model classifier

	return model

def run_experiment(dataset):
	# Store data in dict
	data = {}

	print("Running experiment for " + dataset)
	trainData, testData = load_dataset(dataset)

	if dataset == "SVHN":
		num_classes = 10
	else:
		num_classes = len(trainData.classes)

	# Default Model without Parabolic Dilation
	model = load_vgg16(num_classes)
	print("Standard Model")
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	data["vgg16_standard"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# Default Model with MP Parabolic Dilation
	model = load_vgg16(num_classes)
	channels = -1
	for i, feature in enumerate(model.features):
		if isinstance(feature, nn.Conv2d):
			channels = feature.out_channels
		# print(channels)
		if isinstance(feature, nn.MaxPool2d):
			model.features[i] = ParabolicPool2D_TL(channels, kernel_size=5, stride=2)
			# print(feature)
	
	print("Model with MP Parabolic Dilation")
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	data["vgg16_mp_dilation"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# print("Model with Standard Parabolic Dilation")
	# # Default Model with Standard Parabolic Dilation
	# model = load_vgg16(num_classes)
	# channels = -1
	# for i, feature in enumerate(model.features):
	# 	if isinstance(feature, nn.Conv2d):
	# 		channels = feature.out_channels
	# 	if isinstance(feature, nn.MaxPool2d):
	# 		model.features[i] = ParabolicPool2D_V2_TL(channels, kernel_size=5, stride=2, init='uniform')
	
	# accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	# data["vgg16_std_dilation"] = {
	# 	"accuracy": accuracies,
	# 	"avg_f1": avg_f1,
	# 	"avg_precision": avg_precision,
	# 	"avg_recall": avg_recall,
	# 	"time": times
	# }

	print("Model with Standard Parabolic Dilation with SSI")
	# Default Model with Standard Parabolic Dilation with SSI
	model = load_vgg16(num_classes)
	channels = -1
	for i, feature in enumerate(model.features):
		if isinstance(feature, nn.Conv2d):
			channels = feature.out_channels
		if isinstance(feature, nn.MaxPool2d):
			model.features[i] = ParabolicPool2D_V2_TL(channels, kernel_size=7, stride=2, init='scale_space')
	
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainData, testData)

	data["vgg16_std_dilation_ssi"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}
	
	with open("experiments/performance_transfer_learning_" + dataset + ".json", "w") as outfile:
		json.dump(data, outfile)

run_experiment("SVHN")
run_experiment("CIFAR100")
