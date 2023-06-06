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

from models.standard_lenet import LeNet_Standard
from models.parabolic_lenet import LeNet

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import json

torch.manual_seed(0)

from morphology_package.src.morphological_torch.pooling_operations import ParabolicPool2D_V2_TL, ParabolicPool2D_TL

# define training hyperparameters
INIT_LR = 0.01 # 1e-3 
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

def train_and_eval_model(model, trainDataLoader, testDataLoader):
	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE

	# initialize our optimizer and loss function
	opt = Adam(model.parameters(), lr=INIT_LR)
	lossFn = nn.NLLLoss()
	exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

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

		class_report = classification_report(np.array(targets),
											 np.array(preds),
											 output_dict=True)
		
		accuracies.append(class_report["accuracy"])
		avg_f1.append(class_report["macro avg"]["f1-score"])
		avg_precision.append(class_report["macro avg"]["precision"])
		avg_recall.append(class_report["macro avg"]["recall"])
		times.append(totalTime)
	
	return accuracies, avg_f1, avg_precision, avg_recall, times


def load_standard_model(trainDataLoader, num_classes):
	for (x, _) in trainDataLoader:
			x = x.to(DEVICE)
			in_features = LeNet_Standard(numChannels=3, classes=num_classes).to(DEVICE)._calculate_num_features(x)
			break
	
	model = LeNet_Standard(
		numChannels=3,
		classes=num_classes,
		fc_in_features=in_features).to(DEVICE)
	
	return model

def load_parabolic_model(trainDataLoader, num_classes, pool_method_std, ssi):
	window_size = 7 if ssi else 5

	for (x, _) in trainDataLoader:
			x = x.to(DEVICE)
			in_features = LeNet(
				numChannels=3,
				classes=num_classes,
				ks=window_size,
				pool_std=pool_method_std,
				pool_init='scale_space' if ssi else 'uniform').to(DEVICE)._calculate_num_features(x)
			break
	
	model = LeNet(
				numChannels=3,
				classes=num_classes,
				ks=window_size,
				pool_std=pool_method_std,
				fc_in_features=in_features,
				pool_init='scale_space' if ssi else 'uniform').to(DEVICE)
	
	return model

def run_experiment(dataset):
	# Store data in dict
	data = {}

	print("Running experiment for " + dataset)
	trainData, testData = load_dataset(dataset)

	# initialize the train, validation, and test data loaders
	trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
	testDataLoader = DataLoader(testData, shuffle=False, batch_size=BATCH_SIZE)

	if dataset == "SVHN":
		num_classes = 10
	else:
		num_classes = len(trainDataLoader.dataset.classes)

	# Default Model without Parabolic Dilation
	model = load_standard_model(trainDataLoader, num_classes)

	print("Standard Model")
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainDataLoader, testDataLoader)

	data["standard"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# Default Model with MP Parabolic Dilation
	model = load_parabolic_model(trainDataLoader, num_classes, False, False)

	print("Model with MP Parabolic Dilation")
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainDataLoader, testDataLoader)

	data["mp_dilation"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# Default Model with Standard Parabolic Dilation
	model = load_parabolic_model(trainDataLoader, num_classes, True, False)
	
	print("Model with Standard Parabolic Dilation")
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainDataLoader, testDataLoader)

	data["std_dilation"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}

	# Default Model with Standard Parabolic Dilation with SSI
	model = load_parabolic_model(trainDataLoader, num_classes, True, True)
	
	print("Model with Standard Parabolic Dilation with SSI")
	accuracies, avg_f1, avg_precision, avg_recall, times = train_and_eval_model(model, trainDataLoader, testDataLoader)

	data["std_dilation_ssi"] = {
		"accuracy": accuracies,
		"avg_f1": avg_f1,
		"avg_precision": avg_precision,
		"avg_recall": avg_recall,
		"time": times
	}
	
	with open("experiments/performance_" + dataset + ".json", "w") as outfile:
		json.dump(data, outfile)

run_experiment("CIFAR100")
run_experiment("SVHN")
