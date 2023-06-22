# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import torch
import torch.nn.functional as F
import json
import numpy as np

torch.manual_seed(0)

from models.lenet_learnable_flat import LeNet_LearnableFlat

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 15

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

IMG_SIZE = 28

start_scales = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

data = {}

for ss in start_scales:
	print(f"Starting Scale {ss}:")

	# initialize the LeNet model
	model = LeNet_LearnableFlat(
		numChannels=1,
		classes=len(trainData.dataset.classes),
		ks=13,
		ss=ss).to(device)

	# initialize our optimizer and loss function
	opt = Adam(model.parameters(), lr=INIT_LR)
	lossFn = nn.NLLLoss()
	
	data[ss] = {}
	data[ss]["scales_p1"] = []
	data[ss]["scales_p2"] = []

	for _ in range(EPOCHS):
		model.train()
		for (x, y) in trainDataLoader:
			(x, y) = (x.to(device), y.to(device))
			# print(x.shape)
			# print(in_features)
			pred = model(x)
			loss = lossFn(pred, y)
			opt.zero_grad()
			loss.backward()
			opt.step()
		data[ss]["scales_p1"].append(model.pool1.t.tolist())
		data[ss]["scales_p2"].append(model.pool2.t.tolist())

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
	
	data[ss]["accuracy"] = class_report["accuracy"]

	print(data[ss])

with open("experiments/scale_learnability_experiment_flat.json", "w") as outfile:
	json.dump(data, outfile)
