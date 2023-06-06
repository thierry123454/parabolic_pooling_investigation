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

from models.parabolic_lenet_ss import LeNet_SS

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 5
RUNS = 4

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

def train(scale, in_features):
	# initialize the LeNet model
	model = LeNet_SS(
		numChannels=1,
		classes=len(trainData.dataset.classes),
		ks=13,
		fc_in_features=in_features,
		).to(device)

	# initialize our optimizer and loss function
	opt = Adam(model.parameters(), lr=INIT_LR)
	lossFn = nn.NLLLoss()

	for _ in range(EPOCHS):
		model.train()
		for (x, y) in trainDataLoader:
			if (scale >= 2):
				x = F.interpolate(x, size=(IMG_SIZE * scale, IMG_SIZE * scale), mode='bilinear', align_corners=False)

			(x, y) = (x.to(device), y.to(device))
			# print(x.shape)
			# print(in_features)
			pred = model(x)
			loss = lossFn(pred, y)
			opt.zero_grad()
			loss.backward()
			opt.step()
	
	with torch.no_grad():
		model.eval()

		preds = []

		for (x, y) in testDataLoader:
			x = x.to(device)
			if (scale >= 2):
				x = F.interpolate(x, size=(IMG_SIZE * scale, IMG_SIZE * scale), mode='bilinear', align_corners=False)

			pred = model(x)
			preds.extend(pred.argmax(axis=1).cpu().numpy())

		class_report = classification_report(testData.targets.cpu().numpy(),
												np.array(preds),
												target_names=testData.classes,
												output_dict=True)

	return (model.pool1.t.item(), model.pool2.t.item(), class_report["accuracy"])

def collect_data():
	data = {}

	for img_scale in range(1, 5):
		print(f"Scale {img_scale}:")

		scales_p1 = []
		scales_p2 = []
		acc = []

		for (x, _) in trainDataLoader:
			x = F.interpolate(x, size=(IMG_SIZE * img_scale, IMG_SIZE * img_scale), mode='bilinear', align_corners=False)
			x = x.to(device)
			in_features = LeNet_SS(numChannels=1, classes=len(trainData.dataset.classes)).to(device)._calculate_num_features(x)
			break
		
		print(f"Num features for FCC: {in_features}")

		for r in range(RUNS):
			print(f"Run {r}:")
			(t1, t2, accuracy) = train(img_scale, in_features)
		
			scales_p1.append(t1)
			scales_p2.append(t2)
			acc.append(accuracy)
		
		data[img_scale] = {}
		data[img_scale]["scales_p1"] = scales_p1
		data[img_scale]["scales_p2"] = scales_p2
		data[img_scale]["accuracies"] = acc

		print(data[img_scale])

	with open("experiments/scale_experiment_ss.json", "w") as outfile:
		json.dump(data, outfile)

# Using standard SE
collect_data()