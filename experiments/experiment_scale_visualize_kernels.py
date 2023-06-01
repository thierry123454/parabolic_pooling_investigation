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

from models.parabolic_lenet import LeNet

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 10

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

def train(scale, in_features, use_pool_std):
	# initialize the LeNet model
	model = LeNet(
		numChannels=1,
		classes=len(trainData.dataset.classes),
		ks=13,
		fc_in_features=in_features,
		pool_std=use_pool_std,
		scale=scale).to(device)

	# initialize our optimizer and loss function
	opt = Adam(model.parameters(), lr=INIT_LR)
	lossFn = nn.NLLLoss()

	for _ in range(EPOCHS):
		model.train()
		for (x, y) in trainDataLoader:
			if (scale >= 2):
				x = F.interpolate(x, size=(IMG_SIZE * scale, IMG_SIZE * scale), mode='bilinear', align_corners=False)

			(x, y) = (x.to(device), y.to(device))
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
	
	conv_weights = model.conv1.weight.data.cpu().tolist()

	return (model.pool1.t.tolist(), model.pool2.t.tolist(), class_report["accuracy"], conv_weights)

def collect_data(std_pool):
	data = {}

	print(f"Standard pool: {std_pool}")
	for img_scale in range(1, 5):
		print(f"Scale {img_scale}:")

		for (x, _) in trainDataLoader:
			x = F.interpolate(x, size=(IMG_SIZE * img_scale, IMG_SIZE * img_scale), mode='bilinear', align_corners=False)
			x = x.to(device)
			in_features = LeNet(numChannels=1, classes=len(trainData.dataset.classes), scale=img_scale).to(device)._calculate_num_features(x)
			break
		
		print(f"Num features for FCC: {in_features}")

		(t1, t2, accuracy, conv_weights) = train(img_scale, in_features, std_pool)
	
		scales_p1 = t1
		scales_p2 = t2
		acc = accuracy
		
		data[img_scale] = {}
		data[img_scale]["scales_p1"] = scales_p1
		data[img_scale]["scales_p2"] = scales_p2
		data[img_scale]["conv_weights"] = conv_weights
		data[img_scale]["accuracies"] = acc

		print(data[img_scale])

	with open("experiments/scale_experiment_conv_kernels_standard.json" if std_pool else "experiments/scale_experiment_kernels_normalized.json", "w") as outfile:
		json.dump(data, outfile)

# Using standard SE
collect_data(True)

# # Using normalized SE
# collect_data(False)