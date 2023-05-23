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
import torch
import torch.nn.functional as F
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

avg_ratios_p1 = []
avg_ratios_p2 = []

def train(extrapolate):
	# initialize the LeNet model
	model = LeNet(
		numChannels=1,
		classes=len(trainData.dataset.classes),
		ks=11,
		fc_in_features=(6050 if extrapolate else 800)).to(device)

	# initialize our optimizer and loss function
	opt = Adam(model.parameters(), lr=INIT_LR)
	lossFn = nn.NLLLoss()

	for _ in range(EPOCHS):
		model.train()
		for (x, y) in trainDataLoader:
			if (extrapolate):
				x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)

			(x, y) = (x.to(device), y.to(device))
			pred = model(x)
			loss = lossFn(pred, y)
			opt.zero_grad()
			loss.backward()
			opt.step()
	
	return (model.pool1.t, model.pool2.t)

for r in range(4):
	print(f"Run {r}")
	print("Train without interpolate")
	(t1, t2) = train(False)
	print("Train with interpolate")
	(t3, t4) = train(True)
	print()

	# print("Scale Experiment:")
	# print(f"Scales pool1 before interpolation: {t1}")
	# print(f"Mean scales pool1 before interpolation: {torch.mean(t1)}")
	# print()
	# print(f"Scales pool1 after interpolation: {t3}")
	# print(f"Mean scales pool1 after interpolation: {torch.mean(t3)}")
	# print()
	# print(f"Scales pool2 before interpolation: {t2}")
	# print(f"Mean scales pool2 before interpolation: {torch.mean(t2)}")
	# print()
	# print(f"Scales pool2 after interpolation: {t4}")
	# print(f"Mean scales pool2 after interpolation: {torch.mean(t4)}")
	# print()
	# print(f"Ratio of pool1 after and before interpolation {t3 / t1}")
	# print(f"Ratio of pool2 after and before interpolation {t4 / t2}")
	# print()
	# print(f"Average ratio of pool1 after and before interpolation {torch.mean(t3 / t1)}")
	# print(f"Average ratio of pool2 after and before interpolation {torch.mean(t4 / t2)}")

	avg_ratios_p1.append(torch.mean(t3 / t1).item())
	avg_ratios_p2.append(torch.mean(t4 / t2).item())

data = {
	"avg_ratios_p1": avg_ratios_p1,
	"avg_ratios_p2": avg_ratios_p2
}

with open("experiments/scale_experiment.json", "w") as outfile:
    json.dump(data, outfile)