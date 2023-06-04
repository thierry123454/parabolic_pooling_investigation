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

model = LeNet(
	numChannels=1,
	classes=len(trainData.dataset.classes),
	ks=11,
	pool_std=True,
	check_poi=True).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

print(trainSteps)

total_poi = 0


# Meerdere runs en gemiddelde.

for _ in range(EPOCHS):
	model.train()

	for (x, y) in trainDataLoader:
		(x, y) = (x.to(device), y.to(device))
		pred, poi_counter = model(x)
		loss = lossFn(pred, y)
		opt.zero_grad()
		loss.backward()
		opt.step()
		total_poi += poi_counter

	print(f"Loss: {loss}")

print(total_poi)