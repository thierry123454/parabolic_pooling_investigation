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

# torch.manual_seed(0)

from morphology_package.src.morphological_torch.pooling_operations import ParabolicPool2D_V2, ParabolicPool2D

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 1
RUNS = 20

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# load the ImageNet dataset
trainData = CIFAR100(root="data/cifar100", download=True, train=True,
	transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
testData = CIFAR100(root="data/cifar100", download=True, train=False,
	transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

# calculate the train/validation split
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, shuffle=True, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(weights="DEFAULT").to(device)

for param in model.parameters():
	param.requires_grad = False

num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(trainData.dataset.classes))]) # Add our layer with 4 outputs
model.classifier = nn.Sequential(*features).to(device) # Replace the model classifier

print(model)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

startTime = time.time()
for e in range(EPOCHS):
	print(f"Epoch: {e} / {EPOCHS}")
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	step = 0
	startEpoch = time.time()
	for (x, y) in trainDataLoader:
		(x, y) = (x.to(device), y.to(device))
		pred = model(x)
		loss = lossFn(pred, y)
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
		
		print(f"Step {step} out of {trainSteps}")
		print(f"Loss: {loss}")
		step += 1
	print(f"Epoch duration: {time.time() - startEpoch}")	
	
	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFn(pred, y)
			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()
			
	exp_lr_scheduler.step()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
	valCorrect = valCorrect / len(valDataLoader.dataset)
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))
	
totalTime = time.time() - startTime

# plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(H["train_loss"], label="train_loss")
# plt.plot(H["val_loss"], label="val_loss")
# plt.plot(H["train_acc"], label="train_acc")
# plt.plot(H["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("figures/transfer_learning_training_curve.pdf", format="pdf", bbox_inches="tight")
# plt.show()

with torch.no_grad():
	model.eval()

	preds = []
	targets = []

	for (x, y) in testDataLoader:
		x = x.to(device)
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())
		targets.extend(y.numpy())

	class_report = classification_report(np.array(targets),
										 np.array(preds),
										 target_names=testData.classes,
										 output_dict=True)
	
	print(class_report)

exit()

channels = -1
for i, feature in enumerate(model.features):
	if isinstance(feature, nn.Conv2d):
		channels = feature.out_channels
	# print(channels)
	if isinstance(feature, nn.MaxPool2d):
		model.features[i] = ParabolicPool2D_V2(channels, 5, 2)
		# print(feature)

# print(model)

# 1 Zonder parabolic dilation
# 1 Met parabolic dilation
# 1 Met parabolic dilation en SSI