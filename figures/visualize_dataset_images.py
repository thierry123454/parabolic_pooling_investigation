import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms

transformation = transforms.Compose([
    transforms.ToTensor()
])

trainData = CIFAR10(root="data/CIFAR10", download=True, train=True, transform=transformation)
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=32)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for images, labels in trainDataLoader:
    fig = plt.figure(figsize=(10,4))

    for idx in np.arange(10):   # change the number according to how many images you want to display
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(class_names[labels[idx]])

    plt.tight_layout()
    plt.savefig("figures/cifar_10_train_images.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    break

trainData = SVHN(root="data/SVHN", download=True, split='train', transform=transformation)
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=32)

class_names = [str(x) for x in range(10)]

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for images, labels in trainDataLoader:
    fig = plt.figure(figsize=(10,4))

    for idx in np.arange(10):   # change the number according to how many images you want to display
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(class_names[labels[idx]])

    plt.tight_layout()
    plt.savefig("figures/svhn_train_images.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    break