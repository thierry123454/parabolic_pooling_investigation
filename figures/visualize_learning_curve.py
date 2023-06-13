import matplotlib.pyplot as plt
import numpy as np
import torch

H = {'train_loss': [torch.tensor(0.29495338, dtype=torch.float32), torch.tensor(0.08223312, dtype=torch.float32), torch.tensor(0.05162056, dtype=torch.float32), torch.tensor(0.03380384, dtype=torch.float32), torch.tensor(0.02599436, dtype=torch.float32), torch.tensor(0.02039157, dtype=torch.float32), torch.tensor(0.01633184, dtype=torch.float32), torch.tensor(0.01388431, dtype=torch.float32), torch.tensor(0.01136813, dtype=torch.float32), torch.tensor(0.01323163, dtype=torch.float32)], 'train_acc': [0.9067555555555555, 0.9742222222222222, 0.9841777777777778, 0.9892666666666666, 0.9915111111111111, 0.9933111111111111, 0.9950444444444444, 0.9953333333333333, 0.9963333333333333, 0.9956666666666667], 'val_loss': [torch.tensor(0.12255295, dtype=torch.float32), torch.tensor(0.07374547, dtype=torch.float32), torch.tensor(0.06632999, dtype=torch.float32), torch.tensor(0.07213044, dtype=torch.float32), torch.tensor(0.08395796, dtype=torch.float32), torch.tensor(0.084525, dtype=torch.float32), torch.tensor(0.07504787, dtype=torch.float32), torch.tensor(0.10772106, dtype=torch.float32), torch.tensor(0.08719433, dtype=torch.float32), torch.tensor(0.09622581, dtype=torch.float32)], 'val_acc': [0.9646, 0.9792, 0.9811333333333333, 0.9811333333333333, 0.9786, 0.9784, 0.9830666666666666, 0.9784, 0.9811333333333333, 0.9813333333333333]}
EPOCHS = 10

# Setup LaTeX
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

epochs_list = np.arange(EPOCHS)
plt.figure(figsize=(13, 7))

# Plot train loss with dashes and dots
plt.plot(epochs_list, H["train_loss"], 'o--', label='Train Loss')

# Plot validation loss with dashes and dots
plt.plot(epochs_list, H["val_loss"], 'o--', label='Validation Loss')

plt.legend()
plt.title(f'Learning curve for LeNet using parabolic pooling layers on KMNIST.', fontdict={'fontsize': 15})

# Add x-label and y-label
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Add grid
plt.grid(True)

plt.savefig(f"figures/training_curve.pdf", format="pdf", bbox_inches="tight")

plt.show()