import matplotlib.pyplot as plt
import numpy as np
import json

f = open('experiments/kernel_size_experiment.json')
  
data = json.load(f)

print(data.keys())

EPOCHS = 10

# Setup LaTeX
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

epochs_list = np.arange(EPOCHS)
plt.figure(figsize=(13, 7))

ks = [int(key) for key in data.keys()]

accuracy_list = []
accuracy_std = []

for key in data.keys():
    accuracy_list.append(np.mean(data[key]["accuracy"]))
    accuracy_std.append(np.std(data[key]["accuracy"]))

plt.errorbar(ks, accuracy_list, accuracy_std, marker="o", linestyle='--', capsize=5, label='Average accuracy')

# Plot validation loss with dashes and dots
# plt.plot(epochs_list, H["val_loss"], 'o--', label='Validation Loss')

plt.legend()
plt.title(f'Performance on different metrics versus the kernel size of the parabolic structuring element.', fontdict={'fontsize': 15})

# Add x-label and y-label
plt.xlabel('Kernel Size')

# Add grid
plt.grid(True)

plt.savefig(f"figures/kernel_size_experiment.pdf", format="pdf", bbox_inches="tight")

plt.show()