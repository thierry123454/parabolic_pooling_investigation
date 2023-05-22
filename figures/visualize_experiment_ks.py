import matplotlib.pyplot as plt
import numpy as np
import json

f = open('experiments/kernel_size_experiment.json')
  
data = json.load(f)

print(data.keys())

EPOCHS = 10


epochs_list = np.arange(EPOCHS)
plt.figure(figsize=(13, 7))
# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ks = [int(key) for key in data.keys()]

# Possibly add "avg_f1" and "avg_recall"
metrics = ["accuracy", "avg_precision"]

for metric in metrics:
    metric_list = []
    metric_std = []

    for key in data.keys():
        metric_list.append(np.mean(data[key][metric]))
        metric_std.append(np.std(data[key][metric]))

    label = ""
    if metric == "accuracy":
        label = "Average Accuracy"
    elif metric == "avg_f1":
        label = "Average F1"
    elif metric == "avg_precision":
        label = "Average Precision"
    elif metric == "avg_recall":
        label = "Average Recall"

    plt.errorbar(ks, metric_list, metric_std, marker="o", linestyle='--', capsize=5, label=label, alpha=0.7)

plt.legend()
plt.title(f'Accuracy and precision versus the kernel size of the parabolic structuring element.', fontdict={'fontsize': 15})

# Add x-label
plt.xlabel('Kernel Size')

# Add grid
plt.grid(True)

plt.savefig(f"figures/kernel_size_experiment_accuracy_and_precision.pdf", format="pdf", bbox_inches="tight")

plt.show()

plt.figure(figsize=(13, 7))

ks = [int(key) for key in data.keys()]

time_list = []
time_std = []

for key in data.keys():
    time_list.append(np.mean(data[key]["time"]))
    time_std.append(np.std(data[key]["time"]))

plt.errorbar(ks, time_list, time_std, marker="o", linestyle='--', capsize=5)
plt.title(f'Time taken to train in seconds versus the kernel size of the parabolic structuring element.', fontdict={'fontsize': 15})

# Add x-label and y-label
plt.xlabel('Kernel Size')
plt.ylabel('Time (s)')

# Add grid
plt.grid(True)

plt.savefig(f"figures/kernel_size_experiment_time.pdf", format="pdf", bbox_inches="tight")

plt.show()