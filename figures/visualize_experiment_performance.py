import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/performance_max_pool_vs_parabolic.json') as f:
    data = json.load(f)

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

models = ["Standard LeNet", "LeNet with Parabolic SE"]

fig, axs = plt.subplots(1, 2, figsize=(17, 9))

avg_accuracies = []
std_accuracies = []
avg_f1 = []
std_f1 = []
avg_precision = []
std_precision = []
avg_recall = []
std_recall = []
avg_time = []
std_time = []

for key in data.keys():
    accuracies = np.array(data[key]["accuracy"])
    avg_accuracies.append(np.mean(accuracies))
    std_accuracies.append(np.std(accuracies))
    f1 = np.array(data[key]["avg_f1"])
    avg_f1.append(np.mean(f1))
    std_f1.append(np.std(f1))
    precision = np.array(data[key]["avg_precision"])
    avg_precision.append(np.mean(precision))
    std_precision.append(np.std(precision))
    recall = np.array(data[key]["avg_recall"])
    avg_recall.append(np.mean(recall))
    std_recall.append(np.std(recall))
    time = np.array(data[key]["time"])
    avg_time.append(np.mean(time))
    std_time.append(np.std(time))

x = np.arange(2)

print(avg_accuracies)

bar_width = 0.2

axs[0].bar(x - bar_width*1.5, avg_accuracies, yerr=std_accuracies, capsize=4, width=bar_width, label='Avg. Accuracy')
axs[0].bar(x - bar_width/2, avg_recall, yerr=std_recall, capsize=4,  width=bar_width, label='Avg. Recall')
axs[0].bar(x + bar_width/2, avg_precision, yerr=std_precision, capsize=4,  width=bar_width, label='Avg. Precision')
axs[0].bar(x + bar_width*1.5, avg_f1, yerr=std_f1, capsize=4,  width=bar_width, label='Avg. F1')

# Set the x-axis tick positions and labels
axs[0].set_xticks(x)
axs[0].set_xticklabels(models)

# Set labels and title
axs[0].set_ylabel('Values')

axs[0].legend()
axs[0].set_ylim([0.944, 0.962])
axs[0].grid(True)

axs[1].bar(x, avg_time, yerr=std_time, capsize=4, width=bar_width, label='Avg. Accuracy')

# Set labels and title
axs[1].set_ylabel('Time (s)')

# Set the x-axis tick positions and labels
axs[1].set_xticks(x)
axs[1].set_xticklabels(models)
axs[1].grid(True)

fig.suptitle("Performance of LeNet with and without using a parabolic SE on the KMNIST dataset.", fontsize=30)
fig.tight_layout()

plt.savefig("figures/performance_max_pool_versus_parabolic.pdf", format="pdf", bbox_inches="tight")
plt.show()