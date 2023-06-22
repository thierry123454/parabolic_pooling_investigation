import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/performance_max_pool_vs_mp_vs_parabolic_vs_parabolic_scale_space.json') as f:
    data = json.load(f)

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

models = ["MaxPool", "MorphPool Parabolic SE", "Standard Parabolic SE", "Standard Parabolic SE with SSI"]

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
    print(key)
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

# print(avg_accuracies)

print(avg_time)

x = np.arange(4)

bar_width = 0.2
error_kw=dict(ecolor='black', lw=2, capsize=4)

axs[0].bar(x - bar_width*1.5, avg_accuracies, yerr=std_accuracies, width=bar_width, label='Avg. Accuracy', hatch='/', error_kw=error_kw, alpha=.99)
axs[0].bar(x - bar_width/2, avg_recall, yerr=std_recall, width=bar_width, label='Avg. Recall', hatch='o', error_kw=error_kw, alpha=.99)
axs[0].bar(x + bar_width/2, avg_precision, yerr=std_precision, width=bar_width, label='Avg. Precision', hatch='.', error_kw=error_kw, alpha=.99)
axs[0].bar(x + bar_width*1.5, avg_f1, yerr=std_f1, width=bar_width, label='Avg. F1', hatch='x', error_kw=error_kw, alpha=.99)

# Set the x-axis tick positions and labels
axs[0].set_xticks(x)
axs[0].set_xticklabels(models)

# Set labels and title
axs[0].set_ylabel('Values')

axs[0].legend()
axs[0].set_ylim([0.94, 0.965])
axs[0].grid(True)

axs[1].bar(x, avg_time, yerr=std_time, capsize=4, width=bar_width)

# Set labels and title
axs[1].set_ylabel('Time (s)')

# Set the x-axis tick positions and labels
axs[1].set_xticks(x)
axs[1].set_xticklabels(models)
axs[1].grid(True)

axs[0].set_title('Average accuracy, recall, F1, and precision of the models.')
axs[1].set_title('Time taken to train five epochs of the models.')
fig.suptitle("Performance of LeNet using different pooling methods on the KMNIST dataset.", fontsize=25)
fig.tight_layout()

plt.savefig("figures/performance_max_pool_versus_parabolic.pdf", format="pdf", bbox_inches="tight")
plt.show()