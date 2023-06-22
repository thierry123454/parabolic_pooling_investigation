import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/performance_CIFAR10.json') as f:
    data_cifar10 = json.load(f)


with open('experiments/performance_SVHN.json') as f:
    data_svhn = json.load(f)

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

models = ["MaxPool", "MorphPool Parabolic SE", "Standard Parabolic SE", "Standard Parabolic SE with SSI"]

# CIFAR-10
plt.figure(figsize=(13, 7))
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

for key in data_cifar10.keys():
    print(key)
    accuracies = np.array(data_cifar10[key]["accuracy"])
    avg_accuracies.append(np.mean(accuracies))
    std_accuracies.append(np.std(accuracies))
    f1 = np.array(data_cifar10[key]["avg_f1"])
    avg_f1.append(np.mean(f1))
    std_f1.append(np.std(f1))
    precision = np.array(data_cifar10[key]["avg_precision"])
    avg_precision.append(np.mean(precision))
    std_precision.append(np.std(precision))
    recall = np.array(data_cifar10[key]["avg_recall"])
    avg_recall.append(np.mean(recall))
    std_recall.append(np.std(recall))
    time = np.array(data_cifar10[key]["time"])
    avg_time.append(np.mean(time))
    std_time.append(np.std(time))

print(avg_time)
print(std_time)

x = np.arange(4)

bar_width = 0.2

error_kw=dict(ecolor='black', lw=2, capsize=4)

plt.bar(x - bar_width*1.5, avg_accuracies, yerr=std_accuracies, width=bar_width, label='Avg. Accuracy', hatch='/', error_kw=error_kw, alpha=.99)
plt.bar(x - bar_width/2, avg_recall, yerr=std_recall, width=bar_width, label='Avg. Recall', hatch='o', error_kw=error_kw, alpha=.99)
plt.bar(x + bar_width/2, avg_precision, yerr=std_precision, width=bar_width, label='Avg. Precision', hatch='.', error_kw=error_kw, alpha=.99)
plt.bar(x + bar_width*1.5, avg_f1, yerr=std_f1, width=bar_width, label='Avg. F1', hatch='x', error_kw=error_kw, alpha=.99)

# Set the x-axis tick positions and labels
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(models)

# Set labels and title
ax.set_ylabel('Values')
ax.set_ylim([0.8, 0.85])
plt.grid(True)

plt.legend()
plt.title("Performance of a CNN model using different pooling methods on the CIFAR-10 dataset.", fontdict={'fontsize': 15})
plt.tight_layout()
plt.savefig("figures/performance_cifar10.pdf", format="pdf", bbox_inches="tight")
plt.show()


plt.figure(figsize=(13, 7))
plt.bar(x, avg_time, yerr=std_time, capsize=4, width=bar_width)

ax = plt.gca()

# Set labels and title
ax.set_ylabel('Time (s)')

# Set the x-axis tick positions and labels
ax.set_xticks(x)
ax.set_xticklabels(models)

plt.grid(True)
plt.title("Time taken to train 10 epochs of a CNN model using different pooling methods on the CIFAR-10 dataset.", fontdict={'fontsize': 15})
plt.tight_layout()
plt.savefig("figures/time_svhn.pdf", format="pdf", bbox_inches="tight")
plt.show()

# SVHN
plt.figure(figsize=(13, 7))
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

for key in data_svhn.keys():
    print(key)
    accuracies = np.array(data_svhn[key]["accuracy"])
    avg_accuracies.append(np.mean(accuracies))
    std_accuracies.append(np.std(accuracies))
    f1 = np.array(data_svhn[key]["avg_f1"])
    avg_f1.append(np.mean(f1))
    std_f1.append(np.std(f1))
    precision = np.array(data_svhn[key]["avg_precision"])
    avg_precision.append(np.mean(precision))
    std_precision.append(np.std(precision))
    recall = np.array(data_svhn[key]["avg_recall"])
    avg_recall.append(np.mean(recall))
    std_recall.append(np.std(recall))
    time = np.array(data_svhn[key]["time"])
    avg_time.append(np.mean(time))
    std_time.append(np.std(time))

print(avg_time)
print(std_time)

x = np.arange(4)

bar_width = 0.2

error_kw=dict(ecolor='black', lw=2, capsize=4)

plt.bar(x - bar_width*1.5, avg_accuracies, yerr=std_accuracies, width=bar_width, label='Avg. Accuracy', hatch='/', error_kw=error_kw, alpha=.99)
plt.bar(x - bar_width/2, avg_recall, yerr=std_recall, width=bar_width, label='Avg. Recall', hatch='o', error_kw=error_kw, alpha=.99)
plt.bar(x + bar_width/2, avg_precision, yerr=std_precision, width=bar_width, label='Avg. Precision', hatch='.', error_kw=error_kw, alpha=.99)
plt.bar(x + bar_width*1.5, avg_f1, yerr=std_f1, width=bar_width, label='Avg. F1', hatch='x', error_kw=error_kw, alpha=.99)

# Set the x-axis tick positions and labels
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(models)

# Set labels and title
ax.set_ylabel('Values')
ax.set_ylim([0.9, 0.96])
plt.grid(True)

plt.legend()
plt.title("Performance of a CNN model using different pooling methods on the SVHN dataset.", fontdict={'fontsize': 15})
plt.tight_layout()
plt.savefig("figures/performance_svhn.pdf", format="pdf", bbox_inches="tight")
plt.show()


plt.figure(figsize=(13, 7))
plt.bar(x, avg_time, yerr=std_time, capsize=4, width=bar_width)

ax = plt.gca()

# Set labels and title
ax.set_ylabel('Time (s)')

# Set the x-axis tick positions and labels
ax.set_xticks(x)
ax.set_xticklabels(models)

plt.grid(True)
plt.title("Time taken to train 10 epochs of a CNN model using different pooling methods on the SVHN dataset.", fontdict={'fontsize': 15})
plt.tight_layout()
plt.savefig("figures/time_svhn.pdf", format="pdf", bbox_inches="tight")
plt.show()