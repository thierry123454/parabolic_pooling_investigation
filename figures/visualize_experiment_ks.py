import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/kernel_size_experiment_normalized.json') as f:
    data = json.load(f)

with open('experiments/kernel_size_experiment_standard.json') as f2:
    data_standard = json.load(f2)

print(data.keys())

fig, ax = plt.subplots(figsize=(10, 7))
# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

scales = []

for key in data.keys():
    scales.append(data_standard[key]["scales_p1"][0])

# Creating plot
bp = ax.boxplot(scales)

# Set custom x-axis labels
x_labels = data_standard.keys()
ax.set_xticklabels(x_labels)

plt.show()

fig, ax = plt.subplots(figsize=(10, 7))
# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

scales = []

for key in data.keys():
    scales.append(data_standard[key]["scales_p2"][0])

# Creating plot
bp = ax.boxplot(scales)

# Set custom x-axis labels
x_labels = data_standard.keys()
ax.set_xticklabels(x_labels)

plt.show()

fig, ax = plt.subplots(figsize=(10, 7))
# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

scales = []

for key in data.keys():
    scales.append(data[key]["scales_p1"][0])

# Creating plot
bp = ax.boxplot(scales)

# Set custom x-axis labels
x_labels = data.keys()
ax.set_xticklabels(x_labels)

plt.show()

fig, ax = plt.subplots(figsize=(10, 7))
# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

scales = []

for key in data.keys():
    scales.append(data[key]["scales_p2"][0])

# Creating plot
bp = ax.boxplot(scales)

# Set custom x-axis labels
x_labels = data.keys()
ax.set_xticklabels(x_labels)

plt.show()

# plt.figure(figsize=(13, 7))
# ks = [int(key) for key in data.keys()]

# # Possibly add "avg_f1" and "avg_recall"
# metrics = ["accuracy", "avg_precision"]

# for metric in metrics:
#     metric_list = []
#     metric_std = []

#     for key in data.keys():
#         metric_list.append(np.mean(data[key][metric]))
#         metric_std.append(np.std(data[key][metric]))

#     label = ""
#     if metric == "accuracy":
#         label = "Avg. Accuracy MP"
#     elif metric == "avg_f1":
#         label = "Avg. F1 MP"
#     elif metric == "avg_precision":
#         label = "Avg. Precision MP"
#     elif metric == "avg_recall":
#         label = "Avg. Recall MP"

#     plt.errorbar(ks, metric_list, metric_std, marker="o", linestyle='--', capsize=5, label=label, alpha=0.7)

# for metric in metrics:
#     metric_list = []
#     metric_std = []

#     for key in data_standard.keys():
#         metric_list.append(np.mean(data_standard[key][metric]))
#         metric_std.append(np.std(data_standard[key][metric]))

#     label = ""
#     if metric == "accuracy":
#         label = "Avg. Accuracy Standard"
#     elif metric == "avg_f1":
#         label = "Avg. F1 Standard"
#     elif metric == "avg_precision":
#         label = "Avg. Precision Standard"
#     elif metric == "avg_recall":
#         label = "Avg. Recall Standard"

#     plt.errorbar(ks, metric_list, metric_std, marker="o", linestyle='--', capsize=5, label=label, alpha=0.7)

# plt.legend()
# plt.title("Accuracy and precision versus the kernel size of the parabolic structuring element.", fontdict={'fontsize': 15})

# # Add x-label
# plt.xlabel("Kernel Size")

# # Add grid
# plt.grid(True)

# plt.savefig("figures/kernel_size_experiment_accuracy_and_precision.pdf", format="pdf", bbox_inches="tight")

# plt.show()

# plt.figure(figsize=(13, 7))

# ks = [int(key) for key in data.keys()]

# time_list = []
# time_std = []

# for key in data.keys():
#     time_list.append(np.mean(data[key]["time"]))
#     time_std.append(np.std(data[key]["time"]))

# plt.errorbar(ks, time_list, time_std, marker="o", linestyle="--", capsize=5, label="MP")

# time_list = []
# time_std = []

# for key in data_standard.keys():
#     time_list.append(np.mean(data_standard[key]["time"]))
#     time_std.append(np.std(data_standard[key]["time"]))

# plt.errorbar(ks, time_list, time_std, marker="o", linestyle="--", capsize=5, label="Standard")

# plt.legend()
# plt.title("Time taken to train in seconds versus the kernel size of the parabolic structuring element.", fontdict={"fontsize": 15})

# # Add x-label and y-label
# plt.xlabel("Kernel Size")
# plt.ylabel("Time (s)")

# # Add grid
# plt.grid(True)

# plt.savefig("figures/kernel_size_experiment_time.pdf", format="pdf", bbox_inches="tight")

# plt.show()