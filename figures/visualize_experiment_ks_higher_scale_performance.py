import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/kernel_size_experiment_standard_with_higher_starting_scale.json') as f:
    data_std = json.load(f)

with open('experiments/kernel_size_experiment_mp_with_higher_starting_scale.json') as f:
    data_mp = json.load(f)

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(figsize=(13, 7))
ks = [int(key) for key in data_std.keys()]

# Possibly add "avg_f1" and "avg_recall"
metrics = ["accuracy", "avg_precision"]

for metric in metrics:
    metric_list = []
    metric_std = []

    for key in data_std.keys():
        metric_list.append(np.mean(data_std[key][metric]))
        metric_std.append(np.std(data_std[key][metric]))

    label = ""
    if metric == "accuracy":
        label = "Avg. Accuracy Std. SE"
    elif metric == "avg_f1":
        label = "Avg. F1 Std. SE"
    elif metric == "avg_precision":
        label = "Avg. Precision Std. SE"
    elif metric == "avg_recall":
        label = "Avg. Recall Std. SE"

    plt.errorbar(ks, metric_list, metric_std, marker="o", linestyle='--', capsize=5, label=label, alpha=0.7)

for metric in metrics:
    metric_list = []
    metric_std = []

    for key in data_mp.keys():
        metric_list.append(np.mean(data_mp[key][metric]))
        metric_std.append(np.std(data_mp[key][metric]))

    label = ""
    if metric == "accuracy":
        label = "Avg. Accuracy MP SE"
    elif metric == "avg_f1":
        label = "Avg. F1 MP SE"
    elif metric == "avg_precision":
        label = "Avg. Precision MP SE"
    elif metric == "avg_recall":
        label = "Avg. Recall MP SE"

    plt.errorbar(ks, metric_list, metric_std, marker="o", linestyle='--', capsize=5, label=label, alpha=0.7)


plt.legend()
plt.title("Accuracy and precision versus the window size with all starting scales being 3.", fontdict={'fontsize': 15})

# Add x-label
plt.xlabel("Window Size")

# Add grid
plt.grid(True)

plt.savefig("figures/kernel_size_experiment_accuracy_and_precision_higher_starting_scale_both.pdf", format="pdf", bbox_inches="tight")

plt.show()