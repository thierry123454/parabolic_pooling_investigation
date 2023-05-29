import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/kernel_size_experiment_mp_with_higher_starting_scale.json') as f:
    data = json.load(f)

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create a figure and four subplots
fig, axs = plt.subplots(1, 2, figsize=(17, 9), sharex=True, sharey=True)

def plot_scales(axis, scale_key, data_dict, legend=False):
    min_scales = []
    max_scales = []
    mean_scales = []
    low_quartile = []
    up_quartile = []

    for key in data.keys():
        scales = np.array(data_dict[key][scale_key]).flatten()
        min_scales.append(np.min(scales))
        max_scales.append(np.max(scales))
        mean_scales.append(np.mean(scales))
        low_quartile.append(np.percentile(scales, 25))
        up_quartile.append(np.percentile(scales, 75))

    x = data.keys()

    # Create the area chart
    if legend:
        axis.fill_between(x, min_scales, max_scales, alpha=0.3, label="Area between min and max")
        axis.fill_between(x, low_quartile, up_quartile, alpha=0.3, label="Area between Q1 \& Q3")
        axis.plot(x, mean_scales, marker='o', label="Mean $\sigma$")  
    else:
        axis.fill_between(x, min_scales, max_scales, alpha=0.3)
        axis.fill_between(x, low_quartile, up_quartile, alpha=0.3)
        axis.plot(x, mean_scales, marker='o')

    axis.grid()

# Plot on each subplot
axs[0].set_title('Pool 1')
plot_scales(axs[0], "scales_p1", data, True)

axs[1].set_title('Pool 2')
plot_scales(axs[1], "scales_p2", data)

# Add labels to shared axes
fig.text(0.5, 0.02, 'Window Size', ha='center', fontdict={'fontsize': 15})
fig.text(-0.01, 0.49, "$\sigma$", va='center', rotation='vertical', fontdict={'fontsize': 20})

# Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.1, hspace=0.2)

fig.legend()
fig.suptitle("Scales learned for different window sizes with $\sigma_s = 3$ using MP SE.", fontsize=30)
fig.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.07)
plt.savefig("figures/kernel_size_experiment_scales_higher_starting_scale_mp.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(13, 7))
ks = [int(key) for key in data.keys()]

# Possibly add "avg_f1" and "avg_recall"
metrics = ["accuracy", "avg_precision", "avg_f1", "avg_recall"]

for metric in metrics:
    metric_list = []
    metric_std = []

    for key in data.keys():
        metric_list.append(np.mean(data[key][metric]))
        metric_std.append(np.std(data[key][metric]))

    label = ""
    if metric == "accuracy":
        label = "Avg. Accuracy"
    elif metric == "avg_f1":
        label = "Avg. F1"
    elif metric == "avg_precision":
        label = "Avg. Precision"
    elif metric == "avg_recall":
        label = "Avg. Recall"

    plt.errorbar(ks, metric_list, metric_std, marker="o", linestyle='--', capsize=5, label=label, alpha=0.7)

plt.legend()
plt.title("Accuracy and precision versus the kernel size of the parabolic structuring element.", fontdict={'fontsize': 15})

# Add x-label
plt.xlabel("Kernel Size")

# Add grid
plt.grid(True)

plt.savefig("figures/kernel_size_experiment_accuracy_and_precision_higher_starting_scale_mp.pdf", format="pdf", bbox_inches="tight")

plt.show()

plt.figure(figsize=(13, 7))

ks = [int(key) for key in data.keys()]

time_list = []
time_std = []

for key in data.keys():
    time_list.append(np.mean(data[key]["time"]))
    time_std.append(np.std(data[key]["time"]))

plt.errorbar(ks, time_list, time_std, marker="o", linestyle="--", capsize=5)

plt.title("Time taken to train in seconds versus the kernel size of the standard parabolic structuring element.", fontdict={"fontsize": 15})

# Add x-label and y-label
plt.xlabel("Kernel Size")
plt.ylabel("Time (s)")

# Add grid
plt.grid(True)

plt.savefig("figures/kernel_size_experiment_time_higher_starting_scale_mp.pdf", format="pdf", bbox_inches="tight")

plt.show()