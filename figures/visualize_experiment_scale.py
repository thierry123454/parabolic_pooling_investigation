import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/scale_experiment_normalized.json') as f:
    data = json.load(f)

with open('experiments/scale_experiment_standard.json') as f2:
    data_standard = json.load(f2)

# Filter Data
for key in data.keys():
    accuracies = np.array(data[key]["accuracies"])
    ind_stay = np.where(accuracies != 0.1)[0]

    if (len(ind_stay) <= 3):
        for inner_key in data[key].keys():
            data[key][inner_key] = np.array(data[key][inner_key])[ind_stay]

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create a figure and four subplots
# fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

# def plot_scales(axis, scale_key, data_dict, legend=False):
#     min_scales = []
#     max_scales = []
#     mean_scales = []
#     low_quartile = []
#     up_quartile = []

#     for key in data.keys():
#         scales = np.array(data_dict[key][scale_key]).flatten()
#         min_scales.append(np.min(scales))
#         max_scales.append(np.max(scales))
#         mean_scales.append(np.mean(scales))
#         low_quartile.append(np.percentile(scales, 25))
#         up_quartile.append(np.percentile(scales, 75))

#     x = data_standard.keys()

#     # Create the area chart
#     if legend:
#         axis.fill_between(x, min_scales, max_scales, alpha=0.3, label="Area between min and max")
#         axis.fill_between(x, low_quartile, up_quartile, alpha=0.3, label="Area between Q1 \& Q3")
#         axis.plot(x, mean_scales, marker='o', label="Mean $\sigma$")
#     else:
#         axis.fill_between(x, min_scales, max_scales, alpha=0.3)
#         axis.fill_between(x, low_quartile, up_quartile, alpha=0.3)
#         axis.plot(x, mean_scales, marker='o')

#     axis.grid()

# # Plot on each subplot
# axs[0, 0].set_title('Standard SE / Pool 1')
# plot_scales(axs[0, 0], "scales_p1", data_standard, True)

# axs[0, 1].set_title('Standard SE / Pool 2')
# plot_scales(axs[0, 1], "scales_p2", data_standard)

# axs[1, 0].set_title('MP SE / Pool 1')
# plot_scales(axs[1, 0], "scales_p1", data)

# axs[1, 1].set_title('MP SE / Pool 2')
# plot_scales(axs[1, 1], "scales_p2", data)

# # Add labels to shared axes
# fig.text(0.5, 0.02, 'Scaling Factor', ha='center', fontdict={'fontsize': 15})
# fig.text(0, 0.49, "$\sigma$", va='center', rotation='vertical', fontdict={'fontsize': 20})

# fig.legend()
# fig.suptitle("Scales learned when scaling the input images with a certain factor.", fontsize=20)
# fig.tight_layout()
# plt.subplots_adjust(top=0.9, bottom=0.07)
# plt.savefig("figures/kernel_size_experiment_scales.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(13, 7))
# ks = [int(key) for key in data.keys()]

# avg_accuracies = []
# std_accuracies = []

# for key in data.keys():
#     accuracies = np.array(data[key]["accuracies"])
#     avg_accuracies.append(np.mean(accuracies))
#     std_accuracies.append(np.std(accuracies))

# plt.errorbar(ks, avg_accuracies, std_accuracies, marker="o", linestyle='--', capsize=5, label="MP SE", alpha=0.7)

# avg_accuracies = []
# std_accuracies = []

# for key in data.keys():
#     accuracies = np.array(data_standard[key]["accuracies"])
#     avg_accuracies.append(np.mean(accuracies))
#     std_accuracies.append(np.std(accuracies))

# plt.errorbar(ks, avg_accuracies, std_accuracies, marker="o", linestyle='--', capsize=5, label="Standard SE", alpha=0.7)

# plt.legend()
# plt.title("Accuracy versus the scaling factor of the input images for the standard SE and the MP SE.", fontdict={'fontsize': 15})

# # Add x-label
# plt.xlabel("Scaling Factor")
# plt.ylabel("Average Accuracy")

# # Add grid
# plt.grid(True)

# plt.show()

# Filter Data
with open('experiments/scale_experiment_normalized.json') as f:
    data = json.load(f)

indices_stay = set([0, 1, 2, 3])

for key in data.keys():
    accuracies = np.array(data[key]["accuracies"])
    ind_stay = np.where(accuracies != 0.1)[0]
    indices_stay = indices_stay.intersection(set(ind_stay))

    for inner_key in data[key].keys():
        data[key][inner_key] = np.array(data[key][inner_key])[ind_stay]

usable = list(indices_stay)

for key in data.keys():
    for inner_key in data[key].keys():
        data[key][inner_key] = np.array(data[key][inner_key])[usable]

# Create a figure and four subplots
fig, axs = plt.subplots(2, 2, figsize=(17, 7), sharex=True)

def plot_fraction(axis, scale_key, data_dict, legend=False):
    scales_factor_1 = np.array(data_dict["1"][scale_key]).flatten()
    
    avg_factors = []
    std_factors = []

    for key in data.keys():
        scales = np.array(data_dict[key][scale_key]).flatten()
        factors = scales / scales_factor_1
        avg_factors.append(np.mean(factors))
        std_factors.append(np.std(factors))

    x = ["$S_1 / S_1$", "$S_2 / S_1$", "$S_3 / S_1$", "$S_4 / S_1$"]

    # Add horizontal lines
    y_values = [1, 2, 3, 4]
    bar_width = 0.7  # Adjust the bar width
    for i, y in enumerate(y_values, start=1):
        # print(i)
        middle = (1+2*(i-1)) / (len(x)*2)
        xmin = middle - 0.1
        xmax = middle + 0.11
        axis.axhline(y=y, xmin=xmin, xmax=xmax, color='gray', linestyle='--')
        # axis.axhline(y=y, xmin=1/(len(x)*2), xmax=1, color='gray', linestyle='--')

    axis.bar(x, avg_factors, width=bar_width)
    # axis.bar(x, y_values, width=bar_width, color="green", alpha=0.4)


    # axis.errorbar(x, avg_factors, yerr=std_factors, fmt='o', color='red')

    axis.grid()

# Plot on each subplot
axs[0, 0].set_title('Standard SE / Pool 1')
plot_fraction(axs[0, 0], "scales_p1", data_standard, True)

axs[0, 1].set_title('Standard SE / Pool 2')
plot_fraction(axs[0, 1], "scales_p2", data_standard)

axs[1, 0].set_title('MP SE / Pool 1')
plot_fraction(axs[1, 0], "scales_p1", data)

axs[1, 1].set_title('MP SE / Pool 2')
plot_fraction(axs[1, 1], "scales_p2", data)

fig.text(0.005, 0.5, "Fraction", va='center', rotation='vertical', fontdict={'fontsize': 20})

fig.legend()
fig.suptitle("Scales learned when scaling the input images with a certain factor.", fontsize=20)
fig.tight_layout()
plt.subplots_adjust(left=0.04)
plt.savefig("figures/kernel_size_experiment_scales.pdf", format="pdf", bbox_inches="tight")
plt.show()