import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/scale_learnability_experiment_standard.json') as f:
    data = json.load(f)

EPOCHS = 15

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create a figure and four subplots
fig, axs = plt.subplots(1, 2, figsize=(17, 9), sharex=True, sharey=True)

def plot_scales(axis, scale_key, data_dict, legend=False):
    for key in data.keys():
        min_scales = []
        max_scales = []
        mean_scales = []
        low_quartile = []
        up_quartile = []

        for scales in np.array(data_dict[key][scale_key]):
            min_scales.append(np.min(scales))
            max_scales.append(np.max(scales))
            mean_scales.append(np.mean(scales))
            low_quartile.append(np.percentile(scales, 25))
            up_quartile.append(np.percentile(scales, 75))

        x = range(1, EPOCHS + 1)

        # Create the area chart
        if legend:
            if key == "0.5":
                axis.fill_between(x, min_scales, max_scales, alpha=0.3, label="Area between min and max")
                # axis.fill_between(x, low_quartile, up_quartile, alpha=0.3, label="Area between Q1 \& Q3")
            else:
                axis.fill_between(x, min_scales, max_scales, alpha=0.3)
                # axis.fill_between(x, low_quartile, up_quartile, alpha=0.3)
            axis.plot(x, mean_scales, marker='o', label=f"Mean $s$, $s_0 = {key}$")  
        else:
            axis.fill_between(x, min_scales, max_scales, alpha=0.3)
            # axis.fill_between(x, low_quartile, up_quartile, alpha=0.3)
            axis.plot(x, mean_scales, marker='o')

    axis.grid()

# Plot on each subplot
axs[0].set_title('Pool 1')
plot_scales(axs[0], "scales_p1", data, True)

axs[1].set_title('Pool 2')
plot_scales(axs[1], "scales_p2", data)

# Add labels to shared axes
fig.text(0.5, 0.02, 'Epoch', ha='center', fontdict={'fontsize': 15})
fig.text(-0.01, 0.49, "$s$", va='center', rotation='vertical', fontdict={'fontsize': 20})

# Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.1, hspace=0.2)

fig.legend()
fig.suptitle("Scales learned on different epochs with varying starting scales using the standard SE.", fontsize=25)
fig.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.07)
plt.savefig("figures/learnability_experiment_standard.pdf", format="pdf", bbox_inches="tight")
plt.show()

with open('experiments/scale_learnability_experiment_normalized.json') as f:
    data = json.load(f)

# Create a figure and four subplots
fig, axs = plt.subplots(1, 2, figsize=(17, 9), sharex=True, sharey=True)

def plot_scales(axis, scale_key, data_dict, legend=False):
    for key in data.keys():
        min_scales = []
        max_scales = []
        mean_scales = []
        low_quartile = []
        up_quartile = []

        for scales in np.array(data_dict[key][scale_key]):
            min_scales.append(np.min(scales))
            max_scales.append(np.max(scales))
            mean_scales.append(np.mean(scales))
            low_quartile.append(np.percentile(scales, 25))
            up_quartile.append(np.percentile(scales, 75))

        x = range(1, EPOCHS + 1)

        # Create the area chart
        if legend:
            if key == "0.5":
                axis.fill_between(x, min_scales, max_scales, alpha=0.3, label="Area between min and max")
                # axis.fill_between(x, low_quartile, up_quartile, alpha=0.3, label="Area between Q1 \& Q3")
            else:
                axis.fill_between(x, min_scales, max_scales, alpha=0.3)
                # axis.fill_between(x, low_quartile, up_quartile, alpha=0.3)
            axis.plot(x, mean_scales, marker='o', label=f"Mean $s$, $s_0 = {key}$")  
        else:
            axis.fill_between(x, min_scales, max_scales, alpha=0.3)
            # axis.fill_between(x, low_quartile, up_quartile, alpha=0.3)
            axis.plot(x, mean_scales, marker='o')

    axis.grid()

# Plot on each subplot
axs[0].set_title('Pool 1')
plot_scales(axs[0], "scales_p1", data, True)

axs[1].set_title('Pool 2')
plot_scales(axs[1], "scales_p2", data)

# Add labels to shared axes
fig.text(0.5, 0.02, 'Epoch', ha='center', fontdict={'fontsize': 15})
fig.text(-0.01, 0.49, "$s$", va='center', rotation='vertical', fontdict={'fontsize': 20})

# Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.1, hspace=0.2)

fig.legend()
fig.suptitle("Scales learned on different epochs with varying starting scales using the MorphPool SE.", fontsize=25)
fig.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.07)
plt.savefig("figures/learnability_experiment_normalized.pdf", format="pdf", bbox_inches="tight")
plt.show()