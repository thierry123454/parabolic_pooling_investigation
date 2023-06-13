import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/kernel_size_experiment_normalized.json') as f:
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

        scales_translated = 2 * (int(key) // 2)**2 / (4*scales)

        min_scales.append(np.min(scales_translated))
        max_scales.append(np.max(scales_translated))
        mean_scales.append(np.mean(scales_translated))
        low_quartile.append(np.percentile(scales_translated, 25))
        up_quartile.append(np.percentile(scales_translated, 75))

    x = data.keys()

    print(max_scales)

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
axs[1].set_ylim([0, 500])

# Add labels to shared axes
fig.text(0.5, 0.02, 'Window Size', ha='center', fontdict={'fontsize': 15})
fig.text(0.01, 0.49, "$s$", va='center', rotation='vertical', fontdict={'fontsize': 25})

# Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.1, hspace=0.2)

fig.legend()
fig.suptitle("Scales learned by the MP SE when translated to equivalent scales in the standard SE.", fontsize=20)
fig.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.07, left=0.05)
plt.savefig("figures/kernel_size_experiment_scales_translated.pdf", format="pdf", bbox_inches="tight")
plt.show()
