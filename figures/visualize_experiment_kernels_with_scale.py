import matplotlib.pyplot as plt
import numpy as np
import json

with open('experiments/scale_experiment_conv_kernels_standard.json') as f:
    data = json.load(f)

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

num_channels = 20
img_scales = 4

fig, axs = plt.subplots(img_scales, num_channels, figsize=(19, 8))
for img_scale in range(img_scales):
    data_scale = data[str(img_scale + 1)]
    conv_weights = data_scale["conv_weights"]

    # Normalize weights of convolutional kernels
    conv_weights_zero_min = conv_weights - np.min(conv_weights)
    norm_weights = conv_weights_zero_min / np.max(conv_weights_zero_min)
    for c in range(num_channels):
        scale = data_scale["scales_p1"][c]
        kernel = norm_weights[c][0]
        axs[img_scale, c].imshow(kernel, cmap='gray')
        axs[img_scale, c].set_title(f"{scale:.4f}")

        # Remove tick markers
        axs[img_scale, c].set_xticks([])
        axs[img_scale, c].set_yticks([])

    # Add row title
    # fig.text(0.5, 0.5 - (img_scale - 1) * 0.22, f"Scaling Factor {img_scale + 1}", va='center', rotation='horizontal', fontdict={'fontsize': 16})

# Add labels to shared axes
fig.suptitle("Scales assocated with their respective kernel in the first pooling layer for different image scales.", fontsize=30)
fig.tight_layout()

# Decrease hspace to reduce the vertical distance between rows of subplots
# plt.subplots_adjust(hspace=0)

plt.savefig("figures/experiment_scales_with_kernels.pdf", format="pdf", bbox_inches="tight")
plt.show()