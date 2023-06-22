import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def roll_with_padding(tensor, shift):
    if shift == 0:
        return tensor.clone()
    
    if shift > 0:
        padding = torch.full((shift,), -float('inf'), dtype=tensor.dtype)
        padded_tensor = torch.cat((padding, tensor), 0)
        rolled_tensor = torch.narrow(padded_tensor, 0, 0, len(tensor))
    else:
        padding = torch.full((-shift,), -float('inf'), dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding), 0)
        rolled_tensor = torch.narrow(padded_tensor, 0, -shift, len(tensor))
    
    return rolled_tensor

# k_size oneven?
k_size = 201

class Dilation1D(nn.Module):
    def __init__(self, s):
        super(Dilation1D, self).__init__()
        # scale
        scale = torch.tensor(s, dtype=torch.float32, requires_grad=True)
        self.scale = torch.nn.parameter.Parameter(scale, requires_grad=True)

    def forward(self, input=None):
        if input is None:
            raise ValueError("Input tensor must be provided")
        
        # h(z) = -(||z||**2) / 4s
        z_i =  torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)
        self.z = z_i ** 2
        self.h = -self.z / (4*self.scale)

        self.h.retain_grad()

        out = torch.zeros_like(input)
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        offset = input.shape[0] // 2
        for x in domain_x:
            x = int(x)
            shifted = roll_with_padding(input, -x)
            tmp = torch.add(shifted, self.h)
            out[x + offset] = torch.max(tmp)

        return out
    
domain_x = np.linspace(-100, 100, 201)
f = torch.tensor([math.sin(0.05*(x-40)) + math.cos(0.02*(x-40)) + 0.5 for x in domain_x], dtype=torch.float32)

domain_h_glob = np.linspace(-100, 100, 201)
z_i =  torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)
z = z_i ** 2
h = -z / (4*100)
h_T = z / (4*100) + 2

domain_h = domain_h_glob[70:-70]
domain_h_T = domain_h_glob[90:-50]
print(domain_h_T)
h = h[70:-70]
h_T = h_T[70:-70]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 7))
ax1.plot(domain_x, f, label='$f$')
ax1.plot(domain_h, h, label='$g$')
ax1.set_title('Initial Setup')
ax2.plot(domain_x, f, label='$f$')
ax2.plot(domain_h_T, h_T, label='$g^T$', color="green")
ax2.vlines(x = 20, ymin = 0, ymax = 2, color = 'k', linestyles='dashed')
ax2.text(17, -0.4, '$x$', fontsize=15)
ax2.set_title('Step 1')
ax3.plot(domain_x, f, label='$f$')
h_T = z / (4*100) + 0.759
h_T = h_T[70:-70]
ax3.plot(domain_h_T, h_T, label='$g^T$', color="green")
ax3.plot(20.18, 0.7591, 'ro', markersize=3) 
# ax3.vlines(x = 20.18, ymin = 0.7591, ymax = 1.5, color = 'k', linestyles='dashed')
ax3.plot([20.18, 40], [0.7591, 0.7591], color='k', linestyle='dashed')
ax3.text(42, 0.7, '$(f \oplus g)(x)$')
ax3.set_title('Step 2')
ax4.plot(domain_x, f, label='$f$')
ax4.plot(domain_h_T, h_T, label='$g^T$', color="green")
ax4.plot(20.18, 0.7591, 'ro', markersize=3) 
ax4.plot(domain_x, Dilation1D(100)(f).clone().detach(), color='purple', linestyle='dashed', label=r'$f \oplus g$')
ax4.set_title('Step 3')

for ax in fig.get_axes():
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5) 
    ax.set_axis_off()
    ax.legend()

plt.savefig("figures/dilation.pdf", format="pdf", bbox_inches="tight")
plt.show()