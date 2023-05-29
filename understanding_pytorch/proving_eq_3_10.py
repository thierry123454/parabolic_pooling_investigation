import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

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

# Input
domain_x = torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float64)
f = (domain_x)**2 / (4 * 20)

class Dilation1D(nn.Module):
    def __init__(self, s):
        super(Dilation1D, self).__init__()
        # scale
        scale = torch.tensor(s, dtype=torch.float64)
        self.scale = torch.nn.parameter.Parameter(scale, requires_grad=True)

    def forward(self, input=None):
        if input is None:
            raise ValueError("Input tensor must be provided")
        
        # h(z) = -(||z||**2) / 4s
        z_i =  torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float64)
        self.z = z_i ** 2
        self.h = -self.z / (4*self.scale)

        self.h.retain_grad()

        out = torch.zeros_like(input)
        self.man_grad_tensor = torch.zeros_like(input)
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        offset = input.shape[0] // 2
        for x in domain_x:
            x = int(x)
            shifted = roll_with_padding(input, -x)
            tmp = torch.add(shifted, self.h)
            max_value = torch.max(tmp)

            # Manually calculate gradient of d (f \oplus q^s) / d s using eq 3.10
            with torch.no_grad():
                max_occurences = torch.eq(tmp, max_value)
                max_pos = torch.nonzero(max_occurences).squeeze() - offset
                # print(max_pos)
                man_grad = 1 / max_pos.numel() * torch.sum(max_pos**2 / (4*self.scale**2))
                self.man_grad_tensor[x + offset] = man_grad
                # if (max_pos.numel() != 1):
                #     print(x)

            out[x + offset] = max_value

        return out

model = Dilation1D(20)
pred = model(f)

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot
plt.figure(figsize=(13, 7))
plt.plot(domain_x, f, 'bo', markersize=3)
plt.plot(domain_x, f, label= '$f$')
plt.plot(domain_x, pred.detach().numpy(), label= '$f \oplus q^s$')
plt.legend()
plt.title(f'$f$ and $f \oplus q^s$ with $s = 20$ and the domain being [-100, 100].', fontdict={'fontsize': 15})
plt.savefig(f"understanding_pytorch/dilate_many_poc.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Comparing analytical gradient and PyTorch gradient
pred.retain_grad()
pred.backward(torch.ones(k_size))
print(model.man_grad_tensor)
print(f'd out / d h: {model.h.grad}')
print(f'sum: {torch.sum(model.h.grad)}')
print(f'Gradient of PyTorch: {model.scale.grad}')
grad_h_to_scale = (model.z / (4*(model.scale**2)))
print(f'Eq 3.9: {torch.dot(model.h.grad, grad_h_to_scale)}')
print(f'Eq 3.10: {torch.sum(model.man_grad_tensor)}')
