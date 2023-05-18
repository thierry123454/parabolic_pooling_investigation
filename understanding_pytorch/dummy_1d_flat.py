import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from graphviz import Source

torch.manual_seed(0)  #  for repeatable results

# k_size oneven?
k_size = 41

# Input
f = torch.tensor([2*x for x in range(5)], dtype=torch.float32)
print(f'Input: {f}')

# sin(2t) + cos(4t)

# Define simple 1D dilation Neural Net
class FlatDilation1D(nn.Module):
    def __init__(self, s):
        super(FlatDilation1D, self).__init__()
        # scale
        scale = torch.tensor(s, dtype=torch.float32, requires_grad=True)
        self.scale = torch.nn.parameter.Parameter(scale, requires_grad=True)

    def forward(self, input=None):
        if input is None:
            raise ValueError("Input tensor must be provided")
        
        # h(z) = -(||z||**2) / 4s
        z_i = torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)
        # print(z_i)
        # print(self.scale)
        h = torch.full_like(z_i, -float('inf'))
        mask = torch.abs(z_i) <= self.scale
        h[mask] = 0
        # print(h)

        # print((h == 0).nonzero(as_tuple=True)[0])

        out = torch.zeros_like(input)
        missing = h.shape[0] - input.shape[0]
        padded = nn.functional.pad(input, (missing // 2 + 2, missing // 2 - 2), "constant", -float('inf'))
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        for x in range(input.shape[0]):
            # print(f'Calculating for {x}:')
            # print(h)
            shifted = torch.roll(padded, -x)
            # print((shifted == 0).nonzero(as_tuple=True)[0])
            # print(shifted)
            tmp = torch.add(shifted, h)
            # print(tmp)
            out[x] = torch.max(tmp)

        return out

# Output
g = FlatDilation1D(3)(f).clone().detach()
print(f'Wanted output: {g}')

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

model = FlatDilation1D(1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print("Training:")
print(f'Initial output: {model(f)}')
for i in range(100):
    pred = model(f)
    loss = error(g, pred)
    if ((i + 1) % 10 == 0):
        print(f'Iteration {i + 1}:')
        print(f'Loss = {loss}')
        print(f'Output: {pred}')

    optimizer.zero_grad()
    loss.backward()

    if ((i + 1) % 10 == 0):
        print(f'Scale: {model.scale}')
        print(f'Gradient of scale: {model.scale.grad}')
        print()

    optimizer.step()