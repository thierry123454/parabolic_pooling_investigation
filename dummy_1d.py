import math
import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/1d_dilation")

torch.manual_seed(0)  #  for repeatable results

# k_size oneven?
k_size = 41

# Input
f = torch.tensor([2*x for x in range(21)], dtype=torch.float32)
print(f'Input: {f}')

# Output
g = torch.tensor([4.0000,  6.0000,  8.0000, 10.0000, 12.0000, 14.0000, 16.0000, 18.0000,
        20.0000, 22.0000, 24.0000, 26.0000, 28.0000, 30.0000, 32.0000, 34.0000,
        36.0000, 37.7500, 39.0000, 39.7500, 40.0000])
print(f'Wanted output: {g}')

# scale
s = torch.tensor(0.95, dtype=torch.float32, requires_grad=True)

# print(f'Structuring function: {h}')

# implement 1D parabolic dilation
def parabolic_dilate_1D(input):
    # h(z) = -(||z||**2) / 4s
    z_i = torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)
    z = z_i ** 2
    h = -z / (4*s)

    out = torch.zeros_like(input)
    
    # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
    for x in range(len(input)):
        # print(f'Calculating for {x}:')
        max = 0

        # Loop over h
        for i in range(k_size):
            y = i - k_size // 2
            
            # Check bounds
            if (x - y >= 0 and x - y <= len(input) - 1):
                tmp = input[x-y] + h[i]
                # print(x, y, i)
                # print(tmp)
                if (tmp > max):
                    max = tmp
        # print(f'Final max: {max}')
        out[x] = max

    return out

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

print("Training:")
print(f'Initial output: {parabolic_dilate_1D(f)}')
for i in range(100):
    pred = parabolic_dilate_1D(f)
    loss = error(g, pred)
    if ((i + 1) % 10 == 0):
        print(f'Iteration {i + 1}:')
        print(f'Loss = {loss}')
        print(f'Output: {pred}')
        print(f'Scale = {s}')
        print()

    loss.backward()
    with torch.no_grad():
        s -= s.grad * 0.005
        s.grad.zero_()

# How does it learn?
