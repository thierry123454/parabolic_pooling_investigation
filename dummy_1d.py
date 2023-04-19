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

# Define simple 1D dilation Neural Net
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
        z_i = torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)
        z = z_i ** 2
        h = -z / (4*self.scale)

        out = torch.zeros_like(input)
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        for x in range(input.shape[0]):
            # print(f'Calculating for {x}:')
            max = 0

            # Loop over h
            for i in range(k_size):
                y = i - k_size // 2
                
                # Check bounds
                if (x - y >= 0 and x - y <= input.shape[0] - 1):
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

model = Dilation1D(0.95)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

writer.add_graph(model, input_to_model=f)
writer.close()

print("Training:")
print(f'Initial output: {model(f)}')
for i in range(100):
    pred = model(f)
    loss = error(g, pred)
    if ((i + 1) % 10 == 0):
        print(f'Iteration {i + 1}:')
        print(f'Loss = {loss}')
        print(f'Output: {pred}')
        print()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()