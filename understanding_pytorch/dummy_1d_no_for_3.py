import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)  #  for repeatable results

# k_size oneven?
k_size = 11

# Input
f = torch.tensor([2*x for x in range(41)], dtype=torch.float32)
print(f'Input: {f}')

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

        # print((h == 0).nonzero(as_tuple=True)[0])

        out = torch.zeros_like(input)

        mat = torch.empty((input.shape[0], len(h)), dtype=torch.float32)

        N = mat.shape[0]
        for r in range(N):
            if (r <= len(h) // 2):
                mat[r] = torch.cat((torch.tensor([float('-inf')] * (len(h) // 2 - r)), input[:len(h) - (len(h) // 2 - r)]))
            elif (r >= N - len(h) // 2):
                off = r - (N - len(h) // 2)
                mat[r] = torch.cat((input[N-len(h)+1+off:], torch.tensor([float('-inf')] * (off + 1))))
            else:    
                mat[r] = input[r - len(h) // 2 : r + len(h) // 2 + 1]
        
        repeated_tensors = [h] * f.shape[0]
        h_matrix = torch.stack(repeated_tensors)

        add_inp_h = mat + h_matrix

        out, _ = torch.max(add_inp_h, dim=1)
        
        return out

# Output
g = Dilation1D(1)(f).clone().detach()
print(f'Wanted output: {g}')
# exit()
# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

model = Dilation1D(0.6)

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