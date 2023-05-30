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

        repeated_tensors = [input] * input.shape[0]
        input_repeat = torch.stack(repeated_tensors)
        
        h_matrix = torch.full((f.shape[0], f.shape[0]), float('-inf'))

        # Place the values of the tensor on the diagonals of the matrix
        N = h_matrix.shape[0]
        for i in range(N):
            end = i + len(h) // 2 + 1
            end = end if end <= N else N
            start = max(i - len(h) // 2, 0)
            
            end_vec = end - i - (len(h) // 2 + 1)
            start_vec = max(len(h) // 2 - i, 0)

            if (end_vec == 0):
                h_matrix[i, start:end] = h[start_vec:]
            else:
                h_matrix[i, start:end] = h[start_vec:end_vec]

        add_inp_h = input_repeat + h_matrix

        out, _ = torch.max(add_inp_h, dim=1)
        
        return out

# Output
g = Dilation1D(1)(f).clone().detach()
print(f'Wanted output: {g}')
exit()
# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

model = Dilation1D(0.95)

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