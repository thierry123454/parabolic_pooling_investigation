import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from graphviz import Source

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/1d_dilation")

torch.manual_seed(0)  #  for repeatable results

# k_size oneven?
k_size = 41

# Input
f = torch.tensor([2*x for x in range(5)], dtype=torch.float32)
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

# Output
g = torch.tensor([4.0000, 5.7500, 7.0000, 7.7500, 8.0000], dtype=torch.float32) #Dilation1D(1)(f).clone().detach()
print(f'Wanted output: {g}')

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

model = Dilation1D(0.95)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

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

    optimizer.zero_grad()
    loss.backward()

    if ((i + 1) % 10 == 0):
        print(f'Gradient of scale: {model.scale}')
        print(f'Gradient of scale: {model.scale.grad}')
        print()

    optimizer.step()


# Constructing backpropagation graph
# G = graphviz.Digraph(comment="Backpropagation Graph")

# used_ids = {-1}

# def createDAG(fun):
#     global used_ids
#     id = max(used_ids) + 1
#     used_ids.add(id)

#     if not fun:
#         G.node(str(id), "None" + "_" + str(id))
#         return id

#     string = str(fun).split(" ")[0][1:] + "_" + str(id)
#     G.node(str(id), string)

#     # print(string)
#     # print(fun.next_functions)

#     for child_fun in fun.next_functions:
#         G.edge(str(id), str(createDAG(child_fun[0])))
    
#     return id


# createDAG(loss.grad_fn)

# s = Source(G.source, filename="gradient_cond", format="png")
# s.view()