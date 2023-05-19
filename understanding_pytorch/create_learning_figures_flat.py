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
domain_x = np.linspace(-100, 100, 201)
f = torch.tensor([math.sin(0.1*x) + math.cos(0.05*x) for x in domain_x], dtype=torch.float32)

class FlatDilation1D(nn.Module):
    def __init__(self, s, alpha):
        super(FlatDilation1D, self).__init__()
        # scale
        scale = torch.tensor(s, dtype=torch.float32, requires_grad=True)
        self.scale = torch.nn.parameter.Parameter(scale, requires_grad=True)
        self.alpha = alpha
        self.z_i = torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)

    def forward(self, input=None):
        if input is None:
            raise ValueError("Input tensor must be provided")
        
        # h(z) = -(z/s)**alpha
        self.h = -(self.z_i / self.scale)**self.alpha

        # print(h)

        out = torch.zeros_like(input)
        missing = self.h.shape[0] - input.shape[0]
        padded = nn.functional.pad(input, (missing // 2 + 2, missing // 2 - 2), "constant", -float('inf'))
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        offset = input.shape[0] // 2
        for x in domain_x:
            x = int(x)
            shifted = roll_with_padding(input, -x)
            tmp = torch.add(shifted, self.h)
            max_value = torch.max(tmp)
            
            # with torch.no_grad():
            #     max_occurences = torch.eq(tmp, max_value)
            #     max_pos = torch.nonzero(max_occurences).squeeze() - offset
            #     if (max_pos.numel() != 1): print(f"Meerdere PoC {max_pos.numel()}")

            out[x + offset] = max_value

        return out

alpha = 2**4

t = 50
g = FlatDilation1D(t, alpha)(f).clone().detach()

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

iterations = {10, 50, 100, 200, 400}
n_iterations = 400

gradients = []

def train_and_plot(s):
    print(f"Training s = {s}:")
    model = FlatDilation1D(s, alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

    predictions = []
    struct_funs = []

    startTime = time.time()
    for i in range(n_iterations):
        pred = model(f)
        pred.retain_grad()
        loss = error(g, pred)
        if ((i + 1) in iterations):
            print(f'Iteration {i + 1}:')
            print(f'Loss = {loss}')
            predictions.append((pred.detach(), str(i + 1)))
            struct_funs.append(model.h.detach().numpy())

        optimizer.zero_grad()
        loss.backward()

        if ((i + 1) in iterations):
            print(f'Scale: {model.scale}')
            print(f'Gradient of scale: {model.scale.grad}')
            print()
        
        optimizer.step()

    endTime = time.time()
    print("Total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # Setup LaTeX
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot
    plt.figure(figsize=(13, 7))
    plt.plot(domain_x, f, 'bo', markersize=3)
    plt.plot(domain_x, f, label= '$f$')
    plt.plot(domain_x, g, label= '$f \oplus q^t$')

    for i in range(len(predictions)):
        plt.plot(domain_x, predictions[i][0], label='$f \oplus b^s$ on i = ' + predictions[i][1])
        plt.plot(model.z_i.detach().numpy(), struct_funs[i], label='$b^s$ on $i = $' + predictions[i][1])
    ax = plt.gca()
    ax.set_ylim([-4, 2])
    plt.legend()
    plt.title(f'Learning the scale with $t = {t}$ and the starting scale being {s}.', fontdict={'fontsize': 15})
    plt.show()

train_and_plot(2)
train_and_plot(150)