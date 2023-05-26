import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

torch.manual_seed(0)  #  for repeatable results

def roll_with_padding_2d(tensor, shift_x, shift_y):
    if shift_x == 0 and shift_y == 0:
        return tensor.clone()

    rolled_tensor = torch.clone(tensor)

    if shift_y > 0:
        padding = torch.full((shift_y, rolled_tensor.shape[1]), -float('inf'), dtype=tensor.dtype)
        padded_tensor = torch.cat((padding, rolled_tensor), 0)
        rolled_tensor = padded_tensor[:-shift_y]
    else:
        padding = torch.full((-shift_y, rolled_tensor.shape[1]), -float('inf'), dtype=tensor.dtype)
        padded_tensor = torch.cat((rolled_tensor, padding), 0)
        rolled_tensor = padded_tensor[-shift_y:]
        
    if shift_x > 0:
        padding = torch.full((rolled_tensor.shape[0], shift_x), -float('inf'), dtype=tensor.dtype)
        padded_tensor = torch.cat((padding, rolled_tensor), 1)
        rolled_tensor = padded_tensor[:, :-shift_x]
    else:
        padding = torch.full((rolled_tensor.shape[0], -shift_x), -float('inf'), dtype=tensor.dtype)
        padded_tensor = torch.cat((rolled_tensor, padding), 1)
        rolled_tensor = padded_tensor[:, -shift_x:]

    return rolled_tensor

# k_size oneven?
k_size = 101

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Input
dom_x = np.linspace(-50, 50, k_size)
dom_y = np.linspace(-50, 50, k_size)
X, Y = np.meshgrid(dom_x, dom_y)
f = np.sin(0.2 * np.sqrt(X**2 + Y**2)) + np.cos(0.3*np.sqrt(X**2 + Y**2))

class Dilation2D(nn.Module):
    def __init__(self, s, standard=True):
        super(Dilation2D, self).__init__()
        # scale
        scale = torch.tensor(s, dtype=torch.float32)
        self.scale = torch.nn.parameter.Parameter(scale, requires_grad=True)
        self.standard = standard

    def forward(self, input=None):
        if input is None:
            raise ValueError("Input tensor must be provided")
        
        # h(z) = -(||z||**2) / 4s
        z_i =  torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)
        self.z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2

        self.h = torch.zeros_like(self.z_c)

        if (self.standard):
            self.h = -self.z_c / (4*self.scale)
        else:
            self.z_c = self.z_c / self.z_c[0, 0]
            self.h =  -self.z_c * self.scale

        self.h.retain_grad()

        out = torch.zeros_like(input)
        
        # # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        offset = input.shape[0] // 2
        for x in dom_x:
            for y in dom_y:
                x = int(x)
                y = int(y)
                shifted = roll_with_padding_2d(input, -x, -y)
                tmp = torch.add(shifted, self.h)
                max_value = torch.max(tmp)
                out[x + offset][y + offset] = max_value

        return out

f = torch.from_numpy(f)
g = Dilation2D(100, True)(f).clone().detach()

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

# Create a figure and axes
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, f, cmap='viridis')
ax.plot_surface(X, Y, g, cmap='Accent')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("$f$", fontsize=40)

# Show the plot
plt.show()

# Create a figure and axes
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, g, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("$g$", fontsize=40)

# Show the plot
plt.show()

exit()

iterations = {10, 20, 50, 80, 100}
n_iterations = 100

def train(s, standard):
    print(f"Training s = {s} using standard formula = {standard}:")
    model = Dilation2D(s, standard)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

    predictions = []

    for i in range(n_iterations):
        pred = model(f)
        pred.retain_grad()
        loss = error(g, pred)
        if ((i + 1) in iterations):
            print(f'Iteration {i + 1}:')
            print(f'Loss = {loss}')
            predictions.append((pred.detach(), str(i + 1)))

        optimizer.zero_grad()
        loss.backward()

        if ((i + 1) in iterations):
            print(f'Scale: {model.scale}')
            print(f'Gradient of scale: {model.scale.grad}')
            print()

        optimizer.step()

    return model.h

h = train(50, True)

# Create a figure and axes
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, h, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("$q^s$ with standard formula", fontsize=40)

# Show the plot
plt.show()

h = train(50, False)

# Create a figure and axes
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, h, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("$q^s$ with MP formula", fontsize=40)

# Show the plot
plt.show()