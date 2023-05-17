import math
import torch
import torch.nn as nn
import numpy as np
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

# Input

domain_x = np.linspace(-100, 100, 201)
f = torch.tensor([math.sin(0.1*x) + math.cos(0.05*x) for x in domain_x], dtype=torch.float32)
# f = torch.tensor([math.sin(0.05*x) + math.cos(0.5*x) for x in domain_x], dtype=torch.float32)

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

g = Dilation1D(50)(f).clone().detach()

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

# Learning scale from below
model = Dilation1D(2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

predictions = []
iterations = {10, 50, 100, 200, 400}
n_iterations = 400

gradients = []

print("Training:")
print(f'Initial output: {model(f)}')
for i in range(n_iterations):
    pred = model(f)
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

    grad_h_to_scale = (model.z / (4*(model.scale**2)))
    gradients.append((model.scale.grad, torch.dot(model.h.grad, grad_h_to_scale)))

    optimizer.step()

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot
plt.figure(figsize=(13, 7))
plt.plot(domain_x, f, 'bo', markersize=3)
plt.plot(domain_x, f, label= '$f$')
# plt.plot(domain_x, g, 'bo', markersize=3)
plt.plot(domain_x, g, label= '$f \oplus q^t$')

for prediction in predictions:
    plt.plot(domain_x, prediction[0], label= '$f \oplus q^s$ on i = ' + prediction[1])

plt.legend()
plt.title('Learning the scale with $t = 50$ and the starting scale being 2.', fontdict={'fontsize': 15})
plt.savefig("learning_scale_below.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Learning scale from above
model = Dilation1D(150)

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

predictions = []

print("Training:")
print(f'Initial output: {model(f)}')
for i in range(n_iterations):
    pred = model(f)
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

    grad_h_to_scale = (model.z / (4*(model.scale**2)))
    gradients.append((model.scale.grad, torch.dot(model.h.grad, grad_h_to_scale)))

    optimizer.step()

# Plot
plt.figure(figsize=(13, 7))
plt.plot(domain_x, f, 'bo', markersize=3)
plt.plot(domain_x, f, label= '$f$')
# plt.plot(domain_x, g, 'bo', markersize=3)
plt.plot(domain_x, g, label= '$f \oplus q^t$')

for prediction in predictions:
    plt.plot(domain_x, prediction[0], label= '$f \oplus q^s$ on i = ' + prediction[1])

plt.legend()
plt.title('Learning the scale with $t = 50$ and the starting scale being 150.', fontdict={'fontsize': 15})
plt.savefig("learning_scale_above.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Comparing analytical gradient and PyTorch gradient
diff = []
percentage_diff = []
for gradient in gradients:
    difference = abs(gradient[0] - gradient[1])
    diff.append(difference)
    if (gradient[0] != 0):
        percentage_diff.append(100 * difference / abs(gradient[0]))

print(f'Average difference: {sum(diff) / len(diff)}') # 3.7157277255062127e-10
print(f'Average percentage difference: {sum(percentage_diff) / len(percentage_diff)}') # 4.387165517982794e-06 %
