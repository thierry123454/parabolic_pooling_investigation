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

class Dilation1D(nn.Module):
    def __init__(self, s):
        super(Dilation1D, self).__init__()
        # scale
        scale = torch.tensor(s, dtype=torch.float32)
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
        self.man_grad_tensor = torch.zeros_like(input)
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        offset = input.shape[0] // 2
        for x in domain_x:
            x = int(x)
            shifted = roll_with_padding(input, -x)
            tmp = torch.add(shifted, self.h)
            max_value = torch.max(tmp)

            # Manually calculate gradient using eq 3.10
            with torch.no_grad():
                max_occurences = torch.eq(tmp, max_value)
                max_pos = torch.nonzero(max_occurences).squeeze() - offset
                man_grad = 1 / max_pos.numel() * torch.sum(max_pos**2 / (4*self.scale**2))
                self.man_grad_tensor[x + offset] = man_grad

            out[x + offset] = max_value

        return out

g = Dilation1D(50)(f).clone().detach()

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

iterations = {10, 50, 100, 200, 400}
n_iterations = 400

gradients = []

def train_and_plot(s):
    print(f"Training s = {s}:")
    model = Dilation1D(s)
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
        
        with torch.no_grad():
            grad_h_to_scale = (model.z / (4*(model.scale**2)))
            gradient_tuple = (model.scale.grad, torch.dot(model.h.grad, grad_h_to_scale), torch.dot(pred.grad, model.man_grad_tensor))
            gradients.append(gradient_tuple)

        optimizer.step()

    # Setup LaTeX
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot
    plt.figure(figsize=(13, 7))
    plt.plot(domain_x, f, 'bo', markersize=3)
    plt.plot(domain_x, f, label= '$f$')
    plt.plot(domain_x, g, label= '$f \oplus q^t$')

    for prediction in predictions:
        plt.plot(domain_x, prediction[0], label= '$f \oplus q^s$ on i = ' + prediction[1])

    plt.legend()
    plt.title(f'Learning the scale with $t = 50$ and the starting scale being {s}.', fontdict={'fontsize': 15})
    plt.savefig(f"learning_scale_from_{s}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

train_and_plot(2)
train_and_plot(150)

# Comparing analytical gradient and PyTorch gradient

# First using equation 3.9
diff = []
percentage_diff = []
for gradient in gradients:
    difference = abs(gradient[0] - gradient[1])
    diff.append(difference)
    if (gradient[0] != 0):
        percentage_diff.append(100 * difference / abs(gradient[0]))

print(f'Average difference eq 3.9: {sum(diff) / len(diff)}') # 6.960307330494686e-11
print(f'Average percentage difference eq 3.9: {sum(percentage_diff) / len(percentage_diff)}') # 4.930320301355096e-06 %

# Then using equation 3.10
diff = []
percentage_diff = []
for gradient in gradients:
    difference = abs(gradient[0] - gradient[2])
    diff.append(difference)
    if (gradient[0] != 0):
        percentage_diff.append(100 * difference / abs(gradient[0]))

print(f'Average difference eq 3.10: {sum(diff) / len(diff)}') # 9.038868264976685e-11
print(f'Average percentage difference eq 3.10: {sum(percentage_diff) / len(percentage_diff)}') # 6.047777787898667e-06 %
