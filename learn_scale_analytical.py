import math
import torch
import torch.nn as nn
import numpy as np

# k_size oneven?
k_size = 41

# Input
f = torch.tensor([2*x for x in range(5)], dtype=torch.float32)
print(f'Input: {f}')

s = torch.tensor(0.95, dtype=torch.float32, requires_grad=True)

# Output
g = torch.tensor([4.0000, 5.7500, 7.0000, 7.7500, 8.0000])
print(f'Wanted output: {g}')

# Use simple MSE error
def error(y, y_pred):
    return torch.mean(((y_pred - y)**2))

print("Training:")
for i in range(10):
    # h(z) = -(||z||**2) / 4s
    z_i = torch.linspace(-k_size // 2 + 1, k_size // 2, k_size, dtype=torch.float32)
    z = z_i ** 2
    h = -z / (4*s)
    h.retain_grad()

    out = torch.zeros_like(f)
    missing = h.shape[0] - f.shape[0]
    padded = nn.functional.pad(f, (missing // 2 + 2, missing // 2 - 2), "constant", -float('inf'))

    # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
    for x in range(f.shape[0]):
        shifted = torch.roll(padded, -x)

        if x == 0:
            tmp2 = torch.add(shifted, h)
            maximum2 = torch.max(tmp2)
            print(f'y* = {torch.argmax(tmp2)}')
            out[x] = maximum2
        else:
            tmp = torch.add(shifted, h)
            maximum = torch.max(tmp)
            out[x] = maximum
            print(f'y* = {torch.argmax(tmp)}')
    out.retain_grad()
    tmp.retain_grad()
    tmp2.retain_grad()
    maximum.retain_grad()
    maximum2.retain_grad()
    
    loss = error(g, out)
    loss.backward()

    print(f'Iteration {i + 1}:')
    print(f'Loss = {loss}')
    print(f'Out grad = {out.grad}')
    print(f'Maximum grad for x0 = {maximum2.grad}')
    print(f'Tmp grad for x0 = {tmp2.grad}')
    print(f'H grad = {h.grad}')
    print(f'Gradient of scale by PyTorch: {s.grad}')

    with torch.no_grad():
        grad_h_to_scale = (z / (4*(s**2)))

        print(grad_h_to_scale)
        
        precise_grad_h = h.grad.to(torch.float64)
        
        precise_grad_scale = grad_h_to_scale.to(torch.float64)
        
        grad_analytical = torch.dot(precise_grad_h, precise_grad_scale)
        print(f"Analytically calculated gradient of scale: {'{:.30f}'.format(grad_analytical)}")

        # print(torch.sum(h.grad * grad_h_to_scale))

    print(f'Output: {out}')
    print(f'Scale: {s}')
    print()

    s = s - 0.05 * s.grad
    s.retain_grad()
    # s.grad = torch.tensor(0., dtype=torch.float32)

# Conclusion: The gradient of the output, is mapped back to the index in the addition which caused the maximum and directly
# propagated to H. The rest is easy as h = -z/(4*s) which is a matter of simple calculus. Also, using the analytically calculated
# formula for dH/ds leads to similar but not exactly the same results. Could be caused by some rounding issues or something similar.