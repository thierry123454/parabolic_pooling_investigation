import torch

from torch.nn.functional import unfold
from torch.nn import Unfold

input = torch.randn(2, 5, 3, 4)

print(input)
print(unfold(input, kernel_size=(2,2)))


input = torch.randn(10)
print(input)
print(input.unfold(2, 1, 1))