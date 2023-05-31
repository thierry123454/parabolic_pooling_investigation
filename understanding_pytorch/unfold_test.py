import torch
import numpy as np

from torch.nn.functional import unfold
from torch.nn import Unfold

input = torch.randn(2, 5, 3, 4)

# print(input)
# print(unfold(input, kernel_size=(2,2)))


# input = torch.randn(10)
# print(input)
# print(input.unfold(2, 1, 1))

X = np.array([1,2,3,4,5,4,3,2,1]).reshape(1,1, -1)
X = torch.tensor(X, dtype=torch.float32)
print(X)
print(X.unfold(0, 1, 1))