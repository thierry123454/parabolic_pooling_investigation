import torch
from torch.nn.functional import unfold
import numpy as np

f = torch.tensor([x for x in range(1,15)], dtype=torch.float32)

# Initialize the tensor
x = torch.tensor([3, 2, 1, 0, 1, 2, 3], dtype=torch.float32)

#  Initialize the matrix
mat = torch.empty((f.shape[0], len(x)), dtype=torch.float32)

padded = torch.cat((torch.tensor([float('-inf')] * (len(x) // 2)), f, torch.tensor([float('-inf')] * (len(x) // 2))))

# Place the values of the tensor on the diagonals of the matrix
# N = mat.shape[0]
# for r in range(N):
#     mat[r] = padded[r : r + len(x)]

mat = padded.unfold(0, len(x), 1)
    
# Show the result
print(mat)