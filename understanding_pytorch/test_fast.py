import torch
from torch.nn.functional import unfold
import numpy as np

f = torch.tensor([x for x in range(1,15)], dtype=torch.float32)

# Initialize the tensor
x = torch.tensor([3, 2, 1, 0, 1, 2, 3], dtype=torch.float32)

# # Initialize the target matrix with -inf
mat = torch.empty((f.shape[0], len(x)), dtype=torch.float32)

print(mat)

# print(mat.shape)

# print(f[:len(x) - 1])
# print(torch.tensor(float('-inf')))

# Place the values of the tensor on the diagonals of the matrix
N = mat.shape[0]
for r in range(N):
    if (r <= len(x) // 2):
        mat[r] = torch.cat((torch.tensor([float('-inf')] * (len(x) // 2 - r)), f[:len(x) - (len(x) // 2 - r)]))
    elif (r >= N - len(x) // 2):
        off = r - (N - len(x) // 2)
        mat[r] = torch.cat((f[N-len(x)+1+off:], torch.tensor([float('-inf')] * (off + 1))))
    else:    
        mat[r] = f[r - len(x) // 2 : r + len(x) // 2 + 1]
# Show the result
print(mat)