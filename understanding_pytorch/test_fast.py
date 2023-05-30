import torch
import numpy as np

f = torch.tensor([2*x for x in range(10)], dtype=torch.float32)

# Initialize the tensor
x = torch.tensor([2, 1, 0, 1, 2], dtype=torch.float32)

# Initialize the target matrix with -inf
mat = torch.full((f.shape[0], f.shape[0]), float('-inf'))

print(mat.shape)

# Place the values of the tensor on the diagonals of the matrix
N = mat.shape[0]
for i in range(N):
    end = i + len(x) // 2 + 1
    end = end if end <= N else N
    start = max(i - len(x) // 2, 0)
    
    end_vec = end - i - (len(x) // 2 + 1)
    start_vec = max(len(x) // 2 - i, 0)

    if (end_vec == 0):
        mat[i, start:end] = x[start_vec:]
    else:
        mat[i, start:end] = x[start_vec:end_vec]

# Show the result
print(mat)