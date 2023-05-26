import torch

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

test = torch.tensor(
    [[1, 2, 3],
     [2, 3, 4],
     [4, 5, 6]], dtype=torch.float32
)

print(roll_with_padding_2d(test, -1, 0))