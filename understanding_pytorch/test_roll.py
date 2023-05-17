import torch

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
        print(padded_tensor)
        rolled_tensor = torch.narrow(padded_tensor, 0, -shift, len(tensor))
    
    return rolled_tensor

t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
shift = -2
rolled_t = roll_with_padding(t, shift)
print(rolled_t)
print(torch.roll(t, shift))