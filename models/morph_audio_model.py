from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import MaxPool2d
from torch import flatten
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

class ParabolicPool1D(Module):
	def __init__(self, in_channels, kernel_size=3, stride=2, device: int = 0):
		super(ParabolicPool1D , self).__init__()
		self.in_channels = in_channels
		self.ks = kernel_size
		self.stride = stride
		self.device = device

		t = torch.empty((in_channels, ))
		torch.nn.init.uniform_(t, a=0.0, b=4.0)
		self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

	def _compute_parabolic_kernel(self):
		z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
								dtype=torch.float32).to(torch.device(self.device))
		z_c = z_i ** 2
		z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
		return -z / (4 * self.t.view(-1, 1))

	def forward(self, f: torch.Tensor) -> torch.Tensor:
		h = self._compute_parabolic_kernel()
		out = torch.zeros_like(f)

		# Unfold: nn functional unfold
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}

		# TE TRAAG! Verbeter.
		for b in range(f.shape[0]):
			for c in range(f.shape[1]):
				# HAAL DEZE LOOP WEG!
				for x in range(f.shape[2]):
					max = 0

					# Loop over h
					for i in range(self.ks):
						y = i - self.ks // 2

						# Check bounds
						if (x - y >= 0 and x - y <= f.shape[0] - 1):
							tmp = f[b][c][x-y] + h[c][i]
							if (tmp > max):
								max = tmp

					out[b][c][x] = max
				
				# new_size = len(out[b][c]) // self.stride
				# out[b][c] = torch.as_strided(out[b][c], size=(new_size,), stride=(self.stride,))
		out = torch.as_strided(out, size=(out.shape[0],out.shape[1],out.shape[2] // self.stride), stride=(1, 1, self.stride))
		return out
	
class ParabolicPool1DFast(Module):
	def __init__(self, in_channels, kernel_size=3, stride=2, device: int = 0):
		super(ParabolicPool1DFast , self).__init__()
		self.in_channels = in_channels
		self.ks = kernel_size
		self.stride = stride
		self.device = device

		t = torch.empty((in_channels, ))
		torch.nn.init.uniform_(t, a=10000.0, b=20000.0)
		self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

	def _compute_parabolic_kernel(self):
		z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
								dtype=torch.float32).to(torch.device(self.device))
		z_c = z_i ** 2
		z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
		return -z / (4 * self.t.view(-1, 1))

	def forward(self, f: torch.Tensor) -> torch.Tensor:
		h = self._compute_parabolic_kernel()

		out = torch.zeros_like(f)
		padding = torch.tensor([float('-inf')] * (len(h[0]) // 2)).to(torch.device(self.device))
		input_matrix = torch.empty((f.shape[2], len(h[0])), dtype=torch.float32).to(torch.device(self.device))

        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
		for b in range(f.shape[0]):
			for c in range(f.shape[1]):
				kernel = h[c]
				input_signal = f[b][c]

				padded = torch.cat((padding, input_signal, padding))

				# Place the values of the tensor on the diagonals of the matrix
				input_matrix = padded.unfold(0, len(kernel), 1)

				# print(f"Signal for batch {b} and channel {c}.")
				# print(kernel)
				# print(input_signal)
				# print(f"unique: {torch.unique(input_signal)}")

				add_inp_h = input_matrix + kernel
				output, _ = torch.max(add_inp_h, dim=1)
				# print(output)
				out[b][c] = output
		out = torch.as_strided(out, size=(out.shape[0], out.shape[1], out.shape[2] // self.stride), stride=(1, 1, self.stride))
		return out

class MorphAudioModel(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(MorphAudioModel, self).__init__()
		self.pool1 = ParabolicPool1DFast(numChannels, 2001, 2)
		self.conv1 = Conv1d(in_channels=numChannels, out_channels=3, kernel_size=2001)
		self.relu1 = ReLU()
		self.pool2 = ParabolicPool1DFast(3, 2001, 2)
		self.conv2 = Conv1d(in_channels=3, out_channels=5, kernel_size=2001)
		self.relu2 = ReLU()
		self.pool3 = ParabolicPool1DFast(5, 2001, 2)
		self.fc1 = Linear(in_features=2500, out_features=500)
		self.relu3 = ReLU()
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		x = self.pool1(x)
		# plt.plot((x.detach().cpu())[0][0])
		# plt.show()

		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool2(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool3(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)

		x = self.fc2(x)
		output = self.logSoftmax(x)

		return output