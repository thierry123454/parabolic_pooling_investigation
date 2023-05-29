from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import MaxPool2d
from torch import flatten
import torch

torch.manual_seed(0)

class ParabolicPool1D(Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, device: int = 0):
        super(ParabolicPool1D , self).__init__()
        self.in_channels = in_channels
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
        print(f.shape)
        
        # Calculate (f dilate h)(x) = max{f(x-y) + h(y) for all y in h}
        for x in range(f.shape[0]):
            max = 0

            # Loop over h
            for i in range(self.ks):
                y = i - self.ks // 2
                
                # Check bounds
                if (x - y >= 0 and x - y <= f.shape[0] - 1):
                    tmp = f[x-y] + h[i]
                    if (tmp > max):
                        max = tmp
                        
            out[x] = max

        return torch.as_strided(out, size=(len(out)//self.stride,), stride=(self.stride,))

class MorphAudioModel(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(MorphAudioModel, self).__init__()
		self.conv1 = Conv1d(in_channels=numChannels, out_channels=20, kernel_size=5)
		self.relu1 = ReLU()
		self.pool1 = ParabolicPool1D(20, 11, 2)
		self.conv1 = Conv1d(in_channels=20, out_channels=50, kernel_size=5)
		self.relu2 = ReLU()
		self.pool2 = ParabolicPool1D(50, 11, 2)
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool2(x)

		x = self.fc1(x)
		x = self.relu3(x)

		x = self.fc2(x)
		output = self.logSoftmax(x)

		return output