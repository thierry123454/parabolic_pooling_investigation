import torch

from .pooling_autograd import MaxPool2DAutogradFunction, MinPool2DAutogradFunction, \
    ParameterizedMaxPool2DAutogradFunction, ParameterizedMinPool2DAutogradFunction


class Pool2D(torch.nn.Module):

    def __init__(self, kernel_size: int, stride: int = 2, device: int = 0) -> None:
        super(Pool2D, self).__init__()
        self.ks = kernel_size
        self.stride = stride
        self.device = device

    def extra_repr(self) -> str:
        return f'kernel_size={self.ks}, stride={self.stride}'


class MaxPool2D(Pool2D):

    def __init__(self, kernel_size: int, stride: int = 2, device: int = 0) -> None:
        super(MaxPool2D, self).__init__(kernel_size, stride, device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return MaxPool2DAutogradFunction.apply(f, self.ks, self.stride, self.device)


class MinPool2D(Pool2D):

    def __init__(self, kernel_size: int, stride: int = 2, device: int = 0) -> None:
        super(MinPool2D, self).__init__(kernel_size, stride, device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return MinPool2DAutogradFunction.apply(f, self.ks, self.stride, self.device)


class ParameterizedPool2D(Pool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(ParameterizedPool2D, self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        h = torch.empty((in_channels, kernel_size, kernel_size))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)


class ParameterizedMaxPool2D(ParameterizedPool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(ParameterizedMaxPool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return ParameterizedMaxPool2DAutogradFunction.apply(f, self.h, self.stride, self.device)


class ParameterizedMinPool2D(ParameterizedPool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(ParameterizedMinPool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return ParameterizedMinPool2DAutogradFunction.apply(f, self.h, self.stride, self.device)

class LearnableFlatPool(Pool2D):
    def __init__(self, in_channels, kernel_size=3, stride=2, init='uniform', ss=3.0, alpha=16, device: int = 0):
        super(LearnableFlatPool , self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        
        t = torch.empty((in_channels, ))
        if init == 'uniform':
            torch.nn.init.uniform_(t, a=0.0, b=4.0)
        elif init == 'ones':
            t = torch.full((in_channels, ), 1.0)
        elif init == 'manual':
            t = torch.full((in_channels, ), ss)
        else:
            t = torch.full((in_channels, ), 3.0)

        self.print = True
        self.alpha = alpha
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2

        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)

        # if (self.print):
        #   kernels = -(z / self.t.view(-1, 1, 1))**self.alpha
        #   print("First Kernel")
        #   print(kernels[0])
        #   print("Last Kernel")
        #   print(kernels[self.in_channels - 1])
        #   self.print = False

        # Create the parabolic kernels.
        return -(z / self.t.view(-1, 1, 1))**self.alpha

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        return ParameterizedMaxPool2DAutogradFunction.apply(f, h, self.stride, self.device)

class ParabolicPool2D(Pool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', ss=0.5, device: int = 0):
        super(ParabolicPool2D, self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        t = torch.empty((in_channels, )).to(torch.device(self.device))
        if init == 'zero':
            # Init to make the centre 0 (it always is, and the corner elements -1).
            torch.nn.init.zeros_(t)
        elif init == 'manual':
            t = torch.full((in_channels, ), ss)
        else:
            torch.nn.init.kaiming_uniform_(t)
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        # Normalize, such that the corner elements are 1.
        z_c = z_c / z_c[0, 0]
        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
        # Create the parabolic kernels.
        return - z * self.t.view(-1, 1, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        
        return ParameterizedMaxPool2DAutogradFunction.apply(f, h, self.stride, self.device)

class ParabolicPool2D_V2(Pool2D):
    def __init__(self, in_channels, kernel_size=3, stride=2, init='uniform', ss=0.5, device: int = 0):
        super(ParabolicPool2D_V2 , self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        
        t = torch.empty((in_channels, ))
        if init == 'uniform':
            torch.nn.init.uniform_(t, a=0.0, b=4.0)
        elif init == 'ones':
            t = torch.full((in_channels, ), 1.0)
        elif init == 'scale_space':
            # Assuming min value = 0 and max = 1
    
            # t_min = 1 / 4 makes it so that the element next to the middle 
            # has a weight factor of -1, preserving all detail. 
            t_min = 1 / 4 

            # t_max has a value so that the corner elements are equal to -1,
            # smoothing the input maximally.
            t_max = (self.ks // 2)**2 / 2

            t = torch.linspace(t_min, t_max, in_channels)
        elif init == 'manual':
            t = torch.full((in_channels, ), ss)
        else:
            t = torch.full((in_channels, ), 0.5)

        self.print = True
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2

        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)

        # if (self.print):
        #   kernels = -z / (4 * self.t.view(-1, 1, 1))
        #   print("First Kernel")
        #   print(kernels[0])
        #   print("Last Kernel")
        #   print(kernels[self.in_channels - 1])
        #   self.print = False

        # Create the parabolic kernels.
        return -z / (4 * self.t.view(-1, 1, 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        return ParameterizedMaxPool2DAutogradFunction.apply(f, h, self.stride, self.device)

class ParabolicPool2D_SS(Pool2D):
    def __init__(self, in_channels, kernel_size=3, stride=2, init=0.5, device: int = 0):
        super(ParabolicPool2D_SS , self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.

        self.print = True

        t = torch.tensor(init, dtype=torch.float32, requires_grad=True)
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2

        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)

        # if (self.print):
        #   kernels = -z / (4 * self.t.view(-1, 1, 1))
        #   print("First Kernel")
        #   print(kernels[0])
        #   print("Last Kernel")
        #   print(kernels[self.in_channels - 1])
        #   self.print = False

        # Create the parabolic kernels.
        return -z / (4 * self.t)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        return ParameterizedMaxPool2DAutogradFunction.apply(f, h, self.stride, self.device)

class ParabolicPool2D_TL(Pool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(ParabolicPool2D_TL, self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        t = torch.empty((in_channels, )).to(torch.device(self.device))
        if init == 'zero':
            # Init to make the centre 0 (it always is, and the corner elements -1).
            torch.nn.init.zeros_(t)
        else:
            torch.nn.init.kaiming_uniform_(t)
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        # Normalize, such that the corner elements are 1.
        z_c = z_c / z_c[0, 0]
        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
        # Create the parabolic kernels.
        return - z * self.t.view(-1, 1, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        
        output, _, _ = ParameterizedMaxPool2DAutogradFunction.apply(f, h, self.stride, self.device)
        return output
    
class ParabolicPool2D_V2_TL(Pool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='uniform', device: int = 0):
        super(ParabolicPool2D_V2_TL , self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        
        t = torch.empty((in_channels, ))
        if init == 'uniform':
            torch.nn.init.uniform_(t, a=0.0, b=4.0)
        elif init == 'ones':
            t = torch.full((in_channels, ), 1.0)
        elif init == 'scale_space':
            # Assuming min value = 0 and max = 1
    
            # t_min = 1 / 4 makes it so that the element next to the middle 
            # has a weight factor of -1, preserving all detail. 
            t_min = 1 / 4 

            # t_max has a value so that the corner elements are equal to -1,
            # smoothing the input maximally.
            t_max = (self.ks // 2)**2 / 2

            t = torch.linspace(t_min, t_max, in_channels)
            print("Scale space!")
        else:
            t = torch.full((in_channels, ), 0.5)

        self.t = torch.nn.parameter.Parameter(t.to(torch.device(self.device)), requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2

        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
        # Create the parabolic kernels.
        return -z / (4 * self.t.view(-1, 1, 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = self._compute_parabolic_kernel()
        output, _, _ = ParameterizedMaxPool2DAutogradFunction.apply(f, h, self.stride, self.device)
        return output

class ParabolicMaxPool2D(ParabolicPool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(ParabolicMaxPool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # Compute the parabolic kernel.
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        return ParameterizedMaxPool2DAutogradFunction.apply(f, h, self.stride, self.device)


class ParabolicMinPool2D(ParabolicPool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(ParabolicMinPool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # Compute the parabolic kernel.
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        return ParameterizedMinPool2DAutogradFunction.apply(f, h, self.stride, self.device)
