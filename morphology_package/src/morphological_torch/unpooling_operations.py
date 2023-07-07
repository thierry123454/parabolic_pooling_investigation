import torch

from .pooling_autograd import MaxPool2DAutogradFunction, MinPool2DAutogradFunction, \
    ParameterizedMaxPool2DAutogradFunction, ParameterizedMinPool2DAutogradFunction
from .unpooling_autograd import SparseUnpool2DAutogradFunction, MaxUnpool2DAutogradFunction, \
    DoubleMaxUnpool2DAutogradFunction, MinUnpool2DAutogradFunction, DoubleMinUnpool2DAutogradFunction, \
    ParameterizedMaxUnpool2DAutogradFunction, ParameterizedDoubleMaxUnpool2DAutogradFunction, \
    ParameterizedMinUnpool2DAutogradFunction, ParameterizedDoubleMinUnpool2DAutogradFunction


class SparseUnpool2D(torch.nn.Module):
    """
        This is the regular unpooling by pooling, like the nn.Unpool2D that torch implements,
        but now with my provenance registration.
    """

    def __init__(self, kernel_size, stride=2, device: int = 0) -> None:
        super(SparseUnpool2D, self).__init__()
        self.ks = kernel_size
        self.stride = stride
        self.device = device

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Input the features, the provenances from the previous pool, and the desired size.
        """
        return SparseUnpool2DAutogradFunction.apply(f, provenance, size, self.ks, self.stride, self.device)

    def extra_repr(self) -> str:
        return f'kernel_size={self.ks}, stride={self.stride}'


class Unpool2D(torch.nn.Module):
    """
        Unparameterized Morphological unpooling, that is unpooling, but then also dilate the up-sampled
        signal with a flat structuring element using either max or min pooling.
    """

    def __init__(self, kernel_size, times='single', stride=2, device: int = 0) -> None:
        super(Unpool2D, self).__init__()
        self.unpool_ks = kernel_size
        self.stride = stride
        self.device = device
        # Now establish which interpolation scheme is used, for which one of two options is possible:
        # - single: Interpolate morphologically once, at a bigger kernel size.
        # - double: Interpolate morphologically twice, at the same kernel size.
        if times == 'single':
            self.interpolate_ks = (kernel_size * 2) - 1
        elif times == 'double':
            self.interpolate_ks = kernel_size
        else:
            raise NotImplementedError("Either interpolate once or twice, using [single | double]")

    def extra_repr(self) -> str:
        return f'unpool kernel_size={self.unpool_ks}, interpolation kernel_size={self.interpolate_ks}, ' \
               f'stride={self.stride}'


class MaxUnpool2D(Unpool2D):
    """
        Unparameterized Morphological unpooling, that is max unpooling, but then also dilate the up-sampled
        signal with a flat structuring element.
    """

    def __init__(self, kernel_size, times='single', stride=2, device: int = 0) -> None:
        super(MaxUnpool2D, self).__init__(kernel_size, times, stride, device)
        # Depending on the interpolation scheme, determine which autograd function to use.
        if times == 'single':
            self.autograd_func = MaxUnpool2DAutogradFunction
        else:
            self.autograd_func = DoubleMaxUnpool2DAutogradFunction

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Input the features, the provenances from the previous pool, and the desired size.
        """
        return self.autograd_func.apply(f, provenance, size, self.unpool_ks,
                                        self.interpolate_ks, self.stride, self.device)


class MinUnpool2D(Unpool2D):
    """
        Unparameterized Morphological unpooling, that is max unpooling, but then also dilate the up-sampled
        signal with a flat structuring element.
    """

    def __init__(self, kernel_size, times='single', stride=2, device: int = 0) -> None:
        super(MinUnpool2D, self).__init__(kernel_size, times, stride, device)
        # Depending on the interpolation scheme, determine which autograd function to use.
        if times == 'single':
            self.autograd_func = MinUnpool2DAutogradFunction
        else:
            self.autograd_func = DoubleMinUnpool2DAutogradFunction

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Input the features, the provenances from the previous pool, and the desired size.
        """
        return self.autograd_func.apply(f, provenance, size, self.unpool_ks,
                                        self.interpolate_ks, self.stride, self.device)


class ParameterizedUnpool2D(Unpool2D):
    """
        Parameterized Morphological unpooling, that is max unpooling, but then also dilate the up-sampled
        signal with a free structuring element.
    """

    def __init__(self, in_channels, kernel_size, times='single', stride=2, init='zero', device: int = 0) -> None:
        super(ParameterizedUnpool2D, self).__init__(kernel_size, times, stride, device)
        self.in_channels = in_channels
        h = torch.empty((in_channels, self.interpolate_ks, self.interpolate_ks))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels},unpool kernel_size={self.unpool_ks}, ' \
               f'interpolation kernel_size={self.interpolate_ks},stride={self.stride}'


class ParameterizedMaxUnpool2D(ParameterizedUnpool2D):

    def __init__(self, in_channels, kernel_size, times='single', stride=2, init='zero', device: int = 0) -> None:
        super(ParameterizedMaxUnpool2D, self).__init__(in_channels, kernel_size, times, stride, init, device)
        # Depending on the interpolation scheme, determine which autograd function to use.
        if times == 'single':
            self.autograd_func = ParameterizedMaxUnpool2DAutogradFunction
        else:
            self.autograd_func = ParameterizedDoubleMaxUnpool2DAutogradFunction

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        return self.autograd_func.apply(f, self.h, provenance, size, self.unpool_ks,
                                        self.interpolate_ks, self.stride, self.device)


class ParameterizedMinUnpool2D(ParameterizedUnpool2D):

    def __init__(self, in_channels, kernel_size, times='single', stride=2, init='zero', device: int = 0) -> None:
        super(ParameterizedMinUnpool2D, self).__init__(in_channels, kernel_size, times, stride, init, device)
        # Depending on the interpolation scheme, determine which autograd function to use.
        if times == 'single':
            self.autograd_func = ParameterizedMinUnpool2DAutogradFunction
        else:
            self.autograd_func = ParameterizedDoubleMinUnpool2DAutogradFunction

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        return self.autograd_func.apply(f, self.h, provenance, size, self.unpool_ks,
                                        self.interpolate_ks, self.stride, self.device)


class ParabolicUnpool2D(Unpool2D):
    """
        Parabolic Morphological unpooling, that is max unpooling, but then also dilate the up-sampled
        signal with a parabolic structuring element.
    """

    def __init__(self, in_channels, kernel_size, times='single', stride=2, init='zero', device: int = 0) -> None:
        super(ParabolicUnpool2D, self).__init__(kernel_size, times, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        t = torch.empty((in_channels,))
        if init == 'zero':
            # Init to make the centre 0 (it always is, and the corner elements -1).
            torch.nn.init.zeros_(t)
        else:
            torch.nn.init.kaiming_uniform_(t)
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.interpolate_ks // 2 + 1, self.interpolate_ks // 2, self.interpolate_ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        # Normalize, such that the corner elements are 1.
        z_c = z_c / z_c[0, 0]
        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
        # Create the parabolic kernels.
        return - z * self.t.view(-1, 1, 1)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels},unpool kernel_size={self.unpool_ks}, ' \
               f'interpolation kernel_size={self.interpolate_ks},stride={self.stride}'


class ParabolicMaxUnpool2D(ParabolicUnpool2D):

    def __init__(self, in_channels, kernel_size, times='single', stride=2, init='zero', device: int = 0) -> None:
        super(ParabolicMaxUnpool2D, self).__init__(in_channels, kernel_size, times, stride, init, device)
        # Depending on the interpolation scheme, determine which autograd function to use.
        if times == 'single':
            self.autograd_func = ParameterizedMaxUnpool2DAutogradFunction
        else:
            self.autograd_func = ParameterizedDoubleMaxUnpool2DAutogradFunction

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        # Compute the parabolic kernel.
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        return self.autograd_func.apply(f, h, provenance, size, self.unpool_ks, self.interpolate_ks,
                                        self.stride, self.device)


class ParabolicMinUnpool2D(ParabolicUnpool2D):

    def __init__(self, in_channels, kernel_size, times='single', stride=2, init='zero', device: int = 0) -> None:
        super(ParabolicMinUnpool2D, self).__init__(in_channels, kernel_size, times, stride, init, device)
        # Depending on the interpolation scheme, determine which autograd function to use.
        if times == 'single':
            self.autograd_func = ParameterizedMinUnpool2DAutogradFunction
        else:
            self.autograd_func = ParameterizedDoubleMinUnpool2DAutogradFunction

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        # Compute the parabolic kernel.
        h = self._compute_parabolic_kernel()
        # And return the standard CUDA dilation.
        return self.autograd_func.apply(f, h, provenance, size, self.unpool_ks, self.interpolate_ks,
                                        self.stride, self.device)


class NonProvenanceUnpool2D(torch.nn.Module):
    """
        Unparameterized unpooling, by simply placing back elements in a strided fashion.
        No provenance is needed.
    """

    def __init__(self, kernel_size, stride=2, device: int = 0) -> None:
        super(NonProvenanceUnpool2D, self).__init__()
        self.stride = stride
        self.interpolate_ks = kernel_size
        self.device = device

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple, fill_value=-100.) -> torch.Tensor:
        h, w = size
        b, c = f.shape[0], f.shape[1]
        upsampled_ins = torch.full((b, c, h, w), dtype=torch.float32, requires_grad=True,
                                   fill_value=fill_value).to(torch.device(self.device))
        upsampled_ins[:, :, ::self.stride, ::self.stride] = f
        return upsampled_ins

    def extra_repr(self) -> str:
        return f'interpolation kernel_size={self.interpolate_ks}, stride={self.stride}'


class NonProvenanceMaxUnpool2D(NonProvenanceUnpool2D):

    def __init__(self, kernel_size, stride=2, device: int = 0) -> None:
        super(NonProvenanceMaxUnpool2D, self).__init__(kernel_size, stride, device)

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Input the features, the provenances from the previous pool, and the desired size.
        """
        upsampled_ins = super(NonProvenanceMaxUnpool2D, self).forward(f, provenance, size, -100.)
        return MaxPool2DAutogradFunction.apply(upsampled_ins, self.interpolate_ks, 1, self.device)[0]


class NonProvenanceMinUnpool2D(NonProvenanceUnpool2D):

    def __init__(self, kernel_size, stride=2, device: int = 0) -> None:
        super(NonProvenanceMinUnpool2D, self).__init__(kernel_size, stride, device)

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Input the features, the provenances from the previous pool, and the desired size.
        """
        upsampled_ins = super(NonProvenanceMinUnpool2D, self).forward(f, provenance, size, 100.)
        return MinPool2DAutogradFunction.apply(upsampled_ins, self.interpolate_ks, 1, self.device)[0]


class ParameterizedNonProvenanceUnpool2D(NonProvenanceUnpool2D):

    def __init__(self, in_channels, kernel_size, stride=2, init='zero', device: int = 0) -> None:
        super(ParameterizedNonProvenanceUnpool2D, self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        h = torch.empty((in_channels, self.interpolate_ks, self.interpolate_ks))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels},interpolation kernel_size={self.interpolate_ks},stride={self.stride}'


class ParameterizedNonProvenanceMaxUnpool2D(ParameterizedNonProvenanceUnpool2D):

    def __init__(self, in_channels, kernel_size, stride=2, init='zero', device: int = 0) -> None:
        super(ParameterizedNonProvenanceMaxUnpool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        upsampled_ins = super(ParameterizedNonProvenanceMaxUnpool2D, self).forward(f, provenance, size, -100.)
        return ParameterizedMaxPool2DAutogradFunction.apply(upsampled_ins, self.h, 1, self.device)[0]


class ParameterizedNonProvenanceMinUnpool2D(ParameterizedNonProvenanceUnpool2D):

    def __init__(self, in_channels, kernel_size, stride=2, init='zero', device: int = 0) -> None:
        super(ParameterizedNonProvenanceMinUnpool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        upsampled_ins = super(ParameterizedNonProvenanceMinUnpool2D, self).forward(f, provenance, size, 100.)
        return ParameterizedMinPool2DAutogradFunction.apply(upsampled_ins, self.h, 1, self.device)[0]


class ParabolicNonProvenanceUnpool2D(NonProvenanceUnpool2D):

    def __init__(self, in_channels, kernel_size, stride=2, init='zero', device: int = 0) -> None:
        super(ParabolicNonProvenanceUnpool2D, self).__init__(kernel_size, stride, device)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        t = torch.empty((in_channels,))
        if init == 'zero':
            # Init to make the centre 0 (it always is, and the corner elements -1).
            torch.nn.init.zeros_(t)
        else:
            torch.nn.init.kaiming_uniform_(t)
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)

    def _compute_parabolic_kernel(self):
        z_i = torch.linspace(-self.interpolate_ks // 2 + 1, self.interpolate_ks // 2, self.interpolate_ks,
                             dtype=torch.float32).to(torch.device(self.device))
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        # Normalize, such that the corner elements are 1.
        z_c = z_c / z_c[0, 0]
        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
        # Create the parabolic kernels.
        return - z * self.t.view(-1, 1, 1)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels},interpolation kernel_size={self.interpolate_ks},stride={self.stride}'


class ParabolicNonProvenanceMaxUnpool2D(ParabolicNonProvenanceUnpool2D):

    def __init__(self, in_channels, kernel_size, stride=2, init='zero', device: int = 0) -> None:
        super(ParabolicNonProvenanceMaxUnpool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        upsampled_ins = super(ParabolicNonProvenanceMaxUnpool2D, self).forward(f, provenance, size, -100.)
        h = super(ParabolicNonProvenanceMaxUnpool2D, self)._compute_parabolic_kernel()
        return ParameterizedMaxPool2DAutogradFunction.apply(upsampled_ins, h, 1, self.device)[0]


class ParabolicNonProvenanceMinUnpool2D(ParabolicNonProvenanceUnpool2D):

    def __init__(self, in_channels, kernel_size, stride=2, init='zero', device: int = 0) -> None:
        super(ParabolicNonProvenanceMinUnpool2D, self).__init__(in_channels, kernel_size, stride, init, device)

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        upsampled_ins = super(ParabolicNonProvenanceMinUnpool2D, self).forward(f, provenance, size, 100.)
        h = super(ParabolicNonProvenanceMinUnpool2D, self)._compute_parabolic_kernel()
        return ParameterizedMinPool2DAutogradFunction.apply(upsampled_ins, h, 1, self.device)[0]
