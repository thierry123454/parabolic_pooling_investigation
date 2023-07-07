import torch
from .config import INF


class BaselinePool2D(torch.nn.Module):

    def __init__(self, kernel_size=3, stride=2):
        super(BaselinePool2D, self).__init__()
        self.ks = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        # Make sure to pad to achieve the following:
        # - With stride == 1, input_shape = output_shape
        # - With stride == 2, output_shape = ceil(input_shape / 2)
        if self.ks % 2 == 0:
            self.pad_with = (0, self.ks // 2, 0, self.ks // 2)
        else:
            self.pad_with = (self.ks // 2, self.ks // 2, self.ks // 2, self.ks // 2)

    def forward(self, xs: torch.Tensor) -> tuple:
        padded_xs = torch.nn.functional.pad(xs, self.pad_with, value=-INF)
        pooled_xs, ind = self.pool(padded_xs)
        return pooled_xs, ind


class BaselineParameterizedPool2D(BaselinePool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero'):
        super(BaselineParameterizedPool2D, self).__init__(kernel_size, stride)
        self.in_channels = in_channels
        h = torch.empty((1, kernel_size ** 2, in_channels))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)

    def _pad(self, xs):
        return torch.nn.functional.pad(xs, self.pad_with, value=-INF)

    def forward(self, xs: torch.Tensor) -> tuple:
        _, _, original_h, original_w = xs.shape
        padded_xs = self._pad(xs)
        b, _, h, w = padded_xs.shape
        pooled_xs = torch.empty(xs.shape, dtype=torch.float32, device=xs.device)
        provenances_xs = torch.empty(xs.shape, dtype=torch.int64, device=xs.device)
        for c in range(self.in_channels):
            unfolded_channel = torch.nn.Unfold(self.ks)(padded_xs[:, c, :, :].view(b, 1, h, w))
            added_channel = unfolded_channel + self.h[:, :, c].view(1, self.ks ** 2, 1)
            maxes, provenances = torch.max(added_channel, dim=1)
            maxes, provenances = maxes.view(b, original_h, original_w), provenances.view(b, original_h, original_w)
            pooled_xs[:, c, :, :], provenances_xs[:, c, :, :] = maxes, provenances
        return pooled_xs[:, :, ::self.stride, ::self.stride], provenances_xs[:, :, ::self.stride, ::self.stride]


class BaselineMaxUnpool2D(torch.nn.Module):

    def __init__(self, input_kernel_size, stride=2):
        super().__init__()
        self.in_ks = input_kernel_size
        self.unpool = torch.nn.MaxUnpool2d(input_kernel_size, stride=stride)

    def _unpad(self, xs: torch.Tensor) -> torch.Tensor:
        unpad = self.in_ks // 2
        if self.in_ks % 2 == 1:
            return xs[:, :, unpad:-unpad, unpad:-unpad]
        return xs[:, :, :-unpad, :-unpad]

    def _unfold(self, xs: torch.Tensor) -> torch.Tensor:
        b, c_in, h, w = xs.shape
        pad = self.out_ks // 2
        padded_xs = torch.nn.functional.pad(xs, (pad, pad, pad, pad), value=-INF)
        return self.unfold_func(padded_xs).view(b, c_in, self.out_ks ** 2, h * w)

    def forward(self, x, indices, size=None):
        # If padding was used in forward pooling, we need to provide the sizes to unpooling explicitly.
        if self.in_ks % 2 == 1:
            up_size = (size[0] + self.in_ks - 1, size[1] + self.in_ks - 1)
        # Unpooling into the same size we put in, except for padding.
        else:
            up_size = (size[0] + 1, size[1] + 1)
        x_upsampled = self.unpool(x, indices, up_size)
        return self._unpad(x_upsampled)


class BaselineUnpool2D(torch.nn.Module):

    def __init__(self, input_kernel_size, stride=2):
        super().__init__()
        self.in_ks = input_kernel_size
        self.out_ks = (input_kernel_size * 2) - 1
        self.unpool = torch.nn.MaxUnpool2d(input_kernel_size, stride=stride)
        self.unfold_func = torch.nn.Unfold(self.out_ks)

    def _unpad(self, xs: torch.Tensor) -> torch.Tensor:
        unpad = self.in_ks // 2
        if self.in_ks % 2 == 1:
            return xs[:, :, unpad:-unpad, unpad:-unpad]
        return xs[:, :, :-unpad, :-unpad]

    def _unfold(self, xs: torch.Tensor) -> torch.Tensor:
        b, c_in, h, w = xs.shape
        pad = self.out_ks // 2
        padded_xs = torch.nn.functional.pad(xs, (pad, pad, pad, pad), value=-INF)
        return self.unfold_func(padded_xs).view(b, c_in, self.out_ks ** 2, h * w)

    def forward(self, x, indices, size=None):
        # If padding was used in forward pooling, we need to provide the sizes to unpooling explicitly.
        if self.in_ks % 2 == 1:
            up_size = (size[0] + self.in_ks - 1, size[1] + self.in_ks - 1)
        # Unpooling into the same size we put in, except for padding.
        else:
            up_size = (size[0] + 1, size[1] + 1)
        x_upsampled = self.unpool(x, indices, up_size)
        x_upsampled = self._unpad(x_upsampled)
        # Now unfold and do an unparameterized dilation.
        desired_shape = x_upsampled.shape
        x_unfolded = self._unfold(x_upsampled)
        up, _ = torch.max(x_unfolded, dim=2)
        return up.view(*desired_shape)


class BaselineParameterizedUnpool2D(BaselineUnpool2D):

    def __init__(self, in_channels, input_kernel_size, stride=2, init='zero'):
        super(BaselineParameterizedUnpool2D, self).__init__(input_kernel_size, stride=stride)
        self.in_channels = in_channels
        h = torch.empty((1, in_channels, self.out_ks ** 2, 1))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        h = torch.arange(0, in_channels * self.out_ks ** 2, dtype=torch.float32).view((1, in_channels, self.out_ks ** 2, 1))
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)

    def forward(self, x, indices, size=None):
        # If padding was used in forward pooling, we need to provide the sizes to unpooling explicitly.
        if self.in_ks % 2 == 1:
            up_size = (size[0] + self.in_ks - 1, size[1] + self.in_ks - 1)
        # Unpooling into the same size we put in, except for padding.
        else:
            up_size = (size[0] + 1, size[1] + 1)
        x_upsampled = self.unpool(x, indices, up_size)
        x_upsampled = torch.where(x_upsampled == torch.tensor([-0.], dtype=torch.float32).cuda(),
                                  torch.tensor([-INF], dtype=torch.float32).cuda(), x_upsampled)
        x_upsampled = self._unpad(x_upsampled)
        # Now unfold and do an parameterized dilation.
        desired_shape = x_upsampled.shape
        x_unfolded = self._unfold(x_upsampled)
        x_added = x_unfolded + self.h
        up, _ = torch.max(x_added, dim=2)
        return up.view(*desired_shape), x_upsampled
