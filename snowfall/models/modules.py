#!/usr/bin/env python3

# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import math
import torch
import torch.nn.functional as F
from typing import Tuple

class _LearnedNonlin(torch.autograd.Function):

    @staticmethod
    def _compute(x, cutoffs, scales):
        x = x.unsqueeze(-1) + cutoffs
        x = x.relu()
        x = x * scales
        x = x.sum(-1)
        return x

    @staticmethod
    def forward(ctx, x, cutoffs, scales):
        ctx.save_for_backward(x.detach(), cutoffs.detach(),
                              scales.detach())
        return _LearnedNonlin._compute(x, cutoffs, scales)

    @staticmethod
    def backward(ctx, output_grad):
        x, cutoffs, scales = ctx.saved_tensors
        with torch.enable_grad():
            x.requires_grad = True
            cutoffs.requires_grad = True
            scales.requires_grad = True
            y = _LearnedNonlin._compute(x, cutoffs, scales)
            torch.autograd.backward(y, grad_tensors=(output_grad,))
            return (x.grad, cutoffs.grad, scales.grad)

class LearnedNonlin(torch.nn.Module):
    """
    A learned nonlinearity that initializes itself to ReLU.  It is a single
    nonlinearity: not per channel (so there are few parameters).  This kind of
    thing will tend to learn quite slowly, which is why we make it a "permanent"
    parameter (can be learned from model to model).

    """
    def __init__(self, N = 21):
        super(LearnedNonlin, self).__init__()
        self.cutoffs_perm = torch.nn.Parameter(torch.zeros(N))
        self.scales_perm = torch.nn.Parameter(torch.zeros(N))
        self.offset_perm = torch.nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            N = self.cutoffs_perm.numel()
            # cutoffs range from -2 to 2.
            cutoffs = ((4.0 / (N-1)) * torch.arange(N)) - 2.0
            self.cutoffs_perm.copy_(cutoffs)
            scales = torch.zeros(N)
            # this initialization makes it ReLU.
            scales[N//2] = 1.0
            self.scales_perm.copy_(scales)

    def forward(self, x):
        return _LearnedNonlin.apply(x, self.cutoffs_perm,
                                    self.scales_perm) + self.offset_perm




class Conv1dCompressed(torch.nn.Module):
    """
    Defaults to depthwise convolution which acts only in
    the time dimension, separately for each channel.

    Compressed means we limit the time coefficients to a subspace
    shared between all channels, that's a permanent parameter.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = -1,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = -1,
                 full_kernel_size: int = 65,
                 compressed_kernel_size: int = 4,
                 bias: bool = True):
        super(Conv1dCompressed, self).__init__()
        if out_channels < 0:
            out_channels = in_channels
        if groups < 0:
            groups = in_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.nn.Parameter(torch.empty(
	    out_channels, in_channels // groups, compressed_kernel_size))

        # "_perm" suffix means it will be stored from run to run.
        self.meta_weight_perm = torch.nn.Parameter(torch.empty(compressed_kernel_size,
                                                               full_kernel_size,))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:

        self.reset_parameters()

    def reset_parameters(self):
        init = torch.nn.init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.meta_weight_perm)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        """
        Input: (N, C, L) = (batch,channel,length)
        Output: (N, C, L) = (batch,channel,length)
        """
        full_kernel_size = self.meta_weight_perm.shape[-1]
        padding = full_kernel_size // 2

        weight = torch.matmul(self.weight, self.meta_weight_perm)
        bias = self.bias

        return F.conv1d(input, weight, bias, self.stride,
                        padding, self.dilation, self.groups)



class _Normalize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float, alpha: float, dim: int) -> torch.Tensor:
        """
        View all dimensions of x except `dim` as the batch dimension,
        treating all the resulting x's as separate vectors.   We aim
        to normalize each such x_vec to have unit scale (after using eps as ballast),
        i.e.  x_vec -> x_vec * (alpha / sqrt(x_vec.x_vec + eps)).
        """
        x = x.detach()
        scales = (x**2).sum(dim, keepdim=True)
        scales.add_(eps)
        scales.pow_(-0.5)
        scales.mul_(alpha)
        ctx.save_for_backward(x, scales)
        ctx.dim = dim
        ctx.alpha = alpha
        return x * scales

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> Tuple[torch.Tensor,None,None,None]:
        output_grad = output_grad.detach()
        x, scales = ctx.saved_tensors
        scales_grad = (x * output_grad).sum(ctx.dim, keepdim=True)

        # scales = alpha * (sums + eps)^{-0.5}
        # dscales/dsums = alpha * -0.5 * (sums + eps)^{-1.5)
        #               = (-0.5/alpha^2) * scales**3.
        # -->   sums_grad = scales_grad * ((-0.5/alpha) * scales**3)
        twice_sums_grad = scales_grad * ((-1.0/(ctx.alpha*ctx.alpha)) * scales**3)

        # now: sum_i = (x_i . x_i), so d(sum_i)/d(x_i) = 2 * x_i.
        # so: x_grad += 2 * sums_grad * x
        #  or: x_grad += twice_sums_grad * x

        x_grad = output_grad * scales + \
                 twice_sums_grad * x

        return x_grad, None, None, None


def test_normalize():
    x = torch.outer(torch.arange(10), torch.arange(1.,6.)).detach()
    x.requires_grad = True
    b = _Normalize.apply(x, 1.0e-05, math.sqrt(5), 1)
    b.sum().backward()
    xgrad1 = x.grad.clone()
    print(f'x={x}, b={b}, x_grad={x.grad}')

    x.grad = None
    b = normalize_simple(x, 1.0e-05, math.sqrt(5), 1)
    b.sum().backward()
    print(f'With ref method: x={x}, b={b}, x_grad={x.grad}')
    assert torch.allclose(xgrad1, x.grad)

class Normalize(torch.nn.Module):
    """
    This is like LayerNorm (but without the offset part, only the variance
    part), but acting on a configurable.  dimension (not necessarily the last
    dim).  It also has affine (weight/bias) parameters, optionally.  The naming
    is after Kaldi's NormalizeLayer.
    """

    def __init__(self, num_features, dim=1, eps=1.0e-05, affine=True):
        self.num_features = num_features
        self.alpha = math.sqrt(self.num_features)
        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features))
            self.bias = torch.nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.eps = eps
        self.reset_parameters()


    def reset_parameters():
        init = torch.nn.init
        init.constant_(self.bias, 0.)
        init.constant_(self.weight, 0.)

    def forward(x):
        assert x.shape[dim] == self.num_features
        x = _Normalize.apply(x, self.eps, self.alpha, self.dim)
        if self.weight is not None:
            x = (x * self.weight) + self.bias
        return x


class ConvModule(torch.nn.Module):
    """ this is resnet-like."""
    def __init__(self, idim, odim,
                 hidden_dim, stride=1, dropout=0.0):
        super(ConvModule, self).__init__()

        self.layers = torch.nn.Sequential(*
            [ torch.nn.Conv1d(idim, hidden_dim, stride=1, kernel_size=1, bias=False),
              LearnedNonlin(),
              torch.nn.Dropout(dropout),
              Conv1dCompressed(hidden_dim),
              Normalize(num_features=hidden_dim, dim=1, affine=True),
              LearnedNonlin(),
              torch.nn.Conv1d(hidden_dim, odim, stride=stride, kernel_size=1, bias=False) ])

        self.norm = Normalize(num_features=odim, dim=1, affine=True)

        self.final_norm = Normalize(num_features=odim, dim=1, affine=True)

        if stride != 1 or odim != idim:
            self.bypass_conv = torch.nn.Conv1d(odim, idim, stride=stride,
                                               kernel_size=1, bias=False)
        else:
            self.register_parameter('bypass_conv', None)
        self.reset_parameters()

    def reset_parameters(self):
        init = torch.nn.init
        # the following should make it train more easily.
        init.uniform_(self.norm.weight, 0.25)


    def forward(self, x):
        """
        Input: (N, C, L) = (batch,channel,length)
        Output: (N, C, L) = (batch,channel,length)
        """
        bypass = self.bypass_conv(x) if self.bypass_conv is not None else x
        x = self.layers(x)
        x = self.norm(x)
        return self.final_norm(x + bypass)
        return out


def test_conv():
    batch_size = 2
    num_channels = 3
    T = 20
    input = torch.ones(batch_size, num_channels, T)

    layer = Conv1dCompressed(num_channels)

    output = layer(input)
    print("output = ", output)


def main():
    test_conv()
    test_normalize()

    n = LearnedNonlin()
    a = torch.randn(10,10).detach()
    a.requires_grad = True
    b = n(a)
    print(f"a = {a}, b = {b}")
    b.sum().backward()
    print("a_grad = ", a.grad)
    print("scales_grad = ", n.scales_perm.grad)
    print("cutoffs_grad = ", n.cutoffs_perm.grad)
    print("offset_grad = ", n.offset_perm.grad)



if __name__ == '__main__':
    main()
