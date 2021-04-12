#!/usr/bin/env python3

# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import math
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Any

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
            self.register_parameter('bias', None)

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



def AuxLossWrapper(x, func, include_orig_loss: bool):
    """
     Args:
         x: The input, which you only want the main objective function to
            be backpropped to.  (i.e. it gets the derivative w.r.t. the
            return value y, but not the derivative from `aux_loss`).
      func: A function or other callable object (e.g. torch.nn.Module) where f(x)
            will return (y, aux_loss) with aux_loss an auxiliary loss function as
            a scalar Tensor.  This function will return y.  It will be as if you
            replaced the original loss function with `aux_loss` (or added
            them together, if include_orig_loss==True), in terms of its affects
            on the derivatives of quantities implicitly passed in via `func`.
include_orig_loss:  If true, the original loss (that we are backpropping
            conventionally) will also be propagated to leaf nodes that
            are accessed implicitly via `func`.  If false, just the aux_loss
            will affect their derivatives.
     Return:
            Returns y, which is the first element returned by func(x).  But
            the second value returned by func(x) will be treated as a loss function
            and will affect all gradients that do *not* pass through x.
            (e.g. neural net parameters used by `func` will have their .grad
            affected by `aux_loss`.)
        """
    return _AuxLossWrapper.apply(x, func, include_orig_loss)

class _AuxLossWrapper(torch.autograd.Function):
    """
    This is for when you have an auxiliary objective function whose
    derivatives you only want to be respected for some subset of
    parameters, e.g. those in a module.   This is intended to
    be used only in AuxLossWrapper().
    """
    @staticmethod
    def forward(ctx, x: Tensor, func: Any, include_orig_loss: bool) -> Tensor:
        """
        Args:
          ctx: The context object for backprop
            x: The input, which you only want the main objective function to
               be backpropped to.  (i.e. it gets the derivative w.r.t. the
               return value y, but not the derivative from `aux_loss`).
         func: A function or other callable object (e.g. torch.nn.Module) where f(x)
               will return (y, aux_loss) with aux_loss an auxiliary loss function as
               a scalar Tensor.  This function will return y.  It will be as if you
               added `aux_loss` to the overall loss function, but its derivatives are
               only added to the tensors that come in implicitly via `func` (e.g.
               neural network parameters), and `aux_loss` will have no effect on
               the derivatives w.r.t. x.
 include_orig_loss:  If true, the original loss will be included in the derivatives
               w.r.t. any parameters impliciitly passed in via `func`.  If false,
               only the auxiliary objective will be used to train them.
        Return:
               Returns y, which is the first element returned by func(x).  But
               the second value returned by func(x) will be treated as a loss function
               and will affect all gradients that do *not* pass through x.
               (e.g. neural net parameters used by `func` will have their .grad
               affected by `aux_loss`.)
        """
        orig_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad = orig_requires_grad
        ctx.include_orig_loss = include_orig_loss
        with torch.enable_grad():
            # by saving these as names, instead of with save_for_backward(), we
            # store them as actual Python objects with nothing stripped out.
            # We retain the graph here..
            ctx.x = x
            (ctx.y, ctx.aux_loss) = func(x)
            # We will need the grad_fn on `y` in the backward pass, so we create
            # a new copy of it because returning from this object will cause
            # the grad_fn to be set to our `backward`.
            return ctx.y.detach()

    def backward(ctx, y_deriv: Tensor) -> Tuple[Tensor,None,None]:
        with torch.enable_grad():
            device = ctx.x.device
            aux_loss_deriv = torch.Tensor([1.0])[0].to(device=ctx.x.device)

            # Temporarily set requires_grad=False for x (our version of x is a leaf because
            # we did .detach()...  this will disable the next call to "backward" from
            # propagating the gradient to it (we don't need that gradient, we want the
            # gradient with only the ctx.y part).
            ctx.x.requires_grad=False

            # Note: the reason we do torch.autograd.backward with one call including
            # the y and aux_loss parts, even though we will later have to redo the
            # `y` part, is that we hope to avoid repetition of computations.. whether
            # or not this actually saves time will depend on the graph structure.
            if ctx.include_orig_loss:
                torch.autograd.backward([ctx.y, ctx.aux_loss],
                                        grad_tensors=[y_deriv, aux_loss_deriv],
                                        retain_graph=True)
            else:
                torch.autograd.backward(ctx.aux_loss, retain_graph=True)

            if ctx.needs_input_grad[0]:
                # retain_graph=True below is in case the user had retain_graph=True in
                # the outer backprop.  It's OK; once this object is let go of, this part
                # of the graph will be freed.
                ctx.x.requires_grad=True
                (x_grad,) = torch.autograd.grad([ctx.y], [ctx.x], grad_outputs=[y_deriv],
                                                retain_graph=True)
                return (x_grad, None, None)
            else:
                return (None, None, None)



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

def normalize_simple(x: torch.Tensor, eps: float, alpha: float, dim: int) -> torch.Tensor:
    """
    This is like _Normalize.apply(...) except it uses Torch's native autograd, which
    may be a bit less memory efficient but more time efficient.  It is for testing
    purposes, see test_normalize().
    """
    scales = (x*x).sum(dim, keepdim=True)
    scales = scales + eps
    scales = scales ** -0.5
    scales = scales * alpha
    return x * scales


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

    def __init__(self, num_features, dim=1, eps=1.0e-05,
                 affine=True):
        """
        Constructor
          num_features:  Number of features, e.g. 256, over which
                        we normalize
          dim:          The dimension of the input that corresponds
                        to the feature dimension (may be negative)
          eps:          Epsilon to prevent division by zero
          affine:       If true, include offset and weight parameters
        """

        super(Normalize, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.eps = eps
        self.alpha = math.sqrt(self.num_features)
        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features))
            self.bias = torch.nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init = torch.nn.init
        init.constant_(self.bias, 0.)
        init.constant_(self.weight, 1.)

    def forward(self, x):
        assert x.shape[self.dim] == self.num_features
        x = _Normalize.apply(x, self.eps, self.alpha, self.dim)
        if self.weight is not None:
            weight = self.weight
            bias = self.bias
            dim = self.dim if self.dim >= 0 else x.ndim + self.dim
            for _ in range(dim, x.ndim - 1):
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            x = (x * weight) + bias
        return x


class SlowBatchnorm(torch.nn.Module):
    """
    This ensures that its output is close to having zero mean and unit stddev,
    by adding a weight and a bias.
    The weight and bias are trained so as to minimize an auxiliary objective that encourages
    the output to be zero-mean.  It's best if you follow this layer with
    something that has a conventionally trainable offset/bias, to retain modeling power.
    We suggest following this with NormalizeLayer, which due to this LayerNorm component
    has the additional advantage of keeping the output bounded which will avoid some
    types of divergence.
    """
    def __init__(self, num_features, dim=1,
                 normalize_scale=0.1):
        """
        Constructor
          num_features:  Number of features, e.g. 256, over which
                        we normalize
          dim:          The dimension of the input that corresponds
                        to the feature dimension (may be negative)
          normalize_scale:  Scale that affects the magnitude of the
                        derivatives.   (Note:


                        that scales the objective function.
                        Note: we first divide by the number of vectors
                        being normalized, then the auxiliary objective is the
                        sum-of-squares of the mean times normalize_scale.
                        You should aim that the scale of this is comparable
                        with the scale of the objective function you are
                        training with.
        """
        super(SlowBatchnorm, self).__init__()
        self.bias = torch.nn.Parameter(torch.empty(num_features))
        self.weight = torch.nn.Parameter(torch.empty(num_features))
        self.dim = dim
        self.normalize_scale = normalize_scale
        self.reset_parameters()

    def reset_parameters(self):
        init = torch.nn.init
        init.constant_(self.bias, 0.)
        init.constant_(self.weight, 1.)

    def forward(self, x):
        num_features = self.bias.numel()
        assert x.shape[self.dim] == num_features
        bias = self.bias
        weight = self.weight
        num_vecs = x.numel() / num_features
        dim = self.dim
        if dim < 0:
            dim += x.ndim
        for _ in range(dim, x.ndim - 1):
            bias = bias.unsqueeze(-1)
            weight = weight.unsqueeze(-1)
        normalize_scale = self.normalize_scale
        def _forward(x: Tensor) -> Tuple[Tensor,Tensor]:
            """Returns a pair (y, aux_loss); this becomes the
            'func' passed to AuxLossWrapper().
            """
            y = (x * weight) + bias
            dim_to_normalize=dim
            ymean = torch.sum(y, dim=[ i for i in range(y.ndim)
                                       if i != dim_to_normalize ]) * (1.0 / num_vecs)

            y2mean = torch.sum(y**2, dim=[ i for i in range(y.ndim)
                                           if i != dim_to_normalize ]) * (1.0 / num_vecs)
            aux_loss = ((ymean**2).sum() + ((y2mean - 1.0) ** 2).sum()) * normalize_scale
            return (y, aux_loss)
        return AuxLossWrapper(x, _forward, False)


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
              SlowBatchnorm(num_features=hidden_dim, dim=1),
              Normalize(num_features=hidden_dim, dim=1, affine=True),
              LearnedNonlin(),
              torch.nn.Conv1d(hidden_dim, odim, stride=stride, kernel_size=1, bias=False),
              SlowBatchnorm(num_features=hidden_dim, dim=1) ])

        # keep self.norm separate, we initialize it in reset_parameters().
        self.norm = Normalize(num_features=odim, dim=1, affine=True)

        self.final_norm = torch.nn.Sequential(
            SlowBatchnorm(num_features=hidden_dim, dim=1),
            Normalize(num_features=odim, dim=1, affine=True))


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


def test_slow_batchnorm():
    for n in (-1,1):
        print("With n = ", n)
        x = torch.ones(10,11) * ((n) ** torch.arange(10).unsqueeze(1))
        x.requires_grad = True
        b = SlowBatchnorm(11, dim=1, normalize_scale=1.0)
        y = b(x)
        objf = y.sum() * 1.0e-20
        objf.backward()
        print(f"x={x}, y={y}, x-grad = {x.grad}, bias_grad={b.bias.grad}, weight_grad={b.weight.grad}")


def main():
    test_conv()
    test_normalize()
    test_slow_batchnorm()

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
