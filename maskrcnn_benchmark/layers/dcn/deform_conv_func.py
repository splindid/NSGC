# Modified to use torchvision.ops.deform_conv2d instead of custom _C extension
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from torchvision.ops import deform_conv2d


class DeformConvFunction(Function):
    """
    Deformable Convolution using torchvision.ops.deform_conv2d.
    Note: torchvision's deform_conv2d handles gradients automatically.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        weight,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.dim()))
        
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        # torchvision.ops.deform_conv2d expects offset in shape (N, 2*Kh*Kw, H, W)
        # The original code also uses this format
        # Note: deformable_groups in original is called offset_groups in torchvision
        output = deform_conv2d(
            input,
            offset,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=None
        )
        
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # torchvision handles gradients automatically through autograd
        # This won't be called since we're using autograd-compatible function
        raise NotImplementedError("backward is handled by autograd")

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


class ModulatedDeformConvFunction(Function):
    """
    Modulated Deformable Convolution using torchvision.ops.deform_conv2d with mask.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1
    ):
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        # torchvision.ops.deform_conv2d supports mask parameter
        output = deform_conv2d(
            input,
            offset,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask
        )
        
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # torchvision handles gradients automatically through autograd
        raise NotImplementedError("backward is handled by autograd")

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


def deform_conv(
    input,
    offset,
    weight,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    deformable_groups=1,
    im2col_step=64
):
    """
    Deformable convolution wrapper using torchvision.ops.deform_conv2d.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    
    return deform_conv2d(
        input,
        offset,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=None
    )


def modulated_deform_conv(
    input,
    offset,
    mask,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    deformable_groups=1
):
    """
    Modulated deformable convolution wrapper using torchvision.ops.deform_conv2d.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    
    return deform_conv2d(
        input,
        offset,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask
    )
