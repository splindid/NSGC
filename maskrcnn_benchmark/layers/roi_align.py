# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified to use torchvision.ops.roi_align instead of custom _C extension
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align as tv_roi_align


def roi_align(input, rois, output_size, spatial_scale, sampling_ratio):
    """
    ROI Align pooling using torchvision implementation.
    
    Arguments:
        input (Tensor[N, C, H, W]): input feature map
        rois (Tensor[K, 5]): the box coordinates in (batch_index, x1, y1, x2, y2) format
        output_size: output spatial size (H, W)
        spatial_scale (float): scale of the input feature map 
        sampling_ratio (int): number of sampling points in the interpolation grid
    
    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    output_size = _pair(output_size)
    return tv_roi_align(
        input, 
        rois, 
        output_size, 
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=False  # Use original ROIAlign behavior
    )


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
