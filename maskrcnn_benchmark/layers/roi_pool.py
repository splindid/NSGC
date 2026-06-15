# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified to use torchvision.ops.roi_pool instead of custom _C extension
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_pool as tv_roi_pool


def roi_pool(input, roi, output_size, spatial_scale):
    """
    ROI Pooling using torchvision implementation.
    
    Arguments:
        input (Tensor[N, C, H, W]): input feature map
        roi (Tensor[K, 5]): the box coordinates in (batch_index, x1, y1, x2, y2) format
        output_size: output spatial size (H, W)
        spatial_scale (float): scale of the input feature map 
    
    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    output_size = _pair(output_size)
    return tv_roi_pool(input, roi, output_size, spatial_scale=spatial_scale)


class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
