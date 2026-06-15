# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified to use torchvision.ops.nms instead of custom _C extension
import torch
from torchvision.ops import nms as tv_nms


def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the boxes according to their
    intersection-over-union (IoU).
    
    Arguments:
        boxes (Tensor[N, 4]): boxes to perform NMS on. They are expected to be 
            in (x1, y1, x2, y2) format.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with 
            IoU > iou_threshold
    
    Returns:
        keep (Tensor): indices of the elements that have been kept by NMS,
            sorted in decreasing order of scores
    """
    # torchvision's nms expects (boxes, scores, iou_threshold)
    return tv_nms(boxes, scores, iou_threshold)
