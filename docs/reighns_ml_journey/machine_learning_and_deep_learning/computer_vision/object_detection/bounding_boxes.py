import torch
from typing import *


def voc2coco(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert pascal_voc to coco format.

    voc  => [xmin, ymin, xmax, ymax]
    coco => [xmin, ymin, w, h]

    Args:
        bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, xmax, ymax].

    Returns:
        coco_bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, w, h].
    """

    # don't perform in place to avoid mutation
    coco_bboxes = bboxes.clone()
    for index, each_bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = each_bbox
        w, h = xmax - xmin, ymax - ymin
        coco_bboxes[index] = torch.tensor([xmin, ymin, w, h])

    return coco_bboxes


def coco2voc(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert coco to pascal_voc format.

    coco => [xmin, ymin, w, h]
    voc  => [xmin, ymin, xmax, ymax]


    Args:
        bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, w, h].

    Returns:
        voc_bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, xmax, ymax].
    """

    # don't perform in place to avoid mutation
    voc_bboxes = bboxes.clone()
    for index, each_bbox in enumerate(bboxes):
        xmin, ymin, w, h = each_bbox
        xmax, ymax = xmin + w, ymin + h
        voc_bboxes[index] = torch.tensor([xmin, ymin, xmax, ymax])

    return voc_bboxes
