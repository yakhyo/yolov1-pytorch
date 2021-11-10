import torch
import torch.nn as nn


def pad(k, p):
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=None, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, pad(k, p), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.03, eps=1e-3)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


def nms(boxes, scores, nms_thresh=0.45):
    """ Apply non maximum supression.
    Args:
    Returns:
    """
    threshold = nms_thresh

    x1 = boxes[:, 0]  # [n,]
    y1 = boxes[:, 1]  # [n,]
    x2 = boxes[:, 2]  # [n,]
    y2 = boxes[:, 3]  # [n,]
    areas = (x2 - x1) * (y2 - y1)  # [n,]

    _, ids_sorted = scores.sort(0, descending=True)  # [n,]
    ids = []
    while ids_sorted.numel() > 0:
        # Assume `ids_sorted` size is [m,] in the beginning of this iter.

        i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
        ids.append(i)

        if ids_sorted.numel() == 1:
            break  # If only one box is left (i.e., no box to supress), break.

        inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i])  # [m-1, ]
        inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i])  # [m-1, ]
        inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i])  # [m-1, ]
        inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i])  # [m-1, ]
        inter_w = (inter_x2 - inter_x1).clamp(min=0)  # [m-1, ]
        inter_h = (inter_y2 - inter_y1).clamp(min=0)  # [m-1, ]

        inters = inter_w * inter_h  # intersections b/w/ box `i` and other boxes, sized [m-1, ].
        unions = areas[i] + areas[ids_sorted[1:]] - inters  # unions b/w/ box `i` and other boxes, sized [m-1, ].
        ious = inters / unions  # [m-1, ]

        # Remove boxes whose IoU is higher than the threshold.
        ids_keep = (
                ious <= threshold).nonzero().squeeze()  # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
        if ids_keep.numel() == 0:
            break  # If no box left, break.
        ids_sorted = ids_sorted[ids_keep + 1]  # `+1` is needed because `ids_sorted[0] = i`.

    return torch.LongTensor(ids)
