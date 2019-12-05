import torch


def area(boxes):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return area

def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
        box1: (BoxList) bounding boxes, sized [N,4].
        box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
        (tensor) iou, sized [N,M].
    """

    # N = boxlist1.shape[0]
    # M = boxlist2.shape[1]

    area1 = area(boxlist1)
    area2 = area(boxlist2)

    lt = torch.max(boxlist1[:, None, :2], boxlist2[:, :2])  # [N,M,2]
    rb = torch.min(boxlist1[:, None, 2:], boxlist2[:, 2:])  # [N,M,2]

    wh = (rb - lt ).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def boxlist_distance(boxlist1, boxlist2, tau=1):
    """Compute the Euclidean distance between centers of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
        box1: (BoxList) bounding boxes, sized [N,4].
        box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
        (tensor) distance, sized [N,M].
    """
    center1 = torch.cat((((boxlist1[:, None, 2] - boxlist1[:, None, 0]) / 2) + boxlist1[:, None,0], ((boxlist1[:, None, 3] - boxlist1[:, None, 1]) / 2) + boxlist1[:, None,1]), dim=1)
    center2 = torch.cat((((boxlist2[:, None, 2] - boxlist2[:, None, 0]) / 2) + boxlist2[:, None,0], ((boxlist2[:, None, 3] - boxlist2[:, None, 1]) / 2) + boxlist2[:, None,1]), dim=1)

    center1 = center1.unsqueeze(1)
    center2 = center2.unsqueeze(0)

    d = torch.sqrt((center1[:, :, 0] - center2[:, :, 0]) ** 2 + (center1[:, :, 1] - center2[:, :, 1]) ** 2)

    return torch.exp(-1*tau*d)