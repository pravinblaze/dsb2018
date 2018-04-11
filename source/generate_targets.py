import torch
from constants import AnchorShapes


def generateTargets(gtbbox, data_shape, heatmap_shape):

    anchors = generateAnchros(data_shape, heatmap_shape)

    ious = jaccard(gtbbox, anchors)

    clstarget = clsTargets(ious, heatmap_shape)
    bboxtarget = bboxTargets(ious, clstarget, anchors, gtbbox, heatmap_shape)

    return clstarget, bboxtarget, anchors


def bboxTargets(ious, clstarget, anchors, gtbbox, heatmap_shape):
    """ Creates bounding box targets.
        For each anchor, finds the delta needed to be added to become the closest ground truth box """

    k = len(AnchorShapes)
    posidx = list(range(0, 2*k, 2))
    negidx = list(range(1, 2*k, 2))
    clstarget = torch.cat((clstarget[0][posidx].contiguous().view(-1, 1), 
                            clstarget[0][negidx].contiguous().view(-1,1)), dim=1)
    fgidx = (clstarget[:, 0]>0).nonzero()[:,0]
    nearestgt = ious.max(dim=0)
    nearestgtidx = nearestgt[1]
    bboxtarget = torch.zeros(ious.size()[1], 4).cuda()
    targetidx = nearestgtidx[fgidx]
    bboxtarget[fgidx] = gtbbox[targetidx]
    bboxtarget = center_size(bboxtarget)
    anchors = center_size(anchors)
    anchors[(clstarget[:, 0]<1).nonzero()[:, 0], :] = 0
    bboxtarget = bboxtarget - anchors

    cxidx = list(range(0, 4 * k - 3, 4))
    cyidx = list(range(1, 4 * k - 2, 4))
    widx = list(range(2, 4 * k - 1, 4))
    hidx = list(range(3, 4 * k, 4))
    cx = bboxtarget[:, 0].contiguous()
    cx = cx.view(k, heatmap_shape[0], heatmap_shape[1])
    cy = bboxtarget[:, 1].contiguous()
    cy = cy.view(k, heatmap_shape[0], heatmap_shape[1])
    w = bboxtarget[:, 2].contiguous()
    w = w.view(k, heatmap_shape[0], heatmap_shape[1])
    h = bboxtarget[:, 3].contiguous()
    h = h.view(k, heatmap_shape[0], heatmap_shape[1])
    bboxtarget = torch.zeros(1, 4 * k, heatmap_shape[0], heatmap_shape[1]).cuda()
    bboxtarget[0][cxidx] = cx
    bboxtarget[0][cyidx] = cy
    bboxtarget[0][widx] = w
    bboxtarget[0][hidx] = h

    return bboxtarget


def clsTargets(ious, heatmap_shape):
    """ Creates classification targets.
        Classifies positive for every anchor with iou > tpos with any ground truth box or
        the anchor with the highest iou for a ground truth box.
        Classifies negative for anchors with iou < tneg with every ground truth box."""

    tpos = 0.5
    tneg = 0.1
    k = len(AnchorShapes)

    maxanchor = ious.max(dim=0)
    # Eliminate in-betweens
    dcidx = ((maxanchor[0] < tpos)*(maxanchor[0] > tneg)).nonzero()

    # Finding positive target mask
    fgidx = (maxanchor[0] > tpos).nonzero()
    targetpos = torch.zeros(ious.size()[1]).cuda()
    if len(dcidx) > 0:
        targetpos[dcidx[:, 0]] = -1
    if len(fgidx) > 0:
        targetpos[fgidx[:, 0]] = 1
    maxgt = ious.max(dim=1)
    maxgtidx = (maxgt[0] < tpos).nonzero()
    if len(maxgtidx) > 0:
        missingfgidx = maxgt[1][maxgtidx[:, 0]]
        targetpos[missingfgidx] = 1

    # Finding negative target mask
    bgidx = (maxanchor[0] < tneg).nonzero()
    targetneg = torch.zeros(ious.size()[1]).cuda()
    if len(dcidx) > 0:
        targetneg[dcidx[:, 0]] = -1
    if len(bgidx) > 0:
        targetneg[bgidx[:, 0]] = 1
    if len(fgidx) > 0:
        targetneg[fgidx[:, 0]] = 0
    if len(maxgtidx) > 0:
        targetneg[missingfgidx] = 0

    # building cross entropy mask
    clstarget = torch.zeros(1, 2*k, heatmap_shape[0], heatmap_shape[1]).cuda()
    targetpos = targetpos.view(k, heatmap_shape[0], heatmap_shape[1])
    targetneg = targetneg.view(k, heatmap_shape[0], heatmap_shape[1])
    posidx = torch.cuda.LongTensor(list(range(0, 2*k-1, 2)))
    negidx = torch.cuda.LongTensor(list(range(1, 2*k, 2)))
    clstarget[0][posidx] = targetpos
    clstarget[0][negidx] = targetneg

    1+1
    return clstarget


def generateAnchros(input_shape, heatmap_shape):

    k = len(AnchorShapes)
    baseanchors = torch.cuda.FloatTensor(AnchorShapes)
    anchors = torch.zeros((4*k, heatmap_shape[0], heatmap_shape[1])).cuda()
    anchorindecies = (anchors > -1).nonzero()[:, 1:]
    anchorindecies = anchorindecies[0:heatmap_shape[0]*heatmap_shape[1], :]
    yratio = input_shape[0] // heatmap_shape[0]
    xratio = input_shape[1] // heatmap_shape[1]
    anchoryindicies = anchorindecies[:, 0] * yratio
    anchorxindicies = anchorindecies[:, 1] * xratio
    anchors[torch.cuda.LongTensor(list(range(0, 4*k, 2)))] = anchors[list(range(0, 4*k, 2))] + \
                                      anchorxindicies.contiguous().view(1, heatmap_shape[0],
                                                                        heatmap_shape[1]).type(torch.cuda.FloatTensor)
    anchors[torch.cuda.LongTensor(list(range(1, 4*k, 2)))] = anchors[list(range(1, 4*k, 2))] + \
                                      anchoryindicies.contiguous().view(1, heatmap_shape[0],
                                                                        heatmap_shape[1]).type(torch.cuda.FloatTensor)
    xminidx = torch.cuda.LongTensor(list(range(0, 4*k, 4)))
    yminidx = torch.cuda.LongTensor(list(range(1, 4*k, 4)))
    xmaxidx = torch.cuda.LongTensor(list(range(2, 4*k, 4)))
    ymaxidx = torch.cuda.LongTensor(list(range(3, 4*k, 4)))
    anchors[xminidx] = anchors[xminidx] - baseanchors[:, 0].contiguous().view(k, 1, 1)/2
    anchors[yminidx] = anchors[yminidx] - baseanchors[:, 1].contiguous().view(k, 1, 1)/2
    anchors[xmaxidx] = anchors[xmaxidx] + baseanchors[:, 0].contiguous().view(k, 1, 1)/2
    anchors[ymaxidx] = anchors[ymaxidx] + baseanchors[:, 1].contiguous().view(k, 1, 1)/2
    anchors = torch.cat((anchors[xminidx].contiguous().view(-1, 1),
                         anchors[yminidx].contiguous().view(-1, 1),
                         anchors[xmaxidx].contiguous().view(-1, 1),
                         anchors[ymaxidx].contiguous().view(-1, 1)), dim=1)

    anchormin = anchors.min(dim=1)
    negofidx = (anchormin[0] < 0).nonzero()[:, 0]
    anchors[negofidx] = 0
    anchorxmax = anchors[:, 2].contiguous().view(-1,1).max(dim=1)
    anchorymax = anchors[:, 3].contiguous().view(-1,1).max(dim=1)
    xposofidx = (anchorxmax[0] > input_shape[1]).nonzero()[:, 0]
    yposofidx = (anchorymax[0] > input_shape[0]).nonzero()[:, 0]
    anchors[xposofidx] = 0
    anchors[yposofidx] = 0

    return anchors


def intersect(box_a, box_b):
    # Function taken from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    # Function taken from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def center_size(boxes):
    # Function taken from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
    """ Converts boxes of the form (xmin, ymin, xmax, ymax) to
        (cx, cy, w, h) """

    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), dim=1)  # w, h
