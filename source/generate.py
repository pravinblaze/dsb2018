import torch
from constants import AnchorShapes
from bboxutils import jaccard
from bboxutils import center_size


def generateTargets(gtbbox, data_shape, heatmap_shape):

    anchors = generateAnchors(data_shape, heatmap_shape)

    ious = jaccard(gtbbox, anchors)

    clstarget = clsTargets(ious, heatmap_shape)
    bboxtarget = bboxTargets(ious, clstarget, anchors, gtbbox, heatmap_shape)

    return clstarget, bboxtarget, anchors


def bboxTargets(ious, clstarget, anchors, gtbbox, heatmap_shape):
    """ Creates bounding box targets.
        For each anchor, finds the delta needed to be added to become the closest ground truth box """

    k = len(AnchorShapes)
    posidx = list(range(0, 2*k, 2))
    clstarget = clstarget[0][posidx].contiguous().view(-1)
    fgidx = (clstarget > 0).nonzero()[:, 0]
    nearestgt = ious.max(dim=0)
    nearestgtidx = nearestgt[1]
    bboxtarget = torch.zeros(ious.size()[1], 4).cuda()
    targetidx = nearestgtidx[fgidx]
    bboxtarget[fgidx] = gtbbox[targetidx]
    bboxtarget = center_size(bboxtarget)
    anchors = center_size(anchors)
    anchors[(clstarget < 1).nonzero()[:, 0], :] = 0
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


def generateAnchors(input_shape, heatmap_shape):
    """ Generates array of anchors in the form xmin, ymin, xmax, ymax """

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

    # Eliminating cross boundary anchors
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
