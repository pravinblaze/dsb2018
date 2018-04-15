import torch
from constants import AnchorShapes
from torch.autograd import Variable
from bboxutils import center_size
import numpy as np


def bboxdeltaloss(bboxtargets, bboxdelta, clstargets, anchors, criterion):
    k = len(AnchorShapes)
    cxidx = list(range(0, 4 * k, 4))
    cyidx = list(range(1, 4 * k, 4))
    widx = list(range(2, 4 * k, 4))
    hidx = list(range(3, 4 * k, 4))
    bboxtargets = torch.cat((bboxtargets[0][cxidx].contiguous().view(-1, 1),
                             bboxtargets[0][cyidx].contiguous().view(-1, 1),
                             bboxtargets[0][widx].contiguous().view(-1, 1),
                             bboxtargets[0][hidx].contiguous().view(-1, 1)), dim=1)
    bboxdelta = torch.cat((bboxdelta[0][cxidx].contiguous().view(-1, 1),
                           bboxdelta[0][cyidx].contiguous().view(-1, 1),
                           bboxdelta[0][widx].contiguous().view(-1, 1),
                           bboxdelta[0][hidx].contiguous().view(-1, 1)), dim=1)
    posidx = list(range(0, 2 * k, 2))
    postarget = clstargets[0][posidx].contiguous().view(-1)
    postargetidx = (postarget > 0).nonzero()[:, 0]

    bboxdelta = bboxdelta[postargetidx]
    bboxtargets = Variable(bboxtargets[postargetidx], requires_grad=False)
    anchors = Variable(anchors[postargetidx], requires_grad=False)
    anchors = center_size(anchors)

    tdelta = torch.cat((bboxdelta[:, :2] / anchors[:, 2:],
                        torch.log((bboxdelta[:, 2:] + anchors[:, 2:] + 1e-3) / anchors[:, 2:])), dim=1)
    ttarget = torch.cat((bboxtargets[:, :2] / anchors[:, 2:],
                         torch.log((bboxtargets[:, 2:] + anchors[:, 2:] + 1e-3) / anchors[:, 2:])), dim=1)

    return criterion(tdelta, ttarget)


def classificationloss(classification, clstargets, criterion, N=256, training=False):
    posidx = list(range(0, 2 * len(AnchorShapes) - 1, 2))
    negidx = list(range(1, 2 * len(AnchorShapes), 2))

    clspos = clstargets[0][posidx]
    clsneg = clstargets[0][negidx]
    clspredpos = classification[0][posidx]
    clspredneg = classification[0][negidx]

    targetpos = clspos[clspos > 0].contiguous().view(-1)
    targetneg = clspos[clsneg > 0].contiguous().view(-1)
    predictionpos = torch.cat((clspredpos[clspos > 0].contiguous().view(-1, 1),
                               clspredneg[clspos > 0].contiguous().view(-1, 1)), dim=1)
    predictionneg = torch.cat((clspredpos[clsneg > 0].contiguous().view(-1, 1),
                               clspredneg[clsneg > 0].contiguous().view(-1, 1)), dim=1)

    targetposidx = list(range(targetpos.size()[0]))
    targetnegidx = list(range(targetneg.size()[0]))

    if training:
        ceidxpos = torch.cuda.LongTensor(np.random.choice(targetposidx, N//2))
        ceidxneg = torch.cuda.LongTensor(np.random.choice(targetnegidx, N//2))
    else:
        ceidxpos = targetposidx
        ceidxneg = targetnegidx
    target = Variable(torch.cat((targetpos[ceidxpos], targetneg[ceidxneg]), dim=0))
    prediction = torch.cat((predictionpos[ceidxpos], predictionneg[ceidxneg]), dim=0)

    return criterion(prediction, target.type(torch.cuda.LongTensor))