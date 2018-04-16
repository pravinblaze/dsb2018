from nnmod import rpn
import torch
from datamod import rpnDataset
from constants import DATA
from constants import AnchorShapes
from torch.autograd import Variable
import numpy as np
from bboxutils import center_size
from bboxutils import point_form
from bboxutils import nms
import cv2
from nnmod import backbone01
from nnmod import maskgen2
from generate import generateAnchors

rpnnet = rpn().cuda()
rpnnet.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
clsnet = backbone01().cuda()
clsnet.load_state_dict(torch.load(DATA + 'models/backbone01.torch'))
masknet = maskgen2().cuda()
masknet.load_state_dict(torch.load(DATA + 'models/maskgen2.torch'))

dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for data in dataloader:
    image = torch.cuda.FloatTensor(np.array(data['image']))
    gtbbox = data['bbox'].type(torch.cuda.FloatTensor)

    classification, bboxdelta = rpnnet(Variable(image))

    posidx = list(range(0, 2 * len(AnchorShapes) - 1, 2))
    objscore = classification[0][posidx].contiguous().view(-1)
    k = len(AnchorShapes)
    cxidx = list(range(0, 4 * k, 4))
    cyidx = list(range(1, 4 * k, 4))
    widx = list(range(2, 4 * k, 4))
    hidx = list(range(3, 4 * k, 4))
    bboxdelta = torch.cat((bboxdelta[0][cxidx].contiguous().view(-1, 1),
                           bboxdelta[0][cyidx].contiguous().view(-1, 1),
                           bboxdelta[0][widx].contiguous().view(-1, 1),
                           bboxdelta[0][hidx].contiguous().view(-1, 1)), dim=1)

    anchors = generateAnchors(image.size()[-2:], classification.size()[-2:])
    bboxdeltacpu = bboxdelta.data.cpu()
    anchors = center_size(anchors)
    anchors = anchors.cpu()
    bboxarray = bboxdeltacpu + anchors

    bboxarray = point_form(bboxarray)
    objscore = objscore.data.cpu()

    keep, count = nms(bboxarray, objscore, overlap=0.2, top_k=200)
    bboxarray = bboxarray[keep[:count]]

    bboxarray = bboxarray.numpy()
    imgray = np.array(data['image'], np.uint8)[0, 0]
    crop_array = np.zeros((0, 32, 32), np.uint8)
    for bbox in bboxarray:
        bbox = np.rint(bbox).astype(np.uint8)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        imcrop = imgray[y1:y2, x1:x2]
        try:
            imcrop = cv2.resize(imcrop, (32, 32))
        except Exception:
            imcrop = np.zeros((32, 32), np.uint8)
        crop_array = np.append(crop_array, np.expand_dims(imcrop, axis=0), axis=0)
        pass
    crops = Variable(torch.cuda.FloatTensor(crop_array))
    crops = crops.view(-1, 1, 32, 32)
    clspred = clsnet(crops)
    masks = masknet(crops)
    clspred.max(dim=1)
    1+1
    pass


