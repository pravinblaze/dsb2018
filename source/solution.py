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
import os
from torch import nn
import matplotlib.pyplot as plot
from testcode import drawRectP

rpnnet = rpn().cuda()
rpnnet.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
clsnet = backbone01().cuda()
clsnet.load_state_dict(torch.load(DATA + 'models/backbone01.torch'))
masknet = maskgen2().cuda()
masknet.load_state_dict(torch.load(DATA + 'models/maskgen2.torch'))
threshold = nn.Sigmoid()

# mode = 'valid'
# mode = 'test'
mode = 'final'

if mode == 'valid':
    dataset_path = DATA + 'dataset/rpn-validataion-set'
if mode == 'test':
    dataset_path = DATA + 'dataset/rpn-test/'
if mode == 'final':
    dataset_path = DATA + 'dataset/final/'
dataset = rpnDataset(dataset_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for counter, data in enumerate(dataloader):
    image = torch.cuda.FloatTensor(np.array(data['image']))

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
        bbox = np.rint(bbox).astype(np.int)
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
    masks = threshold(masks)
    clspred = clspred.max(dim=1)[1]
    masks = masks.data * clspred.data.view(clspred.size()[0], 1, 1, 1).type(torch.cuda.FloatTensor)
    masks = masks.view(-1, 32, 32).cpu().numpy()
    th = 0.6
    masks[masks >= th] = 1
    masks[masks < th] = 0

    masks = masks * 255
    masks = masks.astype(np.uint8)
    maskunique = np.zeros(imgray.shape, np.uint8)
    for i, mask in enumerate(masks, 0):
        if mask.max() > 0:
            maskout = np.zeros(imgray.shape, np.uint8)
            bbox = np.rint(bboxarray[i]).astype(np.int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            x2 = maskout.shape[1] if x2 > maskout.shape[1] else x2
            y2 = maskout.shape[0] if y2 > maskout.shape[0] else y2
            try:
                maskout[y1:y2, x1:x2] = cv2.resize(mask, (abs(x2-x1), abs(y2-y1)))
            except Exception:
                pass
            maskout = maskout - maskout * (maskunique/200)
            maskunique = maskunique+maskout
            if mode == 'valid':
                result_path = 'validation-set'
            if mode == 'test':
                result_path = 'test-set'
            if mode == 'final':
                result_path = 'final'
            if not os.path.isdir(DATA + 'results/'+result_path+'/' + data['id'][0] + '/'):
                os.makedirs(DATA + 'results/'+result_path+'/' + data['id'][0] + '/')
            cv2.imwrite(DATA + 'results/'+result_path+'/' + data['id'][0] + '/' + str(i) + '.png', maskout)
        if i % 10 == 0:
            print("Solving ... {:.2f} %".format(100*counter/len(dataset)))
        pass
    pass
