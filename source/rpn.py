import time
import torch
from torch.autograd import Variable
import torch.optim as optimize
import numpy as np
from constants import DATA
from constants import AnchorShapes
from nnmod01 import rpn
from datamod01 import rpnDataset
from generate_targets import generateTargets
from generate_targets import center_size
import matplotlib.pyplot as plot
import cv2


def visualizeRPN():
    net = rpn().cuda()
    net.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
    dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    dataiter = iter(dataloader)
    data = dataiter.next()
    image = torch.cuda.FloatTensor(np.array(data['image']))
    gtbbox = data['bbox'].type(torch.cuda.FloatTensor)
    classification, bboxdelta = net(Variable(image))
    clstargets, bboxtargets, anchors = generateTargets(gtbbox[0], image.size()[-2:], classification.size()[-2:])
    posidx = list(range(0, 2 * len(AnchorShapes) - 1, 2))
    clspos = clstargets[0][posidx]
    targetpos = clspos[clspos > 0].contiguous().view(-1)
    k = len(AnchorShapes)
    cxidx = list(range(0, 4*k, 4))
    cyidx = list(range(1, 4*k, 4))
    widx = list(range(2, 4*k, 4))
    hidx = list(range(3, 4*k, 4))
    bboxdelta = torch.cat((bboxdelta[0][cxidx].contiguous().view(-1, 1), 
                                bboxdelta[0][cyidx].contiguous().view(-1, 1), 
                                bboxdelta[0][widx].contiguous().view(-1, 1),
                                bboxdelta[0][hidx].contiguous().view(-1, 1)), dim=1)
    postargetidx = (targetpos > 0.8).nonzero()[:, 0].cpu()
    bboxdeltacpu = bboxdelta.data.cpu()
    anchors = anchors.cpu()
    bboxarray = bboxdeltacpu[postargetidx]+anchors[postargetidx]
    bboxarray = bboxarray.numpy()
    img = np.array(data['image'])
    img = img[0, 0, :, :]

    for bbox in bboxarray:
        cv2.rectangle(img, (int(round(bbox[0]-bbox[2]/2)), int(round(bbox[1]-bbox[3]/2))), 
                        (int(round(bbox[0]+bbox[2]/2)), int(round(bbox[1]+bbox[3]/2))), 
                        (0,255,0) , 2)
        pass
    
    plot.imshow(img)
    plot.show()
    
    pass


def validateRPN():

    net = rpn().cuda()
    net.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
    dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    criterioncls = torch.nn.CrossEntropyLoss().cuda()
    criterionbbox = torch.nn.SmoothL1Loss().cuda()
    loop_counter = 0
    running_clsloss = 0
    running_bboxloss = 0
    for data in dataloader:
        loop_counter += 1
        image = torch.cuda.FloatTensor(np.array(data['image']))
        gtbbox = data['bbox'].type(torch.cuda.FloatTensor)
        classification, bboxdelta = net(Variable(image))
        clstargets, bboxtargets, anchors = generateTargets(gtbbox[0], image.size()[-2:], classification.size()[-2:])
        clsloss = classificationloss(classification, clstargets, criterioncls)
        bboxloss = bboxdeltaloss(bboxtargets, bboxdelta, clstargets, anchors, criterionbbox)
        running_clsloss += clsloss
        running_bboxloss += bboxloss
        print('progress {:3.3f} %, clsloss {:1.5f}'.format(100 * loop_counter / dataset_size, clsloss.data[0]))
    print('average validation clsloss {:1.5f}, bboxloss {:1.5f}'.format(running_clsloss.data[0] / dataset_size,
                                                                        running_bboxloss.data[0] / dataset_size))

    pass


def trainRPN(epochs=4, a=0.001, b=0.1, train_backbone=False, init_backbone=False, init_rpn=False, save_model=False):

    start_time = time.time()
    net = rpn(train_backbone, init_backbone).cuda()
    if init_rpn:
        net.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
    dataset = rpnDataset(DATA+'dataset/rpn/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    criterioncls = torch.nn.CrossEntropyLoss().cuda()
    criterionbbox = torch.nn.SmoothL1Loss().cuda()
    N = 100
    dataset_size = len(dataset)

    clambda = 1
    blambda = 10
    for epoch in range(epochs):
        optimizer = optimize.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=a * (b ** epoch))
        loop_counter = 0
        for data in dataloader:

            loop_counter += 1
            image = torch.cuda.FloatTensor(np.array(data['image']))
            gtbbox = data['bbox'].type(torch.cuda.FloatTensor)
            classification, _ = net(Variable(image).cuda())
            clstargets, bboxtargets, anchors = generateTargets(gtbbox[0], image.size()[-2:], classification.size()[-2:])

            for i in range(10):
                optimizer.zero_grad()
                classification, bboxdelta = net(Variable(image))
                clsloss = classificationloss(classification, clstargets, criterioncls, N, True)
                bboxloss = bboxdeltaloss(bboxtargets, bboxdelta, clstargets, anchors, criterionbbox)
                if str(bboxloss.data[0]) != 'nan':
                    totalloss = clambda * clsloss + blambda * bboxloss
                    totalloss.backward()
                    optimizer.step()

            print('Epoch {} progress {:3.3f} %, clsloss {:1.5f}, bboxloss {:1.5f}'.format(epoch,
                                                                     100*loop_counter/dataset_size,
                                                                     clsloss.data[0], blambda*bboxloss.data[0]))
    end_time = time.time()
    print('Finished training in {:4.2f} s i.e, {:3.2f} min'.format(end_time-start_time, (end_time-start_time)/60))
    if save_model:
        print('Saving model to disk ... ')
        torch.save(net.state_dict(), DATA+'models/'+net.__class__.__name__+'.torch')
        print('Model saved to disk !')
    pass


def bboxdeltaloss(bboxtargets, bboxdelta, clstargets, anchors, criterion):
    k = len(AnchorShapes)
    cxidx = list(range(0, 4*k, 4))
    cyidx = list(range(1, 4*k, 4))
    widx = list(range(2, 4*k, 4))
    hidx = list(range(3, 4*k, 4))
    bboxtargets = torch.cat((bboxtargets[0][cxidx].contiguous().view(-1, 1), 
                                bboxtargets[0][cyidx].contiguous().view(-1, 1), 
                                bboxtargets[0][widx].contiguous().view(-1, 1),
                                bboxtargets[0][hidx].contiguous().view(-1, 1)), dim=1)
    bboxdelta = torch.cat((bboxdelta[0][cxidx].contiguous().view(-1, 1), 
                                bboxdelta[0][cyidx].contiguous().view(-1, 1), 
                                bboxdelta[0][widx].contiguous().view(-1, 1),
                                bboxdelta[0][hidx].contiguous().view(-1, 1)), dim=1)
    posidx = list(range(0, 2*k, 2))
    postarget = clstargets[0][posidx].contiguous().view(-1)
    postargetidx = (postarget > 0).nonzero()[:, 0]
    
    bboxdelta = bboxdelta[postargetidx]
    bboxtargets = Variable(bboxtargets[postargetidx], requires_grad=False)
    anchors = Variable(anchors[postargetidx], requires_grad=False)
    anchors = center_size(anchors)

    tdelta = torch.cat((bboxdelta[:, :2] / anchors[:, 2:],
                        torch.log((bboxdelta[:, 2:]+anchors[:, 2:]+1e-3)/anchors[:, 2:])), dim=1)
    ttarget = torch.cat((bboxtargets[:, :2] / anchors[:, 2:],
                                torch.log((bboxtargets[:, 2:]+anchors[:, 2:]+1e-3)/anchors[:, 2:])), dim=1)

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