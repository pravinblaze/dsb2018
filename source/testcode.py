from bboxutils import center_size
from bboxutils import point_form
from bboxutils import nms
import matplotlib.pyplot as plot
import cv2
from nnmod import rpn
import torch
from constants import DATA
from datamod import rpnDataset
from constants import AnchorShapes
from torch.autograd import Variable
from generate import generateTargets
import numpy as np
from loss import bboxdeltaloss
from loss import classificationloss
from torchvision import transforms
from datamod import backboneDatasetLoader
from torch import nn
from datamod import loadCropBatches
import pandas as pd
from datamod import pickleData
from datamod import loadMainBatches
from nnmod import maskgen
from nnmod import rpnheatmap
from loss import maskgenloss
from nnmod import maskgen2


def visualizeMaskgen2():

    masknet = maskgen2().cuda()
    masknet.load_state_dict(torch.load(DATA + 'models/' + masknet.__class__.__name__ + '.torch'))
    threshold = nn.Sigmoid().cuda()

    dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader, 0):
        targetmasks = Variable(torch.cuda.FloatTensor(np.array(data['masks'])))
        crops = Variable(torch.cuda.FloatTensor(np.array(data['crops'])))
        targetmasks, crops = targetmasks / 255, crops / 255
        crops = crops[0].view(-1, 1, 32, 32)
        targetmasks = targetmasks[0].view(-1, 1, 32, 32)

        masks = masknet(crops)
        masks = threshold(masks)
        length = len(masks)
        index = np.random.choice(length)

        target = targetmasks.data.cpu()[index, 0].numpy()
        mask = masks.data.cpu()[index, 0, :, :].numpy()
        crop = crops.data.cpu()[index, 0, :, :].numpy()

        th = 0.6
        mask[mask > th] = 1
        mask[mask < th] = 0

        fig = plot.figure()
        fig.add_subplot(1, 3, 1)
        disp1 = plot.imshow(crop)
        fig.add_subplot(1, 3, 2)
        disp2 = plot.imshow(target)
        fig.add_subplot(1, 3, 3)
        disp3 = plot.imshow(mask)
        plot.show()

    pass


def visualizeMaskgen():

    masknet = maskgen().cuda()
    masknet.load_state_dict(torch.load(DATA + 'models/' + masknet.__class__.__name__ + '.torch'))
    heatmap = rpnheatmap().cuda()
    heatmap.load_state_dict(torch.load(DATA + 'models/rpn.torch'))

    dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader, 0):
        targetmasks = Variable(torch.cuda.FloatTensor(np.array(data['masks'])))
        crops = Variable(torch.cuda.FloatTensor(np.array(data['crops'])))
        targetmasks, crops = targetmasks / 255, crops / 255
        crops = crops[0].view(-1, 1, 32, 32)
        targetmasks = targetmasks[0].view(-1, 1, 32, 32)

        features = heatmap(crops)
        masks = masknet(features)
        length = len(masks)
        index = np.random.choice(length)

        target = targetmasks.data.cpu()[index, 0].numpy()
        mask = masks.data.cpu()[index, 0, :, :].numpy()
        crop = crops.data.cpu()[index, 0, :, :].numpy()

        fig = plot.figure()
        fig.add_subplot(1, 3, 1)
        disp1 = plot.imshow(crop)
        fig.add_subplot(1, 3, 2)
        disp2 = plot.imshow(target)
        fig.add_subplot(1, 3, 3)
        disp3 = plot.imshow(mask)
        plot.show()

    pass


def validateMaskgen2(masknet, init=True):

    masknet.cuda()
    if init:
        masknet.load_state_dict(torch.load(DATA + 'models/' + masknet.__class__.__name__ + '.torch'))
    criterion = nn.CrossEntropyLoss().cuda()

    dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    runningloss = 0
    for i, data in enumerate(dataloader, 0):
        targetmasks = Variable(torch.cuda.LongTensor(np.array(data['masks'])))
        targetmasks.requires_grad = False
        crops = Variable(torch.cuda.FloatTensor(np.array(data['crops'])))
        targetmasks, crops = targetmasks / 255, crops / 255
        targetmasks = targetmasks[0].view(-1, 1, 32, 32)
        crops = crops[0].view(-1, 1, 32, 32)
        masks = masknet(crops)
        loss = maskgenloss(targetmasks, masks, criterion)
        runningloss += loss
        if i == 40:
            break
    runningloss = runningloss/40
    print("Cross entropy : {:.4f}".format(runningloss.data[0]))
    pass


def validateMaskgen(masknet, init=True):

    masknet = masknet.cuda()
    if init:
        masknet.load_state_dict(torch.load(DATA + 'models/' + masknet.__class__.__name__ + '.torch'))
    heatmap = rpnheatmap().cuda()
    heatmap.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
    criterion = nn.CrossEntropyLoss().cuda()

    dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    runningloss = 0
    for i, data in enumerate(dataloader, 0):
        targetmasks = Variable(torch.cuda.LongTensor(np.array(data['masks'])))
        targetmasks.requires_grad = False
        crops = Variable(torch.cuda.FloatTensor(np.array(data['crops'])))
        targetmasks, crops = targetmasks / 255, crops / 255
        targetmasks = targetmasks[0].view(-1, 1, 32, 32)
        crops = crops[0].view(-1, 1, 32, 32)

        features = heatmap(crops)
        masks = masknet(features)

        loss = maskgenloss(targetmasks, masks, criterion)
        runningloss += loss
    runningloss = runningloss/len(dataset)
    print(" Mean NLL loss over validation set :  ", runningloss)
    pass


def visualizeRPN(donms=True, o=0.3, topk=200):
    net = rpn().cuda()
    net.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
    # net.load_state_dict(torch.load(DATA + 'models/rpn_tp5tn1.torch'))
    # net.load_state_dict(torch.load(DATA + 'models/rpn_tp7tn3.torch'))

    dataset = rpnDataset(DATA + 'dataset/rpn-validation-set/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for data in dataloader:
        image = torch.cuda.FloatTensor(np.array(data['image']))
        gtbbox = data['bbox'].type(torch.cuda.FloatTensor)

        classification, bboxdelta = net(Variable(image))
        clstargets, bboxtargets, anchors = generateTargets(gtbbox[0], image.size()[-2:], classification.size()[-2:])

        posidx = list(range(0, 2 * len(AnchorShapes) - 1, 2))
        clspos = clstargets[0][posidx].contiguous().view(-1)
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

        postargetidx = (clspos > 0).nonzero()[:, 0].cpu()
        bboxdeltacpu = bboxdelta.data.cpu()
        anchors = center_size(anchors)
        anchors = anchors.cpu()
        bboxarray = bboxdeltacpu + anchors

        bboxarray = point_form(bboxarray)
        objscore = objscore.data.cpu()
        if donms:
            keep, count = nms(bboxarray, objscore, overlap=o, top_k=topk)
            bboxarray = bboxarray[keep[:count]]
            print("Number of proposals : ", count)

        bboxarray = bboxarray.numpy()
        img = np.array(data['image'])
        img = img[0, 0, :, :]
        imgposanchors = np.array(img)
        imgbboxpred = np.array(img)
        gtbbox = gtbbox[0].cpu().numpy()

        fig = plot.figure()

        drawRectP(gtbbox, img)
        disp1 = fig.add_subplot(1, 3, 1)
        disp1.axis('off')
        plot.imshow(img)

        drawRectC(anchors[postargetidx], imgposanchors)
        disp2 = fig.add_subplot(1, 3, 2)
        disp2.axis('off')
        plot.imshow(imgposanchors)

        drawRectP(bboxarray, imgbboxpred)
        disp3 = fig.add_subplot(1, 3, 3)
        disp3.axis('off')
        plot.imshow(imgbboxpred)

        plot.show()
        # break

    pass


def drawRectC(bboxarray, img):
    for bbox in bboxarray:
        cv2.rectangle(img, (int(np.round(bbox[0] - bbox[2] / 2)), int(round(bbox[1] - bbox[3] / 2))),
                      (int(round(bbox[0] + bbox[2] / 2)), int(round(bbox[1] + bbox[3] / 2))),
                      (255, 255, 255), 1)
        pass


def drawRectP(bboxarray, img):
    for bbox in bboxarray:
        cv2.rectangle(img, (int(np.round(bbox[0])), int(round(bbox[1]))),
                      (int(round(bbox[2])), int(round(bbox[3]))),
                      (255, 255, 255), 1)
        pass


def validateRPN(init=True):
    net = rpn().cuda()
    if init:
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


def validateBackboneNet(net, init=True, data_transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.Grayscale(),
                        transforms.ToTensor()])):

    net.cuda()
    _, dataloader = backboneDatasetLoader(batch_size=2000, data_transform=data_transform)

    if init:
        net.load_state_dict(torch.load(DATA+'models/'+net.__class__.__name__+'.torch'))
    criterion = nn.CrossEntropyLoss()

    dataiter = iter(dataloader)
    data = dataiter.next()

    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = net(inputs.cuda())
    loss = criterion(outputs.cpu(), labels)

    # print statistics

    print('Loss over validation set : %.5f' % (loss,))

    pass


def analyzeOriginalCrops():

    load = loadCropBatches()
    shapes = pd.DataFrame(columns=['rows', 'columns'])

    loop_counter = 0
    for data in load:
        for crop in data:
            loop_counter += 1
            rows, columns, _ = np.shape(crop)
            shapes = shapes.append(pd.DataFrame([[rows, columns]], columns=['rows', 'columns']))
        print("Reading original crops ... {} % ...".format((loop_counter*100)//29461))

    pickleData(DATA + 'pickle/analysis/originalCropsStats.p', shapes)
    return shapes


def analyzeImageAndMaks():

    load = loadMainBatches("train")
    image_shapes = pd.DataFrame(columns=['rows', 'columns'])
    mask_sizes = pd.DataFrame(['area'])

    loop_counter = 0
    for data in load:

        for id in list(data.keys()):
            loop_counter += 1
            rows, columns, _ = data[id]['shape']
            image_shapes = image_shapes.append(pd.DataFrame([[rows, columns]], columns=['rows', 'columns']))

            for mask in data[id]['masks']:
                area = np.sum(mask)
                mask_sizes = mask_sizes.append(pd.DataFrame([[area]], columns=['area']))

                pass
        print("Reading original crops ... {} % ...".format((loop_counter * 100) // 670))

    pickleData(DATA+'pickle/analysis/OriginaImagAndMaskStats.p', (image_shapes, mask_sizes))
    return image_shapes, mask_sizes


def visualizeDataN1(data):

    num_examples = np.shape(data['images'])[0]

    example_image_id = np.random.choice(num_examples, 1)[0]

    print("number of training images loaded : ", num_examples)
    print("image id : ", data['id'][example_image_id])
    print("original shape of data : ", data['original_shape'][example_image_id])

    fig = plot.figure()

    fig.add_subplot(1,3,1)
    disp1 = plot.imshow(data['images'][example_image_id])

    fig.add_subplot(1,3,2)
    disp2 = plot.imshow(data['masks'][example_image_id])

    fig.add_subplot(1,3,3)
    disp3 = plot.imshow(data['centroids'][example_image_id])

    plot.show()
