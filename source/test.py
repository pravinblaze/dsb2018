''' Testing solution '''

# from solution import solve
# solve()

''' Testing maskgen2 '''

# from train import trainmaskgen2
# trainmaskgen2(epochs=4,
#               a=0.001, b=0.1, d=1,
#               save_model=True)

from testcode import validateMaskgen2
from nnmod import maskgen2
validateMaskgen2(maskgen2(), init=True, perturb=True)

# from testcode import visualizeMaskgen2
# visualizeMaskgen2()

''' Testing maskgen '''

# from testcode import validateMaskgen
# from nnmod import maskgen
# validateMaskgen(maskgen(), init=True)

# from testcode import visualizeMaskgen
# visualizeMaskgen()

''' Train maskgen '''


# from train import trainMaskgen
#
# trainMaskgen(epochs=4, a=0.001, b=0.1, d=1)

''' Train nucleusDetect '''

# from train import trainNucleusDetect
# trainNucleusDetect(epochs=4, batch_size=100, a=0.01, b=0.1)

''' Testing visualizerpn '''

# from testcode import visualizeRPN
# visualizeRPN(donms=True, o=0.05, topk=200, perturb=True)

''' Testing validateRPN '''

# from testcode import validateRPN
# validateRPN(init=True, perturb=True)

''' Testing trainRPN '''

from train import trainRPN
#
# trainRPN(epochs=4, a=0.001, b=0.1,
#          train_backbone=True,
#          init_backbone=False,
#          init_rpn=False,
#          save_model=True)

''' Testing generateAnchors '''

# from generate_targets import generateAnchros
# anchors = generateAnchros([100, 100], [10, 10])
# print(anchors)

''' Testing generateTargets '''

# from generate import generateTargets
# from datamod import rpnDataset
# from constants import DATA
# import torch
#
# dataset = rpnDataset(DATA+'dataset/rpn/')
# data = dataset.__getitem__(0)
# # data 0 has shape 256, 256 so using heat shape 32
# gtbbox = torch.FloatTensor(data['bbox'])
# generateTargets(gtbbox, [256, 256], [32, 32])


''' Testing jaccard i.e, iou function '''
# import torch
# from generate import jaccard
# box_a = torch.FloatTensor([[0, 0, 2, 2], [0, 0, 1, 1]])
# box_b = torch.FloatTensor([[1, 0, 3, 2], [0, 0, 1, 1]])
# iou = jaccard(box_a, box_b)
# print('iou = ', iou)
# pass

''' Testing rpnDataset '''

# import torch
# from constants import DATA
# from datamod import rpnDataset
# import numpy as np
# import matplotlib.pyplot as plot
# from testcode import drawRectP
#
# dataset = rpnDataset(DATA+'dataset/rpn/')
#
# data = dataset.__getitem__(0)
# print('Size of data set :', len(dataset))
#
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
#                                          shuffle=True)
#
# for data in dataloader:
#     image = data['image'][0, 0, :, :]
#     gtbbox = data['bbox'][0]
#     drawRectP(gtbbox, image.numpy())
#     1+1
#     pass
#
# 1+1
# pass


''' Testing prepareRPNdataset '''

# from datamod import prepareRPNdataset
#
# prepareRPNdataset(dataset='final')

# import os
# from datamod import loadPickle
# from constants import DATA
# import random
#
# paths = os.listdir(DATA+'dataset/rpn/')
# path = DATA+'dataset/rpn/' + random.choice(paths)
# data = loadPickle(path)
# print(data)

''' Testing backboneFeatures '''

# from datamod import backboneDatasetLoader
# from nnmod import backboneFeature
# import torch
# from constants import DATA
# from torch.autograd import Variable
#
# loader, _ = backboneDatasetLoader()
# dataiter = iter(loader)
# data = dataiter.next()
# images, targets = data
#
# net = backboneFeature().cuda()
# net.load_state_dict(torch.load(DATA + 'models/backbone02.torch'))
#
# features = net(Variable(images).cuda())
# features = features.cpu()
#
# print(features)
# pass


''' Testing backbone02 '''

# from nnmod import backbone02
# from train import trainBackbone
# from testcode import validateBackboneNet
#
# # trainBackbone(backbone02(), batch_size=100, epochs=3,
# #               a=0.0001, b=0.1,
# #               save_model=True)
# #
# validateBackboneNet(backbone02(), init=True, perturb=False)

''' Testing validateBackboneNet '''

# from testcode import validateBackboneNet
# from nnmod import backbone01
#
# validateBackboneNet(backbone01())

''' Testing trainBackbone01'''

# from train import trainBackbone
# from nnmod import backbone01
#
# trainBackbone(backbone01(), batch_size=100, epochs=4,
#                 a=0.001, b=0.1,
#                 save_model=True)


''' Testing backboneDatasetLoader '''

# from datamod import backboneDatasetLoader
#
# loader, _ = backboneDatasetLoader()
# for data in loader:
#     images, targets = data
#     for img in images:
#         pass
#     pass


''' Testing encodeCIFAR2jpg '''

# from datamod import encodeCIFAR2jpg
# encodeCIFAR2jpg()

''' Testing encodeCrop2jpg '''

# from datamod import encodeCrop2jpg
# encodeCrop2jpg()

''' Testing analyzeOriginalCrops '''

# from datamod import  analyzeOriginalCrops
# shapes = analyzeOriginalCrops()
#
# pass
#
# import matplotlib.pyplot as plt
# from datamod import loadPickle
# from constants import DATA
#
# shapes = loadPickle(DATA+'pickle/analysis/originalCropsStats.p')
#
# fig = plt.figure()
# r = fig.add_subplot(1, 2, 1)
# r.set_xlabel('Row size')
# shapes['rows'].plot.hist()
# c = fig.add_subplot(1, 2, 2)
# c.set_xlabel('Column size')
# shapes['columns'].plot.hist()
# plt.show()

''' Testing createCropBatches '''

# from datamod import createCropBatches
#
# createCropBatches()

''' Testing createMainDataBatches '''

# from datamod import createMainDataBatches
# from datamod import loadPickle
# from constants import DATA
# createMainDataBatches('valid')
# print("Created mini data batches successfully")

# data = loadPickle(DATA+'pickle/main_test/dataMain1.p')
# print("debugging ...")

''' Testing analyzeImageAndMaks on main data sets '''

# from datamod import analyzeImageAndMaks
# from datamod import loadPickle
# from constants import DATA
# import matplotlib.pyplot as plot
# shapes, sizes = analyzeImageAndMaks()
# shapes, sizes = loadPickle(DATA+'pickle/analysis/OriginaImagAndMaskStats.p')
# fig = plot.figure()
# r = fig.add_subplot(1, 3, 1)
# r.set_xlabel('Row size')
# shapes['rows'].plot.hist()
# c = fig.add_subplot(1, 3, 2)
# c.set_xlabel('Column size')
# shapes['columns'].plot.hist()
# a = fig.add_subplot(1, 3, 3)
# a.set_xlabel('Mask area')
# sizes['area'].plot.hist()
# plot.show()
