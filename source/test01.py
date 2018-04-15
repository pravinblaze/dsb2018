''' Testing visualizerpn '''

# from rpn import visualizeRPN
# visualizeRPN()

''' Testing validateRPN '''

# from rpn import validateRPN
# validateRPN()

''' Testing trainRPN '''

from rpn import trainRPN

trainRPN(epochs=4, a=0.001, b=0.1,
         train_backbone=True,
         init_backbone=False,
         init_rpn=False,
         save_model=True)

''' Testing generateAnchors '''

# from generate_targets import generateAnchros
# anchors = generateAnchros([100, 100], [10, 10])
# print(anchors)

''' Testing generateTargets '''

# from src.generate_targets import generateTargets
# from src.datamod01 import rpnDataset
# from src.constants import DATA
# import torch
#
# dataset = rpnDataset(DATA+'dataset/rpn/')
# data = dataset.__getitem__(0)
# # data 0 has shape 256, 256 so using heat shape 32
# gtbbox = torch.FloatTensor(data['bbox'])
# generateTargets(gtbbox, [256, 256], [32, 32])


''' Testing jaccard i.e, iou function '''
# import torch
# from src.generate_targets import jaccard
# box_a = torch.FloatTensor([[0, 0, 2, 2], [0, 0, 1, 1]])
# box_b = torch.FloatTensor([[1, 0, 3, 2], [0, 0, 1, 1]])
# iou = jaccard(box_a, box_b)
# print('iou = ', iou)
# pass

''' Testing rpnDataset '''

# import torch
# from constants import DATA
# from datamod01 import rpnDataset
# import numpy as np
# import matplotlib.pyplot as plot
# from rpn import drawRectP
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

# from datamod01 import prepareRPNdataset
#
# prepareRPNdataset()

# import os
# from src.datamod01 import loadPickle
# from src.constants import DATA
# import random
#
# paths = os.listdir(DATA+'dataset/rpn/')
# path = DATA+'dataset/rpn/' + random.choice(paths)
# data = loadPickle(path)
# print(data)

''' Testing backboneFeatures '''

# from src.datamod01 import backboneDatasetLoader
# from src.nnmod01 import backboneFeature
# import torch
# from src.constants import DATA
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

# from src.nnmod01 import backbone02
# from src.nnmod01 import trainBackbone
# from src.nnmod01 import validateBackboneNet
#
# # trainBackbone(backbone02(), batch_size=100, epochs=4,
# #               a=0.0001, b=0.1,
# #               save_model=True)
#
# validateBackboneNet(backbone02())

''' Testing validateBackboneNet '''

# from src.nnmod01 import validateBackboneNet
# from src.nnmod01 import backbone01
#
# validateBackboneNet(backbone01())

''' Testing trainBackbone01'''

# from src.nnmod01 import trainBackbone01
# from src.nnmod01 import backbone01
#
# trainBackbone(backbone01(), batch_size=100, epochs=4,
#                 a=0.0001, b=0.1,
#                 save_model=True)


''' Testing backboneDatasetLoader '''

# from src.datamod01 import backboneDatasetLoader
#
# loader, _ = backboneDatasetLoader()
# for data in loader:
#     images, targets = data
#     for img in images:
#         pass
#     pass


''' Testing encodeCIFAR2jpg '''

# from src.datamod01 import encodeCIFAR2jpg
# encodeCIFAR2jpg()

''' Testing encodeCrop2jpg '''

# from src.datamod01 import encodeCrop2jpg
# encodeCrop2jpg()

''' Testing analyzeOriginalCrops '''

# from src.datamod01 import  analyzeOriginalCrops
# shapes = analyzeOriginalCrops()
#
# pass

# import matplotlib.pyplot as plt
# from src.datamod01 import loadPickle
# from src.constants import DATA
#
# shapes = loadPickle(DATA+'pickle/analysis/originalCropsStats.p')
#
# fig = plt.figure()
#
# fig.add_subplot(1, 2, 1)
# disp1 = shapes['rows'].plot.hist()
#
# fig.add_subplot(1, 2, 2)
# disp2 = shapes['columns'].plot.hist()
#
# plt.show()

''' Testing createCropBatches '''

# from src.datamod01 import createCropBatches
#
# createCropBatches()

''' Testing createMainDataBatches '''

# from src.datamod01 import createMainDataBatches
# from src.datamod01 import loadPickle
# from src.constants import DATA
# createMainDataBatches('train')
# print("Created mini data batches successfully")

# data = loadPickle(DATA+'pickle/main_test/dataMain1.p')
# print("debugging ...")

''' Testing analyzeImageAndMaks on main data sets '''

# from src.datamod01 import analyzeImageAndMaks
# shapes, sizes = analyzeImageAndMaks()
# shapes['rows'].plot.hist()
# shapes['columns'].plot.hist()
# sizes['area'].plot.hist()
