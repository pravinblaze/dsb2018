# This module has functions and classes related to neural network implementation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimize
from torchvision import transforms
from torch.autograd import Variable
from datamod01 import backboneDatasetLoader
from constants import DATA
from constants import AnchorShapes


class rpn(nn.Module):

    def __init__(self, train_backbone=False, init_backbone=False):
        super(rpn, self).__init__()
        self.k = len(AnchorShapes)
        self.convdepth = 128    # arbitrary convolution depth for rpn first layer
        self.backbone = backboneFeature()
        if init_backbone:
            self.backbone.load_state_dict(torch.load(DATA + 'models/backbone02.torch'))
        if not train_backbone:
            for p in self.backbone.parameters():    # turning off backprop on backbone
                p.requires_grad = False
        self.bn = nn.BatchNorm2d(self.convdepth)
        self.bn2k = nn.BatchNorm2d(2*self.k)
        self.bn4k = nn.BatchNorm2d(4*self.k)
        self.c = nn.Conv2d(32, self.convdepth, 3, padding=1)
        self.classify = nn.Conv2d(self.convdepth, 2*self.k, 1)
        self.boxdelta = nn.Conv2d(self.convdepth, 4*self.k, 1)
        pass

    def forward(self, x):

        features = self.backbone(x)
        x = F.relu(self.bn(self.c(features)))
        classification = F.relu(self.bn2k(self.classify(x)))
        boxdelta = F.relu(self.bn4k(self.boxdelta(x)))

        return classification, boxdelta
    pass


class backboneFeature(nn.Module):
    def __init__(self):
        super().__init__()

        self.r = F.relu
        self.pool = nn.MaxPool2d(2, 2)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)

        self.c1 = nn.Conv2d(1, 8, 5, padding=2)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.c3 = nn.Conv2d(8, 8, 3, padding=1)
        self.c4 = nn.Conv2d(8, 8, 3, padding=1)
        self.c5 = nn.Conv2d(8, 8, 3, padding=1)
        self.c6 = nn.Conv2d(8, 16, 3, padding=1)
        self.c7 = nn.Conv2d(16, 16, 3, padding=1)
        self.c8 = nn.Conv2d(16, 16, 3, padding=1)
        self.c9 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32*4*4, 2)

        pass

    def forward(self, x):
        # assuming x is 1*32*32

        out1 = self.r(self.bn8(self.c1(x)))
        out1 = self.pool(out1)  # s 8*16*16 f 6

        out2 = self.r(self.bn8(self.c2(out1)))
        out3 = self.r(self.bn8(self.c3(out2)))
        out3 = self.r(self.bn8(out3+out1))

        out4 = self.r(self.bn8(self.c2(out3)))
        out5 = self.r(self.bn8(self.c3(out4)))
        out5 = self.r(self.bn8(out5 + out3))

        out6 = self.r(self.bn16(self.c6(out5)))
        out6 = self.pool(out6)  # s 16*8*8 f 52

        out7 = self.r(self.bn16(self.c7(out6)))
        out8 = self.r(self.bn16(self.c8(out7)))
        out8 = self.r(self.bn16(out8 + out6))

        out9 = self.r(self.bn32(self.c9(out8)))
        out9 = self.pool(out9)  # s 32*4*4 f 152

        return out9
    pass


class backbone02(nn.Module):

    def __init__(self):
        super().__init__()

        self.r = F.relu
        self.pool = nn.MaxPool2d(2, 2)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)

        self.c1 = nn.Conv2d(1, 8, 5, padding=2)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.c3 = nn.Conv2d(8, 8, 3, padding=1)
        self.c4 = nn.Conv2d(8, 8, 3, padding=1)
        self.c5 = nn.Conv2d(8, 8, 3, padding=1)
        self.c6 = nn.Conv2d(8, 16, 3, padding=1)
        self.c7 = nn.Conv2d(16, 16, 3, padding=1)
        self.c8 = nn.Conv2d(16, 16, 3, padding=1)
        self.c9 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32*4*4, 2)

        self.heatmap = None

        pass

    def forward(self, x):
        # assuming x is 1*32*32

        out1 = self.r(self.bn8(self.c1(x)))
        out1 = self.pool(out1)  # s 8*16*16 f 6

        out2 = self.r(self.bn8(self.c2(out1)))
        out3 = self.r(self.bn8(self.c3(out2)))
        out3 = self.r(self.bn8(out3+out1))

        out4 = self.r(self.bn8(self.c2(out3)))
        out5 = self.r(self.bn8(self.c3(out4)))
        out5 = self.r(self.bn8(out5 + out3))

        out6 = self.r(self.bn16(self.c6(out5)))
        out6 = self.pool(out6)  # s 16*8*8 f 52

        out7 = self.r(self.bn16(self.c7(out6)))
        out8 = self.r(self.bn16(self.c8(out7)))
        out8 = self.r(self.bn16(out8 + out6))

        out9 = self.r(self.bn32(self.c9(out8)))
        out9 = self.pool(out9)  # s 32*4*4 f 152

        out9flat = out9.view(-1, 32 * 4 * 4)
        out = self.r(self.fc(out9flat))

        return out


class backbone01(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv5 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv6 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8*8*8, 2)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        out1 = F.relu(self.bn(self.conv1(x)))  # out - 8*32*32

        out2 = F.relu(self.bn(self.conv2(out1)))
        out3 = self.conv3(out2)
        out3 = F.relu(self.bn(out3 + out1))  # out - 8*32*32

        out4 = F.relu(self.bn(self.conv4(out3)))
        out5 = self.conv5(out4)
        out5 = F.relu(self.bn(out3 + out5))
        out5 = self.pool(out5)  # out - 8*16*16

        out6 = F.relu(self.bn(self.conv6(out5)))
        out6 = self.pool(out6)  # out - 8*8*8
        out6 = out6.view(-1, 8*8*8)

        out = F.relu(self.fc(out6))

        return out


def xavier_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)


def trainBackbone(net, batch_size=100,
                  epochs=3,
                  a=0.0001, b=0.1,
                  save_model=False,
                  data_transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.Grayscale(),
                        transforms.ToTensor()])):

    net.apply(xavier_init)  # Applying xavier normal initialization on convolution layer parameters
    net.cuda()

    trainloader, validloader = backboneDatasetLoader(batch_size=batch_size, data_transform=data_transform)
    criterion = nn.CrossEntropyLoss()

    validiter = iter(validloader)
    validdata = validiter.next()
    validinputs, validlabels = validdata
    validinputs = Variable(validinputs).cuda()
    validlabels = Variable(validlabels)

    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer = optimize.Adam(net.parameters(), lr=a * (b ** (epoch//2)))
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.cpu(), labels)
            loss.backward()
            optimizer.step()

            # validataion loss
            validoutputs = net(validinputs)
            validloss = criterion(validoutputs.cpu(), validlabels)

            # print statistics
            if i % (1+1000//batch_size) == 0:
                print('Epoch %d, batch %5d, training loss: %.5f, validation loss: %.5f'
                      % (epoch + 1, i + 1, loss.data[0], validloss.data[0]))

    print('Training finished successfully !')
    if save_model:
        print('Saving model to disk ... ')
        torch.save(net.state_dict(), DATA+'models/'+net.__class__.__name__+'.torch')
        print('Model saved to disk !')

    pass


def validateBackboneNet(net, data_transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.Grayscale(),
                        transforms.ToTensor()])):

    net.cuda()
    _, dataloader = backboneDatasetLoader(batch_size=2000, data_transform=data_transform)

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


class singleFilterConv(nn.Module):

    def __init__(self, w):
        ''' Takes w as a 2d numpy array
            Constructing network for convolution of hand engineered feature on input image '''
        super().__init__()
        # 1 input image channel, 1 output channels, 7x7 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 1, np.shape(w))

        w = torch.from_numpy(w).float()
        w = w.unsqueeze(0)
        w = w.unsqueeze(0)

        self.conv1.weight = nn.Parameter(w)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = F.relu(x)
        return x