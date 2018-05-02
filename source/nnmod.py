# This module has functions and classes related to neural network implementation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import DATA
from constants import AnchorShapes


class maskgen2(nn.Module):
    def __init__(self):
        super(maskgen2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 9, 1, 4)
        self.conv2 = nn.Conv2d(16, 16, 9, 1, 4)
        self.conv3 = nn.Conv2d(16, 16, 9, 1, 4)
        self.conv4 = nn.Conv2d(16, 16, 9, 1, 4)
        self.conv5 = nn.Conv2d(16, 16, 9, 1, 4)
        self.conv6 = nn.Conv2d(16, 16, 9, 1, 4)
        self.conv7 = nn.Conv2d(16, 16, 9, 1, 4)
        self.conv8 = nn.Conv2d(16, 1, 9, 1, 4)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(1)
        pass

    def forward(self, x):

        out1 = F.relu(self.bn16(self.conv1(x)))
        out2 = F.relu(self.bn16(self.conv2(out1)))
        out3 = F.relu(self.bn16(self.conv3(out2)))
        out4 = F.relu(self.bn16(self.conv4(out3)))
        out5 = F.relu(self.bn16(self.conv5(out4)))
        out5 = F.relu(self.bn16(out5+out3))
        out6 = F.relu(self.bn16(self.conv6(out5)))
        out6 = F.relu(self.bn16(out6+out2))
        out7 = F.relu(self.bn16(self.conv7(out6)))
        out7 = F.relu(self.bn16(out7+out1))
        out = F.relu(self.bn1(self.conv8(out7)))

        return out


class maskgen(nn.Module):
    def __init__(self):
        super(maskgen, self).__init__()
        self.conv00 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv01 = nn.Conv2d(32, 32, 3, 1, 1)
        self.tconv1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.tconv2 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.tconv3 = nn.ConvTranspose2d(8, 4, 2, 2)
        self.conv11 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv21 = nn.Conv2d(8, 8, 3, 1, 1)
        self.conv22 = nn.Conv2d(8, 8, 3, 1, 1)
        self.conv31 = nn.Conv2d(4, 4, 3, 1, 1)
        self.conv32 = nn.Conv2d(4, 1, 3, 1, 1)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(4)
        self.bn1 = nn.BatchNorm2d(1)

    def forward(self, heatmap):     # assuming heatmap in shape 32*4*4
        out1 = F.relu(self.bn32(self.conv00(heatmap)))
        out1 = F.relu(self.bn32(self.conv01(out1)))
        out1 = F.relu(self.bn16(self.tconv1(out1)))  # out 8*8
        out1 = F.relu(self.bn16(self.conv11(out1)))
        out1 = F.relu(self.bn16(self.conv12(out1)))

        out2 = F.relu(self.bn8(self.tconv2(out1)))  # out 16*16
        out2 = F.relu(self.bn8(self.conv21(out2)))
        out2 = F.relu(self.bn8(self.conv22(out2)))

        out3 = F.relu(self.bn4(self.tconv3(out2)))  # out 32*32
        out3 = F.relu(self.bn4(self.conv31(out3)))
        out3 = F.relu(self.bn1(self.conv32(out3)))

        return out3


class nucleusDetect(nn.Module):

    def __init__(self):
        super(nucleusDetect, self).__init__()
        self.fc = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        x = x.view(-1, 32 * 4 * 4)
        out = self.fc(x)
        return out


class rpnheatmap(nn.Module):

    def __init__(self):
        super(rpnheatmap, self).__init__()
        self.k = len(AnchorShapes)
        self.convdepth = 128
        self.backbone = backboneFeature()
        self.bn = nn.BatchNorm2d(self.convdepth)
        self.bn2k = nn.BatchNorm2d(2*self.k)
        self.bn4k = nn.BatchNorm2d(4*self.k)
        self.c = nn.Conv2d(32, self.convdepth, 3, padding=1)
        self.classify = nn.Conv2d(self.convdepth, 2*self.k, 1)
        self.boxdelta = nn.Conv2d(self.convdepth, 4*self.k, 1)
        pass

    def forward(self, x):
        features = self.backbone(x)
        return features
    pass


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
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8*16*16, 2)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))  # out - 8*32*32

        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))  # out - 8*32*32
        out4 = F.relu(self.conv4(out3))
        out5 = F.relu(self.conv5(out4))
        out5 = self.pool(out5)  # out - 8*16*16
        out5 = out5.view(-1, 8*16*16)

        out = F.relu(self.fc(out5))

        return out


def xavier_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)


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


def linearFilter(k, direction=0):
    ''' takes k as an integer'''
    filter = np.zeros((k, k))

    coeff = {0: (1, 0), 1: (1, -1), 2: (0, -1), 3: (-1, -1), 4: (-1, 0), 6: (0, 1), 5: (-1, 1), 7: (1, 1)}
    x, y = 0, 0
    b, a = coeff[direction]
    shift = k // 2

    for i in range(np.shape(filter)[0]):
        for j in range(np.shape(filter)[1]):

            update = 0
            y = -(i - shift) * b
            x = (j - shift) * a
            line = x + y

            if (line > 0):
                update = 1
            elif (line == 0):
                pass
            else:
                update = -1

            filter[i, j] = update
    return filter