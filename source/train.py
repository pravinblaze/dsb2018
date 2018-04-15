import time
import torch
from torch.autograd import Variable
import torch.optim as optimize
import numpy as np
from constants import DATA
from nnmod import rpn
from datamod import rpnDataset
from generate import generateTargets
from loss import classificationloss
from loss import bboxdeltaloss
from torchvision import transforms
from nnmod import xavier_init
from datamod import backboneDatasetLoader
from torch import nn


def trainRPN(epochs=4, a=0.001, b=0.1, train_backbone=False, init_backbone=False, init_rpn=False, save_model=False):

    start_time = time.time()
    net = rpn(train_backbone, init_backbone).cuda()
    if init_rpn:
        net.load_state_dict(torch.load(DATA + 'models/rpn.torch'))
    dataset = rpnDataset(DATA+'dataset/rpn/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    criterioncls = torch.nn.CrossEntropyLoss().cuda()
    criterionbbox = torch.nn.L1Loss().cuda()
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
