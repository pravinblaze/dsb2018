# This module contains functions and classes for data loading, transformation and analysis

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
from constants import DATA
from torchvision import transforms, datasets
import torchvision
import torch


class rpnDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform=None):
        super().__init__()
        self.filelist = os.listdir(path)
        self.path = path
        self.transform = transform
        pass

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        data = loadPickle(self.path + self.filelist[index])
        data['image'] = np.expand_dims(cv2.cvtColor(data['image'], cv2.COLOR_BGR2GRAY), axis=0)
        if self.transform:
            data = self.transform(data)
        return data


def prepareRPNdataset():

    load = loadMainBatches('train')
    loop_counter = 0
    for data in load:
        for imgid in data.keys():
            loop_counter += 1
            img = data[imgid]['image']
            img = img.astype(np.uint8)
            shape = data[imgid]['shape']
            bbox_array = np.zeros((0, 4), np.uint8)
            mask_array = np.zeros((0, 32, 32), np.uint8)

            for mask in data[imgid]['masks']:

                mask_uint8 = mask.astype(np.uint8)

                im2, contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    x, y, w, h = cv2.boundingRect(contours[0])
                else:
                    # print("Debugging: no contour found ...")
                    pass
                m = 0.3  # margin
                x1 = max(0, round(x - w * m))
                y1 = max(0, round(y - h * m))
                x2 = min(shape[1], round(x + w * (1 + m)) + 1)
                y2 = min(shape[0], round(y + h * (1 + m)) + 1)

                maskcrop = mask_uint8[y1:y2, x1:x2]
                maskcrop = cv2.resize(maskcrop, (32, 32))
                mask_array = np.append(mask_array, np.expand_dims(maskcrop, axis=0), axis=0)

                bbox = np.array([x1, y1, x2, y2])
                bbox_array = np.append(bbox_array, np.expand_dims(bbox, axis=0), axis=0)

            datadict = {"id": imgid, "image": img, "bbox": bbox_array, "masks": mask_array}
            if loop_counter % 10 == 0:
                print("pickling rpn data %.3f %%" % ((100*loop_counter)/670,))
            pickleData(DATA + 'dataset/rpn/' + str(imgid) + '.p', datadict)

    pass


def backboneDatasetLoader(batch_size=100,
                          data_transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.Grayscale(),
                            transforms.ToTensor()])):

    train_dataset = datasets.ImageFolder(root=DATA+'dataset/backbone/training', transform=data_transform)
    validation_dataset = datasets.ImageFolder(root=DATA+'dataset/backbone/validation', transform=data_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True, num_workers=4)

    return train_loader, valid_loader


def encodeCIFAR2jpg():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(DATA + 'data', train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=10)

    loop_counter = 0
    for data in trainloader:
        images, targets = data
        for img in images:
            loop_counter += 1
            img = np.dstack((img[0], img[1], img[2]))
            img = 255 * img
            cv2.imwrite(DATA + 'dataset/backbone/background/' + str(loop_counter) + '.jpg', img.astype(np.uint8))

        if loop_counter > 28000:
            break

        print("Encoding images {} % ...".format((100*loop_counter)//28000))
        pass


def encodeCrop2jpg():

    load = loadCropBatches()

    loop_counter = 0
    for data in load:
        print("Encoding crops ... {} % ...".format((loop_counter * 100) // 29461))
        for crop in data:
            loop_counter += 1
            cv2.imwrite(DATA+'dataset/backbone/nucleus/'+str(loop_counter)+'.jpg', crop.astype(np.uint8))

    pass


def createCropBatches():
    ''' Creates crop images of nuclei from main dataset for training backbone network '''

    load = loadMainBatches('train')
    loop_counter = 0
    crop_list = []
    batch_size = 1000

    for data in load:
        for imgid in data.keys():

            img = data[imgid]['image']
            shape = data[imgid]['shape']

            for mask in data[imgid]['masks']:

                mask_uint8 = mask.astype(np.uint8)

                im2, contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    x, y, w, h = cv2.boundingRect(contours[0])
                else:
                    # print("Debugging: no contour found ...")
                    pass
                m = 0.3  # margin
                x1 = max(0, round(x-w*m))
                y1 = max(0, round(y-h*m))
                x2 = min(shape[1], round(x+w*(1+m))+1)
                y2 = min(shape[0], round(y+h*(1+m))+1)

                crop = img[y1:y2, x1:x2]
                crop_list.append(crop)

                loop_counter += 1
                if loop_counter % batch_size == 0:
                    print("Pickling ... {}% ...".format((loop_counter*100)//29461))
                    pickleData(DATA+'pickle/training_crops/crop'+str(loop_counter//batch_size)+'.p', crop_list)
                    crop_list = []
    if len(crop_list) > 0:
        pickleData(DATA + 'pickle/training_crops/crop' + str(loop_counter // batch_size) + '.p', crop_list)

    pass


class loadCropBatches():

    def __init__(self):
        self.data_pickle_dir = DATA + 'pickle/training_crops/crop'
        self.batch_no = 0
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_no += 1
        data_path = self.data_pickle_dir + str(self.batch_no) + ".p"
        if os.path.isfile(data_path):
            return loadPickle(data_path)
        else:
            raise StopIteration


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


def createMainDataBatches(data_set='train'):
    ''' Creates batches of unaltered data and pickles them with the following structure
        {id :{'image':numpyArray, 'shape':tupple, 'masks':numpyArray}, ...}
    '''

    data_set_paths = {'train': 'stage1_train/', 'test': 'stage1_test/'}
    data_dir = DATA + data_set_paths[data_set]
    data_pickle_paths = {'train': 'pickle/main_train/', 'test': 'pickle/main_test/'}
    data_pickle_dir = DATA + data_pickle_paths[data_set]
    img_id_list = os.listdir(data_dir)
    batch_size = 10
    image_list_size = len(img_id_list)

    # img_id_list = img_id_list[:5]

    data = dict()
    loop_counter = 0

    for img_id in img_id_list:

        # reading image data
        path = data_dir + img_id + '/images/' + img_id + '.png'
        img = cv2.imread(path)
        img_shape = np.shape(img)
        print("pickling image {} of {} ".format(loop_counter,image_list_size))
        print("image shape : ", img_shape)

        mask_array = np.zeros((0,)+img_shape[:-1], np.float32)

        data[img_id] = {'image': img, 'shape': img_shape}

        # reading training masks
        if data_set == 'train':

            mask_dir = data_dir + img_id + '/masks/'
            mask_id_list = os.listdir(mask_dir)
            mask_list_size = len(mask_id_list)
            print("number of masks : ", mask_list_size)

            for mask_id in mask_id_list:

                path = data_dir + img_id + '/masks/' + mask_id
                mask = cv2.imread(path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_array = np.append(mask_array, np.expand_dims(mask, axis=0), axis=0)

            data[img_id]['masks'] = mask_array

            pass

        loop_counter += 1
        if loop_counter % batch_size == 0:

            pickleData(data_pickle_dir+'dataMain'+str(loop_counter//batch_size)+'.p', data)
            data = dict()

            pass
    if len(list(data.keys())) > 0:
        pickleData(data_pickle_dir + 'dataMain' + str(loop_counter // batch_size) + '.p', data)


class loadMainBatches():

    def __init__(self, data_set='train'):
        self.data_pickle_paths = {'train': 'pickle/main_train/', 'test': 'pickle/main_test/'}
        self.data_pickle_dir = DATA + self.data_pickle_paths[data_set]
        self.batch_no = 0
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_no += 1
        data_path = self.data_pickle_dir + "dataMain" + str(self.batch_no) + ".p"
        if os.path.isfile(data_path):
            return loadPickle(data_path)
        else:
            raise StopIteration


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


def loadDataN1(data_set='train'):
    ''' Data loader '''
    data_set_path = {'train': 'stage1_train/', 'test': 'stage1_test/'}
    data_dir = DATA + data_set_path[data_set]
    img_id_list = os.listdir(data_dir)

    # img_id_list = img_id_list[:5]

    data = dict()
    image_id = []
    image_data = np.zeros((0, 300, 300), np.float32)
    mask_data = np.zeros((0, 300, 300), np.float32)
    original_shape = np.zeros((0, 2), np.int)
    mask_centroids = np.zeros((0, 300, 300), np.float32)

    for img_id in img_id_list:

        image_id.append(img_id)

        # reading image data
        path = data_dir + img_id + '/images/' + img_id + '.png'
        img = cv2.imread(path)
        original_shape = np.append(original_shape, np.expand_dims(np.shape(img)[:-1], axis=0), axis=0)

        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img_g / 255
        img = cv2.resize(img, (300, 300))

        image_data = np.append(image_data, np.expand_dims(img, axis=0), axis=0)

        # reading training masks
        if data_set == 'train':
            mask_dir = data_dir + img_id + '/masks/'
            mask_id_list = os.listdir(mask_dir)

            mask_append = np.zeros((300, 300), np.float32)
            centroid_append = np.zeros((300, 300), np.float32)

            for mask_id in mask_id_list:

                path = data_dir + img_id + '/masks/' + mask_id
                mask = cv2.imread(path)
                mask_g = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = mask_g / 255
                mask = cv2.resize(mask, (300, 300))
                mask_append += mask

                # creating centroid maps

                centroid = np.zeros((300, 300))
                M = cv2.moments(mask)
                if (M['m00'] != 0):
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    s = 1
                    centroid[cy - s:cy + s, cx - s:cx + s] = 1.0
                    centroid_append += centroid

            mask_data = np.append(mask_data, np.expand_dims(mask_append, axis=0), axis=0)
            mask_centroids = np.append(mask_centroids, np.expand_dims(centroid_append, axis=0), axis=0)

    data['id'] = image_id
    data['images'] = image_data
    data['masks'] = mask_data
    data['original_shape'] = original_shape
    data['centroids'] = mask_centroids

    return data


def visualizeDataN1(data):

    num_examples = np.shape(data['images'])[0]

    example_image_id = np.random.choice(num_examples, 1)[0]

    print("number of training images loaded : ", num_examples)
    print("image id : ", data['id'][example_image_id])
    print("original shape of data : ", data['original_shape'][example_image_id])

    fig = plt.figure()

    fig.add_subplot(1,3,1)
    disp1 = plt.imshow(data['images'][example_image_id])

    fig.add_subplot(1,3,2)
    disp2 = plt.imshow(data['masks'][example_image_id])

    fig.add_subplot(1,3,3)
    disp3 = plt.imshow(data['centroids'][example_image_id])

    plt.show()

# Serialization and de-serialization code


def pickleData(fileName, obj):
    with open(fileName, 'wb') as pickleOut:
        pickle.dump(obj, pickleOut)


def loadPickle(fileName):
    with open(fileName,'rb') as pickleIn:
        return pickle.load(pickleIn)
