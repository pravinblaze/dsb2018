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

    def __init__(self, path, transform=None, perturb=False):
        super().__init__()
        self.filelist = os.listdir(path)
        self.path = path
        self.transform = transform
        self.perturb = perturb
        pass

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        data = loadPickle(self.path + self.filelist[index])
        data['image'] = np.expand_dims(cv2.cvtColor(data['image'], cv2.COLOR_BGR2GRAY), axis=0)
        if self.perturb:
            rnd = np.random.randint(0, 25, data['image'].shape)
            data['image'] = data['image'].astype(np.int) + rnd
            data['image'][data['image']>255] = 255
            data['image'] = data['image'].astype(np.uint8)
        if self.transform:
            data = self.transform(data)
        return data


def prepareRPNdataset():

    datasetlist = ['train', 'valid']
    if not os.path.exists(DATA + 'dataset/rpn/'):
        os.makedirs(DATA + 'dataset/rpn/')
    if not os.path.exists(DATA + 'dataset/rpn-valid/'):
        os.makedirs(DATA + 'dataset/rpn-valid/')
    for dataset in datasetlist:
        print("Praring RPN Dataset: "+dataset)
        load = loadMainBatches(dataset)
        loop_counter = 0
        for data in load:
            loop_counter += 1
            for imgid in data.keys():
                img = data[imgid]['image']
                img = img.astype(np.uint8)
                imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                shape = data[imgid]['shape']
                bbox_array = np.zeros((0, 4), np.uint8)
                mask_array = np.zeros((0, 32, 32), np.uint8)
                crop_array = np.zeros((0, 32, 32), np.uint8)

                for mask in data[imgid]['masks']:

                    mask_uint8 = mask.astype(np.uint8)

                    im2, contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        x, y, w, h = cv2.boundingRect(contours[0])
                    else:
                        # print("Debugging: no contour found ...")
                        pass
                    m = 0.1  # margin
                    x1 = max(0, round(x - w * m))
                    y1 = max(0, round(y - h * m))
                    x2 = min(shape[1], round(x + w * (1 + m)))
                    y2 = min(shape[0], round(y + h * (1 + m)))

                    maskcrop = mask_uint8[y1:y2, x1:x2]
                    maskcrop = cv2.resize(maskcrop, (32, 32))
                    imcrop = imgray[y1:y2, x1:x2]
                    imcrop = cv2.resize(imcrop, (32, 32))
                    mask_array = np.append(mask_array, np.expand_dims(maskcrop, axis=0), axis=0)
                    crop_array = np.append(crop_array, np.expand_dims(imcrop, axis=0), axis=0)

                    bbox = np.array([x1, y1, x2, y2])
                    bbox_array = np.append(bbox_array, np.expand_dims(bbox, axis=0), axis=0)

                    datadict = {"id": imgid, "image": img, "bbox": bbox_array, "masks": mask_array, "crops": crop_array}
                if dataset == 'train':
                    pickleData(DATA + 'dataset/rpn/' + str(imgid) + '.p', datadict)
                if dataset == 'valid':
                    pickleData(DATA + 'dataset/rpn-valid/' + str(imgid) + '.p', datadict)
            print("pickling rpn data {:.3f} %".format(100*loop_counter/len(load)))
        print("Done !")

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
    trainset = torchvision.datasets.CIFAR10(DATA + 'dataset/', train=True, transform=transform, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=10)
    if not os.path.exists(DATA + 'dataset/backbone/training/background/'):
        os.makedirs(DATA + 'dataset/backbone/training/background/')
    if not os.path.exists(DATA + 'dataset/backbone/validation/background/'):
        os.makedirs(DATA + 'dataset/backbone/validation/background/')
    print('Encoding CIFAR10 images as negative examples for backbone network')
    loop_counter = 0
    for data in trainloader:
        images, targets = data
        for img in images:
            loop_counter += 1
            img = np.dstack((img[0], img[1], img[2]))
            img = 255 * img
            if loop_counter > 2800:
                cv2.imwrite(DATA + 'dataset/backbone/training/background/'
                            + str(loop_counter) + '.jpg', img.astype(np.uint8))
            else:
                cv2.imwrite(DATA + 'dataset/backbone/validation/background/'
                            + str(loop_counter) + '.jpg', img.astype(np.uint8))

        if loop_counter > 28000:
            break

        print("Encoding images {} % ...".format((100*loop_counter)//28000))
        pass


def encodeCrop2jpg():

    load = loadCropBatches()
    loop_counter = 0
    if not os.path.exists(DATA+'dataset/backbone/training/nucleus/'):
        os.makedirs(DATA+'dataset/backbone/training/nucleus/')
    if not os.path.exists(DATA + 'dataset/backbone/validation/nucleus/'):
        os.makedirs(DATA + 'dataset/backbone/validation/nucleus/')
    print("Encoding nuclei crop images as positive example for backbone network")
    i = 0
    for data in load:
        print("Encoding crops ... {} % ...".format((loop_counter * 100) // 29461))
        i += 1
        for crop in data:
            loop_counter += 1
            if i > 1:
                cv2.imwrite(DATA+'dataset/backbone/training/nucleus/'
                            + str(loop_counter)+'.jpg', crop.astype(np.uint8))
            else:
                cv2.imwrite(DATA + 'dataset/backbone/validation/nucleus/'
                            + str(loop_counter) + '.jpg', crop.astype(np.uint8))
    pass


class loadCropBatches:

    def __init__(self):
        self.data_pickle_dir = DATA + 'pickle/crops/'
        self.batch_no = 0
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_no += 1
        data_path = self.data_pickle_dir + 'crop' + str(self.batch_no) + ".p"
        if os.path.isfile(data_path):
            return loadPickle(data_path)
        else:
            raise StopIteration


def createCropBatches():
    ''' Creates crop images of nuclei from main dataset for training backbone network '''

    load = loadMainBatches('train')
    loop_counter = 0
    crop_list = []
    batch_size = 1000

    if not os.path.exists(DATA + 'pickle/crops/'):
        os.makedirs(DATA + 'pickle/crops/')

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
                    pickleData(DATA+'pickle/crops/crop'+str(loop_counter//batch_size)+'.p', crop_list)
                    crop_list = []
    if len(crop_list) > 0:
        pickleData(DATA + 'pickle/crops/crop' + str(loop_counter // batch_size) + '.p', crop_list)
    print("Done !")

    pass


class loadMainBatches:

    def __init__(self, data_set='train'):
        self.data_pickle_paths = {'train': 'pickle/train/', 'valid': 'pickle/valid/'}
        self.data_pickle_dir = DATA + self.data_pickle_paths[data_set]
        self.batch_no = 0
        self.len = len(os.listdir(self.data_pickle_dir))
        pass

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        self.batch_no += 1
        data_path = self.data_pickle_dir + "dataMain" + str(self.batch_no) + ".p"
        if os.path.isfile(data_path):
            return loadPickle(data_path)
        else:
            raise StopIteration


def createMainDataBatches(data_set='train'):
    ''' Creates batches of unaltered data and pickles them with the following structure
        {id :{'image':numpyArray, 'shape':tupple, 'masks':numpyArray}, ...}
    '''

    print("Creating data batches for dataset: "+data_set)
    data_set_path = 'stage1_train/'
    data_dir = DATA + data_set_path
    data_pickle_paths = {'train': 'pickle/train/', 'valid': 'pickle/valid/'}
    data_pickle_dir = DATA + data_pickle_paths[data_set]
    if not os.path.exists(data_pickle_dir):
        os.makedirs(data_pickle_dir)
    img_id_list = os.listdir(data_dir)
    batch_size = 10
    image_list_size = len(img_id_list)
    if data_set == 'train':
        img_id_list = img_id_list[image_list_size//10:]
    elif data_set == 'valid':
        img_id_list = img_id_list[:image_list_size//10]
    image_list_size = len(img_id_list)
    data = dict()
    for i, img_id in enumerate(img_id_list):

        # reading image data
        path = data_dir + img_id + '/images/' + img_id + '.png'
        img = cv2.imread(path)
        img_shape = np.shape(img)
        mask_array = np.zeros((0,)+img_shape[:-1], np.float32)
        data[img_id] = {'image': img, 'shape': img_shape}

        # reading training masks
        mask_dir = data_dir + img_id + '/masks/'
        mask_id_list = os.listdir(mask_dir)

        for mask_id in mask_id_list:

            path = data_dir + img_id + '/masks/' + mask_id
            mask = cv2.imread(path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_array = np.append(mask_array, np.expand_dims(mask, axis=0), axis=0)

        data[img_id]['masks'] = mask_array

        if i % batch_size == 0:
            print("Pickeling data batches ... {} of {}".format(i//batch_size, image_list_size//batch_size))
            pickleData(data_pickle_dir+'dataMain'+str(i//batch_size)+'.p', data)
            data = dict()

            pass
    if len(list(data.keys())) > 0:
        pickleData(data_pickle_dir + 'dataMain' + str(i//batch_size) + '.p', data)
    print("Done !")


def loadDataN1(data_set='train'):
    ''' Data loader '''
    data_set_path = {'train': 'stage1_train/', 'test': 'stage1_test/', 'final': 'stage2_test'}
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


# Serialization and de-serialization code
def pickleData(fileName, obj):
    with open(fileName, 'wb') as pickleOut:
        pickle.dump(obj, pickleOut)


def loadPickle(fileName):
    with open(fileName,'rb') as pickleIn:
        return pickle.load(pickleIn)
