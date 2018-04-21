'''
Fast inplementation of Run-Length Encoding algorithm
Takes only 200 seconds to process 5635 mask files
'''

import numpy as np
from PIL import Image
import os
from constants import DATA


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


if __name__ == '__main__':
    input_path = DATA + 'results/test-set/'
    imgid_list = os.listdir(input_path)

    encodings = []
    for i, imgid in enumerate(imgid_list):
        img = Image.open(os.path.join(input_path, imgid+''))
        x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
        x = x // 255
        encodings.append(rle_encoding(x))

    # check output
    conv = lambda l: ' '.join(map(str, l))  # list -> string
    subject, img = 1, 1
    print('\n{},{},{}'.format(subject, img, conv(encodings[0])))
