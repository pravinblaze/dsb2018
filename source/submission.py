'''
Fast inplementation of Run-Length Encoding algorithm
Takes only 200 seconds to process 5635 mask files
'''

import numpy as np
from PIL import Image
import os
from constants import DATA
import pandas as pd
from datamod import pickleData
from datamod import loadPickle


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev+1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


if __name__ == '__main__':
    input_path = DATA + 'results/final/'
    imgid_list = os.listdir(input_path)
    oids = os.listdir(DATA + 'stage2_test')
    encodings = []
    imgids = []
    submission_df = pd.DataFrame()

    if not os.path.isfile(DATA+'pickle/submission.p'):
        for i, imgid in enumerate(imgid_list):
            maskpath_list = os.listdir(input_path+imgid+'/')
            for j, maskpath in enumerate(maskpath_list):
                img = Image.open(input_path+imgid+'/'+maskpath)
                x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
                x = x // 255
                encodings.append(rle_encoding(x))
                imgids.append(imgid)
            print("Encoding {:.2f} %".format(100* i/len(imgid_list)))

        submission_df['ImageId'] = imgids
        submission_df['EncodedPixels'] = pd.Series(encodings).apply(lambda x: ' '.join(str(y) for y in x))
        pickleData(DATA + 'pickle/submission.p', submission_df)
    else:
        submission_df = loadPickle(DATA+'pickle/submission.p')
    missingids = set(oids).difference(set(np.unique(submission_df["ImageId"])))
    missingids = list(missingids)
    if not len(np.unique(submission_df["ImageId"])) == len(oids):
        print("Submission is not complete")
        print("Missing test ids: {0}".format(missingids))
    else:
        print("Submission is complete")
    submission_df['ImageId'].append(pd.Series(missingids))
    submission_df.to_csv(DATA+'results/submission.csv', index=False)
    pass
