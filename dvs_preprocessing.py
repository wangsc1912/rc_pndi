'''
Accumulate the dvs videos into 5 frames
Checked the raw data processing script, the 1st axis is the time axis
'''

import os
import sys
import torch
from torch import nn
import numpy as np

from dvs_dataset import DvsDataset
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

start_f = 6
num_frame_split = 32
img_width = 28

'''DATA LOADING'''
train_dataset = DvsDataset(DATADIR='dataset/DVS_C10_TS1_1024', train=True, use_raw=False)
test_dataset = DvsDataset(DATADIR='dataset/DVS_C10_TS1_1024', train=False, use_raw=False)
trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=True)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

# save dir
SAVE_DIR = 'dataset/'
if os.path.exists(SAVE_DIR) is False:
    os.mkdir(SAVE_DIR)
# # save dir for 11 classes
for i in range(11):
    subdir = os.path.join(SAVE_DIR, str(i))
    if os.path.exists(subdir) is False:
        os.mkdir(subdir)
'''data pre-processing'''
# traversing raw dataset
for tp, Loader in zip(['tr', 'te'], [trainDataLoader, testDataLoader]):
    processed_dataset, labels = [], []
    for i, (data, label) in enumerate(Loader):
        '''choose part of the classes'''
        if label.item() in [0, 1, 2, 3, 4]:
            data = data[0]
            print(f'label: {label}, data shape: {data.shape}')

            # sort the points according to time
            cloud = data[data[:, 0].argsort()]  # the 0th axis is the time axis

            # normalize the position to [0, 28]
            cloud_coord = cloud[:, 1:]
            cloud_coord = (cloud_coord - cloud_coord.min()) / (cloud_coord.max() - cloud_coord.min()) * img_width
            cloud_coord = torch.where(cloud_coord == img_width, torch.tensor(img_width -1).to(cloud_coord.dtype), cloud_coord) 
            cloud[:, 1:] = cloud_coord.to(torch.int)

            cloud_new_idx = list(torch.split(cloud, int(1024 / num_frame_split)))
            images = torch.zeros((num_frame_split, img_width, img_width))
            for i in range(num_frame_split):
                cloud_new_idx[i] = torch.cat((torch.ones_like(cloud_new_idx[i].unsqueeze(-1))[:, 0] * i, cloud_new_idx[i][:, 1:]), dim=1)

            cloud_new_idx = torch.cat(cloud_new_idx, dim=0)
            cloud_new_idx = cloud_new_idx.to(torch.long)

            # put the points into images
            for point in cloud_new_idx:
                images[point[0], point[1], point[2]] = 1

            # for i in cloud_new_idx:
            images_5frame = images[start_f: :5][:5]

            # processed_dataset.append((cloud_new, label))
            processed_dataset.append(images_5frame)
            labels.append(label)

    # to tensor
    processed_dataset = torch.stack(processed_dataset)
    labels = torch.tensor(labels)
    torch.save((processed_dataset, labels), os.path.join(SAVE_DIR, f'dvs_proced_{tp}_5cls_w28_{num_frame_split}frame_s{start_f}i5.pt'))
